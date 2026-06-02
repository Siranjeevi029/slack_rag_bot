"""Incremental update: fetch only new Slack messages, rebuild tree, carry forward summaries.

Flow:
  1. Load last_scrape_ts from data/last_scrape_ts.txt
  2. Fetch only messages newer than that timestamp from Slack
  3. Append new messages to existing messages.md (insert into correct day sections)
  4. Fix any unclosed code fences (so PageIndex parses all headers)
  5. Rebuild tree (Phase 1 — fast, pure parsing)
  6. Migrate summaries from old tree for unchanged nodes (no Gemini needed)
  7. Fill summaries for new/changed nodes only (Gemini, sequential, key rotation)
  8. Update doc_id.txt + last_scrape_ts

Run:  python -m bot.update
"""
import asyncio
import re
import sys
import uuid

from bot import config
from bot.scrape import (
    _client, _retry, build_user_map, build_channel_map,
    fetch_new_messages, append_to_markdown,
    load_last_scrape_ts, save_last_scrape_ts,
)
from bot.index import (
    get_client, _iter_nodes, migrate_summaries, fill_summaries,
    SUMMARY_TOKEN_THRESHOLD,
)

sys.path.insert(0, str(config.ROOT / "PageIndex"))
from pageindex.page_index_md import md_to_tree  # noqa: E402
from pageindex.utils import print_tree  # noqa: E402


def fix_unclosed_fences(path) -> None:
    """Close unclosed ``` fences before ## day headers so PageIndex parses all nodes."""
    lines = open(path, encoding="utf-8").readlines()
    out = []
    in_code = False
    fixes = 0
    for line in lines:
        if line.rstrip("\n").startswith("## ") and in_code:
            out.append("```\n")
            in_code = False
            fixes += 1
        if line.strip().startswith("```"):
            in_code = not in_code
        out.append(line)
    open(path, "w", encoding="utf-8").writelines(out)
    if fixes:
        print(f"Fixed {fixes} unclosed code fences.")


def main():
    slack = _client()
    channel = config.SLACK_CHANNEL_ID
    if not channel:
        sys.exit("SLACK_CHANNEL_ID missing in .env")

    if not config.MESSAGES_MD.exists():
        sys.exit("messages.md not found — run `python -m bot.scrape` first for initial full scrape.")

    last_ts = load_last_scrape_ts()
    print(f"Last scrape ts: {last_ts} — fetching messages newer than that...")

    users = build_user_map(slack)
    chans = build_channel_map(slack)

    new_msgs = fetch_new_messages(slack, channel, oldest_ts=last_ts + 0.001 if last_ts else 0.0)
    print(f"New messages: {len(new_msgs)}")

    if not new_msgs:
        print("Nothing new. Exiting.")
        return

    affected_days = append_to_markdown(new_msgs, users, chans)
    print(f"Affected days: {affected_days}")

    fix_unclosed_fences(config.MESSAGES_MD)

    max_ts = max(float(m["ts"]) for m in new_msgs)

    # Read channel name
    first = config.MESSAGES_MD.read_text(encoding="utf-8").splitlines()[0]
    channel = first.split(":", 1)[1].strip() if ":" in first else "the"

    # Snapshot old doc_id before rebuilding
    old_doc_id = config.DOC_ID_FILE.read_text(encoding="utf-8").strip() \
        if config.DOC_ID_FILE.exists() else None

    print("Phase 1: rebuilding tree from updated messages.md ...")
    result = asyncio.run(md_to_tree(
        md_path=str(config.MESSAGES_MD),
        if_thinning=False,
        if_add_node_summary="no",
        summary_token_threshold=SUMMARY_TOKEN_THRESHOLD,
        model=config.INDEX_MODEL,
        if_add_doc_description="no",
        if_add_node_text="yes",
        if_add_node_id="yes",
    ))

    new_doc_id = str(uuid.uuid4())
    pi_client = get_client()
    pi_client.documents[new_doc_id] = {
        "id": new_doc_id,
        "type": "md",
        "path": str(config.MESSAGES_MD),
        "doc_name": result.get("doc_name", ""),
        "doc_description": "",
        "line_count": result.get("line_count", 0),
        "structure": result["structure"],
    }
    pi_client._save_doc(new_doc_id)
    config.DOC_ID_FILE.write_text(new_doc_id, encoding="utf-8")
    print(f"New doc_id = {new_doc_id}")

    # Migrate unchanged summaries from old tree → skip Gemini for those nodes
    if old_doc_id:
        migrate_summaries(new_doc_id, old_doc_id)

    print("Phase 2: filling summaries for new/changed nodes ...")
    fill_summaries(new_doc_id, channel)

    save_last_scrape_ts(max_ts)
    print(f"Done. last_scrape_ts updated to {max_ts}.")


if __name__ == "__main__":
    main()
