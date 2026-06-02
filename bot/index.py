"""Build a PageIndex tree from scraped markdown, then fill summaries reliably.

Two phases:
  1. PageIndex builds the day-node tree with NODE TEXT kept and summaries OFF
     (so PageIndex makes zero LLM summary calls — nothing to rate-limit/fail).
  2. We generate every node's summary ourselves via bot.llm.complete, which
     rotates all 9 Gemini keys + retries + waits. Sequential, retry-until-nonempty,
     fallback to raw text -> no node is ever left without a summary.

The saved workspace JSON keeps each node's `text` (needed for md retrieval) and
a generated `summary` / `prefix_summary` (used for SELECT routing).
"""
import asyncio
import glob
import json
import sys
import uuid

from bot import config
from bot.llm import complete

# PageIndex is cloned, not pip-installed -> add to path
sys.path.insert(0, str(config.ROOT / "PageIndex"))
from pageindex import PageIndexClient  # noqa: E402
from pageindex.page_index_md import md_to_tree  # noqa: E402
from pageindex.utils import count_tokens, print_tree  # noqa: E402

SUMMARY_TOKEN_THRESHOLD = 200   # below this, a node's own text IS its summary
SUMMARY_RETRIES = 6             # extra retries per node if model returns empty

SUMMARY_PROMPT = (config.ROOT / "node_summarizer.txt").read_text(encoding="utf-8")


def get_client() -> PageIndexClient:
    return PageIndexClient(
        model=config.INDEX_MODEL,
        retrieve_model=config.RAG_MODEL,
        workspace=str(config.WORKSPACE),
    )


def _iter_nodes(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_nodes(v)
    elif isinstance(obj, list):
        for i in obj:
            yield from _iter_nodes(i)


def _workspace_json(doc_id: str) -> str:
    for f in glob.glob(str(config.WORKSPACE / "*.json")):
        try:
            if doc_id in open(f, encoding="utf-8").read():
                return f
        except OSError:
            continue
    raise FileNotFoundError(f"workspace json for {doc_id} not found")


def _summarize_node(node: dict, channel: str) -> str:
    text = (node.get("text") or "").strip()
    if not text:
        return ""
    # Short nodes: the raw text is already the best "summary".
    if count_tokens(text, model=None) < SUMMARY_TOKEN_THRESHOLD:
        return text
    prompt = SUMMARY_PROMPT.format(channel=channel, text=text[:12000])
    for _ in range(SUMMARY_RETRIES):
        r = (complete(prompt, max_all_key_waits=30) or "").strip()
        if r:
            return r
    return text  # never empty: fall back to raw text


def migrate_summaries(new_doc_id: str, old_doc_id: str) -> None:
    """Copy summaries from old workspace JSON into new one where node title + text match.
    Call this after Phase 1 so unchanged nodes skip Gemini in fill_summaries."""
    try:
        old_path = _workspace_json(old_doc_id)
    except FileNotFoundError:
        return
    old_data = json.load(open(old_path, encoding="utf-8"))
    old_map = {
        n["title"]: n for n in _iter_nodes(old_data)
        if isinstance(n, dict) and n.get("node_id") and n.get("title")
    }

    new_path = _workspace_json(new_doc_id)
    new_data = json.load(open(new_path, encoding="utf-8"))
    migrated = 0
    for node in _iter_nodes(new_data):
        if not isinstance(node, dict) or not node.get("node_id"):
            continue
        old = old_map.get(node.get("title", ""))
        if not old:
            continue
        old_summ = (old.get("summary") or old.get("prefix_summary") or "").strip()
        if not old_summ:
            continue
        # Only migrate if text is identical (day didn't change)
        if old.get("text", "").strip() == node.get("text", "").strip():
            if node.get("nodes"):
                node["prefix_summary"] = old.get("prefix_summary", "")
            else:
                node["summary"] = old.get("summary", "")
            migrated += 1
    json.dump(new_data, open(new_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Migrated {migrated} unchanged summaries from old tree.")


def fill_summaries(doc_id: str, channel: str) -> None:
    path = _workspace_json(doc_id)
    data = json.load(open(path, encoding="utf-8"))
    nodes = [n for n in _iter_nodes(data)
             if isinstance(n, dict) and n.get("node_id")]
    print(f"Generating summaries for {len(nodes)} nodes (sequential, 9-key rotation)...")
    for i, node in enumerate(nodes, 1):
        existing = (node.get("summary") or node.get("prefix_summary") or "").strip()
        if existing:
            print(f"  [{i}/{len(nodes)}] {node['node_id']} {node.get('title','')} -> skip (already has summary)")
            continue
        summ = _summarize_node(node, channel)
        if node.get("nodes"):
            node["prefix_summary"] = summ
        else:
            node["summary"] = summ
        print(f"  [{i}/{len(nodes)}] {node['node_id']} {node.get('title','')} -> {len(summ)} chars")
        json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("All summaries written.")


def main():
    if not config.MESSAGES_MD.exists():
        sys.exit(f"{config.MESSAGES_MD} not found — run `python -m bot.scrape` first")

    # channel name from the markdown's first line: "# Slack channel: <name>"
    first = config.MESSAGES_MD.read_text(encoding="utf-8").splitlines()[0]
    channel = first.split(":", 1)[1].strip() if ":" in first else "the"

    print(f"Phase 1: building tree (summaries OFF, text kept) with {config.INDEX_MODEL} ...")
    # Call md_to_tree directly so we control the flags: no summary calls here,
    # keep node text. (client.index hardcodes summaries on.)
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

    doc_id = str(uuid.uuid4())
    client = get_client()
    client.documents[doc_id] = {
        "id": doc_id,
        "type": "md",
        "path": str(config.MESSAGES_MD),
        "doc_name": result.get("doc_name", ""),
        "doc_description": result.get("doc_description", ""),
        "line_count": result.get("line_count", 0),
        "structure": result["structure"],
    }
    client._save_doc(doc_id)
    config.DOC_ID_FILE.write_text(doc_id, encoding="utf-8")
    print(f"doc_id = {doc_id}")

    print("\nPhase 2: filling summaries ourselves (sequential, 9-key rotation) ...")
    fill_summaries(doc_id, channel)

    # reload + show tree
    structure = json.loads(get_client().get_document_structure(doc_id))
    print("\n--- Tree ---")
    print_tree(structure)


if __name__ == "__main__":
    main()
