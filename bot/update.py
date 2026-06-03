"""Incremental update: fetch new Slack messages, rebuild FAISS index.

Flow:
  1. Load last_scrape_ts
  2. Fetch messages newer than that from Slack
  3. Append to messages.md (insert into correct day sections)
  4. Fix unclosed code fences
  5. Rebuild FAISS index from scratch (guarantees no stale chunks)
  6. Save last_scrape_ts

Run:  python -m bot.update
"""
import sys

from bot import config
from bot.scrape import (
    _client, build_user_map, build_channel_map,
    fetch_new_messages, append_to_markdown,
    load_last_scrape_ts, save_last_scrape_ts,
)


def fix_unclosed_fences(path) -> None:
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
        sys.exit("messages.md not found — run `python -m bot.scrape` + `python -m bot.embed` first.")

    last_ts = load_last_scrape_ts()
    print(f"Last scrape ts: {last_ts} — fetching newer messages...")

    users = build_user_map(slack)
    chans = build_channel_map(slack)
    new_msgs = fetch_new_messages(slack, channel,
                                  oldest_ts=last_ts + 0.001 if last_ts else 0.0)
    print(f"New messages: {len(new_msgs)}")

    if not new_msgs:
        print("Nothing new. Exiting.")
        return

    affected_days = append_to_markdown(new_msgs, users, chans)
    print(f"Affected days: {affected_days}")
    fix_unclosed_fences(config.MESSAGES_MD)

    max_ts = max(float(m["ts"]) for m in new_msgs)

    # Rebuild full FAISS index — guarantees threads with new replies are re-embedded
    print("Rebuilding FAISS index from updated messages.md...")
    from bot.embed import main as embed_main
    embed_main()

    save_last_scrape_ts(max_ts)
    print("Update complete.")


if __name__ == "__main__":
    main()
