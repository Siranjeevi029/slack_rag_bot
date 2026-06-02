"""Scrape entire Slack channel history (messages + thread replies) to markdown.

Output is structured so PageIndex can build a useful tree:
  # Channel
  ## YYYY-MM-DD            (one node per day)
  **Name** [HH:MM]: text   (top-level message)
      ↳ **Name** [HH:MM]: text   (thread reply, indented)
"""
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

# Slack timestamps are UTC epoch. The team is in IST (UTC+5:30, no DST),
# so render times and group days in IST — otherwise both are shifted -5:30
# and late-night/early-morning messages fall into the wrong day bucket.
IST = timezone(timedelta(hours=5, minutes=30))

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from bot import config


def _client() -> WebClient:
    if not config.SCRAPE_BOT_TOKEN:
        sys.exit("SCRAPE_BOT_TOKEN / SLACK_BOT_TOKEN missing in .env")
    return WebClient(token=config.SCRAPE_BOT_TOKEN)


def _retry(fn, **kwargs):
    """Call a Slack API method, honoring 429 Retry-After."""
    while True:
        try:
            return fn(**kwargs)
        except SlackApiError as e:
            if e.response.status_code == 429:
                wait = int(e.response.headers.get("Retry-After", 5))
                print(f"  rate limited, sleeping {wait}s")
                time.sleep(wait)
                continue
            raise


def build_user_map(client: WebClient) -> dict:
    """id -> display name, via paginated users.list."""
    users = {}
    cursor = None
    while True:
        resp = _retry(client.users_list, limit=200, cursor=cursor)
        for u in resp["members"]:
            prof = u.get("profile", {})
            name = (prof.get("display_name") or prof.get("real_name")
                    or u.get("name") or u["id"])
            users[u["id"]] = name
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    print(f"Resolved {len(users)} users")
    return users


def build_channel_map(client: WebClient) -> dict:
    """id -> channel name, for resolving <#C..|> mentions."""
    chans = {}
    cursor = None
    while True:
        resp = _retry(client.conversations_list, limit=200, cursor=cursor,
                      types="public_channel,private_channel")
        for c in resp["channels"]:
            chans[c["id"]] = c.get("name", c["id"])
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return chans


def clean_text(text: str, users: dict, chans: dict) -> str:
    """Replace <@U..>, <#C..|>, <url|label> markup with readable text."""
    if not text:
        return ""
    text = re.sub(r"<@([A-Z0-9]+)>", lambda m: "@" + users.get(m.group(1), m.group(1)), text)
    text = re.sub(r"<#([A-Z0-9]+)\|([^>]*)>", lambda m: "#" + (m.group(2) or chans.get(m.group(1), m.group(1))), text)
    text = re.sub(r"<(https?://[^|>]+)\|([^>]+)>", r"\2 (\1)", text)
    text = re.sub(r"<(https?://[^>]+)>", r"\1", text)
    # Escape lines starting with '#' → prevent false markdown headers in PageIndex tree.
    text = re.sub(r"(?m)^(#+)", r"\\\1", text)
    # Close any unclosed triple-backtick fences → Slack uses ``` for inline code blocks
    # and sometimes omits the closing fence; PageIndex skips all ## headers inside fences.
    fence_count = text.count("```")
    if fence_count % 2 != 0:
        text = text + "\n```"
    return text.strip()


def fetch_all_messages(client: WebClient, channel: str) -> list:
    """All top-level messages, oldest-first, with thread replies attached."""
    messages = []
    cursor = None
    while True:
        resp = _retry(client.conversations_history, channel=channel,
                      limit=200, cursor=cursor)
        messages.extend(resp["messages"])
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(1)  # be gentle with tier-3 limits

    messages.sort(key=lambda m: float(m["ts"]))
    print(f"Fetched {len(messages)} top-level messages")

    # Attach thread replies
    for m in messages:
        if m.get("reply_count", 0) > 0:
            replies = _retry(client.conversations_replies,
                             channel=channel, ts=m["thread_ts"])["messages"]
            # first item is the parent itself -> skip
            m["_replies"] = replies[1:]
            time.sleep(1)
    return messages


def fmt_msg(m: dict, users: dict, chans: dict) -> str:
    who = users.get(m.get("user", ""), m.get("username", "unknown"))
    t = datetime.fromtimestamp(float(m["ts"]), tz=IST).strftime("%H:%M")
    body = clean_text(m.get("text", ""), users, chans)
    return f"**{who}** [{t}]: {body}" if body else f"**{who}** [{t}]: _(no text)_"


def to_markdown(messages: list, users: dict, chans: dict, channel_name: str) -> str:
    by_day = defaultdict(list)
    for m in messages:
        day = datetime.fromtimestamp(float(m["ts"]), tz=IST).strftime("%Y-%m-%d")
        by_day[day].append(m)

    out = [f"# Slack channel: {channel_name}\n"]
    for day in sorted(by_day):
        out.append(f"\n## {day}\n")
        for m in by_day[day]:
            out.append(fmt_msg(m, users, chans))
            for r in m.get("_replies", []):
                out.append("    ↳ " + fmt_msg(r, users, chans))
    return "\n".join(out) + "\n"


def load_last_scrape_ts() -> float:
    if config.LAST_SCRAPE_FILE.exists():
        try:
            return float(config.LAST_SCRAPE_FILE.read_text(encoding="utf-8").strip())
        except ValueError:
            pass
    return 0.0


def save_last_scrape_ts(ts: float) -> None:
    config.LAST_SCRAPE_FILE.write_text(str(ts), encoding="utf-8")


def fetch_new_messages(client: WebClient, channel: str, oldest_ts: float) -> list:
    """Fetch only messages newer than oldest_ts, with thread replies."""
    messages = []
    cursor = None
    while True:
        resp = _retry(client.conversations_history, channel=channel,
                      limit=200, cursor=cursor, oldest=str(oldest_ts))
        messages.extend(resp["messages"])
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(1)

    messages.sort(key=lambda m: float(m["ts"]))

    for m in messages:
        if m.get("reply_count", 0) > 0:
            replies = _retry(client.conversations_replies,
                             channel=channel, ts=m["thread_ts"])["messages"]
            m["_replies"] = replies[1:]
            time.sleep(1)
    return messages


def append_to_markdown(new_messages: list, users: dict, chans: dict) -> list[str]:
    """Merge new messages into existing messages.md. Returns list of affected days."""
    if not new_messages:
        return []

    # Group new messages by IST day, skip system messages
    by_day: dict[str, list] = defaultdict(list)
    for m in new_messages:
        if m.get("type") != "message" or m.get("subtype") in (
                "channel_join", "channel_leave", "bot_message"):
            continue
        if not (m.get("text") or m.get("_replies")):
            continue
        day = datetime.fromtimestamp(float(m["ts"]), tz=IST).strftime("%Y-%m-%d")
        by_day[day].append(m)

    if not by_day:
        return []

    content = config.MESSAGES_MD.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)

    for day in sorted(by_day.keys()):
        new_lines = []
        for m in by_day[day]:
            new_lines.append(fmt_msg(m, users, chans) + "\n")
            for r in m.get("_replies", []):
                new_lines.append("    ↳ " + fmt_msg(r, users, chans) + "\n")

        day_header = f"## {day}"
        day_idx = next((i for i, l in enumerate(lines)
                        if l.rstrip("\n") == day_header), None)

        if day_idx is not None:
            # Existing day: find end of section (next ## or EOF), insert before it
            end_idx = len(lines)
            for i in range(day_idx + 1, len(lines)):
                if lines[i].startswith("## "):
                    end_idx = i
                    break
            lines = lines[:end_idx] + new_lines + lines[end_idx:]
        else:
            # New day: append section at end
            if lines and not lines[-1].endswith("\n"):
                lines.append("\n")
            lines.append(f"\n{day_header}\n\n")
            lines.extend(new_lines)

    config.MESSAGES_MD.write_text("".join(lines), encoding="utf-8")
    return sorted(by_day.keys())


def main():
    client = _client()
    channel = config.SLACK_CHANNEL_ID
    if not channel:
        sys.exit("SLACK_CHANNEL_ID missing in .env")

    info = _retry(client.conversations_info, channel=channel)["channel"]
    channel_name = info.get("name", channel)
    print(f"Scraping #{channel_name} ({channel})")

    users = build_user_map(client)
    chans = build_channel_map(client)
    messages = fetch_all_messages(client, channel)
    md = to_markdown(messages, users, chans, channel_name)

    config.MESSAGES_MD.write_text(md, encoding="utf-8")
    if messages:
        save_last_scrape_ts(max(float(m["ts"]) for m in messages))
    print(f"Wrote {config.MESSAGES_MD}  ({len(md)} chars, {len(messages)} msgs)")


if __name__ == "__main__":
    main()
