"""Parse messages.md into thread-based chunks for hybrid RAG.

Chunk = top-level Slack message + all its ↳ replies = 1 semantic unit.
Multi-line message bodies are joined (fixes silent truncation in old per-line parser).

Run standalone to inspect: python -m bot.chunk
"""
import hashlib
import re
from pathlib import Path

from bot import config

MSG_RE = re.compile(r"^\*\*(.+?)\*\* \[(\d{2}:\d{2})\]: (.*)$")
REPLY_RE = re.compile(r"^    ↳ \*\*(.+?)\*\* \[(\d{2}:\d{2})\]: (.*)$")


def _chunk_id(day: str, time: str, user: str) -> str:
    return hashlib.md5(f"{day}|{time}|{user}".encode()).hexdigest()[:12]


def parse_chunks(md_path: Path) -> list[dict]:
    """Return list of chunk dicts.

    Each dict: {chunk_id, day, start_time, end_time, users, is_thread, text, msg_count}
    text = full concatenated content shown to the LLM.
    """
    lines = md_path.read_text(encoding="utf-8").splitlines()
    chunks: list[dict] = []
    current_day: str | None = None

    # mutable state for the message being assembled
    cur_user: str | None = None
    cur_time: str | None = None
    cur_lines: list[str] = []
    cur_replies: list[dict] = []  # each: {user, time, lines[]}
    cur_target: str | None = None  # "msg" | "reply"

    def flush():
        if cur_user is None or current_day is None:
            return
        body = " ".join(cur_lines).strip()
        parts = [f"**{cur_user}** [{current_day} {cur_time}]: {body}"]
        users = [cur_user]
        end_time = cur_time
        # structured per-message records (index 0 = thread parent)
        msgs = [{"user": cur_user, "time": cur_time, "text": body, "is_reply": False}]
        for r in cur_replies:
            rbody = " ".join(r["lines"]).strip()
            parts.append(f"  ↳ **{r['user']}** [{current_day} {r['time']}]: {rbody}")
            if r["user"] not in users:
                users.append(r["user"])
            end_time = r["time"]
            msgs.append({"user": r["user"], "time": r["time"], "text": rbody, "is_reply": True})
        chunks.append({
            "chunk_id":   _chunk_id(current_day, cur_time, cur_user),
            "day":        current_day,
            "start_time": cur_time,
            "end_time":   end_time,
            "users":      users,
            "is_thread":  len(cur_replies) > 0,
            "text":       "\n".join(parts),
            "messages":   msgs,
            "msg_count":  1 + len(cur_replies),
        })

    for line in lines:
        # Day header
        if line.startswith("## "):
            flush()
            cur_user = cur_time = cur_target = None
            cur_lines = []
            cur_replies = []
            current_day = line[3:].strip()
            continue

        if current_day is None:
            continue

        # Thread reply
        m = REPLY_RE.match(line)
        if m:
            cur_replies.append({"user": m.group(1), "time": m.group(2), "lines": [m.group(3)]})
            cur_target = "reply"
            continue

        # Top-level message
        m = MSG_RE.match(line)
        if m:
            flush()
            cur_user, cur_time = m.group(1), m.group(2)
            cur_lines = [m.group(3)]
            cur_replies = []
            cur_target = "msg"
            continue

        # Continuation line (multi-line body)
        stripped = line.strip()
        if stripped:
            if cur_target == "msg" and cur_user:
                cur_lines.append(stripped)
            elif cur_target == "reply" and cur_replies:
                cur_replies[-1]["lines"].append(stripped)

    flush()
    return chunks


if __name__ == "__main__":
    chunks = parse_chunks(config.MESSAGES_MD)
    threads = sum(1 for c in chunks if c["is_thread"])
    total_msgs = sum(c["msg_count"] for c in chunks)
    print(f"Chunks : {len(chunks)}")
    print(f"Threads: {threads}  Standalone: {len(chunks) - threads}")
    print(f"Total messages covered: {total_msgs}")
    print(f"\nSample chunk:\n{chunks[10]['text'][:400]}")
