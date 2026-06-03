"""Slack bot (Socket Mode).

Behaviour:
  - @mention in a channel        -> reply in a thread (handled by app_mention event).
  - @mention inside a thread      -> reply in that thread with thread context.
  - Non-mention reply in a thread the bot already participated in -> reply there
    with the whole thread as context (handled by message event).
  - DM                            -> always reply.

Thread "tracking" is stateless: fetch the thread and check whether the bot's own
user id appears in it. The thread itself is the memory (stored in Slack).
"""
import re

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from bot import config
from bot.rag import RAGBot

app = App(token=config.SLACK_BOT_TOKEN)
bot = RAGBot()  # load tree once at startup
BOT_USER_ID = app.client.auth_test()["user_id"]

MENTION_RE = re.compile(r"<@[A-Z0-9]+>")
_name_cache: dict[str, str] = {}


def _name(client, uid: str) -> str:
    if uid == BOT_USER_ID:
        return "Assistant"
    if uid in _name_cache:
        return _name_cache[uid]
    try:
        prof = client.users_info(user=uid)["user"]["profile"]
        nm = prof.get("display_name") or prof.get("real_name") or uid
    except Exception:  # noqa: BLE001
        nm = uid
    _name_cache[uid] = nm
    return nm


def _clean(text: str) -> str:
    return MENTION_RE.sub("", text or "").strip()


def _resolve(client, text: str) -> str:
    """Replace <@USERID> with display names; strip bot self-mention only."""
    def replacer(m):
        uid = m.group(0)[2:-1]  # strip <@ and >
        if uid == BOT_USER_ID:
            return ""
        return _name(client, uid)
    return MENTION_RE.sub(replacer, text or "").strip()


def _format_thread(client, messages: list) -> str:
    lines = []
    for m in messages:
        if m.get("subtype"):
            continue
        body = _resolve(client, m.get("text", ""))
        if body:
            lines.append(f"{_name(client, m.get('user', ''))}: {body}")
    return "\n".join(lines)


def _reply(text: str, thread_text: str | None = None, client=None) -> str:
    q = _resolve(client, text) if client else _clean(text)
    if not q and not thread_text:
        return "Ask me something about the channel history. e.g. *what blogs are pending?*"
    try:
        return bot.ask(q, thread_text=thread_text)
    except Exception as e:  # noqa: BLE001
        return f"Error answering: {e}"


def _send(say, text: str, thread_ts: str | None = None) -> None:
    """Send answer, splitting into chunks if > 3900 chars to avoid Slack limits."""
    limit = 3900
    kwargs = {"thread_ts": thread_ts} if thread_ts else {}
    if len(text) <= limit:
        say(text=text, **kwargs)
        return
    # try paragraph breaks, then line breaks, then hard-split
    parts = []
    current = ""
    separators = ["\n\n", "\n"]
    lines = text.split("\n")
    for line in lines:
        candidate = (current + "\n" + line) if current else line
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                parts.append(current)
            # if single line itself > limit, hard split it
            while len(line) > limit:
                parts.append(line[:limit])
                line = line[limit:]
            current = line
    if current:
        parts.append(current)
    for part in parts:
        say(text=part, **kwargs)


def _reply_in_thread(client, channel: str, thread_ts: str, text: str, say):
    replies = client.conversations_replies(channel=channel, ts=thread_ts).get("messages", [])
    thread_text = _format_thread(client, replies)
    _send(say, _reply(text, thread_text=thread_text, client=client), thread_ts=thread_ts)


def _bot_ids(context) -> set:
    """All ids that count as 'the bot': what we post as + what we're mentioned as."""
    ids = {BOT_USER_ID}
    if context and context.get("bot_user_id"):
        ids.add(context["bot_user_id"])
    return ids


@app.event("app_mention")
def on_mention(event, say, client):
    print(f"[app_mention] channel={event.get('channel')} thread={event.get('thread_ts')}")
    channel = event["channel"]
    thread_ts = event.get("thread_ts")
    if thread_ts:
        _reply_in_thread(client, channel, thread_ts, event.get("text", ""), say)
    else:
        _send(say, _reply(event.get("text", ""), client=client), thread_ts=event["ts"])


@app.event("message")
def on_message(event, say, client, context):
    # ignore bot/system messages, edits, joins, file-shares, etc.
    if event.get("bot_id") or event.get("subtype"):
        return
    bot_ids = _bot_ids(context)
    user = event.get("user")
    if not user or user in bot_ids:
        return

    text = event.get("text", "")
    print(f"[message] user={user} thread={event.get('thread_ts')} text={text[:50]!r}")

    # Mentions are handled by the app_mention event -> avoid double reply
    if any(f"<@{bid}>" in text for bid in bot_ids):
        return

    # Direct message -> always answer
    if event.get("channel_type") == "im":
        _send(say, _reply(text, client=client))
        return

    # Non-mention reply in a thread -> answer only if bot already participated
    thread_ts = event.get("thread_ts")
    if thread_ts:
        replies = client.conversations_replies(channel=event["channel"], ts=thread_ts).get("messages", [])
        if any(m.get("user") in bot_ids for m in replies):
            thread_text = _format_thread(client, replies)
            _send(say, _reply(text, thread_text=thread_text, client=client), thread_ts=thread_ts)


def main():
    print(f"Bot starting (Socket Mode), bot user = {BOT_USER_ID} ... Ctrl+C to stop.")
    SocketModeHandler(app, config.SLACK_APP_TOKEN).start()


if __name__ == "__main__":
    main()
