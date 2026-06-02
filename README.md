# Slack Channel RAG Bot

Scrapes a Slack channel's full history, indexes it with **PageIndex**
(vectorless, reasoning-based RAG), and answers questions in Slack via **Groq** (free).

## Architecture

```
Slack channel ──scrape──▶ data/messages.md ──PageIndex(Groq)──▶ tree (workspace)
                                                                      │
@mention / DM ──▶ RAGBot: Groq picks day-nodes ─▶ fetch text ─▶ Groq answers ─▶ reply
```

## Stack

| Part            | Tool                                   |
|-----------------|----------------------------------------|
| Scrape + bot    | `slack-bolt` (Socket Mode)             |
| Index / RAG     | PageIndex (cloned in `PageIndex/`)     |
| LLM             | Groq via litellm (`groq/llama-3.3-70b-versatile`) |

## Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install litellm pymupdf PyPDF2 pyyaml slack-bolt websocket-client
```

`.env` (already present):
```
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
SLACK_CHANNEL_ID=C...
GROQ_API_KEY=gsk_...
```

Bot must be **invited to the channel** to read history.

## Run (in order)

```powershell
# 1. scrape channel -> data/messages.md
.venv\Scripts\python.exe -m bot.scrape

# 2. build PageIndex tree (Groq)
.venv\Scripts\python.exe -m bot.index

# 3a. test RAG in terminal
.venv\Scripts\python.exe -m bot.rag

# 3b. OR run the Slack bot
.venv\Scripts\python.exe -m bot.app
```

Then in Slack: `@bot what blogs are pending?` or DM the bot.

## Notes

- Groq free tier = 12k tokens/min; `bot/rag.py` backs off on rate limits.
- Re-run `scrape` + `index` to refresh after new channel activity.
- Markdown is grouped by day (one PageIndex node per day).
