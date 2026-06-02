# Slack Channel RAG Bot

Scrapes a Slack channel's full history, indexes it with **PageIndex**
(vectorless, reasoning-based RAG), and answers questions in Slack via **Gemini 2.5 Flash**.

## Architecture

```
Slack channel ──scrape──▶ data/messages.md ──PageIndex(Gemini)──▶ tree (workspace)
                                                                        │
@mention / DM ──▶ RAGBot: SELECT day-nodes ─▶ fetch text ─▶ Gemini answers ─▶ reply in thread
```

## Stack

| Part            | Tool                                          |
|-----------------|-----------------------------------------------|
| Scrape + bot    | `slack-bolt` (Socket Mode)                    |
| Index / RAG     | PageIndex (cloned in `PageIndex/`)            |
| LLM             | Gemini 2.5 Flash via litellm (up to 16 keys) |
| Key rotation    | `bot/llm.py` — rotates all keys, retries 503 |

## Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install litellm pymupdf PyPDF2 pyyaml slack-bolt websocket-client
git clone https://github.com/VectifyAI/PageIndex.git
```

### `.env` required keys

```
SLACK_BOT_TOKEN=xoxb-...          # slack_rag_bot token
SLACK_APP_TOKEN=xapp-...          # for Socket Mode
SLACK_CHANNEL_ID=C...             # target channel
SCRAPE_BOT_TOKEN=xoxb-...        # bot that is a member of the channel

GEMINI_API_KEY_1=AIza...
GEMINI_API_KEY_2=AIza...
# ... up to GEMINI_API_KEY_16
```

## Commands

```powershell
# Full initial scrape + index (first time only)
.venv\Scripts\python.exe -m bot.scrape
.venv\Scripts\python.exe -m bot.index

# Incremental update (new messages only, carries forward old summaries)
.venv\Scripts\python.exe -m bot.update

# Test RAG in terminal
.venv\Scripts\python.exe -m bot.rag

# Run live Slack bot
.venv\Scripts\python.exe -m bot.app
```

## Bot behaviour

- `@mention` in channel → replies in a new thread
- Any reply in a thread the bot participated in → bot responds (no mention needed)
- DM → always replies
- Thread context passed to Gemini for follow-up awareness
- IST timezone awareness — resolves "today", "yesterday", "20 days ago"

## Prompts (editable, no code change needed)

| File | Purpose |
|------|---------|
| `codex_prompt.txt` | SELECT (node routing) + ANSWER + ANSWER_THREAD |
| `node_summarizer.txt` | Summary prompt used during indexing |

## Incremental update flow

```
bot.update:
  1. Fetch only new messages since last_scrape_ts
  2. Append to messages.md (correct day sections)
  3. Fix unclosed code fences
  4. Rebuild tree (Phase 1 — pure parsing, seconds)
  5. Migrate summaries from old tree (unchanged nodes = 0 Gemini calls)
  6. Fill summaries for new/changed days only
  7. Save new doc_id + last_scrape_ts
```

## Key rotation

All 16 Gemini keys rotated automatically on TPM/RPM/503 errors.
Saves per-node during indexing — safe to stop and resume anytime.

## Notes

- `bot.index` skip-existing logic: re-running is safe, already-summarised nodes are skipped
- messages.md grouped by IST day (one PageIndex node per day)
- Unclosed code fence bug in Slack messages auto-fixed during scrape and update
