"""Central config: loads .env, exposes tokens + model names."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# override=True so .env wins over any stale OS/shell env vars (e.g. a leftover
# SLACK_BOT_TOKEN from another app shadowing the one in .env).
load_dotenv(ROOT / ".env", override=True)

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "").strip()
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "").strip()
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID", "").strip()
# Token for scraping only; falls back to the live bot token if not set.
SCRAPE_BOT_TOKEN = os.getenv("SCRAPE_BOT_TOKEN", "").strip() or SLACK_BOT_TOKEN
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Primary + backup Groq keys, in rotation order. Empty ones dropped.
GROQ_KEYS = [k.strip() for k in (
    os.getenv("GROQ_API_KEY", ""),
    os.getenv("GROQ_API_KEY_1", ""),
    os.getenv("GROQ_API_KEY_2", ""),
    os.getenv("GROQ_API_KEY_3", ""),
) if k.strip()]

# Gemini key pool: 8 backup keys + 1 dedicated scrape key = 9 total.
# All are used in rotation (indexing + runtime) to spread RPM across keys.
GEMINI_KEYS = [k.strip() for k in (
    os.getenv("GEMINI_API_KEY_1", ""),
    os.getenv("GEMINI_API_KEY_2", ""),
    os.getenv("GEMINI_API_KEY_3", ""),
    os.getenv("GEMINI_API_KEY_4", ""),
    os.getenv("GEMINI_API_KEY_5", ""),
    os.getenv("GEMINI_API_KEY_6", ""),
    os.getenv("GEMINI_API_KEY_7", ""),
    os.getenv("GEMINI_API_KEY_8", ""),
    os.getenv("GEMINI_API_KEY_9", ""),
    os.getenv("GEMINI_API_KEY_10", ""),
    os.getenv("GEMINI_API_KEY_11", ""),
    os.getenv("GEMINI_API_KEY_12", ""),
    os.getenv("GEMINI_API_KEY_13", ""),
    os.getenv("GEMINI_API_KEY_14", ""),
    os.getenv("GEMINI_API_KEY_15", ""),
    os.getenv("GEMINI_API_KEY_16", ""),
) if k.strip()]
# PageIndex indexing calls litellm directly and reads GEMINI_API_KEY from env.
if GEMINI_KEYS:
    os.environ["GEMINI_API_KEY"] = GEMINI_KEYS[0]

# Seconds to wait when ALL keys have hit their TPM/RPM limit
ALL_KEYS_TPM_WAIT = 60
# Retries for Gemini 503 "model overloaded / high demand" on a single key
GEMINI_503_RETRIES = 5

# litellm-style model strings (Gemini)
INDEX_MODEL = os.getenv("INDEX_MODEL", "gemini/gemini-2.5-flash").strip()
RAG_MODEL = os.getenv("RAG_MODEL", "gemini/gemini-2.5-flash").strip()

# Paths
DATA_DIR = ROOT / "data"
MESSAGES_MD = DATA_DIR / "messages.md"
WORKSPACE = DATA_DIR / "pageindex_workspace"
DOC_ID_FILE = DATA_DIR / "doc_id.txt"
LAST_SCRAPE_FILE = DATA_DIR / "last_scrape_ts.txt"
PROMPT_FILE = ROOT / "codex_prompt.txt"

DATA_DIR.mkdir(exist_ok=True)
