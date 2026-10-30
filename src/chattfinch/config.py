"""Paths, region settings, and API keys loaded from .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # regenerable API pulls, not committed
MANUAL_DIR = DATA_DIR / "manual"  # hand-curated inputs, committed
DUMPS_DIR = DATA_DIR / "dumps"  # published artifacts, committed
PRIVATE_DIR = DATA_DIR / "private"  # not redistributable (IUCN terms), never committed
INDEX_DIR = DATA_DIR / "index"  # Chroma vector store, rebuilt by scripts/build_index.py
CORPUS_PATH = DUMPS_DIR / "corpus.jsonl.gz"  # what the index is built from

REGION_CODE = "KZ"
REGION_NAME = "Kazakhstan"

# Optional ranking of the species list by eBird reporting frequency. Off by default:
# eBird coverage of KZ is too sparse for frequencies to mean much. To enable it
# (e.g. for a well-covered region), download the region's bar-chart data from the
# eBird website, save it under data/manual/, and point this at the file. When set,
# the list is cut to the top TARGET_SPECIES_COUNT by frequency (plus notable
# species); when None, every species on the region's eBird list is kept.
FREQUENCY_FILE: Path | None = None
TARGET_SPECIES_COUNT = 200

EBIRD_API_KEY = os.getenv("EBIRD_API_KEY")
IUCN_API_TOKEN = os.getenv("IUCN_API_TOKEN")
WIKIPEDIA_USER_AGENT = os.getenv("WIKIPEDIA_USER_AGENT", "Chattfinch/0.1")

# --- Retrieval index ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # input limit: 256 tokens
CHUNK_TOKENS = 200  # cap per chunk incl. header and special tokens; limit leaves headroom
CHUNK_OVERLAP_TOKENS = 30  # a chunk's last sentence repeats in the next if this short
INDEX_COLLECTION = "birds"

# --- Answer generation ---
RETRIEVAL_K = 6  # chunks shown to the model per question (~1,200 tokens of context)
# Haiku is the cheapest current Claude model ($0.10 in / $0.50 out per million tokens),
# chosen to keep costs down; a question costs a small fraction of a cent.
LLM_MODEL = "claude-haiku-5-5"
LLM_EFFORT = "medium"  # low | medium | high | xhigh | max: thoroughness vs. cost and speed

# --- Query routing (day 4) ---
# Live "seen recently" answers: how far back to ask eBird (its maximum is 30 days),
# how many sightings to list, and how long to wait. One retry only: a chat reply
# can't sit through the batch client's backoff.
RECENT_DAYS = 30
RECENT_SIGHTINGS_SHOWN = 5
LIVE_TIMEOUT_S = 8.0
# How long the bot keeps a live answer: enough for its "Show passages" button and a
# quick repeat, then the next ask fetches fresh sightings.
LIVE_CACHE_SECONDS = 600
IUCN_SNAPSHOT_PATH = PRIVATE_DIR / "iucn_snapshot.json"

# --- Telegram bot ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # from @BotFather; a secret
# Comma-separated numeric Telegram user IDs allowed to ask questions. Everyone else
# gets a "this bot is private" reply and no model call. Empty = nobody is allowed.
TELEGRAM_ALLOWED_USER_IDS = os.getenv("TELEGRAM_ALLOWED_USER_IDS", "")
# A second guard on spending: at most this many new (uncached) questions per user
# in this many seconds.
TELEGRAM_RATE_LIMIT_QUESTIONS = 20
TELEGRAM_RATE_LIMIT_SECONDS = 3600
TELEGRAM_MAX_QUESTION_CHARS = 500

# Storage allotment for the committed dumps in data/dumps/. The server has about
# 15 GB in total, shared with the vector store (~70 MB), the raw cache, the model
# download and the app, so this check is a loose guard against a runaway dump, not a
# tight limit. The dumps are ~2 MB today.
DUMPS_BUDGET_MB = 15_000

# The frozen target species list. Committed; build_dumps.py reads it instead of
# calling eBird, so the list only changes when you run it with --refresh-species-list.
SPECIES_LIST_PATH = DUMPS_DIR / "species_list.csv"
