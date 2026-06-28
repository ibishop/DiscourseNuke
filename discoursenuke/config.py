"""Central config for the DiscourseNuke feed generator.

Reads from .env / environment (same dotenv pattern as auth.py) with sensible
defaults so local testing works with minimal setup. The only value you must set
once you have a tunnel is FEEDGEN_HOSTNAME.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Identity / hosting -----------------------------------------------------
# Public hostname the tunnel exposes (no scheme), e.g. "abc.trycloudflare.com".
FEEDGEN_HOSTNAME = os.getenv("FEEDGEN_HOSTNAME", "example.invalid")
# Service DID — defaults to did:web of the hostname.
FEEDGEN_SERVICE_DID = os.getenv("FEEDGEN_SERVICE_DID", f"did:web:{FEEDGEN_HOSTNAME}")
# The DID of the account that publishes the feed record (set after publishing).
FEEDGEN_PUBLISHER_DID = os.getenv("FEEDGEN_PUBLISHER_DID", "")

SERVICE_ENDPOINT = f"https://{FEEDGEN_HOSTNAME}"

# --- Feed record metadata ---------------------------------------------------
FEED_RKEY = os.getenv("FEED_RKEY", "discoursenuke")
FEED_DISPLAY_NAME = os.getenv("FEED_DISPLAY_NAME", "Mutuals minus US politics")
FEED_DESCRIPTION = os.getenv(
    "FEED_DESCRIPTION",
    "Posts from my Bluesky mutuals, with US political discourse filtered out.",
)

# --- Indexing ---------------------------------------------------------------
JETSTREAM_URL = os.getenv("JETSTREAM_URL", "wss://jetstream1.us-east.bsky.network/subscribe")
# Public Jetstream hosts to round-robin on reconnect.
JETSTREAM_HOSTS = [
    "jetstream1.us-east.bsky.network",
    "jetstream2.us-east.bsky.network",
    "jetstream1.us-west.bsky.network",
    "jetstream2.us-west.bsky.network",
]
CLASSIFIER_THRESHOLD = float(os.getenv("CLASSIFIER_THRESHOLD", "0.08"))
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "3"))
INDEX_REPLIES = os.getenv("INDEX_REPLIES", "true").lower() == "true"

# --- Storage ----------------------------------------------------------------
DATA_DIR = Path("feed_data")
DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "feed.db")))
MUTUALS_PATH = DATA_DIR / "mutuals.json"
CURSOR_PATH = DATA_DIR / "jetstream_cursor.txt"

# --- Server -----------------------------------------------------------------
PORT = int(os.getenv("PORT", "8080"))


def feed_uri() -> str:
    """at:// URI of the published generator record (needs publisher DID)."""
    return f"at://{FEEDGEN_PUBLISHER_DID}/app.bsky.feed.generator/{FEED_RKEY}"


def load_mutual_dids() -> list[str]:
    """Return the list of mutual DIDs from feed_data/mutuals.json."""
    if not MUTUALS_PATH.exists():
        raise SystemExit(
            f"{MUTUALS_PATH} not found. Run "
            "`python -m discoursenuke.pipeline.fetch_mutuals` first to build the "
            "mutuals list."
        )
    data = json.loads(MUTUALS_PATH.read_text())
    return list(data.keys())
