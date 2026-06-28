"""Seed the feed DB from the existing filtered corpus so it isn't empty on day one.

Loads feed_data/filtered_feed.json (kept, non-political posts produced by
filter_feed.py) and inserts them into the feed.db that the server reads. The
indexer will keep it fresh from there.

Usage:
    python seed_db.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import config
import db

SRC = config.DATA_DIR / "filtered_feed.json"


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found. Run `python filter_feed.py` first.")

    posts = json.loads(Path(SRC).read_text())["posts"]
    conn = db.connect()
    db.init_db(conn)

    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (p["uri"], "", p.get("created_at") or now, now, float(p.get("score", 0.0)))
        for p in posts
        if p.get("uri")
    ]
    db.insert_many(conn, rows)
    print(f"Seeded {len(rows)} posts into {config.DB_PATH} (db total {db.count(conn)}).")


if __name__ == "__main__":
    main()
