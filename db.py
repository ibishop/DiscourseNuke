"""SQLite store for kept (non-political) posts.

The indexer process writes while the server process reads the same file, so we
enable WAL mode and a busy timeout to avoid 'database is locked'. Only posts that
pass the classifier (non-political) are ever stored.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS post (
    uri        TEXT PRIMARY KEY,   -- at://did/app.bsky.feed.post/rkey (dedup)
    did        TEXT NOT NULL,      -- author DID (a mutual)
    created_at TEXT NOT NULL,      -- record.createdAt (ISO); ordering + cursor
    indexed_at TEXT NOT NULL,      -- when we stored it (ISO, UTC)
    score      REAL NOT NULL       -- classifier margin at index time
);
CREATE INDEX IF NOT EXISTS idx_post_created_uri ON post (created_at DESC, uri DESC);
CREATE INDEX IF NOT EXISTS idx_post_created ON post (created_at);
"""


def connect(path: Path | str = config.DB_PATH) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def insert_many(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """rows: (uri, did, created_at, indexed_at, score). Dedup on uri."""
    if not rows:
        return
    conn.executemany(
        "INSERT OR IGNORE INTO post (uri, did, created_at, indexed_at, score) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def delete_post(conn: sqlite3.Connection, uri: str) -> None:
    conn.execute("DELETE FROM post WHERE uri = ?", (uri,))
    conn.commit()


def get_feed_page(
    conn: sqlite3.Connection, limit: int, cursor: str | None = None
) -> list[sqlite3.Row]:
    """Newest-first keyset page. Cursor format: '<created_at>::<uri>'."""
    if cursor and "::" in cursor:
        c_created, c_uri = cursor.split("::", 1)
        return conn.execute(
            "SELECT uri, created_at FROM post "
            "WHERE created_at < ? OR (created_at = ? AND uri < ?) "
            "ORDER BY created_at DESC, uri DESC LIMIT ?",
            (c_created, c_created, c_uri, limit),
        ).fetchall()
    return conn.execute(
        "SELECT uri, created_at FROM post ORDER BY created_at DESC, uri DESC LIMIT ?",
        (limit,),
    ).fetchall()


def prune(conn: sqlite3.Connection, older_than_iso: str) -> int:
    cur = conn.execute("DELETE FROM post WHERE created_at < ?", (older_than_iso,))
    conn.commit()
    return cur.rowcount


def count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM post").fetchone()[0]
