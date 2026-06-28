"""Jetstream indexer: stream mutuals' posts, drop US politics, store the rest.

Connects to Bluesky's Jetstream over a websocket, filters server-side to
app.bsky.feed.post records from our mutuals (via wantedDids), classifies each
post, and persists only the non-political ones to SQLite. The server reads from
that same DB.

Run:  python -m indexer   (or: python indexer.py)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import websockets

import config
import db
from classifier import PoliticalClassifier

BATCH_SIZE = 32
FLUSH_INTERVAL = 1.0   # seconds
PRUNE_INTERVAL = 3600  # seconds


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_cursor() -> str | None:
    if config.CURSOR_PATH.exists():
        return config.CURSOR_PATH.read_text().strip() or None
    return None


def _write_cursor(time_us: int) -> None:
    config.CURSOR_PATH.write_text(str(time_us))


def _parse_commit(event: dict) -> tuple[str, str, str, str] | None:
    """Return (uri, did, text, created_at) for an indexable post create, else None."""
    if event.get("kind") != "commit":
        return None
    commit = event.get("commit") or {}
    if commit.get("operation") != "create":
        return None
    if commit.get("collection") != "app.bsky.feed.post":
        return None
    record = commit.get("record") or {}
    text = (record.get("text") or "").strip()
    if not text:
        return None
    if record.get("reply") and not config.INDEX_REPLIES:
        return None
    did = event["did"]
    uri = f"at://{did}/app.bsky.feed.post/{commit['rkey']}"
    created_at = record.get("createdAt") or _utcnow_iso()
    return uri, did, text, created_at


def _handle_delete(conn, event: dict) -> None:
    commit = event.get("commit") or {}
    if commit.get("operation") == "delete" and commit.get("collection") == "app.bsky.feed.post":
        uri = f"at://{event['did']}/app.bsky.feed.post/{commit['rkey']}"
        db.delete_post(conn, uri)


def _flush(conn, clf: PoliticalClassifier, batch: list[tuple]) -> int:
    """batch: list of (uri, did, text, created_at). Returns # kept (stored)."""
    if not batch:
        return 0
    verdicts = clf.classify_many([b[2] for b in batch])
    now = _utcnow_iso()
    rows = [
        (uri, did, created_at, now, round(v.score, 4))
        for (uri, did, _text, created_at), v in zip(batch, verdicts)
        if not v.is_political
    ]
    db.insert_many(conn, rows)
    return len(rows)


async def _consume(conn, clf, mutual_dids: list[str]) -> None:
    host = config.JETSTREAM_URL
    url = f"{host}?wantedCollections=app.bsky.feed.post"
    cursor = _read_cursor()
    if cursor:
        url += f"&cursor={cursor}"

    async with websockets.connect(url, max_size=None) as ws:
        # Push the full mutual DID set via options_update (too long for the URL).
        await ws.send(json.dumps({
            "type": "options_update",
            "payload": {
                "wantedCollections": ["app.bsky.feed.post"],
                "wantedDids": mutual_dids,
            },
        }))
        print(f"Connected to {host} — filtering {len(mutual_dids)} mutual DIDs.")

        batch: list[tuple] = []
        kept_total = stored_seen = 0
        last_flush = asyncio.get_event_loop().time()
        last_prune = last_flush

        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=FLUSH_INTERVAL)
                event = json.loads(raw)
                if "time_us" in event:
                    _write_cursor(event["time_us"])
                parsed = _parse_commit(event)
                if parsed:
                    batch.append(parsed)
                else:
                    _handle_delete(conn, event)
            except asyncio.TimeoutError:
                pass  # fall through to time-based flush

            now = asyncio.get_event_loop().time()
            if batch and (len(batch) >= BATCH_SIZE or now - last_flush >= FLUSH_INTERVAL):
                seen = len(batch)
                kept = _flush(conn, clf, batch)
                kept_total += kept
                stored_seen += seen
                batch = []
                last_flush = now
                print(f"\r  seen {stored_seen} posts, kept {kept_total} "
                      f"(db total {db.count(conn)})", end="", flush=True)

            if now - last_prune >= PRUNE_INTERVAL:
                cutoff = (datetime.now(timezone.utc) - timedelta(days=config.RETENTION_DAYS)).isoformat()
                removed = db.prune(conn, cutoff)
                last_prune = now
                if removed:
                    print(f"\n  pruned {removed} posts older than {config.RETENTION_DAYS}d")


async def run_indexer() -> None:
    conn = db.connect()
    db.init_db(conn)
    clf = PoliticalClassifier(threshold=config.CLASSIFIER_THRESHOLD)
    mutual_dids = config.load_mutual_dids()
    print(f"Loaded {len(mutual_dids)} mutual DIDs. Warming up classifier ...")
    clf.classify("warmup")  # trigger model load up front

    backoff = 1.0
    host_idx = 0
    while True:
        try:
            await _consume(conn, clf, mutual_dids)
        except Exception as exc:  # noqa: BLE001 - keep the daemon alive
            print(f"\nConnection error: {exc!r}; reconnecting in {backoff:.0f}s ...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
            # Round-robin to the next public Jetstream host.
            host_idx = (host_idx + 1) % len(config.JETSTREAM_HOSTS)
            config.JETSTREAM_URL = f"wss://{config.JETSTREAM_HOSTS[host_idx]}/subscribe"
        else:
            backoff = 1.0


if __name__ == "__main__":
    try:
        asyncio.run(run_indexer())
    except KeyboardInterrupt:
        print("\nStopped.")
