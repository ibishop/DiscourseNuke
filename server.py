"""Feed generator HTTP service (FastAPI).

Serves the did:web document and the two XRPC endpoints Bluesky needs. It only
reads pre-classified posts from SQLite — no model is loaded here, so responses
are fast.

Run:  uvicorn server:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

import config
import db

app = FastAPI(title="DiscourseNuke Feed Generator")

# One shared read connection (WAL allows concurrent reader + the indexer writer).
_conn = db.connect()
db.init_db(_conn)


@app.get("/")
def root():
    return {
        "service": "DiscourseNuke feed generator",
        "feed": config.FEED_DISPLAY_NAME,
        "indexed_posts": db.count(_conn),
        "service_did": config.FEEDGEN_SERVICE_DID,
    }


@app.get("/.well-known/did.json")
def did_document():
    return {
        "@context": ["https://www.w3.org/ns/did/v1"],
        "id": config.FEEDGEN_SERVICE_DID,
        "service": [
            {
                "id": "#bsky_fg",
                "type": "BskyFeedGenerator",
                "serviceEndpoint": config.SERVICE_ENDPOINT,
            }
        ],
    }


@app.get("/xrpc/app.bsky.feed.describeFeedGenerator")
def describe_feed_generator():
    return {
        "did": config.FEEDGEN_SERVICE_DID,
        "feeds": [{"uri": config.feed_uri()}],
    }


@app.get("/xrpc/app.bsky.feed.getFeedSkeleton")
def get_feed_skeleton(
    feed: str = Query(default=""),
    limit: int = Query(default=50, ge=1, le=100),
    cursor: str | None = Query(default=None),
):
    # Note: the inter-service JWT (Authorization header) is intentionally not
    # verified — this is a single shared feed with no per-user personalization.
    rows = db.get_feed_page(_conn, limit=limit, cursor=cursor)
    body: dict = {"feed": [{"post": r["uri"]} for r in rows]}
    if len(rows) == limit:
        last = rows[-1]
        body["cursor"] = f"{last['created_at']}::{last['uri']}"
    return JSONResponse(body)
