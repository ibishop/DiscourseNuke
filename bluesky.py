"""Pull posts from Bluesky's public AppView (no authentication required).

For the PoC we read public author feeds via the unauthenticated XRPC endpoint
at public.api.bsky.app. This lets us test the classifier against real posts
without storing any credentials yet. Auth'd timeline pulling comes later.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

PUBLIC_APPVIEW = "https://public.api.bsky.app"


@dataclass
class Post:
    author: str       # handle
    text: str
    uri: str
    created_at: str


def get_author_feed(actor: str, limit: int = 50) -> list[Post]:
    """Fetch recent posts for a given handle/DID from the public AppView."""
    url = f"{PUBLIC_APPVIEW}/xrpc/app.bsky.feed.getAuthorFeed"
    posts: list[Post] = []
    cursor = None
    while len(posts) < limit:
        params = {"actor": actor, "limit": min(100, limit - len(posts))}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("feed", []):
            post = item.get("post", {})
            record = post.get("record", {})
            text = record.get("text", "")
            if not text:
                continue
            posts.append(
                Post(
                    author=post.get("author", {}).get("handle", actor),
                    text=text,
                    uri=post.get("uri", ""),
                    created_at=record.get("createdAt", ""),
                )
            )
        cursor = data.get("cursor")
        if not cursor:
            break
    return posts[:limit]