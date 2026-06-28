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


def get_timeline(client, limit: int | None = 50, since_days: int | None = None,
                 max_posts: int = 5000) -> list[Post]:
    """Fetch the authenticated user's home timeline (posts from follows).

    Requires a logged-in atproto Client (see auth.get_client). This is the
    private feed the user actually consumes.

    - limit: stop after this many posts (ignored if since_days is set).
    - since_days: keep paginating back until posts are older than N days.
    - max_posts: hard safety cap so a deep pull can't run away.
    """
    from datetime import datetime, timedelta, timezone

    cutoff = None
    if since_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)

    posts: list[Post] = []
    cursor = None
    while len(posts) < max_posts:
        page_size = 100
        if limit is not None and since_days is None:
            page_size = min(100, limit - len(posts))
            if page_size <= 0:
                break
        resp = client.get_timeline(cursor=cursor, limit=page_size)
        reached_cutoff = False
        for item in resp.feed:
            post = item.post
            text = getattr(post.record, "text", "") or ""
            if not text:
                continue
            created = getattr(post.record, "created_at", "") or ""
            # Effective "when I'd have seen it" time for the cutoff. For a repost
            # that's item.reason.indexed_at; otherwise the post's own indexed_at.
            # (record.created_at is wrong for reposts — it's the original's date.)
            reason = getattr(item, "reason", None)
            if reason is not None and "Repost" in type(reason).__name__:
                indexed = getattr(reason, "indexed_at", "") or created
            else:
                indexed = getattr(post, "indexed_at", "") or created
            if cutoff is not None and indexed:
                try:
                    when = datetime.fromisoformat(indexed.replace("Z", "+00:00"))
                    if when < cutoff:
                        reached_cutoff = True
                        break
                except ValueError:
                    pass
            posts.append(
                Post(
                    author=post.author.handle,
                    text=text,
                    uri=post.uri,
                    created_at=created,
                )
            )
        cursor = resp.cursor
        if reached_cutoff or not cursor:
            break
    return posts if since_days is not None else posts[: (limit or len(posts))]


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