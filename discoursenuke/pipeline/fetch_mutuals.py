"""Pull the mutuals-only feed and store it under feed_data/ (git-ignored).

This builds the local corpus we'll use for the quantization work. Mutuals are
a minority of the timeline, so reaching N mutual posts means scanning a larger
slice of the home timeline (bounded by --scan).

Usage:
    python -m discoursenuke.pipeline.fetch_mutuals --limit 2000 --scan 20000
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ..auth import get_client
from ..bluesky import get_mutuals, get_timeline

OUT_DIR = Path("feed_data")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch mutuals feed into feed_data/.")
    ap.add_argument("--limit", type=int, default=2000, help="Target number of mutual posts.")
    ap.add_argument("--scan", type=int, default=20000, help="Max timeline posts to scan.")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    client = get_client()

    print("Computing mutuals (follows ∩ followers) ...")
    mutuals = get_mutuals(client)
    print(f"  {len(mutuals)} mutuals.")
    (OUT_DIR / "mutuals.json").write_text(json.dumps(mutuals, indent=2))

    print(f"Pulling up to {args.limit} mutual posts (scanning up to {args.scan}) ...")
    posts = get_timeline(client, limit=args.limit, authors=set(mutuals), max_posts=args.scan)

    records = [p.__dict__ for p in posts]
    out = OUT_DIR / "mutuals_feed.json"
    out.write_text(json.dumps(
        {
            "pulled_at": datetime.now(timezone.utc).isoformat(),
            "count": len(records),
            "posts": records,
        },
        indent=2,
        ensure_ascii=False,
    ))
    span = f"{records[-1]['created_at'][:10]} -> {records[0]['created_at'][:10]}" if records else "n/a"
    print(f"Saved {len(records)} mutual posts to {out} (span {span}).")


if __name__ == "__main__":
    main()