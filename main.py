"""DiscourseNuke PoC: pull Bluesky posts and nuke US political discourse.

Usage:
    python main.py --actor someone.bsky.social --limit 50
    python main.py --actor someone.bsky.social --threshold 0.05 --show-kept

With no --actor, runs against a small built-in sample feed so you can sanity
check the classifier offline (no network, but still downloads the model once).
"""

from __future__ import annotations

import argparse

from bluesky import Post, get_author_feed, get_timeline
from classifier import PoliticalClassifier

SAMPLE_FEED = [
    Post("demo", "Congress just passed the new spending bill along party lines.", "", ""),
    Post("demo", "I baked sourdough for the first time and it actually worked!", "", ""),
    Post("demo", "The Supreme Court's ruling today will reshape the election.", "", ""),
    Post("demo", "My cat knocked a glass off the table again. Classic.", "", ""),
    Post("demo", "Trump and Biden are neck and neck in the latest swing-state poll.", "", ""),
    Post("demo", "This new indie game has the best soundtrack I've heard all year.", "", ""),
    Post("demo", "Protesters gathered at the Capitol over the immigration policy.", "", ""),
    Post("demo", "Finally hiked to the summit — the view was unreal.", "", ""),
]


def truncate(text: str, n: int = 90) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def main() -> None:
    ap = argparse.ArgumentParser(description="Nuke US political discourse from a Bluesky feed.")
    ap.add_argument("--actor", help="Pull a specific user's public author feed.")
    ap.add_argument("--timeline", action="store_true", help="Pull your authenticated home timeline.")
    ap.add_argument("--sample", action="store_true", help="Use the built-in offline sample feed.")
    ap.add_argument("--limit", type=int, default=50, help="Max posts to pull.")
    ap.add_argument("--threshold", type=float, default=0.05, help="Political margin threshold.")
    ap.add_argument("--show-kept", action="store_true", help="Also print kept (non-political) posts.")
    args = ap.parse_args()

    if args.actor:
        print(f"Pulling up to {args.limit} posts from @{args.actor} ...")
        posts = get_author_feed(args.actor, limit=args.limit)
    elif args.sample:
        print("Using built-in sample feed.")
        posts = SAMPLE_FEED
    else:  # default: authenticated home timeline
        from auth import get_client

        print(f"Pulling up to {args.limit} posts from your home timeline ...")
        client = get_client()
        posts = get_timeline(client, limit=args.limit)

    print(f"Loaded {len(posts)} posts. Loading model + classifying ...\n")

    clf = PoliticalClassifier(threshold=args.threshold)
    verdicts = clf.classify_many([p.text for p in posts])

    nuked = 0
    for post, v in zip(posts, verdicts):
        if v.is_political:
            nuked += 1
            print(f"  \U0001f4a5 NUKE  [{v.score:+.3f}] @{post.author}: {truncate(post.text)}")
        elif args.show_kept:
            print(f"  ✅ keep  [{v.score:+.3f}] @{post.author}: {truncate(post.text)}")

    print(f"\nNuked {nuked}/{len(posts)} posts as US political discourse.")


if __name__ == "__main__":
    main()