"""Filter US political discourse out of a stored feed.

Loads feed_data/mutuals_feed.json, classifies every post, and writes:
  - feed_data/filtered_feed.json  -> the cleaned feed (US politics removed)
  - feed_data/nuked_feed.json     -> what was removed (for inspection)

Usage:
    python -m discoursenuke.pipeline.filter_feed --threshold 0.08
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..classify.classifier import PoliticalClassifier

DATA = Path("feed_data")
SRC = DATA / "mutuals_feed.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter US politics out of the stored feed.")
    ap.add_argument("--threshold", type=float, default=0.08, help="US-political margin threshold.")
    ap.add_argument("--src", default=str(SRC), help="Source feed JSON.")
    args = ap.parse_args()

    blob = json.loads(Path(args.src).read_text())
    posts = blob["posts"]
    print(f"Loaded {len(posts)} posts from {args.src}. Classifying ...")

    clf = PoliticalClassifier(threshold=args.threshold)
    verdicts = clf.classify_many([p["text"] for p in posts])

    # Show sensitivity so the threshold choice is informed.
    print("\nUS-political count by threshold:")
    for t in [0.05, 0.08, 0.10, 0.12, 0.15]:
        n = sum(1 for v in verdicts if v.label == "us_political" and v.score >= t)
        print(f"  thresh {t:.2f}: {n}")

    kept, nuked = [], []
    for post, v in zip(posts, verdicts):
        rec = {**post, "us_score": round(v.us, 3), "score": round(v.score, 3), "label": v.label}
        (nuked if v.is_political else kept).append(rec)

    (DATA / "filtered_feed.json").write_text(json.dumps(
        {"threshold": args.threshold, "count": len(kept), "posts": kept},
        indent=2, ensure_ascii=False))
    (DATA / "nuked_feed.json").write_text(json.dumps(
        {"threshold": args.threshold, "count": len(nuked), "posts": nuked},
        indent=2, ensure_ascii=False))

    print(f"\nAt threshold {args.threshold}: nuked {len(nuked)}/{len(posts)} "
          f"({100*len(nuked)/len(posts):.1f}%), kept {len(kept)}.")
    print(f"  -> feed_data/filtered_feed.json ({len(kept)} posts)")
    print(f"  -> feed_data/nuked_feed.json ({len(nuked)} posts)")


if __name__ == "__main__":
    main()