"""Crawl mutuals' author feeds to build a large corpus (~100k posts).

The home timeline can't reach far enough back, so instead we pull each mutual's
public author feed directly (unauthenticated) and pull ~per-author posts from
every mutual for even coverage. Output is JSONL (one post per line) since the
corpus is large.

Usage:
    python -m discoursenuke.pipeline.crawl_mutuals --per-author 120 --workers 5
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .. import config
from ..bluesky import get_author_feed


def fetch_author(did: str, per_author: int, retries: int = 4) -> list:
    """Pull up to per_author posts for one author, retrying on rate limits."""
    for attempt in range(retries):
        try:
            return get_author_feed(did, limit=per_author)
        except Exception as exc:  # noqa: BLE001 - resilience over precision
            wait = 2 ** attempt
            if attempt == retries - 1:
                return []  # give up on this author
            time.sleep(wait)
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description="Crawl mutuals' author feeds into a JSONL corpus.")
    ap.add_argument("--per-author", type=int, default=120, help="Max posts per mutual.")
    ap.add_argument("--workers", type=int, default=5, help="Concurrent fetchers (mind rate limits).")
    ap.add_argument("--target", type=int, default=0, help="Stop after this many posts (0 = no cap).")
    ap.add_argument("--out", default=str(config.DATA_DIR / "mutuals_corpus.jsonl"),
                    help="Output JSONL path.")
    args = ap.parse_args()

    config.DATA_DIR.mkdir(exist_ok=True)
    mutuals = json.loads(config.MUTUALS_PATH.read_text())  # {did: handle}
    dids = list(mutuals.keys())
    print(f"Crawling {len(dids)} mutuals, up to {args.per_author} posts each "
          f"({args.workers} workers) -> {args.out}")

    seen: set[str] = set()
    lock = threading.Lock()
    total = done = 0
    start = time.time()

    with open(args.out, "w", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch_author, did, args.per_author): did for did in dids}
            for fut in as_completed(futures):
                posts = fut.result()
                with lock:
                    done += 1
                    for p in posts:
                        if not p.uri or p.uri in seen:
                            continue
                        seen.add(p.uri)
                        fh.write(json.dumps(p.__dict__, ensure_ascii=False) + "\n")
                        total += 1
                    if done % 25 == 0 or done == len(dids):
                        rate = total / max(1e-9, time.time() - start)
                        print(f"\r  {done}/{len(dids)} authors | {total} posts "
                              f"| {rate:.0f} posts/s", end="", flush=True)
                if args.target and total >= args.target:
                    print(f"\n  reached target {args.target}; stopping early.")
                    for f in futures:
                        f.cancel()
                    break

    print(f"\nDone. Wrote {total} unique posts from {done} authors to {args.out} "
          f"in {time.time() - start:.0f}s.")


if __name__ == "__main__":
    main()
