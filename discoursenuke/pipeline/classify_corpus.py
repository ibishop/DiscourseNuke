"""Run the geography-aware politics classifier over the corpus in NOMIC space.

Same contrastive logic as classifier.py (US-political vs foreign-political vs
neutral seeds, nearest-seed cosine), but using the nomic corpus embeddings we
already computed — we just embed the seeds with nomic and score. Lets us compare
the nomic filter against the MiniLM one.

Usage:
    python -m discoursenuke.pipeline.classify_corpus --threshold 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import config
from ..classify.classifier import (FOREIGN_POLITICAL_SEEDS, NEUTRAL_SEEDS,
                                    US_POLITICAL_SEEDS)
from ..classify.embedder import Embedder


def nearest(posts: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Max cosine of each post to any seed (posts & seeds are L2-normalized)."""
    return (posts @ seeds.T).max(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Politics classifier over corpus in nomic space.")
    ap.add_argument("--emb", default=str(config.DATA_DIR / "corpus_emb_nomic.npy"))
    ap.add_argument("--meta", default=str(config.DATA_DIR / "corpus_emb_nomic_meta.jsonl"))
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="US must beat max(foreign, neutral) by this margin.")
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    emb = np.load(args.emb).astype(np.float32)
    meta = [json.loads(l) for l in Path(args.meta).open(encoding="utf-8")]
    print(f"Loaded {emb.shape} posts. Embedding seeds with nomic ...")

    embedder = Embedder(preset="nomic")
    us_s = embedder.encode(US_POLITICAL_SEEDS)
    fr_s = embedder.encode(FOREIGN_POLITICAL_SEEDS)
    ne_s = embedder.encode(NEUTRAL_SEEDS)

    us = nearest(emb, us_s)
    fr = nearest(emb, fr_s)
    ne = nearest(emb, ne_s)
    margin = us - np.maximum(fr, ne)
    is_us_top = (us >= fr) & (us >= ne)

    n = emb.shape[0]
    print("\nUS-political count by margin threshold (nomic):")
    for t in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15]:
        flagged = is_us_top & (margin >= t)
        print(f"  thresh {t:.2f}: {int(flagged.sum())} ({100*flagged.mean():.1f}%)")

    political = is_us_top & (margin >= args.threshold)
    print(f"\nAt threshold {args.threshold}: nuked {int(political.sum())} / {n} "
          f"({100*political.mean():.1f}%)")

    order = np.argsort(-margin)
    print(f"\n=== top {args.show} flagged (highest US margin) ===")
    for i in order[:args.show]:
        if not political[i]:
            continue
        print(f"  [{margin[i]:+.3f}] @{meta[i]['author']}: {meta[i]['text'][:90].replace(chr(10),' ')}")

    # Spot-check geography: things that look foreign-political (us not top).
    foreign_like = np.argsort(-fr)
    print(f"\n=== sample strongly-foreign posts (should be SPARED) ===")
    shown = 0
    for i in foreign_like:
        if not is_us_top[i]:
            print(f"  us={us[i]:.2f} for={fr[i]:.2f} @{meta[i]['author']}: "
                  f"{meta[i]['text'][:70].replace(chr(10),' ')}")
            shown += 1
            if shown >= 6:
                break


if __name__ == "__main__":
    main()
