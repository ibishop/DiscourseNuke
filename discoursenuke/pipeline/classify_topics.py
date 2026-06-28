"""General multi-topic classifier over the nomic corpus.

Assigns each post to its nearest topic (us_politics / foreign_politics / ai /
finance), or neutral if it doesn't beat the neutral seeds by a margin, and adds
a NEWS vs COMMENTARY type. Prints counts and per-topic samples so AI and finance
buckets can be eyeballed.

Usage:
    python -m discoursenuke.pipeline.classify_topics --threshold 0.03
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import config
from ..classify.embedder import Embedder
from ..classify.topics import (CHATTER_SEEDS, COMMENTARY_SEEDS, NEUTRAL_SEEDS,
                               NEWS_SEEDS, TOPICS)


def nearest(posts: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    return (posts @ seeds.T).max(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-topic classifier (nomic).")
    ap.add_argument("--emb", default=str(config.DATA_DIR / "corpus_emb_nomic.npy"))
    ap.add_argument("--meta", default=str(config.DATA_DIR / "corpus_emb_nomic_meta.jsonl"))
    ap.add_argument("--threshold", type=float, default=0.03,
                    help="How much the best topic must beat neutral to not be neutral.")
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()

    emb = np.load(args.emb).astype(np.float32)
    meta = [json.loads(l) for l in Path(args.meta).open(encoding="utf-8")]
    texts = [m["text"] for m in meta]
    print(f"Loaded {emb.shape}. Embedding topic + type seeds with nomic ...")

    e = Embedder(preset="nomic")
    topic_names = list(TOPICS)
    topic_scores = np.stack([nearest(emb, e.encode(TOPICS[t])) for t in topic_names], axis=1)
    neutral = nearest(emb, e.encode(NEUTRAL_SEEDS))
    news = nearest(emb, e.encode(NEWS_SEEDS))
    comm = nearest(emb, e.encode(COMMENTARY_SEEDS))
    chat = nearest(emb, e.encode(CHATTER_SEEDS))

    best = topic_scores.argmax(axis=1)
    best_score = topic_scores.max(axis=1)
    is_topic = (best_score - neutral) >= args.threshold
    # 3-way type axis: news vs substantive commentary vs chatter/filler.
    type_idx = np.stack([news, comm, chat], axis=1).argmax(axis=1)  # 0 news,1 comm,2 chat
    TYPES = ["news", "commentary", "chatter"]

    n = len(texts)
    print(f"\nTopic counts (threshold {args.threshold}):")
    print(f"  {'neutral':<18}: {int((~is_topic).sum()):>6}  ({100*(~is_topic).mean():.1f}%)")
    for ti, t in enumerate(topic_names):
        mask = is_topic & (best == ti)
        parts = " / ".join(f"{TYPES[k]} {int((mask & (type_idx==k)).sum())}" for k in range(3))
        print(f"  {t:<18}: {int(mask.sum()):>6}  ({100*mask.mean():.1f}%)   {parts}")

    def show(ti, type_k, title):
        # Most-central-to-topic posts of a given type.
        mask = is_topic & (best == ti) & (type_idx == type_k)
        order = [i for i in np.argsort(-best_score) if mask[i]][: args.show]
        print(f"\n=== {title} ===")
        for i in order:
            print(f"  @{meta[i]['author']}: {texts[i][:88].replace(chr(10),' ')}")

    for ti, t in enumerate(topic_names):
        if t in ("us_politics", "ai"):  # spot-check news vs commentary per topic
            show(ti, 0, f"{t.upper()} NEWS")
            show(ti, 1, f"{t.upper()} COMMENTARY")


if __name__ == "__main__":
    main()
