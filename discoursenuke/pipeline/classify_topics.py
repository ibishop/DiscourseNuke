"""Multi-topic + type classifier over the nomic corpus (CLI summary).

Assigns each post a topic (us_politics / foreign_politics / ai / finance /
neutral) and a type (news / commentary / chatter), then prints counts and
per-topic samples. All scoring lives in classify.taxonomy.

Usage:
    python -m discoursenuke.pipeline.classify_topics --threshold 0.03
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import config
from ..classify.taxonomy import TYPES, TopicClassifier


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-topic classifier (nomic).")
    ap.add_argument("--emb", default=str(config.DATA_DIR / "corpus_emb_nomic.npy"))
    ap.add_argument("--meta", default=str(config.DATA_DIR / "corpus_emb_nomic_meta.jsonl"))
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()

    emb = np.load(args.emb).astype(np.float32)
    meta = [json.loads(l) for l in Path(args.meta).open(encoding="utf-8")]
    texts = [m["text"] for m in meta]
    print(f"Loaded {emb.shape}. Classifying ...")

    tax = TopicClassifier(threshold=args.threshold).classify(emb)

    n = len(texts)
    print(f"\nCategories (threshold {args.threshold}):")
    neu = int((~tax.is_topic).sum())
    print(f"  {'neutral':<18}: {neu:>6}  ({100*neu/n:.1f}%)")
    for t in tax.topic_names:
        mask = tax.topic == t
        parts = " / ".join(f"{ty} {int((mask & (tax.type == ty)).sum())}" for ty in TYPES)
        print(f"  {t:<18}: {int(mask.sum()):>6}  ({100*mask.mean():.1f}%)   {parts}")

    def show(topic, typ):
        mask = (tax.topic == topic) & (tax.type == typ)
        order = [i for i in np.argsort(-tax.topic_score) if mask[i]][: args.show]
        print(f"\n=== {topic.upper()} {typ.upper()} ===")
        for i in order:
            print(f"  @{meta[i]['author']}: {texts[i][:88].replace(chr(10),' ')}")

    for topic in ("us_politics", "ai"):
        show(topic, "news")
        show(topic, "commentary")


if __name__ == "__main__":
    main()
