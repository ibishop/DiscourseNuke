"""Multi-axis political classifier over the nomic corpus.

Two axes on top of the politics filter:
  1. geography : US-political vs foreign-political vs neutral
  2. type      : (for US politics) NEWS (reporting events) vs META commentary
                 (takes about politics / media / strategy / the discourse)

Categories emitted: us_news, us_commentary, foreign_political, neutral.
Prints counts + the most-characteristic samples per category so the seed
definitions can be judged and refined.

Usage:
    python -m discoursenuke.pipeline.classify_multi --pol-threshold 0.03
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

# Reporting of concrete political events / facts.
NEWS_SEEDS = [
    "President Trump signed an executive order today.",
    "The Supreme Court ruled 6-3 in the case.",
    "Federal agents arrested protesters near the Capitol.",
    "Congress passed the spending bill on a party-line vote.",
    "A new poll shows the two candidates tied at 47 percent.",
    "ICE conducted immigration raids in several cities this week.",
    "The senator introduced legislation to reform healthcare.",
    "The governor signed a state law banning the practice.",
    "Officials announced new tariffs on imported goods.",
    "A federal judge blocked the administration's order.",
    "The House voted to advance the resolution.",
    "The candidate won the primary by a narrow margin.",
]

# Takes / opinion / discourse about politics, media, and strategy.
COMMENTARY_SEEDS = [
    "The media keeps framing this story the wrong way.",
    "Democrats need a better message to win over swing voters.",
    "Political journalists can't imagine a world where things change.",
    "Hot take: the whole strategy here is fundamentally misguided.",
    "The discourse around this issue is exhausting and misses the point.",
    "People keep saying this but they don't understand how it actually works.",
    "This is just more proof that both sides are grifters.",
    "The way pundits cover this race tells you everything you need to know.",
    "Everyone online is overreacting to this, as usual.",
    "My honest opinion is that the left needs to rethink its whole approach.",
    "It's wild how the narrative on this keeps shifting.",
    "The real problem is how we even talk about these elections.",
]


def nearest(posts: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    return (posts @ seeds.T).max(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-axis political classifier (nomic).")
    ap.add_argument("--emb", default=str(config.DATA_DIR / "corpus_emb_nomic.npy"))
    ap.add_argument("--meta", default=str(config.DATA_DIR / "corpus_emb_nomic_meta.jsonl"))
    ap.add_argument("--pol-threshold", type=float, default=0.03,
                    help="How much political must beat neutral to count as political.")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    emb = np.load(args.emb).astype(np.float32)
    meta = [json.loads(l) for l in Path(args.meta).open(encoding="utf-8")]
    texts = [m["text"] for m in meta]
    print(f"Loaded {emb.shape}. Embedding seed sets with nomic ...")

    e = Embedder(preset="nomic")
    us = nearest(emb, e.encode(US_POLITICAL_SEEDS))
    fr = nearest(emb, e.encode(FOREIGN_POLITICAL_SEEDS))
    ne = nearest(emb, e.encode(NEUTRAL_SEEDS))
    news = nearest(emb, e.encode(NEWS_SEEDS))
    comm = nearest(emb, e.encode(COMMENTARY_SEEDS))

    # Axis 1: political at all, and which geography.
    pol_strength = np.maximum(us, fr) - ne
    is_political = pol_strength >= args.pol_threshold
    is_us = is_political & (us >= fr)
    is_foreign = is_political & (fr > us)
    # Axis 2: news vs commentary (only meaningful for US political).
    is_news = news >= comm

    cat = np.full(len(texts), "neutral", dtype=object)
    cat[is_foreign] = "foreign_political"
    cat[is_us & is_news] = "us_news"
    cat[is_us & ~is_news] = "us_commentary"

    n = len(texts)
    print(f"\nCategories (pol-threshold {args.pol_threshold}):")
    for c in ["us_news", "us_commentary", "foreign_political", "neutral"]:
        cnt = int((cat == c).sum())
        print(f"  {c:<18}: {cnt:>6}  ({100*cnt/n:.1f}%)")

    def show(mask, rank, title):
        order = np.argsort(-rank)
        order = [i for i in order if mask[i]][: args.show]
        print(f"\n=== {title} ===")
        for i in order:
            print(f"  news={news[i]:.2f} comm={comm[i]:.2f} @{meta[i]['author']}: "
                  f"{texts[i][:85].replace(chr(10),' ')}")

    show(is_us & is_news, news - comm, "US NEWS (most news-like)")
    show(is_us & ~is_news, comm - news, "US META COMMENTARY (most commentary-like)")

    foreign_order = np.argsort(-fr)
    print(f"\n=== FOREIGN POLITICAL (sample) ===")
    shown = 0
    for i in foreign_order:
        if is_foreign[i]:
            print(f"  us={us[i]:.2f} for={fr[i]:.2f} @{meta[i]['author']}: "
                  f"{texts[i][:75].replace(chr(10),' ')}")
            shown += 1
            if shown >= args.show:
                break


if __name__ == "__main__":
    main()
