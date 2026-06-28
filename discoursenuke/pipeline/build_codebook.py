"""Fit a k-means codebook over the corpus embeddings (the quantization step).

Each cluster centroid is a "token". Every post maps to its nearest token, giving
the discrete representation we want to select on. Saves the codebook + per-post
token assignments, and prints representative posts per token so you can see what
each cluster captures.

Usage:
    python -m discoursenuke.pipeline.build_codebook --k 512
    python -m discoursenuke.pipeline.build_codebook --emb feed_data/corpus_emb_nomic_256d.npy --k 1024
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .. import config


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit a k-means codebook over embeddings.")
    ap.add_argument("--emb", default=str(config.DATA_DIR / "corpus_emb_nomic_256d.npy"))
    ap.add_argument("--meta", default=str(config.DATA_DIR / "corpus_emb_nomic_meta.jsonl"))
    ap.add_argument("--k", type=int, default=512, help="Number of tokens (clusters).")
    ap.add_argument("--out", default="", help="Output .npz (default: codebook_k<k>.npz).")
    ap.add_argument("--show", type=int, default=12, help="How many clusters to preview.")
    ap.add_argument("--per-cluster", type=int, default=4, help="Posts shown per cluster.")
    args = ap.parse_args()

    emb = np.load(args.emb).astype(np.float32)
    texts = [json.loads(l)["text"] for l in Path(args.meta).open(encoding="utf-8")]
    authors = [json.loads(l)["author"] for l in Path(args.meta).open(encoding="utf-8")]
    n, d = emb.shape
    print(f"Clustering {n} x {d} -> {args.k} tokens ...")

    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=args.k, random_state=0, batch_size=4096,
                         n_init=3, max_iter=200)
    labels = km.fit_predict(emb)
    centroids = l2norm(km.cluster_centers_.astype(np.float32))
    print(f"Done in {time.time()-t0:.0f}s.")

    # Cluster size distribution.
    sizes = np.bincount(labels, minlength=args.k)
    print(f"\nCluster sizes: min {sizes.min()}, median {int(np.median(sizes))}, "
          f"mean {sizes.mean():.0f}, max {sizes.max()}  (empty: {(sizes==0).sum()})")

    out = args.out or str(config.DATA_DIR / f"codebook_k{args.k}.npz")
    np.savez(out, centroids=centroids, labels=labels.astype(np.int32), sizes=sizes)
    print(f"Saved codebook -> {out}")

    # Preview a spread of clusters: most-representative posts (nearest to centroid).
    rng = np.random.default_rng(0)
    shown = rng.choice(np.where(sizes > 0)[0], size=min(args.show, (sizes > 0).sum()), replace=False)
    print(f"\n=== {len(shown)} sample tokens (most representative posts) ===")
    for tok in shown:
        members = np.where(labels == tok)[0]
        sims = emb[members] @ centroids[tok]
        top = members[np.argsort(-sims)[: args.per_cluster]]
        print(f"\n● token {tok}  ({sizes[tok]} posts)")
        for i in top:
            txt = texts[i].replace("\n", " ")
            print(f"    @{authors[i]}: {txt[:95]}")


if __name__ == "__main__":
    main()
