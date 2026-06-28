"""Product-quantize the corpus embeddings into per-post tokens.

Splits each D-dim embedding into M contiguous subspaces, learns an independent
k-means codebook per subspace, and assigns each post one token per subspace.
Result: every post becomes a tuple of M tokens you can select on.

  post -> (t_0, t_1, ..., t_{M-1}),  t_m in [0, codes)

Outputs feed_data/pq_m<M>_c<codes>.npz:
  - centroids : float32 [M, codes, D/M]   (per-subspace codebooks)
  - codes     : uint16  [N, M]            (per-post tokens)
  - params    : M, codes, dim

Usage:
    python -m discoursenuke.pipeline.build_pq --subspaces 8 --codes 256
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .. import config


def main() -> None:
    ap = argparse.ArgumentParser(description="Product-quantize embeddings into per-post tokens.")
    ap.add_argument("--emb", default=str(config.DATA_DIR / "corpus_emb_nomic_256d.npy"))
    ap.add_argument("--subspaces", type=int, default=8, help="M: chunks the vector is split into.")
    ap.add_argument("--codes", type=int, default=256, help="Codebook size per subspace.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    emb = np.load(args.emb).astype(np.float32)
    n, d = emb.shape
    m = args.subspaces
    if d % m != 0:
        raise SystemExit(f"dim {d} not divisible by subspaces {m}")
    ds = d // m
    print(f"PQ: {n} posts, {d} dims -> {m} subspaces x {ds} dims, {args.codes} codes each "
          f"=> {m} tokens/post")

    centroids = np.zeros((m, args.codes, ds), dtype=np.float32)
    codes = np.zeros((n, m), dtype=np.uint16)

    t0 = time.time()
    for sub in range(m):
        chunk = emb[:, sub * ds:(sub + 1) * ds]
        km = MiniBatchKMeans(n_clusters=args.codes, random_state=0, batch_size=4096,
                             n_init=3, max_iter=100)
        codes[:, sub] = km.fit_predict(chunk).astype(np.uint16)
        centroids[sub] = km.cluster_centers_.astype(np.float32)
        print(f"\r  subspace {sub+1}/{m} done", end="", flush=True)
    print(f"\nFit {m} codebooks in {time.time()-t0:.0f}s.")

    out = args.out or str(config.DATA_DIR / f"pq_m{m}_c{args.codes}.npz")
    np.savez(out, centroids=centroids, codes=codes,
             params=np.array([m, args.codes, d]))
    print(f"Saved -> {out}  (codes {codes.shape} {codes.dtype})")

    # Sanity: show a few posts' token tuples, and how 'selectable' tokens are.
    print("\nExample per-post tokens (first 5 posts):")
    for i in range(5):
        print(f"  post {i}: {tuple(int(x) for x in codes[i])}")

    # How many distinct full tuples vs how shared individual subspace tokens are.
    uniq_tuples = len({tuple(row) for row in codes[:5000]})
    print(f"\nAmong first 5000 posts: {uniq_tuples} distinct full {m}-token tuples "
          f"(near-unique → expressive).")
    # Subspace-0 token populations (selectability of a single token).
    counts = np.bincount(codes[:, 0], minlength=args.codes)
    print(f"Subspace 0: token populations min {counts.min()}, "
          f"median {int(np.median(counts[counts>0]))}, max {counts.max()} "
          f"(selecting one subspace-0 token picks ~that many posts).")


if __name__ == "__main__":
    main()
