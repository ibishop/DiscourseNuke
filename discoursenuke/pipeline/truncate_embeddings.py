"""Matryoshka-truncate nomic embeddings to smaller dims, and measure the loss.

nomic-embed-text-v1.5 is trained with Matryoshka representation learning, so the
first k dims of the 768-d vector are a usable embedding on their own. We slice to
each target dim and L2-renormalize (this matches sentence-transformers' own
truncate_dim behavior), then evaluate how well nearest-neighbor structure is
preserved vs the full 768-d space — that's what matters for the codebook stage.

Usage:
    python -m discoursenuke.pipeline.truncate_embeddings
    python -m discoursenuke.pipeline.truncate_embeddings --dims 256 128 64
"""

from __future__ import annotations

import argparse

import numpy as np

from .. import config


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Per-vector layer norm — nomic's recommended Matryoshka pre-step.

    Invariant to the L2 scaling already applied at encode time, so this gives the
    same result as running it on the raw (un-normalized) embeddings.
    """
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def matryoshka(full: np.ndarray, dim: int) -> np.ndarray:
    """nomic recipe: layer-norm the full vector, take first `dim`, L2-renormalize."""
    return l2norm(layer_norm(full)[:, :dim].copy())


def topk(sims: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-sims, k, axis=1)[:, :k]
    return idx


def nn_overlap(full: np.ndarray, trunc: np.ndarray, q_idx, c_idx, k=10) -> float:
    """Mean fraction of each query's top-k neighbors preserved after truncation."""
    Cf, Ct = full[c_idx], trunc[c_idx]
    sf = full[q_idx] @ Cf.T
    st = trunc[q_idx] @ Ct.T
    tf, tt = topk(sf, k), topk(st, k)
    overlaps = [len(set(a) & set(b)) / k for a, b in zip(tf, tt)]
    return float(np.mean(overlaps))


def main() -> None:
    ap = argparse.ArgumentParser(description="Matryoshka-truncate nomic embeddings.")
    ap.add_argument("--src", default=str(config.DATA_DIR / "corpus_emb_nomic.npy"))
    ap.add_argument("--dims", type=int, nargs="+", default=[512, 256, 128, 64])
    ap.add_argument("--no-save", action="store_true", help="Only evaluate, don't write files.")
    args = ap.parse_args()

    full = np.load(args.src)
    n, d0 = full.shape
    print(f"Loaded {args.src} -> {full.shape}")

    rng = np.random.default_rng(0)
    q_idx = rng.choice(n, size=min(500, n), replace=False)
    c_idx = rng.choice(n, size=min(20000, n), replace=False)

    print(f"\nNearest-neighbor preservation vs full {d0}-d (recall@10, "
          f"{len(q_idx)} queries / {len(c_idx)} candidates):")
    print(f"  {d0:>4}d : 1.000  (reference)")
    for dim in args.dims:
        if dim >= d0:
            continue
        trunc = matryoshka(full, dim)
        rec = nn_overlap(full, trunc, q_idx, c_idx, k=10)
        size_mb = trunc.nbytes / 1e6
        note = ""
        if not args.no_save:
            out = config.DATA_DIR / f"corpus_emb_nomic_{dim}d.npy"
            np.save(out, trunc.astype(np.float32))
            note = f" -> {out.name} ({size_mb:.0f} MB)"
        print(f"  {dim:>4}d : {rec:.3f}{note}")


if __name__ == "__main__":
    main()
