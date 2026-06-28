"""Embed the mutuals corpus into a single matrix for the quantization stage.

Reads feed_data/mutuals_corpus.jsonl, embeds every post, and writes:
  - feed_data/corpus_emb.npy        float32 [N, 384], L2-normalized
  - feed_data/corpus_emb_meta.jsonl one line per row (uri, author, created_at, text)

Row i of the .npy aligns with line i of the meta file.

Usage:
    python -m discoursenuke.pipeline.embed_corpus
    python -m discoursenuke.pipeline.embed_corpus --limit 1000   # quick test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .. import config
from ..classify.embedder import Embedder


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed the mutuals corpus.")
    ap.add_argument("--model", default="minilm", choices=["minilm", "nomic"],
                    help="Embedding model preset.")
    ap.add_argument("--src", default=str(config.DATA_DIR / "mutuals_corpus.jsonl"))
    ap.add_argument("--out", default="", help="Output .npy (default: corpus_emb_<model>.npy).")
    ap.add_argument("--meta", default="", help="Output meta JSONL (default: corpus_emb_<model>_meta.jsonl).")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="Only embed the first N posts (0 = all).")
    args = ap.parse_args()

    args.out = args.out or str(config.DATA_DIR / f"corpus_emb_{args.model}.npy")
    args.meta = args.meta or str(config.DATA_DIR / f"corpus_emb_{args.model}_meta.jsonl")

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"{src} not found. Run crawl_mutuals first.")

    # Load corpus (skip empty text). Keep meta aligned to embedding rows.
    records = []
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            p = json.loads(line)
            if not p.get("text"):
                continue
            records.append(p)
            if args.limit and len(records) >= args.limit:
                break
    texts = [r["text"] for r in records]
    print(f"Loaded {len(texts)} posts. Embedding with '{args.model}' (batch={args.batch_size}) ...")

    embedder = Embedder(preset=args.model)
    start = time.time()
    vecs = embedder.encode(texts, batch_size=args.batch_size, show_progress=True)
    elapsed = time.time() - start
    print(f"Embedded {vecs.shape[0]} posts -> {vecs.shape} in {elapsed:.0f}s "
          f"({vecs.shape[0]/max(1e-9,elapsed):.0f} posts/s)")

    np.save(args.out, vecs)
    with open(args.meta, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(
                {"uri": r.get("uri", ""), "author": r.get("author", ""),
                 "created_at": r.get("created_at", ""), "text": r["text"]},
                ensure_ascii=False) + "\n")

    mb = vecs.nbytes / 1e6
    print(f"Saved {args.out} ({mb:.0f} MB) and {args.meta}")


if __name__ == "__main__":
    main()
