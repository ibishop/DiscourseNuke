"""Reusable text embedder (sentence-transformers all-MiniLM-L6-v2).

Wraps the same model the classifier uses, but exposes a generic batched encode
for arbitrary text — used to embed the full corpus for the quantization work.
Embeddings are L2-normalized so cosine similarity == dot product.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from .classifier import MODEL_NAME


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

    @cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name)

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: list[str],
        batch_size: int = 256,
        show_progress: bool = False,
    ) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        ).astype(np.float32)
