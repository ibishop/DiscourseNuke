"""Reusable text embedder with selectable model presets.

Supports two models:
  - "minilm": all-MiniLM-L6-v2 (384-dim, fast; what the classifier uses)
  - "nomic" : nomic-ai/nomic-embed-text-v1.5 (768-dim, 8192-token context,
              stronger clustering). Requires trust_remote_code and a task prefix.

Embeddings are L2-normalized so cosine similarity == dot product.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from .classifier import MODEL_NAME

# preset -> (hf model name, trust_remote_code, text prefix)
MODELS = {
    "minilm": (MODEL_NAME, False, ""),
    "nomic": ("nomic-ai/nomic-embed-text-v1.5", True, "clustering: "),
}


class Embedder:
    def __init__(self, preset: str = "minilm"):
        if preset not in MODELS:
            raise ValueError(f"unknown preset {preset!r}; choose from {list(MODELS)}")
        self.preset = preset
        self.model_name, self.trust_remote_code, self.prefix = MODELS[preset]

    @cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name, trust_remote_code=self.trust_remote_code)

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: list[str],
        batch_size: int = 256,
        show_progress: bool = False,
    ) -> np.ndarray:
        if self.prefix:
            texts = [self.prefix + t for t in texts]
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        ).astype(np.float32)
