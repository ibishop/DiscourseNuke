"""Embedding-based classifier for US political discourse.

Approach (PoC, no quantization yet):
  - Embed each post with a local sentence-transformers model.
  - Keep two small reference sets: US-political seeds and neutral seeds.
  - Score a post by how much closer it sits to the political seeds than to
    the neutral seeds (contrastive cosine similarity).
  - A post is "political" when that margin clears a threshold.

This is intentionally simple so we can see whether the embedding space alone
separates US politics before adding clustering / tokenization.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

# Seed phrases that exemplify US political discourse. These are the "positive"
# anchors. Broad coverage matters more than precision here.
POLITICAL_SEEDS = [
    "The president signed a new executive order today.",
    "Congress is debating the federal budget and a government shutdown.",
    "Republicans and Democrats clashed over the new bill in the Senate.",
    "The Supreme Court issued a ruling on abortion rights.",
    "Voters head to the polls in the upcoming midterm elections.",
    "Trump rallied supporters ahead of the primary.",
    "Biden's administration announced a new immigration policy.",
    "The governor signed legislation on gun control.",
    "Debate over taxes, tariffs, and the economy dominated the campaign.",
    "Protesters gathered at the Capitol over the new policy.",
    "The senator introduced a bill to reform healthcare.",
    "Polling shows a tight race between the two candidates.",
    "The House passed a resolution along party lines.",
    "Discussion of MAGA, the GOP, and progressive Democrats.",
    "Election integrity and voter ID laws are being contested in court.",
]

# Seed phrases for ordinary, non-political content. The "negative" anchors.
NEUTRAL_SEEDS = [
    "I just baked sourdough bread for the first time.",
    "Check out this photo of my cat sleeping in the sun.",
    "The new sci-fi movie was absolutely incredible.",
    "Here's my recipe for a great weeknight pasta.",
    "Finally finished that 1000-piece jigsaw puzzle.",
    "The hike up the mountain had amazing views.",
    "I'm learning to play the guitar this year.",
    "This new indie game is so much fun.",
    "Coffee tastes better on a rainy morning.",
    "My garden tomatoes are finally ripening.",
    "Watched a great documentary about the ocean.",
    "Trying out a new workout routine this week.",
    "The concert last night was unforgettable.",
    "Just adopted a puppy from the shelter.",
    "Working on a new watercolor painting.",
]


@dataclass
class Verdict:
    is_political: bool
    score: float  # margin: political_similarity - neutral_similarity


class PoliticalClassifier:
    def __init__(self, threshold: float = 0.05, model_name: str = MODEL_NAME):
        self.threshold = threshold
        self.model_name = model_name

    @cached_property
    def _model(self):
        # Imported lazily so importing this module is cheap.
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name)

    @cached_property
    def _political_proto(self) -> np.ndarray:
        return self._encode(POLITICAL_SEEDS).mean(axis=0)

    @cached_property
    def _neutral_proto(self) -> np.ndarray:
        return self._encode(NEUTRAL_SEEDS).mean(axis=0)

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True)

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        # Inputs are already L2-normalized, so dot product == cosine sim.
        return float(np.dot(a, b))

    def classify(self, text: str) -> Verdict:
        vec = self._encode([text])[0]
        pol = self._cos(vec, self._political_proto)
        neu = self._cos(vec, self._neutral_proto)
        margin = pol - neu
        return Verdict(is_political=margin >= self.threshold, score=margin)

    def classify_many(self, texts: list[str]) -> list[Verdict]:
        if not texts:
            return []
        vecs = self._encode(texts)
        out = []
        for vec in vecs:
            pol = self._cos(vec, self._political_proto)
            neu = self._cos(vec, self._neutral_proto)
            margin = pol - neu
            out.append(Verdict(is_political=margin >= self.threshold, score=margin))
        return out
