"""Multi-topic + type classification over corpus embeddings (nomic space).

Single source of truth for the topic/type taxonomy used by the pipeline scripts
and HTML renderers. Assigns each post:
  - topic : nearest of TOPICS (us_politics/foreign_politics/ai/finance) if it
            beats the neutral seeds by `threshold`, else "neutral".
  - type  : news vs commentary vs chatter, via z-scored argmax (news embeds more
            tightly than reactions, so axes are standardized to compete fairly).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embedder import Embedder
from .topics import (CHATTER_SEEDS, COMMENTARY_SEEDS, NEUTRAL_SEEDS,
                     NEWS_SEEDS, TOPICS)

TYPES = ["news", "commentary", "chatter"]


@dataclass
class Taxonomy:
    topic: np.ndarray        # str per post ("neutral" or a topic name)
    type: np.ndarray         # str per post (news/commentary/chatter)
    topic_score: np.ndarray  # affinity to the assigned topic
    is_topic: np.ndarray     # bool: cleared the neutral threshold
    topic_names: list[str]


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


class TopicClassifier:
    def __init__(self, embedder: Embedder | None = None, threshold: float = 0.03):
        self.embedder = embedder or Embedder(preset="nomic")
        self.threshold = threshold

    def _nearest(self, emb: np.ndarray, seeds: list[str]) -> np.ndarray:
        """Max cosine of each post to any seed (both L2-normalized)."""
        return (emb @ self.embedder.encode(seeds).T).max(axis=1)

    def classify(self, emb: np.ndarray) -> Taxonomy:
        names = list(TOPICS)
        topic_scores = np.stack([self._nearest(emb, TOPICS[t]) for t in names], axis=1)
        neutral = self._nearest(emb, NEUTRAL_SEEDS)
        news = _zscore(self._nearest(emb, NEWS_SEEDS))
        comm = _zscore(self._nearest(emb, COMMENTARY_SEEDS))
        chat = _zscore(self._nearest(emb, CHATTER_SEEDS))

        best = topic_scores.argmax(axis=1)
        best_score = topic_scores.max(axis=1)
        is_topic = (best_score - neutral) >= self.threshold
        type_idx = np.stack([news, comm, chat], axis=1).argmax(axis=1)

        topic = np.where(is_topic, np.asarray(names)[best], "neutral")
        typ = np.asarray(TYPES)[type_idx]
        return Taxonomy(topic=topic, type=typ, topic_score=best_score,
                        is_topic=is_topic, topic_names=names)
