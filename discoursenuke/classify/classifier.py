"""Embedding-based classifier for *US* political discourse.

Approach (PoC, no quantization yet):
  - Embed each post with a local sentence-transformers model.
  - Keep three small reference sets: US-political, FOREIGN-political, and
    neutral seeds.
  - Score a post by its nearest seed (max cosine) within each set.
  - A post is "US political" only when it is closest to the US-political set,
    beating both the foreign-political and neutral sets by a margin.

The foreign-political set is the key to geography: posts about UK/Canada/EU
politics are political, so they'd beat the neutral set — but they should land
closer to the foreign anchors than the US ones, so they're NOT nuked.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

# US political discourse — entity-rich on purpose so geography is encoded.
US_POLITICAL_SEEDS = [
    "President Trump signed an executive order from the White House.",
    "Biden's administration announced a new federal immigration policy.",
    "Congress and the Senate debated the federal budget on Capitol Hill.",
    "The Supreme Court (SCOTUS) issued a major ruling on abortion rights.",
    "Republicans and Democrats clashed over the bill in the US House.",
    "GOP and MAGA supporters rallied ahead of the Republican primary.",
    "DHS and ICE agents carried out immigration enforcement raids.",
    "Voters head to the polls in the US midterm and presidential elections.",
    "The governor signed a state law on gun control.",
    "Medicare for All (M4A) and US healthcare reform in Congress.",
    "Progressive Democrats and the DSA won a House primary.",
    "JD Vance and the Republican ticket campaigned in swing states.",
    "American politics dominated the news from Washington, DC.",
    "Senator Chris Murphy introduced a bill in the Senate.",
    "Voter fraud, the SAVE Act, and US election integrity laws.",
    "The filibuster and procedure in the United States Senate.",
]

# Political discourse from OTHER countries. These exist to absorb non-US
# politics so it doesn't get misfiled as US politics.
FOREIGN_POLITICAL_SEEDS = [
    "The UK Prime Minister and the Chancellor of the Exchequer.",
    "Labour, the Tories, and Reform UK debated in Parliament.",
    "Margaret Thatcher's council house policy and the Conservatives.",
    "Westminster and Downing Street politics in Britain.",
    "Keir Starmer's Labour government announced a new policy.",
    "Canada's Prime Minister and Parliament in Ottawa.",
    "Canadian federal policy and the provinces like Ontario.",
    "The European Union and the European Parliament in Brussels.",
    "France's National Assembly and the French political spectrum.",
    "Austria's communist party won local elections in Graz.",
    "Mexico's president rolled out universal healthcare.",
    "German federal politics and the Bundestag in Berlin.",
    "Australia's parliament and the prime minister in Canberra.",
    "The Scottish, Welsh, and Northern Irish devolved governments.",
]

# Ordinary, non-political content.
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
    is_political: bool           # specifically US political (the nuke decision)
    score: float                 # us - max(foreign, neutral); margin of confidence
    us: float                    # nearest-seed similarity to US-political set
    foreign: float               # nearest-seed similarity to foreign-political set
    neutral: float               # nearest-seed similarity to neutral set
    label: str                   # 'us_political' | 'foreign_political' | 'neutral'


class PoliticalClassifier:
    def __init__(self, threshold: float = 0.08, model_name: str = MODEL_NAME):
        # threshold = how much US must beat the best competing set by.
        # ~0.08 drops the near-zero-margin noise (posts not close to anything)
        # while keeping genuine US-political posts, which score 0.08+.
        self.threshold = threshold
        self.model_name = model_name

    @cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name)

    @cached_property
    def _us(self) -> np.ndarray:
        return self._encode(US_POLITICAL_SEEDS)

    @cached_property
    def _foreign(self) -> np.ndarray:
        return self._encode(FOREIGN_POLITICAL_SEEDS)

    @cached_property
    def _neutral(self) -> np.ndarray:
        return self._encode(NEUTRAL_SEEDS)

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True)

    def _verdict(self, vec: np.ndarray) -> Verdict:
        # Inputs are L2-normalized, so a matrix-vector dot gives cosine sims.
        us = float(self._us.dot(vec).max())
        foreign = float(self._foreign.dot(vec).max())
        neutral = float(self._neutral.dot(vec).max())

        scores = {"us_political": us, "foreign_political": foreign, "neutral": neutral}
        label = max(scores, key=scores.get)
        margin = us - max(foreign, neutral)
        is_us = label == "us_political" and margin >= self.threshold
        return Verdict(is_us, margin, us, foreign, neutral, label)

    def classify(self, text: str) -> Verdict:
        return self._verdict(self._encode([text])[0])

    def classify_many(self, texts: list[str]) -> list[Verdict]:
        if not texts:
            return []
        return [self._verdict(v) for v in self._encode(texts)]