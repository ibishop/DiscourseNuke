"""Topic + type seed sets for the multi-topic classifier.

Each topic is a set of seed sentences; a post is assigned to its nearest topic
(by max cosine to any seed) if it beats the neutral set by a margin. A separate
NEWS vs COMMENTARY axis (domain-general seeds) splits each topic into factual
reporting vs takes/opinion/discourse.
"""

from __future__ import annotations

from .classifier import (FOREIGN_POLITICAL_SEEDS, NEUTRAL_SEEDS,
                         US_POLITICAL_SEEDS)

AI_SEEDS = [
    "OpenAI, Anthropic, and Google released new large language models.",
    "The new LLM scores higher on coding and math benchmarks.",
    "Prompting, fine-tuning, and RAG for large language models.",
    "Comparing GPT, Claude, Gemini, and Llama models.",
    "AI agents that autonomously use tools and call APIs.",
    "Nvidia GPUs and compute for training neural networks.",
    "The chatbot hallucinates and makes things up.",
    "AI safety, alignment, and AGI timelines.",
    "Diffusion models generate images from text prompts.",
    "Token context windows and inference costs for LLMs.",
    "Open-weight models released on Hugging Face.",
    "Transformer architecture and attention mechanisms.",
]

FINANCE_SEEDS = [
    "The stock market rose sharply today.",
    "The Federal Reserve raised interest rates.",
    "Bond yields climbed after the inflation data.",
    "The company reported quarterly earnings above estimates.",
    "Markets sold off amid recession fears.",
    "The S&P 500 closed at a record high.",
    "Oil prices surged on supply concerns.",
    "The dollar strengthened against the euro.",
    "Investors are pricing in a rate cut.",
    "The IPO valued the company at billions of dollars.",
    "Treasury yields rose and the yield curve inverted.",
    "Bitcoin and crypto prices fluctuated sharply.",
]

# Domain-general: ATTRIBUTED / sourced reporting of an event ("NYT says X").
NEWS_SEEDS = [
    "The New York Times reports that the policy was overturned.",
    "According to Reuters, officials announced new measures today.",
    "Breaking: the court has issued its ruling in the case.",
    "Sources say the company will lay off thousands of workers.",
    "Bloomberg reports the central bank raised interest rates again.",
    "Officials announced the change in a statement this morning.",
    "A new report finds that prices rose four percent last month.",
    "The senator introduced the bill on the floor today.",
    "Data released today shows unemployment fell to a new low.",
    "The AP confirms the deal was finalized after months of talks.",
]

# Domain-general: personal REACTION / opinion ("I can't believe X happened").
COMMENTARY_SEEDS = [
    "I can't believe this actually happened.",
    "This is absolutely outrageous and completely unacceptable.",
    "Honestly this is insane and I am furious about it.",
    "What a disaster this turned out to be.",
    "I'm so tired of watching this happen over and over again.",
    "It's appalling that anyone thinks this is okay.",
    "I genuinely cannot believe they would do this.",
    "This is the dumbest decision I have seen in a long time.",
    "Absolutely shameful. I have no words.",
    "This makes me so angry, it's exactly what's wrong with everything.",
]

# Domain-general: low-content chatter / online-meta / generic replies.
# Exists to ABSORB filler so it isn't misread as commentary.
CHATTER_SEEDS = [
    "lol this is so true.",
    "I'm always saying this.",
    "this is literally me.",
    "everyone is way too online these days.",
    "honestly idk how any of this works.",
    "just general discourse i guess.",
    "log off and go touch grass.",
    "haha yeah exactly, same.",
    "the For You feed is so weird.",
    "anyway, that's my whole point.",
]

# topic name -> seeds
TOPICS = {
    "us_politics": US_POLITICAL_SEEDS,
    "foreign_politics": FOREIGN_POLITICAL_SEEDS,
    "ai": AI_SEEDS,
    "finance": FINANCE_SEEDS,
}

__all__ = ["TOPICS", "NEUTRAL_SEEDS", "NEWS_SEEDS", "COMMENTARY_SEEDS",
           "CHATTER_SEEDS", "AI_SEEDS", "FINANCE_SEEDS"]
