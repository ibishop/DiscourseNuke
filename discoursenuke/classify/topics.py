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

# Domain-general: factual event reporting.
NEWS_SEEDS = [
    "Company announced a new product today.",
    "Officials released a report showing the latest figures.",
    "The agency said it would take new action.",
    "Prices rose by ten percent according to new data.",
    "A new study was published with these findings.",
    "Breaking: the deal was finalized after months of talks.",
    "Regulators approved the proposal in a vote.",
    "The quarterly results came in above expectations.",
    "A court issued a ruling on the matter today.",
    "The organization launched the initiative this week.",
]

# Domain-general: SUBSTANTIVE opinion / analysis / argument about a subject.
COMMENTARY_SEEDS = [
    "I think this policy will backfire, and here is the reason why.",
    "The fundamental problem with this approach is the incentives it creates.",
    "This ruling sets a dangerous precedent for the years ahead.",
    "My read is that the market has gotten this completely wrong.",
    "What everyone is missing is the structural cause behind this.",
    "The argument that this will work does not hold up to scrutiny.",
    "This is overhyped and will not live up to the expectations.",
    "The strategy here is misguided and likely to fail.",
    "This reflects a deeper problem with how the institution operates.",
    "The real lesson is that we keep repeating the same mistake.",
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
