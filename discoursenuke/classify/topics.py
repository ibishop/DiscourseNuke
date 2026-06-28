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

# Domain-general: takes / opinion / discourse-about-the-topic.
COMMENTARY_SEEDS = [
    "Hot take: this whole thing is wildly overhyped.",
    "Everyone online is overreacting to this, as usual.",
    "I think the real problem is how we talk about this.",
    "My honest opinion is that this approach is misguided.",
    "The discourse around this is exhausting and misses the point.",
    "People keep saying this but they don't get how it works.",
    "The way the media covers this tells you everything.",
    "It's wild how the narrative around this keeps shifting.",
    "Unpopular opinion, but I don't think this matters at all.",
    "This is just more proof of what I've been saying for years.",
]

# topic name -> seeds
TOPICS = {
    "us_politics": US_POLITICAL_SEEDS,
    "foreign_politics": FOREIGN_POLITICAL_SEEDS,
    "ai": AI_SEEDS,
    "finance": FINANCE_SEEDS,
}

__all__ = ["TOPICS", "NEUTRAL_SEEDS", "NEWS_SEEDS", "COMMENTARY_SEEDS",
           "AI_SEEDS", "FINANCE_SEEDS"]
