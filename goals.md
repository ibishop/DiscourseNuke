# DiscourseNuke — Goals

## What this is
A tool for "nuking" certain discourses from a Bluesky feed — i.e. identifying
posts that belong to topics/conversations a user wants to stop seeing, and
filtering them out of their feed.

## Core goals
- [ ] Connect to Bluesky (AT Protocol) and read a user's feed.
- [ ] Define what a "discourse" is — a topic, theme, or recurring conversation
      that can be matched against posts (keywords, phrases, and/or semantic
      similarity).
- [ ] Let the user specify which discourses to nuke.
- [ ] Classify incoming posts against the user's nuke list.
- [ ] Remove / hide matching posts from the feed the user sees.

## Open questions
- How is a "discourse" specified? Simple keyword lists vs. embeddings/LLM
  classification vs. a hybrid.
- Where does filtering happen? Client-side display filter, a custom feed
  generator, or a moderation/labeling service.
- How aggressive is matching? Tradeoff between over-filtering (false positives)
  and letting discourse leak through.
- Persistence: where are a user's nuke lists stored?

## Non-goals (for now)
- Posting or otherwise writing to Bluesky on the user's behalf.
- Moderating other users' feeds.

## Stack
- Python (PyCharm project), venv in `.venv`.
- `atproto` (Bluesky client), `sentence-transformers` (embeddings), `numpy`, `requests`, `python-dotenv`.

## PoC #1 — embedding classifier (built 2026-06-28)
Status: **working end-to-end, but not the direction we want to keep.**

What it does:
- `auth.py` — login with a Bluesky app password, cache the session token to
  `.session`, resume from it on later runs (token-based storage).
- `bluesky.py` — pull the authenticated home timeline (`get_timeline`) or any
  user's public author feed (`get_author_feed`).
- `classifier.py` — embed each post with `all-MiniLM-L6-v2`, score via
  nearest-seed cosine against three sets (US-political / foreign-political /
  neutral), nuke only posts closest to the US-political anchors.
- `main.py` — pull + classify + show nuked. Flags: `--timeline` (default),
  `--actor`, `--sample`, `--threshold`, `--show-kept`.

What we learned:
- Embeddings separate "political vs not" well.
- A foreign-political seed set adds usable geography: ~333/1000 timeline posts
  (UK/Canada/EU/etc.) correctly spared while US politics is nuked.
- Limits: precision tops out at the margin — residual false positives on posts
  with a faint US lean (e.g. an Apple/AI Bloomberg post, arXiv links), and
  country-ambiguous fragments are hard. Recall on conversational politics is
  decent but threshold-sensitive.

Verdict: not quite what we're looking for — revisit the approach.
(TODO: capture what we *do* want the matching to look like.)