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

## Stack (TBD)
- Python (PyCharm project).
- Bluesky / AT Protocol client library (e.g. `atproto`).