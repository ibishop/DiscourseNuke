"""Render the filtered feed as a browsable HTML page in html_view/ (git-ignored).

Reads feed_data/filtered_feed.json (kept) and feed_data/nuked_feed.json
(removed) and writes html_view/index.html with two tabs.

Usage:
    python render_html.py
    open html_view/index.html
"""

from __future__ import annotations

import html
import json
from pathlib import Path

DATA = Path("feed_data")
OUT = Path("html_view")


def load(name: str) -> list[dict]:
    p = DATA / name
    return json.loads(p.read_text())["posts"] if p.exists() else []


def post_uri_to_url(author: str, uri: str) -> str:
    # at://did/app.bsky.feed.post/<rkey>  ->  https://bsky.app/profile/<author>/post/<rkey>
    if uri.startswith("at://") and "/app.bsky.feed.post/" in uri:
        rkey = uri.rsplit("/", 1)[-1]
        return f"https://bsky.app/profile/{author}/post/{rkey}"
    return f"https://bsky.app/profile/{author}"


def card(p: dict, show_score: bool) -> str:
    author = html.escape(p.get("author", ""))
    text = html.escape(p.get("text", "")).replace("\n", "<br>")
    when = html.escape((p.get("created_at", "") or "")[:16].replace("T", " "))
    url = post_uri_to_url(p.get("author", ""), p.get("uri", ""))
    badge = ""
    if show_score and "score" in p:
        badge = f'<span class="score">US-political {p["score"]:+.3f}</span>'
    return f"""
    <article class="post">
      <div class="head">
        <a class="author" href="https://bsky.app/profile/{author}" target="_blank">@{author}</a>
        <span class="meta">{when}{badge}</span>
      </div>
      <div class="text">{text}</div>
      <a class="permalink" href="{url}" target="_blank">open ↗</a>
    </article>"""


def main() -> None:
    kept = load("filtered_feed.json")
    nuked = load("nuked_feed.json")
    OUT.mkdir(exist_ok=True)

    kept_html = "\n".join(card(p, show_score=False) for p in kept) or "<p class='empty'>No posts.</p>"
    nuked_html = "\n".join(card(p, show_score=True) for p in
                           sorted(nuked, key=lambda x: -x.get("score", 0))) or "<p class='empty'>No posts.</p>"

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DiscourseNuke — Feed</title>
<style>
  :root {{ --bg:#15181c; --card:#1e2329; --line:#2c333b; --fg:#e7ecf0; --mut:#8b97a3; --accent:#3b82f6; --nuke:#ef4444; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  header {{ position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:16px 20px; }}
  h1 {{ margin:0 0 10px; font-size:18px; }}
  .tabs {{ display:flex; gap:8px; }}
  .tab {{ background:none; border:1px solid var(--line); color:var(--mut); padding:6px 14px; border-radius:999px; cursor:pointer; font:inherit; }}
  .tab.active {{ background:var(--card); color:var(--fg); border-color:var(--accent); }}
  .tab .n {{ opacity:.6; margin-left:6px; }}
  main {{ max-width:640px; margin:0 auto; padding:16px 12px 80px; }}
  .post {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; margin:10px 0; position:relative; }}
  .head {{ display:flex; justify-content:space-between; align-items:baseline; gap:10px; margin-bottom:6px; }}
  .author {{ color:var(--fg); font-weight:600; text-decoration:none; }}
  .author:hover {{ text-decoration:underline; }}
  .meta {{ color:var(--mut); font-size:12px; white-space:nowrap; }}
  .score {{ color:var(--nuke); margin-left:8px; font-variant-numeric:tabular-nums; }}
  .text {{ white-space:pre-wrap; word-wrap:break-word; }}
  .permalink {{ display:inline-block; margin-top:8px; color:var(--mut); font-size:12px; text-decoration:none; }}
  .permalink:hover {{ color:var(--accent); }}
  .empty {{ color:var(--mut); text-align:center; padding:40px; }}
  .view {{ display:none; }} .view.active {{ display:block; }}
</style>
</head>
<body>
<header>
  <h1>DiscourseNuke — mutuals feed</h1>
  <div class="tabs">
    <button class="tab active" data-view="kept">Cleaned feed <span class="n">{len(kept)}</span></button>
    <button class="tab" data-view="nuked">Nuked (US politics) <span class="n">{len(nuked)}</span></button>
  </div>
</header>
<main>
  <section id="kept" class="view active">{kept_html}</section>
  <section id="nuked" class="view">{nuked_html}</section>
</main>
<script>
  document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {{
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.view').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById(t.dataset.view).classList.add('active');
  }}));
</script>
</body>
</html>"""

    out = OUT / "index.html"
    out.write_text(doc, encoding="utf-8")
    print(f"Wrote {out} — {len(kept)} kept, {len(nuked)} nuked.")
    print(f"Open it with:  open {out}")


if __name__ == "__main__":
    main()
