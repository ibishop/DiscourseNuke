"""Render the codebook tokens as a browsable HTML page.

Each token (k-means cluster) is shown as a card with its most-representative
posts — i.e. each candidate "discourse" you could select on. Output goes to
html_view/codebook.html (git-ignored). Includes a client-side search box.

Usage:
    python -m discoursenuke.pipeline.render_codebook_html --codebook feed_data/codebook_k512.npz
    open html_view/codebook.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from .. import config

OUT = Path("html_view")


def post_url(author: str, uri: str) -> str:
    if uri.startswith("at://") and "/app.bsky.feed.post/" in uri:
        return f"https://bsky.app/profile/{author}/post/{uri.rsplit('/', 1)[-1]}"
    return f"https://bsky.app/profile/{author}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Render codebook tokens to HTML.")
    ap.add_argument("--codebook", default=str(config.DATA_DIR / "codebook_k512.npz"))
    ap.add_argument("--emb", default=str(config.DATA_DIR / "corpus_emb_nomic_256d.npy"))
    ap.add_argument("--meta", default=str(config.DATA_DIR / "corpus_emb_nomic_meta.jsonl"))
    ap.add_argument("--per-token", type=int, default=6, help="Representative posts per token.")
    ap.add_argument("--out", default=str(OUT / "codebook.html"))
    args = ap.parse_args()

    cb = np.load(args.codebook)
    centroids, labels, sizes = cb["centroids"], cb["labels"], cb["sizes"]
    emb = np.load(args.emb).astype(np.float32)
    meta = [json.loads(l) for l in Path(args.meta).open(encoding="utf-8")]
    k = centroids.shape[0]

    # Tokens sorted by size (biggest discourses first), skip empties.
    order = [t for t in np.argsort(-sizes) if sizes[t] > 0]

    cards = []
    for tok in order:
        members = np.where(labels == tok)[0]
        sims = emb[members] @ centroids[tok]
        top = members[np.argsort(-sims)[: args.per_token]]
        posts_html = ""
        for i in top:
            m = meta[i]
            author = html.escape(m.get("author", ""))
            text = html.escape(m.get("text", "")).replace("\n", " ")
            url = post_url(m.get("author", ""), m.get("uri", ""))
            posts_html += (
                f'<div class="post"><a class="author" href="https://bsky.app/profile/{author}" '
                f'target="_blank">@{author}</a> '
                f'<a class="permalink" href="{url}" target="_blank">↗</a>'
                f'<div class="text">{text}</div></div>'
            )
        cards.append(
            f'<section class="token"><div class="thead">'
            f'<span class="tid">token {int(tok)}</span>'
            f'<span class="tsize">{int(sizes[tok])} posts</span></div>{posts_html}</section>'
        )

    doc = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DiscourseNuke — Codebook ({k} tokens)</title>
<style>
  :root {{ --bg:#15181c; --card:#1e2329; --line:#2c333b; --fg:#e7ecf0; --mut:#8b97a3; --accent:#3b82f6; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  header {{ position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:14px 18px; z-index:5; }}
  h1 {{ margin:0 0 8px; font-size:17px; }}
  #q {{ width:100%; max-width:520px; padding:8px 12px; border-radius:8px; border:1px solid var(--line);
        background:var(--card); color:var(--fg); font:inherit; }}
  .hint {{ color:var(--mut); font-size:12px; margin-top:6px; }}
  main {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:12px; padding:14px; }}
  .token {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px 14px; }}
  .thead {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px;
            border-bottom:1px solid var(--line); padding-bottom:6px; }}
  .tid {{ font-weight:700; }}
  .tsize {{ color:var(--mut); font-size:12px; }}
  .post {{ margin:8px 0; }}
  .author {{ color:var(--accent); font-weight:600; text-decoration:none; font-size:12px; }}
  .permalink {{ color:var(--mut); text-decoration:none; font-size:12px; }}
  .text {{ white-space:pre-wrap; word-wrap:break-word; }}
</style></head><body>
<header>
  <h1>Codebook — {len(order)} tokens (of {k}), sorted by size</h1>
  <input id="q" placeholder="filter tokens by text… (e.g. iran, cats, fed, election)">
  <div class="hint">Each card is one discourse cluster. Type to keep only tokens containing matching posts.</div>
</header>
<main id="grid">{''.join(cards)}</main>
<script>
  const q = document.getElementById('q');
  const cards = [...document.querySelectorAll('.token')];
  q.addEventListener('input', () => {{
    const s = q.value.toLowerCase();
    for (const c of cards) c.style.display = !s || c.textContent.toLowerCase().includes(s) ? '' : 'none';
  }});
</script>
</body></html>"""

    OUT.mkdir(exist_ok=True)
    Path(args.out).write_text(doc, encoding="utf-8")
    print(f"Wrote {args.out} — {len(order)} tokens.")
    print(f"Open it with:  open {args.out}")


if __name__ == "__main__":
    main()
