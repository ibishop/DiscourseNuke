"""Render the nomic politics classifier's flagged posts to HTML for eyeballing.

Scores every post in nomic space (US vs foreign vs neutral), then lists the
US-political posts sorted by margin (most political first), with divider lines
at candidate thresholds so you can see exactly where each cutoff lands and judge
precision/recall by scrolling.

Usage:
    python -m discoursenuke.pipeline.render_politics_html
    open html_view/politics.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from .. import config
from ..classify.classifier import (FOREIGN_POLITICAL_SEEDS, NEUTRAL_SEEDS,
                                    US_POLITICAL_SEEDS)
from ..classify.embedder import Embedder

OUT = Path("html_view")
THRESHOLDS = [0.10, 0.08, 0.05, 0.03, 0.00]


def post_url(author: str, uri: str) -> str:
    if uri.startswith("at://") and "/app.bsky.feed.post/" in uri:
        return f"https://bsky.app/profile/{author}/post/{uri.rsplit('/', 1)[-1]}"
    return f"https://bsky.app/profile/{author}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Render nomic politics-flagged posts to HTML.")
    ap.add_argument("--emb", default=str(config.DATA_DIR / "corpus_emb_nomic.npy"))
    ap.add_argument("--meta", default=str(config.DATA_DIR / "corpus_emb_nomic_meta.jsonl"))
    ap.add_argument("--cap", type=int, default=8000, help="Max cards to render (top by margin).")
    ap.add_argument("--out", default=str(OUT / "politics.html"))
    args = ap.parse_args()

    emb = np.load(args.emb).astype(np.float32)
    meta = [json.loads(l) for l in Path(args.meta).open(encoding="utf-8")]
    print(f"Loaded {emb.shape}. Embedding seeds + scoring ...")

    embedder = Embedder(preset="nomic")
    us = (emb @ embedder.encode(US_POLITICAL_SEEDS).T).max(axis=1)
    fr = (emb @ embedder.encode(FOREIGN_POLITICAL_SEEDS).T).max(axis=1)
    ne = (emb @ embedder.encode(NEUTRAL_SEEDS).T).max(axis=1)
    margin = us - np.maximum(fr, ne)
    is_us_top = (us >= fr) & (us >= ne)

    idx = np.where(is_us_top & (margin >= 0.0))[0]
    idx = idx[np.argsort(-margin[idx])][: args.cap]
    print(f"{len(idx)} US-political posts (margin>=0), rendering top {len(idx)}.")

    # Counts per threshold for the header.
    counts = {t: int((is_us_top & (margin >= t)).sum()) for t in THRESHOLDS}

    parts, next_t = [], 0
    for i in idx:
        m = float(margin[i])
        while next_t < len(THRESHOLDS) and m < THRESHOLDS[next_t]:
            t = THRESHOLDS[next_t]
            parts.append(f'<div class="divider">— threshold {t:.2f} '
                         f'({counts[t]} posts above) —</div>')
            next_t += 1
        rec = meta[i]
        author = html.escape(rec.get("author", ""))
        text = html.escape(rec.get("text", "")).replace("\n", " ")
        url = post_url(rec.get("author", ""), rec.get("uri", ""))
        parts.append(
            f'<div class="post"><span class="m">{m:+.3f}</span> '
            f'<a class="author" href="https://bsky.app/profile/{author}" target="_blank">@{author}</a> '
            f'<a class="permalink" href="{url}" target="_blank">↗</a>'
            f'<div class="text">{text}</div></div>'
        )

    head = " · ".join(f"≥{t:.2f}: {counts[t]}" for t in THRESHOLDS)
    doc = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DiscourseNuke — Politics filter (nomic)</title>
<style>
  :root {{ --bg:#15181c; --card:#1e2329; --line:#2c333b; --fg:#e7ecf0; --mut:#8b97a3; --accent:#3b82f6; --nuke:#ef4444; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  header {{ position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:12px 16px; z-index:5; }}
  h1 {{ margin:0 0 6px; font-size:16px; }}
  .counts {{ color:var(--mut); font-size:12px; }}
  #q {{ width:100%; max-width:520px; margin-top:8px; padding:7px 11px; border-radius:8px; border:1px solid var(--line); background:var(--card); color:var(--fg); font:inherit; }}
  main {{ max-width:720px; margin:0 auto; padding:12px; }}
  .post {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 12px; margin:8px 0; }}
  .m {{ color:var(--nuke); font-weight:700; font-variant-numeric:tabular-nums; margin-right:8px; }}
  .author {{ color:var(--accent); font-weight:600; text-decoration:none; font-size:12px; }}
  .permalink {{ color:var(--mut); text-decoration:none; }}
  .text {{ white-space:pre-wrap; word-wrap:break-word; margin-top:4px; }}
  .divider {{ text-align:center; color:var(--nuke); font-weight:700; margin:18px 0 6px; font-size:13px; border-top:1px dashed var(--nuke); padding-top:8px; }}
</style></head><body>
<header>
  <h1>Politics filter (nomic) — flagged posts by US-political margin</h1>
  <div class="counts">{head}</div>
  <input id="q" placeholder="search flagged posts…">
</header>
<main id="grid">{''.join(parts)}</main>
<script>
  const q=document.getElementById('q'), posts=[...document.querySelectorAll('.post')];
  q.addEventListener('input',()=>{{const s=q.value.toLowerCase();
    for(const p of posts) p.style.display=!s||p.textContent.toLowerCase().includes(s)?'':'none';}});
</script>
</body></html>"""

    OUT.mkdir(exist_ok=True)
    Path(args.out).write_text(doc, encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Open it with:  open {args.out}")


if __name__ == "__main__":
    main()
