"""Render the multi-topic / 3-way-type classification to a browsable HTML page.

Posts are tagged with topic (us_politics/foreign_politics/ai/finance) and type
(news/commentary/chatter). The page has topic + type filter pills and a search
box so you can scroll any bucket and tune the seeds. Output html_view/topics.html.

Usage:
    python -m discoursenuke.pipeline.render_topics_html
    open html_view/topics.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from .. import config
from ..classify.embedder import Embedder
from ..classify.topics import (CHATTER_SEEDS, COMMENTARY_SEEDS, NEUTRAL_SEEDS,
                               NEWS_SEEDS, TOPICS)

OUT = Path("html_view")
TYPES = ["news", "commentary", "chatter"]


def nearest(posts, seeds):
    return (posts @ seeds.T).max(axis=1)


def post_url(author, uri):
    if uri.startswith("at://") and "/app.bsky.feed.post/" in uri:
        return f"https://bsky.app/profile/{author}/post/{uri.rsplit('/', 1)[-1]}"
    return f"https://bsky.app/profile/{author}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Render topic x type buckets to HTML.")
    ap.add_argument("--emb", default=str(config.DATA_DIR / "corpus_emb_nomic.npy"))
    ap.add_argument("--meta", default=str(config.DATA_DIR / "corpus_emb_nomic_meta.jsonl"))
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--per-bucket", type=int, default=300, help="Max cards per topic x type.")
    ap.add_argument("--out", default=str(OUT / "topics.html"))
    args = ap.parse_args()

    emb = np.load(args.emb).astype(np.float32)
    meta = [json.loads(l) for l in Path(args.meta).open(encoding="utf-8")]
    print(f"Loaded {emb.shape}. Scoring topics + types ...")

    e = Embedder(preset="nomic")
    topic_names = list(TOPICS)
    topic_scores = np.stack([nearest(emb, e.encode(TOPICS[t])) for t in topic_names], axis=1)
    neutral = nearest(emb, e.encode(NEUTRAL_SEEDS))
    # z-score each type axis so commentary (lower absolute sim) competes fairly.
    def zscore(x):
        return (x - x.mean()) / (x.std() + 1e-9)
    news = zscore(nearest(emb, e.encode(NEWS_SEEDS)))
    comm = zscore(nearest(emb, e.encode(COMMENTARY_SEEDS)))
    chat = zscore(nearest(emb, e.encode(CHATTER_SEEDS)))

    best = topic_scores.argmax(axis=1)
    best_score = topic_scores.max(axis=1)
    is_topic = (best_score - neutral) >= args.threshold
    type_idx = np.stack([news, comm, chat], axis=1).argmax(axis=1)

    cards = []
    counts = {}
    for ti, t in enumerate(topic_names):
        for ty in range(3):
            mask = is_topic & (best == ti) & (type_idx == ty)
            counts[(t, TYPES[ty])] = int(mask.sum())
            order = [i for i in np.argsort(-best_score) if mask[i]][: args.per_bucket]
            for i in order:
                author = html.escape(meta[i].get("author", ""))
                text = html.escape(meta[i].get("text", "")).replace("\n", " ")
                url = post_url(meta[i].get("author", ""), meta[i].get("uri", ""))
                cards.append(
                    f'<div class="post" data-topic="{t}" data-type="{TYPES[ty]}">'
                    f'<div class="badges"><span class="b topic">{t}</span>'
                    f'<span class="b type t-{TYPES[ty]}">{TYPES[ty]}</span> '
                    f'<a class="author" href="https://bsky.app/profile/{author}" target="_blank">@{author}</a> '
                    f'<a class="permalink" href="{url}" target="_blank">↗</a></div>'
                    f'<div class="text">{text}</div></div>'
                )

    def pills(name, vals):
        out = f'<span class="lbl">{name}:</span>'
        out += f'<button class="pill active" data-{name}="all">all</button>'
        for v in vals:
            out += f'<button class="pill" data-{name}="{v}">{v}</button>'
        return out

    count_rows = "".join(
        f"<tr><td>{t}</td>" + "".join(f"<td>{counts[(t, ty)]}</td>" for ty in TYPES) + "</tr>"
        for t in topic_names
    )

    doc = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DiscourseNuke — Topics x Types</title>
<style>
  :root {{ --bg:#15181c; --card:#1e2329; --line:#2c333b; --fg:#e7ecf0; --mut:#8b97a3; --accent:#3b82f6;
           --news:#22c55e; --commentary:#eab308; --chatter:#a855f7; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  header {{ position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:12px 16px; z-index:5; }}
  h1 {{ margin:0 0 8px; font-size:16px; }}
  .row {{ margin:6px 0; display:flex; flex-wrap:wrap; gap:6px; align-items:center; }}
  .lbl {{ color:var(--mut); font-size:12px; margin-right:4px; }}
  .pill {{ background:var(--card); border:1px solid var(--line); color:var(--fg); padding:4px 10px;
           border-radius:999px; cursor:pointer; font:inherit; font-size:12px; }}
  .pill.active {{ border-color:var(--accent); background:#24304a; }}
  #q {{ width:100%; max-width:480px; padding:7px 11px; border-radius:8px; border:1px solid var(--line); background:var(--card); color:var(--fg); font:inherit; }}
  table {{ border-collapse:collapse; font-size:12px; color:var(--mut); margin-top:6px; }}
  td,th {{ border:1px solid var(--line); padding:2px 8px; text-align:right; }}
  td:first-child,th:first-child {{ text-align:left; }}
  main {{ max-width:720px; margin:0 auto; padding:12px; }}
  .post {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:9px 12px; margin:8px 0; }}
  .badges {{ font-size:12px; margin-bottom:3px; }}
  .b {{ border-radius:4px; padding:1px 6px; font-weight:600; font-size:11px; }}
  .b.topic {{ background:#2a3344; color:#cbd5e1; }}
  .t-news {{ background:var(--news); color:#06280f; }}
  .t-commentary {{ background:var(--commentary); color:#2a2200; }}
  .t-chatter {{ background:var(--chatter); color:#1c0633; }}
  .author {{ color:var(--accent); text-decoration:none; }} .permalink {{ color:var(--mut); text-decoration:none; }}
  .text {{ white-space:pre-wrap; word-wrap:break-word; }}
</style></head><body>
<header>
  <h1>Topics × Types (top {args.per_bucket}/bucket by topic affinity)</h1>
  <div class="row">{pills("topic", topic_names)}</div>
  <div class="row">{pills("type", TYPES)}</div>
  <div class="row"><input id="q" placeholder="search…"></div>
  <table><tr><th>topic</th><th>news</th><th>commentary</th><th>chatter</th></tr>{count_rows}</table>
</header>
<main id="grid">{''.join(cards)}</main>
<script>
  let fTopic='all', fType='all', q='';
  const posts=[...document.querySelectorAll('.post')];
  function apply(){{
    for(const p of posts){{
      const okT = fTopic==='all'||p.dataset.topic===fTopic;
      const okY = fType==='all'||p.dataset.type===fType;
      const okQ = !q||p.textContent.toLowerCase().includes(q);
      p.style.display = okT&&okY&&okQ ? '' : 'none';
    }}
  }}
  document.querySelectorAll('[data-topic]').forEach(b=>{{ if(b.tagName==='BUTTON') b.onclick=()=>{{
    fTopic=b.dataset.topic; document.querySelectorAll('[data-topic]').forEach(x=>{{if(x.tagName==='BUTTON')x.classList.toggle('active',x===b);}}); apply();
  }};}});
  document.querySelectorAll('[data-type]').forEach(b=>{{ if(b.tagName==='BUTTON') b.onclick=()=>{{
    fType=b.dataset.type; document.querySelectorAll('[data-type]').forEach(x=>{{if(x.tagName==='BUTTON')x.classList.toggle('active',x===b);}}); apply();
  }};}});
  document.getElementById('q').addEventListener('input',ev=>{{q=ev.target.value.toLowerCase();apply();}});
</script>
</body></html>"""

    OUT.mkdir(exist_ok=True)
    Path(args.out).write_text(doc, encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Open it with:  open {args.out}")


if __name__ == "__main__":
    main()
