"""A blind sheet for choosing the best translations among several models.

For each sampled text: the frame with the text's box marked, the crop, the checked Korean, and every model's
translation (one condition, by default with the type's guide) as anonymous lettered cards in a per-text random
order. Identical translations are shown once and credited to every model that gave them. The rater clicks the
best one, or several if they are equally good, or "none is good"; marks stay in the browser and download as a CSV.
The letter-to-model key is written to a separate JSON file, so the sheet itself does not reveal the models.

    python -m eval.model_pick_sheet --runs <chapter_full_results.json ...> --blocks <blocks_v2.csv ...> \
        --calls <dir with one subfolder per model holding calls.jsonl> ... --per-type 4 --out pick.html --key pick_key.json
    python -m eval.model_pick_sheet --key pick_key.json --score model_pick.csv   # share of texts each model wins
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from eval.ocr_baselines import crop
from eval.ocr_check_sheet import b64
from eval.sample_frames import FAMILIES, local_path

LETTERS = "ABCDEFGHIJKLMNOP"

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Best translation</title><style>
:root{--ink:#16161d;--muted:#6b6b78;--line:#e4e4ea;--bg:#fafafb;--card:#fff;--acc:#5b47ff;--acc-t:#ecebff;--bad:#d43c3c}
@media (prefers-color-scheme:dark){:root{--ink:#ececf1;--muted:#9a9aa8;--line:#2c2c36;--bg:#111116;--card:#1a1a21;--acc:#9385ff;--acc-t:#262147}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,"Noto Sans KR","Malgun Gothic",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header b{font-size:16px}.grow{flex:1}.btn{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:7px 14px;cursor:pointer;font:inherit}
.btn.primary{background:var(--acc);color:#fff;border-color:transparent}main{max-width:1180px;margin:0 auto;padding:16px}
.help{color:var(--muted);max-width:820px}.tag{font-size:12px;border-radius:99px;padding:2px 9px;background:var(--line)}
.text{background:var(--card);border:1px solid var(--line);border-radius:14px;margin:0 0 22px;padding:16px;display:grid;grid-template-columns:minmax(0,300px) minmax(0,1fr);gap:18px}
@media (max-width:820px){.text{grid-template-columns:1fr}}
.page{width:100%;border-radius:6px;border:1px solid var(--line)}.crop{max-width:100%;max-height:180px;border:1px solid var(--line);border-radius:6px;background:#fff}
.ko{font-size:20px;margin:8px 0 12px;word-break:keep-all}.n{font-weight:700;color:var(--acc)}
.opts{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px}
.opt{position:relative;text-align:left;border:1.5px solid var(--line);background:var(--bg);color:var(--ink);border-radius:10px;padding:10px 12px 10px 40px;cursor:pointer;font:inherit;font-size:16px;line-height:1.45}
.opt:hover{border-color:var(--acc)}.opt b{position:absolute;left:10px;top:10px;width:22px;height:22px;border-radius:6px;display:grid;place-items:center;background:var(--line);font-size:12px}
.opt.on{border-color:var(--acc);background:var(--acc-t)}.opt.on b{background:var(--acc);color:#fff}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:12px}.none.on{background:var(--bad);color:#fff;border-color:transparent}
input.note{flex:1;min-width:200px;border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:8px;padding:7px 10px;font:inherit}
.done{opacity:.6}
</style></head><body>
<header><b>Best translation</b><span id="count" class="tag"></span><span class="grow"></span>
<label><input type="checkbox" id="only"> only open</label><button class="btn primary" onclick="exportCsv()">Download CSV</button></header>
<main><p class="help">Each card is one line from a Korean webtoon: the frame with the line's box, the line itself, and several English versions in a random order.
Click the version that works best <b>on this page</b> (meaning, tone and how it would sit in that balloon or lettering). If two or more are equally best, click each of them.
If none is acceptable, press <b>None is good</b>. A short note is welcome but optional. Your marks are saved in this browser; press <b>Download CSV</b> when you finish and send the file back.</p>
__BODY__</main>
<script>
const KEY="modelpick:"+__N__;let S={};try{S=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}paint()}
function tog(u,l){const s=S[u]=S[u]||{p:[]};s.none=false;s.p=s.p.includes(l)?s.p.filter(x=>x!==l):[...s.p,l].sort();save()}
function none(u){const s=S[u]=S[u]||{p:[]};s.none=!s.none;if(s.none)s.p=[];save()}
function note(u,t){(S[u]=S[u]||{p:[]}).note=t;try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}}
const answered=s=>s&&(s.none||(s.p||[]).length>0);
function paint(){let n=0;const only=document.getElementById("only").checked;
 document.querySelectorAll(".text").forEach(t=>{const s=S[t.dataset.u]||{p:[]};const a=answered(s);if(a)n++;
  t.querySelectorAll(".opt").forEach(o=>o.classList.toggle("on",(s.p||[]).includes(o.dataset.l)));
  t.querySelector(".none").classList.toggle("on",!!s.none);const i=t.querySelector("input.note");if(document.activeElement!==i)i.value=s.note||"";
  t.classList.toggle("done",a&&!only);t.style.display=only&&a?"none":""});
 document.getElementById("count").textContent=n+" / "+__N__+" answered"}
document.getElementById("only").onchange=paint;
function exportCsv(){const q=s=>'"'+String(s??"").replace(/"/g,'""')+'"';const rows=[["text_id","best","none_good","note"]];
 document.querySelectorAll(".text").forEach(t=>{const s=S[t.dataset.u]||{};rows.push([t.dataset.u,(s.p||[]).join(" "),s.none?1:"",s.note||""])});
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob(["\\ufeff"+rows.map(r=>r.map(q).join(",")).join("\\n")],{type:"text/csv"}));a.download="model_pick.csv";a.click()}
paint();
</script></body></html>"""


def norm(s: str) -> str:
    return " ".join((s or "").split())


def build(a) -> None:
    rng = random.Random(a.seed)
    cls, src = {}, {}
    for f in a.blocks:
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            cls[r["block_uid"]] = r["class_name"]
    # one translation per model and text: model -> uid -> text
    tr = defaultdict(dict)
    for d in a.calls:
        for f in Path(d).glob("*/calls.jsonl"):
            for line in open(f, encoding="utf-8"):
                c = json.loads(line)
                if c.get("condition") == a.condition and c["block_uid"] in cls and c.get("translation") and not c.get("error"):
                    tr[c["model"]][c["block_uid"]] = norm(c["translation"])
                    src[c["block_uid"]] = c.get("source_text", "")
    models = sorted(m for m in tr if not a.models or m in a.models)
    full = [u for u in cls if all(u in tr[m] for m in models)]
    by_type = defaultdict(list)
    for u in sorted(full):
        by_type[cls[u]].append(u)
    picked = []
    for t, us in sorted(by_type.items()):
        rng.shuffle(us)
        seen_frames = set()
        for u in us:  # spread over frames: one text per frame first
            fr = u.rsplit("/", 1)[0]
            if len([x for x in picked if cls[x] == t]) >= a.per_type:
                break
            if fr in seen_frames:
                continue
            seen_frames.add(fr)
            picked.append(u)
    rng.shuffle(picked)

    frames = {}
    for rf in a.runs:
        data = json.loads(Path(rf).read_text(encoding="utf-8"))
        run = Path(rf).parent.name
        for fr in data.get("frames", []):
            for b in fr.get("blocks", []):
                frames[f"{run}/{b.get('frame_id', fr.get('id'))}/{b.get('id')}"] = (fr["image_path"], b["box"])

    key, cards = {}, []
    for n, u in enumerate(picked, 1):
        texts = defaultdict(list)
        for m in models:
            texts[tr[m][u]].append(m)
        items = list(texts.items())
        rng.shuffle(items)
        tid = f"t{n:03d}"
        key[tid] = {"block_uid": u, "class_name": cls[u], "letters": {LETTERS[i]: ms for i, (_, ms) in enumerate(items)}}
        img = cv2.imread(str(local_path(frames[u][0], a.project_root)))
        x0, y0, x1, y1 = map(int, frames[u][1])
        page = img.copy()
        cv2.rectangle(page, (x0, y0), (x1, y1), (60, 60, 230), max(3, img.shape[1] // 200))
        sc = min(1.0, 600 / page.shape[1])
        page = cv2.resize(page, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        opts = "".join(f'<button class="opt" data-l="{LETTERS[i]}" onclick="tog(\'{tid}\',\'{LETTERS[i]}\')"><b>{LETTERS[i]}</b>{html.escape(t)}</button>'
                       for i, (t, _) in enumerate(items))
        cards.append(
            f'<section class="text" data-u="{tid}"><div><img class="page" loading="lazy" src="data:image/jpeg;base64,{b64(page, 78)}"></div>'
            f'<div><span class="n">{n}</span> <img class="crop" src="data:image/jpeg;base64,{b64(crop(img, frames[u][1], pad=8))}">'
            f'<div class="ko" lang="ko">{html.escape(src.get(u, ""))}</div><div class="opts">{opts}</div>'
            f'<div class="row"><button class="btn none" onclick="none(\'{tid}\')">None is good</button>'
            f'<input class="note" placeholder="Note (optional)" oninput="note(\'{tid}\',this.value)"></div></div></section>')
    a.out.write_text(PAGE.replace("__BODY__", "\n".join(cards)).replace("__N__", str(len(picked))), encoding="utf-8")
    a.key.write_text(json.dumps({"condition": a.condition, "models": models, "texts": key}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(picked)} texts, {len(models)} models -> {a.out} (key: {a.key})")
    print("types:", dict(Counter(cls[u] for u in picked)))


def score(a) -> None:
    k = json.loads(a.key.read_text(encoding="utf-8"))
    models = k["models"]
    wins, sole, n, nones = Counter(), Counter(), 0, 0
    fam_wins, fam_n = defaultdict(Counter), Counter()
    for r in csv.DictReader(open(a.score, encoding="utf-8-sig")):
        t = k["texts"].get(r["text_id"])
        if not t or not (r["best"].strip() or r["none_good"].strip()):
            continue
        n += 1
        fam = FAMILIES.get(t["class_name"], "other")
        fam_n[fam] += 1
        if r["none_good"].strip():
            nones += 1
            continue
        best = {m for l in r["best"].split() for m in t["letters"].get(l, [])}
        for m in best:
            wins[m] += 1
            fam_wins[fam][m] += 1
        if len(best) == 1:
            sole[next(iter(best))] += 1
    print(f"{n} texts answered, {nones} with none good\n")
    fams = sorted(fam_n)
    print("| Model | Among the best | Alone best | " + " | ".join(f"{f} (n={fam_n[f]})" for f in fams) + " |")
    print("|---|---|---|" + "---|" * len(fams))
    for m in sorted(models, key=lambda m: -wins[m]):
        print(f"| {m} | {wins[m] / max(1, n):.0%} | {sole[m] / max(1, n):.0%} | " + " | ".join(f"{fam_wins[f][m] / max(1, fam_n[f]):.0%}" for f in fams) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--blocks", nargs="*", default=[])
    ap.add_argument("--calls", nargs="*", default=[], help="folders holding one subfolder per model with calls.jsonl")
    ap.add_argument("--models", nargs="*", default=[], help="model ids to include (default: all with every text)")
    ap.add_argument("--condition", default="true_class")
    ap.add_argument("--per-type", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--score", type=Path, help="the downloaded CSV: print the per-model table and stop")
    a = ap.parse_args()
    score(a) if a.score else build(a)


if __name__ == "__main__":
    main()
