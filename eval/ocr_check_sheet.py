"""A self-contained HTML sheet for checking reference transcriptions by eye.

Every frame of the sample is shown with its text blocks numbered; under it, each block's crop sits beside the
reference reading (from llm_transcribe consensus) or, where the models disagree, beside every model's reading.
The checker marks each block "matches" or "error" (and may pick the right reading or type a note); marks are kept
in the browser and downloaded as a CSV (block_uid, verdict, pick, note). `--apply` turns it into the checked
reference: blocks marked "matches" keep their reading (or the picked one); blocks marked "error" are left out. No Korean is needed: the check is whether every drawn character appears, in order, and nothing else.

    python -m eval.ocr_check_sheet --runs data/output/<runs>/*/chapter_full_results.json \
        --blocks data/study/sample/blocks.csv --llm-ocr data/study/llm_ocr --out data/study/ocr_check.html
    python -m eval.ocr_check_sheet --llm-ocr data/study/llm_ocr --apply ocr_check.csv   # writes reference_checked.csv
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from eval.ocr_baselines import crop
from eval.sample_frames import local_path


def b64(img, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Transcription check</title><style>
:root{--ink:#16161d;--muted:#6b6b78;--line:#e4e4ea;--bg:#fafafb;--card:#fff;--ok:#1f9d55;--bad:#d43c3c;--acc:#5b47ff}
@media (prefers-color-scheme:dark){:root{--ink:#ececf1;--muted:#9a9aa8;--line:#2c2c36;--bg:#111116;--card:#1a1a21;--acc:#9385ff}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,"Noto Sans KR","Malgun Gothic",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header b{font-size:16px}.grow{flex:1}.btn{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:6px 12px;cursor:pointer;font:inherit}
.btn.primary{background:var(--acc);color:#fff;border-color:transparent}main{max-width:1100px;margin:0 auto;padding:16px}
.frame{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:0 0 20px;padding:14px;display:grid;grid-template-columns:minmax(0,320px) minmax(0,1fr);gap:16px}
@media (max-width:760px){.frame{grid-template-columns:1fr}}
.frame img.page{width:100%;border-radius:6px;border:1px solid var(--line)}.blk{display:grid;grid-template-columns:28px minmax(0,220px) minmax(0,1fr);gap:10px;padding:10px 0;border-top:1px solid var(--line);align-items:start}
.blk:first-child{border-top:0}.n{font-weight:700;color:var(--acc)}.crop{max-width:100%;max-height:220px;border:1px solid var(--line);border-radius:4px;background:#fff}
.read{font-size:19px;word-break:keep-all}.alt{font-size:15px;margin:2px 0}.alt label{cursor:pointer}.model{color:var(--muted);font-size:12px}
.acts{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}.acts button.on.ok{background:var(--ok);color:#fff;border-color:transparent}.acts button.on.bad{background:var(--bad);color:#fff;border-color:transparent}
input.note{width:100%;box-sizing:border-box;margin-top:6px;border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:6px;padding:5px 8px;font:inherit}
.tag{font-size:12px;border-radius:99px;padding:1px 8px;background:var(--line)}.tag.dis{background:#ffe3c2;color:#7a4300}
.help{color:var(--muted);font-size:14px;max-width:760px}
</style></head><body>
<header><b>Transcription check</b><span id="count" class="tag"></span><span class="grow"></span>
<label><input type="checkbox" id="only"> only unchecked</label>
<button class="btn" onclick="exportCsv()">Download CSV</button></header>
<main><p class="help">For each numbered text, compare the crop with the reading on the right, character by character. Press <b>Matches</b> when every drawn character is there, in order, with nothing extra (spaces and line breaks do not matter). Press <b>Error</b> otherwise and say briefly what is wrong. Where the models disagree, pick the reading that matches, or none. Marks are saved in this browser; download the CSV when done.</p>
__BODY__</main>
<script>
const KEY="ocrcheck:"+__N__;let S={};try{S=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}paint()}
function mark(u,v){S[u]=Object.assign(S[u]||{},{verdict:v});save()}
function pick(u,p){S[u]=Object.assign(S[u]||{},{pick:p,verdict:p==="none"?"error":"matches"});save()}
function note(u,t){S[u]=Object.assign(S[u]||{},{note:t});save()}
function paint(){let n=0;document.querySelectorAll(".blk").forEach(b=>{const s=S[b.dataset.u]||{};if(s.verdict)n++;
 b.querySelectorAll(".acts button").forEach(x=>x.classList.toggle("on",x.dataset.v===s.verdict));
 b.querySelectorAll("input[type=radio]").forEach(x=>x.checked=(x.value===s.pick));const i=b.querySelector("input.note");if(i&&document.activeElement!==i)i.value=s.note||"";});
 const only=document.getElementById("only").checked;document.querySelectorAll(".frame").forEach(f=>{const open=[...f.querySelectorAll(".blk")].some(b=>!(S[b.dataset.u]||{}).verdict);f.style.display=only&&!open?"none":""});
 document.getElementById("count").textContent=n+" / "+__N__+" checked"}
document.getElementById("only").onchange=paint;
function exportCsv(){const q=s=>'"'+String(s??"").replace(/"/g,'""')+'"';const rows=[["block_uid","verdict","pick","note"]];
 document.querySelectorAll(".blk").forEach(b=>{const s=S[b.dataset.u]||{};rows.push([b.dataset.u,s.verdict||"",s.pick||"",s.note||""])});
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob(["\\ufeff"+rows.map(r=>r.map(q).join(",")).join("\\n")],{type:"text/csv"}));a.download="ocr_check.csv";a.click()}
paint();
</script></body></html>"""


def apply(llm_ocr: Path, checks: Path) -> None:
    ref = {r["block_uid"]: r["gt_text"] for r in csv.DictReader(open(llm_ocr / "reference.csv", encoding="utf-8-sig"))}
    readings = {r["block_uid"]: r for r in csv.DictReader(open(llm_ocr / "readings.csv", encoding="utf-8-sig"))}
    keep, n = [], Counter()
    for c in csv.DictReader(open(checks, encoding="utf-8-sig")):
        u, v = c["block_uid"], c["verdict"]
        n[v or "unchecked"] += 1
        text = readings.get(u, {}).get(c["pick"]) if c.get("pick") not in ("", "none", None) else ref.get(u)
        if v == "matches" and text:
            keep.append([u, text, "checked"])
    with open(llm_ocr / "reference_checked.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["block_uid", "gt_text", "agree"])
        w.writerows(keep)
    print(f"reference_checked.csv: {len(keep)} blocks; marks: {dict(n)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+")
    ap.add_argument("--blocks", type=Path)
    ap.add_argument("--apply", type=Path, help="the downloaded CSV of marks: write reference_checked.csv and stop")
    ap.add_argument("--llm-ocr", type=Path, required=True, help="folder with readings.csv and reference.csv")
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.apply:
        return apply(a.llm_ocr, a.apply)
    if not (a.runs and a.blocks and a.out):
        ap.error("--runs, --blocks and --out are needed to build the sheet")

    want = [r["block_uid"] for r in csv.DictReader(open(a.blocks, encoding="utf-8-sig"))]
    ref = {r["block_uid"]: r["gt_text"] for r in csv.DictReader(open(a.llm_ocr / "reference.csv", encoding="utf-8-sig"))}
    rd = list(csv.DictReader(open(a.llm_ocr / "readings.csv", encoding="utf-8-sig")))
    models = [k for k in rd[0] if k != "block_uid"] if rd else []
    readings = {r["block_uid"]: r for r in rd}
    where, frames = {}, defaultdict(list)
    for f in a.runs:
        run = f.parent.name
        for fr in json.loads(f.read_text(encoding="utf-8")).get("frames", []):
            for b in fr.get("blocks", []):
                u = f"{run}/{b.get('frame_id', fr.get('id'))}/{b.get('id')}"
                where[u] = (fr["image_path"], b["box"])
    for u in want:
        frames[where[u][0]].append(u)

    out = []
    for fi, (path, us) in enumerate(frames.items(), 1):
        img = cv2.imread(str(local_path(path, a.project_root)))
        page = img.copy()
        for i, u in enumerate(us, 1):
            x0, y0, x1, y1 = [int(v) for v in where[u][1]]
            cv2.rectangle(page, (x0, y0), (x1, y1), (255, 71, 91), 3)
            cv2.putText(page, str(i), (x0 + 4, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 71, 91), 3)
        h, w = page.shape[:2]
        page = cv2.resize(page, (640, int(h * 640 / w))) if w > 640 else page
        rows = []
        for i, u in enumerate(us, 1):
            c = b64(crop(img, where[u][1], pad=8))
            if u in ref:
                body = f'<div class="read">{html.escape(ref[u])}</div>'
            else:
                r = readings.get(u, {})
                alts = "".join(
                    f'<div class="alt"><label><input type="radio" name="p{fi}_{i}" value="{html.escape(m)}" onchange="pick(\'{u}\',this.value)"> '
                    f'{html.escape(r.get(m, "") or "(empty)")} <span class="model">{html.escape(m.split("/")[0])}</span></label></div>' for m in models)
                body = (f'<span class="tag dis">models disagree: pick the matching one</span>{alts}'
                        f'<div class="alt"><label><input type="radio" name="p{fi}_{i}" value="none" onchange="pick(\'{u}\',this.value)"> none of these</label></div>')
            rows.append(
                f'<div class="blk" data-u="{html.escape(u)}"><span class="n">{i}</span><img class="crop" loading="lazy" src="data:image/jpeg;base64,{c}"><div>{body}'
                f'<div class="acts"><button class="btn ok" data-v="matches" onclick="mark(\'{u}\',\'matches\')">Matches</button>'
                f'<button class="btn bad" data-v="error" onclick="mark(\'{u}\',\'error\')">Error</button></div>'
                f'<input class="note" placeholder="What is wrong (optional)" onchange="note(\'{u}\',this.value)"></div></div>')
        out.append(f'<section class="frame"><div><div class="model">Frame {fi} of {len(frames)}</div>'
                   f'<img class="page" loading="lazy" src="data:image/jpeg;base64,{b64(page, 80)}"></div><div>{"".join(rows)}</div></section>')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(PAGE.replace("__BODY__", "\n".join(out)).replace("__N__", str(len(want))), encoding="utf-8")
    print(f"wrote {a.out}: {len(frames)} frames, {len(want)} blocks, {len(want) - sum(u in ref for u in want)} where the models disagree")


if __name__ == "__main__":
    main()
