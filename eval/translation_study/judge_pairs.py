"""A language model as judge of the translator study's pairs: with the type's guide against without the type.

The judge gets what a rater gets: the frame with the line's box drawn in red, the Korean, and the two English
versions in a random order (A/B, recorded per pair). It chooses the better version or "equal" and rates the tone of
each version against the page as too flat, fitting or too strong, the scale of the rating application. Pairs whose
two versions are identical are skipped and counted (the guide changed nothing). Calls resume: a pair already judged
by this judge is not asked again.

    python -m eval.translation_study.judge_pairs --runs <chapter_full_results.json ...> --blocks <blocks_v2.csv ...> \
        --calls <translations dir ...> --models anthropic/claude-sonnet-5.5 --judge openai/gpt-6.1-sol --out <dir>
    python -m eval.translation_study.judge_pairs --summary <dir>/*.jsonl --blocks <blocks_v2.csv ...>
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from eval.sample_frames import FAMILIES, local_path

TONE = ("flat", "fine", "strong")

PROMPT = """You are an experienced Korean-to-English comics translator. The image is one frame of a Korean webtoon; the line in question is drawn inside the red box. Look at how it is drawn: the balloon or box, the lettering and the scene.

Korean: {source}
A: {a}
B: {b}

Choose the version a reader of an English comic would get the same meaning and the same tone from. Adding tone that the page does not show is as wrong as losing tone that it does. Choose "equal" if both are equally good or equally bad.
For each version, rate its tone against the page: "flat" (loses tone the page shows), "fine" (fits), or "strong" (adds tone the page does not show).

Answer with JSON only: {{"choice": "A" | "B" | "equal", "tone": {{"A": "flat" | "fine" | "strong", "B": "flat" | "fine" | "strong"}}, "reason": "<one sentence>"}}"""


def frame_jpeg(path: Path, box, max_side: int = 1280) -> str:
    img = cv2.imread(str(path))
    x0, y0, x1, y1 = map(int, box)
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 230), max(3, img.shape[1] // 200))
    sc = min(1.0, max_side / max(img.shape[:2]))
    if sc < 1:
        img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf.tobytes()).decode()


def ask(judge: str, text: str, image_b64: str | None, max_tokens: int) -> dict:
    import requests
    content = [{"type": "text", "text": text}]
    if image_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
    body = {"model": judge, "max_tokens": max_tokens, "temperature": 0, "messages": [{"role": "user", "content": content}]}
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "X-Title": "ComicGlot"}
    last = ""
    for attempt in range(5):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers, timeout=180)
            if r.status_code == 200 and not r.json().get("error"):
                j = r.json()
                ch = (j.get("choices") or [{}])[0]
                return {"raw": ch.get("message", {}).get("content") or "", "finish_reason": ch.get("finish_reason"), "usage": j.get("usage")}
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code not in (408, 429, 500, 502, 503, 504):
                break
        except requests.RequestException as e:
            last = str(e)
        time.sleep(min(30, 2 ** attempt))
    return {"raw": "", "error": last}


def parse(raw: str) -> dict | None:
    m = re.search(r"\{[\s\S]*\}", raw or "")
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    c = str(j.get("choice", "")).strip()
    c = {"a": "A", "b": "B", "tie": "equal"}.get(c.lower(), c)
    tone = {k: str((j.get("tone") or {}).get(k, "")).strip().lower() for k in ("A", "B")}
    if c not in ("A", "B", "equal") or any(t not in TONE for t in tone.values()):
        return None
    return {"choice": c, "tone": tone, "reason": str(j.get("reason", ""))[:300]}


def load_pairs(a) -> list[dict]:
    cls = {}
    for f in a.blocks:
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            cls[r["block_uid"]] = r["class_name"]
    tr = defaultdict(dict)  # (model, uid) -> condition -> call
    for d in a.calls:
        for f in Path(d).glob("*/calls.jsonl"):
            for line in open(f, encoding="utf-8"):
                c = json.loads(line)
                if c["block_uid"] in cls and c.get("translation") and not c.get("error") and (not a.models or c["model"] in a.models):
                    tr[(c["model"], c["block_uid"])][c["condition"]] = c
    pairs = []
    for (model, uid), cs in sorted(tr.items()):
        if "true_class" in cs and "no_class" in cs:
            g, n = cs["true_class"], cs["no_class"]
            pairs.append({"model": model, "block_uid": uid, "class_name": cls[uid], "source": g.get("source_text", ""),
                          "guide": " ".join(g["translation"].split()), "plain": " ".join(n["translation"].split())})
    return pairs


def judge_all(a) -> None:
    pairs = load_pairs(a)
    frames = {}
    for rf in a.runs:
        data = json.loads(Path(rf).read_text(encoding="utf-8"))
        run = Path(rf).parent.name
        for fr in data.get("frames", []):
            for b in fr.get("blocks", []):
                frames[f"{run}/{b.get('frame_id', fr.get('id'))}/{b.get('id')}"] = (fr["image_path"], b["box"])
    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / (a.judge.replace("/", "__") + ("" if a.image else "__text") + ".jsonl")
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            if r.get("parsed") or r.get("identical"):
                done.add((r["model"], r["block_uid"]))
    todo = [p for p in pairs if (p["model"], p["block_uid"]) not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(pairs)} pairs, {len(done)} already judged, {len(todo)} to go -> {out}", flush=True)

    def one(p):
        rec = {k: p[k] for k in ("model", "block_uid", "class_name")}
        if p["guide"] == p["plain"]:
            return {**rec, "identical": True}
        rng = random.Random(f"{a.seed}:{p['model']}:{p['block_uid']}")
        guide_letter = rng.choice("AB")
        a_text, b_text = (p["guide"], p["plain"]) if guide_letter == "A" else (p["plain"], p["guide"])
        img = None
        if a.image:
            path, box = frames[p["block_uid"]]
            img = frame_jpeg(local_path(path, a.project_root), box)
        res = ask(a.judge, PROMPT.format(source=p["source"], a=a_text, b=b_text), img, a.max_tokens)
        parsed = parse(res.get("raw", ""))
        out_rec = {**rec, "judge": a.judge, "image": a.image, "guide_letter": guide_letter, **res, "parsed": parsed}
        if parsed:
            plain_letter = "B" if guide_letter == "A" else "A"
            out_rec["preferred"] = {"equal": "equal", guide_letter: "guide", plain_letter: "plain"}[parsed["choice"]]
            out_rec["tone_guide"], out_rec["tone_plain"] = parsed["tone"][guide_letter], parsed["tone"][plain_letter]
        return out_rec

    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"  {i} / {len(todo)}", flush=True)
    summary(a, [out])


def summary(a, files) -> None:
    for f in files:
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        by = defaultdict(list)
        for r in rows:
            by[r["model"]].append(r)
        print(f"\n{Path(f).name}\n")
        print("| Model | Pairs | Identical | Guide preferred | No type preferred | Equal | Too strong (guide / none) | Too flat (guide / none) | Unparsed |")
        print("|---|---|---|---|---|---|---|---|---|")
        for m, rs in sorted(by.items()):
            ok = [r for r in rs if r.get("preferred")]
            c = Counter(r["preferred"] for r in ok)
            n = max(1, len(ok))
            st = lambda k, t: sum(r[k] == t for r in ok) / n
            print(f"| {m} | {len(rs)} | {sum(1 for r in rs if r.get('identical'))} | {c['guide'] / n:.0%} | {c['plain'] / n:.0%} | {c['equal'] / n:.0%} | "
                  f"{st('tone_guide', 'strong'):.0%} / {st('tone_plain', 'strong'):.0%} | {st('tone_guide', 'flat'):.0%} / {st('tone_plain', 'flat'):.0%} | "
                  f"{sum(1 for r in rs if not r.get('identical') and not r.get('parsed'))} |")
        fam = defaultdict(Counter)
        for r in rows:
            if r.get("preferred"):
                fam[FAMILIES.get(r["class_name"], "other")][r["preferred"]] += 1
        print("\nBy kind of text (all models): " + "; ".join(f"{k}: guide {v['guide']}, none {v['plain']}, equal {v['equal']}" for k, v in sorted(fam.items())))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--blocks", nargs="*", default=[])
    ap.add_argument("--calls", nargs="*", default=[])
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--judge")
    ap.add_argument("--image", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="judge at most this many new pairs (a pilot)")
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--summary", nargs="*")
    a = ap.parse_args()
    summary(a, a.summary) if a.summary else judge_all(a)


if __name__ == "__main__":
    main()
