"""Reference transcriptions from multimodal language models, for scoring the OCR readers.

  read       crops every block of --blocks from its frame and asks each --models model (through
             OpenRouter, key in OPENROUTER_API_KEY) for the Korean exactly as drawn. Every call is
             logged to <out>/<model>.jsonl with the prompt, the raw answer and the time, and a run
             that stops can be started again; done blocks are skipped. Writes <out>/readings.csv
             (block_uid plus one column per model), usable as --readings of ocr_eval.py.
  consensus  builds a reference from readings.csv: a block whose normalised readings (NFC, spaces
             removed) agree for at least --min-agree models gets that reading; the others are listed
             in disagreements.csv for a person to check and are left out until then. Writes
             reference.csv (block_uid, gt_text, agree), the --gt file of ocr_eval.py, and prints
             how often the models agree with each other.

A model whose reading is part of the reference cannot be scored fairly against it; report it apart.

    python -m eval.llm_transcribe read --runs data/output/<runs>/*/chapter_full_results.json \
        --blocks data/study/sample/blocks.csv --models google/gemini-4-argon openai/gpt-6.1-sol \
        anthropic/claude-opus-5.5 --out data/study/llm_ocr
    python -m eval.llm_transcribe consensus --out data/study/llm_ocr --min-agree 2
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import cv2

from eval.ocr_baselines import crop
from eval.sample_frames import local_path

PROMPT = (
    "The image is a text block cut from a Korean webtoon (a balloon, a box, free lettering or a sound effect). "
    "Transcribe the Korean exactly as it is drawn: every character in reading order, with its punctuation "
    "(…, !, ?, ~ and quotation marks). Join lines with a single space. For a sound effect, write its syllables "
    "as drawn, even if they are not a word. Do not translate, correct or explain. "
    "If there is no legible text, answer #ILLEGIBLE. Answer with the transcription only."
)


def norm(s: str) -> str:
    return "".join(unicodedata.normalize("NFC", s or "").split())


def call(model: str, png_b64: str, retries: int = 4) -> dict:
    import requests
    body = {"model": model, "temperature": 0, "max_tokens": 2000, "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}}]}]}
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "X-Title": "ComicGlot"}
    last = ""
    for attempt in range(retries + 1):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers, timeout=120)
            if r.status_code == 200 and not r.json().get("error"):
                j = r.json()
                ch = (j.get("choices") or [{}])[0]
                # Reasoning models spend tokens before answering; the stop reason shows a cut-off answer.
                return {"answer": (ch.get("message", {}).get("content") or "").strip(),
                        "finish_reason": ch.get("finish_reason"), "usage": j.get("usage")}
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in (408, 429, 500, 502, 503, 504):
                break
        except requests.RequestException as e:
            last = str(e)
        time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(last)


def clean(answer: str) -> str:
    """Drop wrapping quotes or code fences a model may add around the transcription."""
    a = answer.strip().strip("`").strip()
    if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'“”":
        a = a[1:-1].strip()
    return a


def read(a) -> None:
    want = [r["block_uid"] for r in csv.DictReader(open(a.blocks, encoding="utf-8-sig"))]
    where: Dict[str, tuple] = {}
    for f in a.runs:
        run = f.parent.name
        for fr in json.loads(f.read_text(encoding="utf-8")).get("frames", []):
            for b in fr.get("blocks", []):
                where[f"{run}/{b.get('frame_id', fr.get('id'))}/{b.get('id')}"] = (fr["image_path"], b["box"])
    missing = [u for u in want if u not in where]
    if missing:
        raise SystemExit(f"{len(missing)} blocks are not in --runs, e.g. {missing[:3]}")
    images: Dict[str, object] = {}
    crops: Dict[str, str] = {}
    for u in want:
        path, box = where[u]
        if path not in images:
            images[path] = cv2.imread(str(local_path(path, a.project_root)))
        ok, buf = cv2.imencode(".png", crop(images[path], box, pad=8))
        crops[u] = base64.b64encode(buf.tobytes()).decode()
    a.out.mkdir(parents=True, exist_ok=True)
    readings: Dict[str, Dict[str, str]] = {u: {} for u in want}
    for model in a.models:
        log = a.out / (model.replace("/", "__") + ".jsonl")
        done = {}
        if log.exists():
            for line in open(log, encoding="utf-8"):
                c = json.loads(line)
                if c.get("answer") and not c.get("error"):  # an empty answer is asked again
                    done[c["block_uid"]] = c["answer"]
        todo = [u for u in want if u not in done]
        if a.dry_run:
            print(f"{model}: {len(todo)} calls to make, {len(done)} done")
            continue

        def one(u):
            try:
                return {"block_uid": u, "model": model, "prompt": PROMPT, **call(model, crops[u]), "error": None, "time": time.time()}
            except Exception as e:  # logged; the next run retries it
                return {"block_uid": u, "model": model, "prompt": PROMPT, "answer": None, "error": str(e)[:300], "time": time.time()}

        failed = 0
        with open(log, "a", encoding="utf-8") as fh, ThreadPoolExecutor(a.workers) as pool:
            for rec in pool.map(one, todo):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if rec["error"] or not rec["answer"]:
                    failed += 1
                else:
                    done[rec["block_uid"]] = rec["answer"]
        print(f"{model}: {len(todo) - failed} read, {failed} failed (run again to retry)")
        for u, ans in done.items():
            if u in readings:
                readings[u][model] = clean(ans)
    if a.dry_run:
        return
    with open(a.out / "readings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["block_uid", *a.models])
        for u in want:
            w.writerow([u, *(readings[u].get(m, "") for m in a.models)])
    print(f"wrote {a.out / 'readings.csv'}")


def consensus(a) -> None:
    rows = list(csv.DictReader(open(a.out / "readings.csv", encoding="utf-8-sig")))
    models = [k for k in rows[0] if k != "block_uid"]
    ref, dis = [], []
    pair = Counter()
    for r in rows:
        vals = {m: r[m] for m in models if r[m]}
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                if m1 in vals and m2 in vals:
                    pair[(m1, m2, "n")] += 1
                    pair[(m1, m2, "same")] += norm(vals[m1]) == norm(vals[m2])
        counts = Counter(norm(v) for v in vals.values())
        best, n = counts.most_common(1)[0] if counts else ("", 0)
        if n >= a.min_agree:
            text = next(v for v in vals.values() if norm(v) == best)
            ref.append([r["block_uid"], "#ILLEGIBLE" if best == "#ILLEGIBLE" else text, n])
        else:
            dis.append([r["block_uid"], *(r[m] for m in models)])
    with open(a.out / "reference.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["block_uid", "gt_text", "agree"])
        w.writerows(ref)
    with open(a.out / "disagreements.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["block_uid", *models])
        w.writerows(dis)
    print(f"reference for {len(ref)} of {len(rows)} blocks (at least {a.min_agree} models agree); {len(dis)} to check by hand")
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            n = pair[(m1, m2, "n")]
            if n:
                print(f"  {m1} = {m2}: {pair[(m1, m2, 'same')]}/{n} identical after normalising")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--runs", type=Path, nargs="+", required=True)
    r.add_argument("--blocks", type=Path, required=True)
    r.add_argument("--models", nargs="+", required=True)
    r.add_argument("--project-root", type=Path, default=Path("."))
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("consensus")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--min-agree", type=int, default=2)
    a = ap.parse_args()
    read(a) if a.cmd == "read" else consensus(a)


if __name__ == "__main__":
    main()
