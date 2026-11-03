"""OCR evaluation against a hand transcription.

Replaces training/recognition/ocr_metrics.py for paper numbers: that script
scores the primary reading against the LLM-corrected text, which measures
agreement with the arbiter, not accuracy.

Input
  --gt        CSV with columns: block_uid, gt_text  (block_uid = <run>/<frame_id>/<block_id>,
              as written by make_ocr_gt_sheet.py). Rows with empty gt_text are
              skipped; rows whose gt_text is "#ILLEGIBLE" are reported separately.
  --runs      one or more chapter_full_results.json files from comic_processor
  --fields    reading fields to score (default: text secondary_text corrected_text)

Output (stdout and --out JSON)
  per field: CER and WER over all blocks and per class, with 95% bootstrap
  intervals resampled over blocks; character counts; the arbitration outcome
  for each block (agree / kept primary / kept secondary / neither) and the CER
  of the corrected text within each outcome, which shows whether the LLM's
  "neither" outputs are repairs or rewrites.

Normalisation: Unicode NFC, whitespace removed for CER (Korean spacing in
comics is inconsistent), collapsed to single spaces for WER. Punctuation is
kept because it carries tone ("!", "...", "~"); pass --strip-punct to drop it.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List

import numpy as np

PUNCT = re.compile(r"[\s\.\,\!\?\~\-…‥·・'\"“”‘’\[\]\(\)<>〈〉「」『』:;]+")


def levenshtein(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm_chars(s: str, strip_punct: bool) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = PUNCT.sub("", s) if strip_punct else re.sub(r"\s+", "", s)
    return s


def norm_words(s: str, strip_punct: bool) -> List[str]:
    s = unicodedata.normalize("NFC", s or "")
    if strip_punct:
        s = PUNCT.sub(" ", s)
    return s.split()


def load_blocks(paths: List[Path]) -> Dict[str, Dict]:
    blocks = {}
    for p in paths:
        run = p.parent.name
        data = json.load(open(p, encoding="utf-8"))
        for frame in data.get("frames", []):
            for b in frame.get("blocks", []):
                uid = f"{run}/{b.get('frame_id', frame.get('id'))}/{b.get('id')}"
                blocks[uid] = b
    return blocks


def outcome(b: Dict) -> str:
    p, s, c = (norm_chars(b.get(k, ""), False) for k in ("text", "secondary_text", "corrected_text"))
    if p == s:
        return "agree"
    if not p or not s:
        return "one_empty"
    if c == p:
        return "kept_primary"
    if c == s:
        return "kept_secondary"
    return "neither"


def bootstrap_ratio(errors: np.ndarray, lengths: np.ndarray, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(errors), size=(n, len(errors)))
    vals = errors[idx].sum(1) / np.maximum(lengths[idx].sum(1), 1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def score(pairs, strip_punct: bool) -> Dict:
    ce = np.array([levenshtein(norm_chars(h, strip_punct), norm_chars(r, strip_punct)) for h, r in pairs], float)
    cl = np.array([len(norm_chars(r, strip_punct)) for _, r in pairs], float)
    we = np.array([levenshtein(norm_words(h, strip_punct), norm_words(r, strip_punct)) for h, r in pairs], float)
    wl = np.array([len(norm_words(r, strip_punct)) for _, r in pairs], float)
    if not len(pairs):
        return {"n": 0}
    return {"n": len(pairs), "chars": int(cl.sum()),
            "cer": float(ce.sum() / max(cl.sum(), 1)), "cer_ci95": bootstrap_ratio(ce, cl),
            "wer": float(we.sum() / max(wl.sum(), 1)), "wer_ci95": bootstrap_ratio(we, wl)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--fields", nargs="+", default=["text", "secondary_text", "corrected_text"])
    ap.add_argument("--strip-punct", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    blocks = load_blocks(args.runs)
    gt = {}
    illegible = 0
    with open(args.gt, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            t = (row.get("gt_text") or "").strip()
            if not t:
                continue
            if t == "#ILLEGIBLE":
                illegible += 1
                continue
            gt[row["block_uid"]] = t
    missing = [u for u in gt if u not in blocks]
    if missing:
        raise SystemExit(f"{len(missing)} ground-truth rows match no block, e.g. {missing[:3]}")

    report = {"blocks_scored": len(gt), "illegible": illegible, "fields": {}}
    for field in args.fields:
        pairs = [(blocks[u].get(field) or "", t) for u, t in gt.items()]
        res = {"all": score(pairs, args.strip_punct), "by_class": {}}
        by = collections.defaultdict(list)
        for u, t in gt.items():
            by[blocks[u].get("class_name", "unknown")].append((blocks[u].get(field) or "", t))
        for cls, ps in sorted(by.items(), key=lambda kv: -len(kv[1])):
            res["by_class"][cls] = score(ps, args.strip_punct)
        report["fields"][field] = res
    by_out = collections.defaultdict(list)
    for u, t in gt.items():
        by_out[outcome(blocks[u])].append((blocks[u].get("corrected_text") or "", t))
    report["arbitration"] = {k: score(v, args.strip_punct) for k, v in by_out.items()}

    for field, res in report["fields"].items():
        a = res["all"]
        if a.get("n"):
            print(f"{field:16s} CER {100 * a['cer']:.2f}% [{100 * a['cer_ci95'][0]:.2f}, {100 * a['cer_ci95'][1]:.2f}]  "
                  f"WER {100 * a['wer']:.2f}%  n={a['n']}")
    for k, v in report["arbitration"].items():
        if v.get("n"):
            print(f"arbiter {k:15s} n={v['n']:4d}  CER of corrected text {100 * v['cer']:.2f}%")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
