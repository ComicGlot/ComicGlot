"""Analysis E: how much of the carrier class can the words alone predict?

If the class were recoverable from the text, passing it to the translator
would add nothing. This script fits text-only classifiers (character n-grams,
and a handful of surface cues such as "!", ellipses, brackets and laughter
jamo) with group-aware cross-validation, and, when crop embeddings are
supplied for the same blocks, an image-only probe and a text-plus-image model
on identical folds.

Inputs, one of:
  --runs RUN.json ...   ComicGlot chapter outputs; groups are run/frame, labels
                        are the detector's classes (say so when reporting)
  --table blocks.csv    columns: group, class, text [, emb_row]; with --emb
                        emb.npy for the image side (emb_row indexes its rows)

Usage:
    python -m research.text_vs_image --runs out/run25/chapter_full_results.json --out out/e_pilot.json
    python -m research.text_vs_image --table crops_ocr.csv --emb emb_dinov2s_q.npy --out out/e_full.json

Pilot on chapters 3 and 4 (488 blocks, 8 classes, detector labels, Oct 2026):
char n-grams reach accuracy 0.64 and macro F1 0.48 (majority class 0.35);
sfx_sound and system_bubble are predictable from text (F1 0.81 and 0.86),
think_bubble, spiked_bubble and scream_bubble are not (0.14 to 0.39).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

SURFACE = ["length", "exclaim", "question", "ellipsis", "bracket", "laugh", "digits", "latin", "tilde", "words"]


def surface_features(texts: List[str]) -> np.ndarray:
    out = []
    for t in texts:
        out.append([len(t), t.count("!"), t.count("?"), t.count("...") + t.count("…"),
                    int(t[:1] in "[<【" or "【" in t), sum(ch in "ㅋㅎ" for ch in t),
                    sum(ch.isdigit() for ch in t), sum("a" <= ch.lower() <= "z" for ch in t),
                    t.count("~"), len(t.split())])
    return np.array(out, float)


def rows_from_runs(paths: List[Path], source: str):
    rows = []
    for p in paths:
        run = p.parent.name
        data = json.load(open(p, encoding="utf-8"))
        for frame in data.get("frames", []):
            for b in frame.get("blocks", []):
                text = (b.get("corrected_text") if source == "corrected" else b.get("text")) or ""
                if text.strip():
                    rows.append({"group": f"{run}/{b.get('frame_id', frame.get('id'))}",
                                 "class": b.get("class_name"), "text": text.strip(), "emb_row": None})
    return rows


def rows_from_table(path: Path):
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if (r.get("text") or "").strip():
            rows.append({"group": r["group"], "class": r["class"], "text": r["text"].strip(),
                         "emb_row": int(r["emb_row"]) if r.get("emb_row") not in (None, "") else None})
    return rows


def evaluate(rows, emb: Optional[np.ndarray], folds: int, min_count: int, C: float):
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    counts = collections.Counter(r["class"] for r in rows)
    keep = sorted(c for c, v in counts.items() if v >= min_count)
    rows = [r for r in rows if r["class"] in keep]
    y = np.array([r["class"] for r in rows])
    g = np.array([r["group"] for r in rows])
    texts = [r["text"] for r in rows]
    cv = GroupKFold(min(folds, len(set(g))))
    report = {"n": len(rows), "classes": {c: int((y == c).sum()) for c in keep},
              "majority_accuracy": max(counts[c] for c in keep) / len(rows), "models": {}}

    def score(name, pred):
        f = f1_score(y, pred, average=None, labels=keep)
        report["models"][name] = {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f.mean()),
                                  "per_class_f1": dict(zip(keep, map(float, f)))}

    char = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2),
                         LogisticRegression(max_iter=3000, C=C, class_weight="balanced"))
    score("text_char_ngrams", cross_val_predict(char, texts, y, cv=cv, groups=g))
    surf = make_pipeline(FunctionTransformer(surface_features), StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced"))
    score("text_surface_cues", cross_val_predict(surf, texts, y, cv=cv, groups=g))
    if emb is not None and all(r["emb_row"] is not None for r in rows):
        X = emb[[r["emb_row"] for r in rows]]
        img = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"))
        score("image_probe", cross_val_predict(img, X, y, cv=cv, groups=g))
        both = np.hstack([X, surface_features(texts)])
        score("image_plus_surface", cross_val_predict(img, both, y, cv=cv, groups=g))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs", type=Path, nargs="+")
    src.add_argument("--table", type=Path)
    ap.add_argument("--source", choices=["corrected", "raw"], default="corrected", help="text field for --runs")
    ap.add_argument("--emb", type=Path)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--C", type=float, default=5.0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = rows_from_runs(args.runs, args.source) if args.runs else rows_from_table(args.table)
    emb = np.load(args.emb) if args.emb else None
    rep = evaluate(rows, emb, args.folds, args.min_count, args.C)
    print(f"blocks {rep['n']}, classes {len(rep['classes'])}, majority {rep['majority_accuracy']:.3f}")
    for name, m in rep["models"].items():
        per = " ".join(f"{c.split('_')[0]}={v:.2f}" for c, v in m["per_class_f1"].items())
        print(f"  {name:20s} acc {m['accuracy']:.3f} macro F1 {m['macro_f1']:.3f} | {per}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(args.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
