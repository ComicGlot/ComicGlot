"""Analyse the filled rating workbooks.

Reads packet_<rater>.xlsx files and key.json, unblinds them and reports:
  - Krippendorff's alpha (ordinal) per scale, with a bootstrap interval over items
  - mean per condition and scale, with bootstrap intervals over items
  - paired comparisons fixed in advance: true_class vs no_class, true_class vs
    wrong_class, true_class vs no_context, true_class vs page_vlm
    (Wilcoxon signed-rank on item means across raters, Holm-corrected)
  - tone fit per carrier class and condition

Usage:
    python -m eval.translation_study.analyse_ratings --packets study_round1/packets --out study_round1/results.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SCALES = ["adequacy_1to5", "tone_fit_1to5", "fluency_1to5"]
COMPARISONS = [("true_class", "no_class"), ("true_class", "wrong_class"),
               ("true_class", "no_context"), ("true_class", "page_vlm")]


def krippendorff_alpha_ordinal(units: List[List[Optional[float]]]) -> float:
    """units: one list of ratings per unit (None = missing). Ordinal metric."""
    vals = sorted({v for u in units for v in u if v is not None})
    if len(vals) < 2:
        return float("nan")
    idx = {v: i for i, v in enumerate(vals)}
    k = len(vals)
    o = np.zeros((k, k))
    for u in units:
        r = [v for v in u if v is not None]
        m = len(r)
        if m < 2:
            continue
        for i, a in enumerate(r):
            for j, b in enumerate(r):
                if i != j:
                    o[idx[a], idx[b]] += 1.0 / (m - 1)
    n_c = o.sum(1)
    n = n_c.sum()
    if n <= 1:
        return float("nan")
    delta = np.zeros((k, k))
    for c in range(k):
        for d in range(k):
            lo, hi = min(c, d), max(c, d)
            delta[c, d] = (n_c[lo:hi + 1].sum() - (n_c[c] + n_c[d]) / 2.0) ** 2
    d_o = (o * delta).sum() / n
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))
    return float(1 - d_o / d_e) if d_e > 0 else float("nan")


def holm(pvals: List[float]) -> List[float]:
    order = np.argsort(pvals)
    adj = [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def read_packets(folder: Path):
    from openpyxl import load_workbook
    key = json.load(open(folder / "key.json", encoding="utf-8"))
    rows = []
    for rater, items in key.items():
        wb = load_workbook(folder / f"packet_{rater}.xlsx", data_only=True)
        ws = wb["rate"]
        head = [c.value for c in ws[1]]
        for r in ws.iter_rows(min_row=2, values_only=True):
            rec = dict(zip(head, r))
            if rec.get("item") is None:
                continue
            item = items[str(int(rec["item"]))]
            cond = item["letters"][rec["candidate"]]
            out = {"rater": rater, "block_uid": item["block_uid"], "condition": cond}
            for s in SCALES:
                v = rec.get(s)
                out[s] = float(v) if v not in (None, "") else None
            rows.append(out)
    return rows


def boot_mean(x: np.ndarray, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packets", type=Path, required=True)
    ap.add_argument("--calls", type=Path, help="calls.jsonl, to report results per carrier class")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    from scipy.stats import wilcoxon

    rows = read_packets(args.packets)
    cls_of = {}
    if args.calls:
        for l in open(args.calls, encoding="utf-8"):
            c = json.loads(l)
            cls_of[c["block_uid"]] = c["class_name"]
    report: Dict = {"ratings": len(rows), "raters": sorted({r["rater"] for r in rows}), "scales": {}}
    for s in SCALES:
        units = collections.defaultdict(list)
        for r in rows:
            units[(r["block_uid"], r["condition"])].append(r[s])
        alpha = krippendorff_alpha_ordinal(list(units.values()))
        item_mean = collections.defaultdict(dict)
        for (uid, cond), v in units.items():
            v = [x for x in v if x is not None]
            if v:
                item_mean[cond][uid] = float(np.mean(v))
        conds = {}
        for cond, d in item_mean.items():
            x = np.array(list(d.values()))
            conds[cond] = {"mean": float(x.mean()), "ci95": boot_mean(x), "items": len(x)}
        comps, ps = [], []
        for a, b in COMPARISONS:
            common = sorted(set(item_mean.get(a, {})) & set(item_mean.get(b, {})))
            if len(common) < 10:
                continue
            xa = np.array([item_mean[a][u] for u in common])
            xb = np.array([item_mean[b][u] for u in common])
            p = float(wilcoxon(xa, xb, zero_method="zsplit").pvalue)
            comps.append({"a": a, "b": b, "items": len(common), "mean_diff": float((xa - xb).mean()),
                          "diff_ci95": boot_mean(xa - xb), "p": p})
            ps.append(p)
        for c, p in zip(comps, holm(ps)):
            c["p_holm"] = p
        by_class = {}
        if cls_of:
            for cond, d in item_mean.items():
                g = collections.defaultdict(list)
                for uid, v in d.items():
                    g[cls_of.get(uid, "unknown")].append(v)
                by_class[cond] = {k: [float(np.mean(v)), len(v)] for k, v in g.items()}
        report["scales"][s] = {"alpha_ordinal": alpha, "conditions": conds, "comparisons": comps, "by_class": by_class}
        print(f"{s}: alpha {alpha:.3f}; " + "; ".join(f"{k} {v['mean']:.2f}" for k, v in sorted(conds.items())))
        for c in comps:
            print(f"   {c['a']} - {c['b']}: {c['mean_diff']:+.2f} (n={c['items']}, Holm p={c['p_holm']:.3g})")
    if args.out:
        json.dump(report, open(args.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
