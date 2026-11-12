"""Analyse the ranking round of the translator study (ranking CSV export of the rating app, one row per version).

In the ranking round a rater puts in order, best first, the versions of a few models for one text: each model with
the type's style guide (`true_class`) and without the type (`no_class`). The models are drawn per rater and text.
Versions with the same wording share a card and so a rank; a decoy (the translation of another text) is the
attention check and belongs at the bottom. The script reports three tables:

  1. The guide against no type, pooled over models, per family of text and overall: within each model shown,
     its two versions make one judgement (the better-ranked one wins; the same rank is a tie). Same summary as
     analyse_pairs.py: shares, Krippendorff's alpha, and a sign test on texts with a bootstrap interval.
  2. The same per translation model.
  3. The models against each other: a Bradley-Terry fit on the pairs implied by each ranking (rank-breaking; a tie
     counts half to each side), between versions of different models. Strengths are given per model and per model
     and condition, as the probability of beating an average version, with 95% intervals from a bootstrap over texts.
     Also the share of comparisons across models that a guided version wins against an unguided one.

Raters who miss more than --max-missed of their attention checks are left out (listed in the output).

    python -m eval.translation_study.rank_analysis --csv comicglot-ranking.csv --out study2026/ranking.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from eval.translation_study.analyse_pairs import FAMILY_ORDER, GUIDE, PLAIN, base_condition, read_rows, summarise, table
from research.common import FAMILIES


def items(rows: List[dict]) -> Dict[str, List[dict]]:
    """Rows grouped by ranking (one rater, one text)."""
    out: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        out[r["item"]].append(r)
    return out


def attention(rows: List[dict]) -> Dict[str, dict]:
    per: Dict[str, List[bool]] = defaultdict(list)
    for r in rows:
        if r.get("is_decoy", "").lower() == "true":
            per[r["rater"]].append(r.get("decoy_last", "").lower() == "true")
    return {k: {"checks": len(v), "passed": sum(v)} for k, v in per.items()}


def judgements(rows: List[dict]) -> List[dict]:
    """Within each model of a ranking, its guided against its unguided version (format of analyse_pairs)."""
    out = []
    for item, rs in items(rows).items():
        by_model: Dict[str, Dict[str, int]] = defaultdict(dict)
        for r in rs:
            if r.get("is_decoy", "").lower() == "true":
                continue
            by_model[r["model"]][base_condition(r["condition"])] = int(r["rank"])
        r0 = rs[0]
        for model, ranks in by_model.items():
            if GUIDE not in ranks or PLAIN not in ranks:
                continue
            g, p = ranks[GUIDE], ranks[PLAIN]
            out.append({
                "rater": r0["rater"],
                "kind": r0.get("rater_kind") or "person",
                "text": r0["block_uid"],
                "unit": f'{r0["block_uid"]}|{model}',
                "model": model,
                "family": FAMILIES.get(r0.get("detector_type", ""), "unknown"),
                "choice": "tie" if g == p else "guide" if g < p else "plain",
                "overdone_guide": False,
                "overdone_plain": False,
                "source_wrong": r0.get("source_wrong", "").lower() == "true",
            })
    return out


def pairs(rows: List[dict], key) -> List[Tuple[str, str, str, float]]:
    """(text, a, b, score of a) for every two versions of different models in one ranking; key names a version."""
    out = []
    for rs in items(rows).values():
        vs = [r for r in rs if r.get("is_decoy", "").lower() != "true"]
        for i, a in enumerate(vs):
            for b in vs[i + 1:]:
                if a["model"] == b["model"]:
                    continue
                ra, rb = int(a["rank"]), int(b["rank"])
                out.append((a["block_uid"], key(a), key(b), 1.0 if ra < rb else 0.5 if ra == rb else 0.0))
    return out


def bradley_terry(ps: List[Tuple[str, str, str, float]], names: List[str], iters: int = 500) -> np.ndarray:
    """Log-strengths (mean 0) by the MM algorithm of Hunter (2004), ties split half and half."""
    ix = {n: i for i, n in enumerate(names)}
    k = len(names)
    w = np.zeros((k, k))
    for _, a, b, s in ps:
        w[ix[a], ix[b]] += s
        w[ix[b], ix[a]] += 1 - s
    n = w + w.T
    s = np.ones(k)
    for _ in range(iters):
        denom = (n / (s[:, None] + s[None, :])).sum(1)
        s = np.where(denom > 0, (w.sum(1) + 1e-9) / np.maximum(denom, 1e-12), s)
        s /= np.exp(np.log(s).mean())
    return np.log(s)


def strengths(ps, names, n_boot: int, seed: int) -> Dict[str, dict]:
    """Probability of beating an average version (log-strength 0), with a bootstrap over texts."""
    est = bradley_terry(ps, names)
    by_text: Dict[str, list] = defaultdict(list)
    for p in ps:
        by_text[p[0]].append(p)
    texts = list(by_text)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(texts), len(texts))
        boots.append(bradley_terry([p for t in pick for p in by_text[texts[t]]], names, iters=200))
    boots = np.array(boots)
    prob = lambda x: 1 / (1 + np.exp(-x))
    wins = defaultdict(lambda: [0.0, 0])
    for _, a, b, s in ps:
        wins[a][0] += s; wins[a][1] += 1
        wins[b][0] += 1 - s; wins[b][1] += 1
    out = {}
    for i, n in enumerate(names):
        lo, hi = np.percentile(boots[:, i], [2.5, 97.5])
        out[n] = {"p_beat_average": float(prob(est[i])), "ci95": [float(prob(lo)), float(prob(hi))],
                  "log_strength": float(est[i]), "comparisons": wins[n][1],
                  "share_won": wins[n][0] / wins[n][1] if wins[n][1] else None}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["log_strength"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-missed", type=float, default=0.25, help="largest share of missed attention checks kept")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = [r for r in read_rows(a.csv) if (r.get("rater_kind") or "person") == "person"]
    att = attention(rows)
    dropped = sorted(k for k, v in att.items() if v["checks"] and (v["checks"] - v["passed"]) / v["checks"] > a.max_missed)
    rows = [r for r in rows if r["rater"] not in dropped]

    js = judgements(rows)
    models = sorted({j["model"] for j in js})
    model_pairs = pairs(rows, lambda r: r["model"])
    cond_key = lambda r: f'{r["model"]}|{"guide" if base_condition(r["condition"]) == GUIDE else "no_type"}'
    cond_pairs = pairs(rows, cond_key)
    cross = [s if a_.endswith("|guide") else 1 - s for _, a_, b_, s in cond_pairs if a_.split("|")[1] != b_.split("|")[1]]

    out = {
        "rankings": len(items(rows)),
        "raters": sorted({r["rater"] for r in rows}),
        "attention": att,
        "dropped_raters": dropped,
        "guide_vs_no_type": table(js),
        "guide_vs_no_type_by_model": {m: summarise([j for j in js if j["model"] == m]) for m in models},
        "guide_vs_no_type_korean_ok": summarise([j for j in js if not j["source_wrong"]]),
        "models": strengths(model_pairs, models, a.boot, a.seed),
        "models_by_condition": strengths(cond_pairs, sorted({k for p in cond_pairs for k in p[1:3]}), a.boot, a.seed),
        "guided_beats_unguided_across_models": {"share": float(np.mean(cross)) if cross else None, "comparisons": len(cross)},
        "families": FAMILY_ORDER,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    g = out["guide_vs_no_type"]["all"]
    print(f'{out["rankings"]} rankings by {len(out["raters"])} raters; dropped: {dropped or "none"}')
    print(f'guide vs no type: {g["guide"]:.2f} / {g["no_type"]:.2f} / tie {g["tie"]:.2f}, sign test p = {g["sign_test_p"]:.3g}' if g["judgements"] else "no judgements")
    for m, v in out["models"].items():
        print(f'  {m:40s} {v["p_beat_average"]:.2f} [{v["ci95"][0]:.2f}, {v["ci95"][1]:.2f}]')


if __name__ == "__main__":
    main()
