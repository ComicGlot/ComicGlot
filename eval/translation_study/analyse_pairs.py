"""Analyse blind pairwise judgements of translations (CSV export, one row per comparison).

Each row is one comparison a rater made between two versions of a text: the type's style guide
(`true_class`) and no type (`no_class`), from the same model. The script reports, per family of
text (outlined balloons; boxes and free text; sound effects) and overall:

  - judgements, and the shares that prefer the guide, prefer no type, or call a tie;
  - how often each version's tone was marked overdone;
  - Krippendorff's alpha (nominal) over texts judged by two or more people;
  - a two-sided exact sign test on texts: each text counts once, for the version most of its raters
    chose (texts where the raters split evenly drop out), with a bootstrap 95% interval over texts
    for the share of texts that prefer the guide.

It does the same per translation model, reports model judges (rater_kind = model) apart from people,
repeats the overall test without texts whose Korean a rater marked as wrong, and lists attention
checks per rater. Practice sets never reach the export.

    python -m eval.translation_study.analyse_pairs --csv judgements.csv --out study_round1/pairs.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from research.common import FAMILIES

FAMILY_ORDER = ["outlined_balloon", "box_or_free_text", "sound_effect"]
GUIDE, PLAIN = "true_class", "no_class"


def base_condition(c: str) -> str:
    """'true_class+page@openai/gpt-x' -> 'true_class'."""
    return c.split("@")[0].removesuffix("+page")


def sign_test(k: int, n: int) -> float:
    """Exact two-sided sign test, p = 0.5."""
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def alpha_nominal(units: Iterable[List[str]]) -> Optional[float]:
    """Krippendorff's alpha, nominal metric; units with fewer than two values are skipped."""
    o: Dict[tuple, float] = defaultdict(float)
    n = 0
    for u in units:
        m = len(u)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[(u[i], u[j])] += 1 / (m - 1)
        n += m
    if n < 2:
        return None
    nc: Dict[str, float] = defaultdict(float)
    for (a, _), v in o.items():
        nc[a] += v
    d_o = sum(v for (a, b), v in o.items() if a != b)
    d_e = sum(nc[a] * nc[b] for a in nc for b in nc if a != b) / (n - 1)
    return None if d_e == 0 else 1 - d_o / d_e


def read_rows(paths: List[Path]) -> List[dict]:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8-sig", newline="") as f:
            rows += list(csv.DictReader(f))
    return rows


def judgements(rows: List[dict]) -> List[dict]:
    """The guide-against-no-type comparisons, one dict per judgement."""
    out = []
    for r in rows:
        if r.get("is_attention_check", "").lower() == "true":
            continue
        conds = [base_condition(c) for c in r["conditions"].split("|")]
        if sorted(conds) != sorted([GUIDE, PLAIN]):
            continue
        choice = r["choice_condition"]
        pick = "tie" if choice == "tie" else {GUIDE: "guide", PLAIN: "plain"}.get(base_condition(choice), "other")
        tone = {}
        for part in filter(None, r.get("tone_by_condition", "").split(";")):
            c, _, t = part.rpartition(":")
            tone[base_condition(c)] = t
        out.append({
            "rater": r["rater"],
            "kind": r.get("rater_kind") or "person",
            "text": r["block_uid"],
            "unit": f'{r["block_uid"]}|{r.get("model", "")}',
            "model": r.get("model") or "(imported)",
            "family": FAMILIES.get(r.get("detector_type", ""), "unknown"),
            "choice": pick,
            "overdone_guide": tone.get(GUIDE) == "overdone",
            "overdone_plain": tone.get(PLAIN) == "overdone",
            "source_wrong": r.get("source_wrong", "").lower() == "true",
        })
    return out


def summarise(js: List[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    n = len(js)
    by_unit: Dict[str, List[dict]] = defaultdict(list)
    for j in js:
        by_unit[j["unit"]].append(j)
    wins = []  # 1 guide, 0 no type, per decided text
    for v in by_unit.values():
        g = sum(x["choice"] == "guide" for x in v)
        p = sum(x["choice"] == "plain" for x in v)
        if g != p:
            wins.append(1 if g > p else 0)
    wins_a = np.array(wins, dtype=float)
    k, m = int(wins_a.sum()), len(wins)
    ci = None
    if m:
        rng = np.random.default_rng(seed)
        boots = wins_a[rng.integers(0, m, size=(n_boot, m))].mean(1)
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    share = lambda c: (sum(j["choice"] == c for j in js) / n) if n else None
    return {
        "judgements": n,
        "texts": len(by_unit),
        "guide": share("guide"),
        "no_type": share("plain"),
        "tie": share("tie"),
        "guide_among_non_ties": (sum(j["choice"] == "guide" for j in js) / max(1, sum(j["choice"] in ("guide", "plain") for j in js))) if n else None,
        "overdone_guide": (sum(j["overdone_guide"] for j in js) / n) if n else None,
        "overdone_no_type": (sum(j["overdone_plain"] for j in js) / n) if n else None,
        "alpha_nominal": alpha_nominal([x["choice"] for x in v] for v in by_unit.values()),
        "texts_decided": m,
        "texts_prefer_guide": k,
        "share_texts_prefer_guide": (k / m) if m else None,
        "share_texts_prefer_guide_ci95": ci,
        "sign_test_p": sign_test(k, m),
    }


def table(js: List[dict]) -> Dict[str, dict]:
    out = {f: summarise([j for j in js if j["family"] == f]) for f in FAMILY_ORDER}
    out["all"] = summarise(js)
    return out


def attention(rows: List[dict]) -> Dict[str, dict]:
    per: Dict[str, List[bool]] = defaultdict(list)
    for r in rows:
        if r.get("is_attention_check", "").lower() == "true":
            per[r["rater"]].append(r.get("attention_passed", "").lower() == "true")
    return {k: {"checks": len(v), "passed": sum(v)} for k, v in sorted(per.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, nargs="+", required=True, help="pairwise judgement CSV file(s)")
    ap.add_argument("--exclude-raters", nargs="*", default=[], help="raters left out (e.g. failed attention checks)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    rows = [r for r in read_rows(a.csv) if r["rater"] not in set(a.exclude_raters)]
    js = judgements(rows)
    people = [j for j in js if j["kind"] != "model"]
    result = {
        "inputs": [str(p) for p in a.csv],
        "excluded_raters": a.exclude_raters,
        "people": table(people),
        "people_without_wrong_source": table([j for j in people if not j["source_wrong"]]),
        "per_model": {m: summarise([j for j in people if j["model"] == m]) for m in sorted({j["model"] for j in people})},
        "model_judges": {r: table([j for j in js if j["rater"] == r]) for r in sorted({j["rater"] for j in js if j["kind"] == "model"})},
        "attention_checks": attention(rows),
        "raters": sorted({j["rater"] for j in people}),
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text, encoding="utf-8")
    al = result["people"]["all"]
    print(f"people: {al['judgements']} judgements on {al['texts']} texts; guide {al['guide']}, no type {al['no_type']}, tie {al['tie']}; "
          f"texts decided {al['texts_decided']}, prefer guide {al['texts_prefer_guide']} (p = {al['sign_test_p']:.4g}); alpha {al['alpha_nominal']}")
    if not a.out:
        print(text)


if __name__ == "__main__":
    main()
