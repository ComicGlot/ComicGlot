"""Descriptive statistics of processed chapters (paper Tables 'arbiter' and 'markers').

Reads ComicGlot chapter outputs (chapter_full_results.json) and reports

  arbiter   per kind of text: blocks where the two readers agree, and, where they
            differ, whether the corrected text equals the general reader
            (secondary_text, team-lucid/trocr-small-korean), the comic reader
            (text, ko-trocr-cg2m) or neither. The same comparison rule is used
            for agreement and for the arbiter's choice. "pipeline" (the paper's
            table) ignores whitespace, case and , . ; : ! ? as
            ComicProcessor.normalize does, which is the rule that decides whether
            a block reaches the arbiter; "space" removes whitespace only.
  markers   per class with at least --min-blocks blocks: share of blocks with
            "!" and an ellipsis in the Korean (corrected_text) and with "!", an
            ellipsis and a fully capitalised word in the translation.

No models, no API calls.

Usage:
    python -m research.run_stats --runs RUN3/chapter_full_results.json RUN4/chapter_full_results.json \
        --out out/run_stats.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Dict, List

KIND = {
    "sfx_sound": "Sound effects",
    "standalone_text": "Free text, titles, background",
    "background_text": "Free text, titles, background",
    "title_panel": "Free text, titles, background",
    "sqare_bubble": "Boxes and system messages",
    "square_bubble": "Boxes and system messages",
    "system_bubble": "Boxes and system messages",
}
BALLOONS = "Balloons"


def kind_of(cls: str) -> str:
    return KIND.get(cls, BALLOONS)


def norm_space(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def norm_pipeline(s: str) -> str:
    return re.sub(r"[\s,.;:!?]", "", (s or "")).lower()


def load_blocks(paths: List[Path]) -> List[dict]:
    blocks = []
    for p in paths:
        data = json.load(open(p, encoding="utf-8"))
        for frame in data.get("frames", []):
            blocks.extend(frame.get("blocks", []))
    return blocks


def arbiter_outcome(b: dict, rule: str) -> str:
    comic, general, corr = b.get("text") or "", b.get("secondary_text") or "", b.get("corrected_text") or ""
    same = norm_pipeline if rule == "pipeline" else norm_space
    if same(comic) == same(general):
        return "agree"
    if not comic.strip() or not general.strip():
        return "one_empty"
    if same(corr) == same(general):
        return "general"
    if same(corr) == same(comic):
        return "comic"
    return "neither"


def arbiter_table(blocks: List[dict], rule: str) -> Dict[str, Dict[str, int]]:
    table: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for b in blocks:
        o = arbiter_outcome(b, rule)
        table[kind_of(b.get("class_name", ""))][o] += 1
        table["All"][o] += 1
    return {k: dict(v, blocks=sum(v.values())) for k, v in table.items()}


ELLIPSIS_KO = re.compile(r"(\.\.|…|‥)")


def has_caps(s: str) -> bool:
    return any(len(w) > 1 and w.isupper() for w in re.findall(r"[A-Za-z']+", s or ""))


def marker_table(blocks: List[dict], min_blocks: int) -> Dict[str, Dict[str, float]]:
    by = collections.defaultdict(list)
    for b in blocks:
        by[b.get("class_name", "")].append(b)
    out = {}
    for cls, bl in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(bl) < min_blocks:
            continue
        ko = [b.get("corrected_text") or "" for b in bl]
        en = [b.get("translated_text") or "" for b in bl]
        share = lambda xs: round(sum(xs) / len(xs), 3)
        out[cls] = {
            "blocks": len(bl),
            "ko_exclaim": share(["!" in s for s in ko]),
            "ko_ellipsis": share([bool(ELLIPSIS_KO.search(s)) for s in ko]),
            "en_exclaim": share(["!" in s for s in en]),
            "en_ellipsis": share(["..." in s or "…" in s for s in en]),
            "en_caps": share([has_caps(s) for s in en]),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--min-blocks", type=int, default=20)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    blocks = load_blocks(args.runs)
    result = {
        "runs": [str(p) for p in args.runs],
        "blocks": len(blocks),
        "arbiter": {rule: arbiter_table(blocks, rule) for rule in ("space", "pipeline")},
        "markers": marker_table(blocks, args.min_blocks),
    }
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
