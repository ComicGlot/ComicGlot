"""Draw the study sample: frames from pipeline runs, spread over text types and chapters.

The study's tests (transcription, translation) use the same sample, so this script fixes it once,
from a seed. Frames are added one at a time: the type
furthest below its quota is served next, with a random frame that contains it, taken from the
chapter used least so far. Frames with more than --max-blocks-per-frame texts are left out (one
page of twenty sound effects would use up the budget), and no chapter gives more than
--max-frames-per-chapter frames. Sampling stops once --target-blocks texts are in, or when every
type has reached its quota or run out of frames; the rest of the target is then filled with random
frames from the least used chapters. Practice frames are drawn from frames outside the sample.


Writes, in --out:
  sample.json   the parameters, the frames with their blocks, the practice frames, and counts per
                type, family and chapter
  blocks.csv    block_uid, class_name, family, chapter: the --sample of run_conditions.py and the
                list for the OCR baselines

    python -m eval.sample_frames --runs data/output/<runs>/*/chapter_full_results.json \
        --target-blocks 250 --per-type 25 --out data/study/sample
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from research.common import FAMILIES


def local_path(src: str, root: Path) -> Path:
    """Run files store image paths as written on the machine that ran them; keep everything from 'data' on."""
    parts = [p for p in src.replace("\\", "/").split("/") if p]
    if "data" in parts:
        parts = parts[parts.index("data"):]
    return root.joinpath(*parts)


def load_frames(run_files: List[Path]) -> List[dict]:
    frames = []
    for f in run_files:
        run = f.parent.name
        data = json.loads(f.read_text(encoding="utf-8"))
        chapter = Path(str(data.get("chapter_path", run)).replace("\\", "/").rstrip("/")).name
        for k, fr in enumerate(data.get("frames", [])):
            blocks = [{"uid": f"{run}/{b.get('frame_id', fr.get('id'))}/{b.get('id')}", "class_name": b.get("class_name")}
                      for b in fr.get("blocks", [])]
            frames.append({"run": run, "chapter": chapter, "frame_index": k, "frame_id": fr.get("id"),
                           "image_path": fr.get("image_path"), "blocks": blocks})
    return frames



def draw(frames: List[dict], target: int, per_type: int, max_blocks: int, max_per_chapter: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    usable = [i for i, f in enumerate(frames) if 0 < len(f["blocks"]) <= max_blocks]
    types = sorted({b["class_name"] for i in usable for b in frames[i]["blocks"]})
    have: Counter = Counter()
    per_chapter: Counter = Counter()
    chosen: List[int] = []
    taken = set()

    def total() -> int:
        return sum(have.values())

    def pick(pool: List[int]) -> int | None:
        pool = [i for i in pool if i not in taken and per_chapter[frames[i]["chapter"]] < max_per_chapter]
        if not pool:
            return None
        least = min(per_chapter[frames[i]["chapter"]] for i in pool)
        return rng.choice([i for i in pool if per_chapter[frames[i]["chapter"]] == least])

    def add(i: int) -> None:
        taken.add(i)
        chosen.append(i)
        per_chapter[frames[i]["chapter"]] += 1
        have.update(b["class_name"] for b in frames[i]["blocks"])

    open_types = set(types)
    while total() < target and open_types:
        t = min(open_types, key=lambda x: (have[x] / per_type, have[x], x))
        if have[t] >= per_type:
            open_types.discard(t)
            continue
        i = pick([i for i in usable if any(b["class_name"] == t for b in frames[i]["blocks"])])
        if i is None:
            open_types.discard(t)  # no frame left for this type
            continue
        add(i)
    while total() < target:
        i = pick(usable)
        if i is None:
            break
        add(i)
    rng.shuffle(chosen)
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--target-blocks", type=int, default=250)
    ap.add_argument("--per-type", type=int, default=25)
    ap.add_argument("--max-blocks-per-frame", type=int, default=6)
    ap.add_argument("--max-frames-per-chapter", type=int, default=8)
    ap.add_argument("--practice", type=int, default=2, help="practice frames, drawn from outside the sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    frames = load_frames(a.runs)
    chosen = draw(frames, a.target_blocks, a.per_type, a.max_blocks_per_frame, a.max_frames_per_chapter, a.seed)
    rng = random.Random(a.seed + 1)
    rest = [i for i, f in enumerate(frames) if 1 <= len(f["blocks"]) <= 3 and i not in set(chosen)]
    practice = rng.sample(rest, min(a.practice, len(rest)))

    sample = [frames[i] for i in chosen]
    blocks = [b for f in sample for b in f["blocks"]]
    by_type = Counter(b["class_name"] for b in blocks)
    available = Counter(b["class_name"] for f in frames for b in f["blocks"])
    out = {
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k != "runs"},
        "runs": [str(p) for p in a.runs],
        "counts": {
            "frames": len(sample),
            "blocks": len(blocks),
            "by_type": dict(by_type.most_common()),
            "available_by_type": dict(available.most_common()),
            "by_family": dict(Counter(FAMILIES.get(b["class_name"], "unknown") for b in blocks).most_common()),
            "frames_by_chapter": dict(sorted(Counter(f["chapter"] for f in sample).items(), key=lambda x: x[0])),
        },
        "frames": sample,
        "practice": [frames[i] for i in practice],
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "sample.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(a.out / "blocks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["block_uid", "class_name", "family", "chapter"])
        for fr in sample:
            for b in fr["blocks"]:
                w.writerow([b["uid"], b["class_name"], FAMILIES.get(b["class_name"], "unknown"), fr["chapter"]])

    c = out["counts"]
    print(f"{c['frames']} frames, {c['blocks']} blocks; practice {len(practice)} frames")
    print("by type (sample / available):", ", ".join(f"{t} {by_type[t]}/{available[t]}" for t, _ in available.most_common()))
    print("by family:", c["by_family"])
    print("frames by chapter:", c["frames_by_chapter"])


if __name__ == "__main__":
    main()
