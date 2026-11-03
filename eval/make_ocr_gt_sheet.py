"""Build a transcription sheet for OCR ground truth.

Samples blocks from pipeline outputs (chapter_full_results.json), stratified by
class, crops each block from its source page and writes:

  <out>/crops/<n>.png      the block image, numbered in random order
  <out>/sheet.csv          item, block_uid, class_name, gt_text (empty), notes
  <out>/key.json           item -> block_uid and box, kept away from transcribers

Transcribers see the crop only. Model readings are left out by default, since
showing them anchors the transcriber on the model's errors; pass
--show-readings for a correction pass instead of a blind transcription.

Paths in the JSON are Windows paths from the original machine
(E:\\Projects\\Python\\Manhua_Scanner\\data\\...). Boxes are in the coordinates of
the frame's segment image (data/output/<run>/segments/segment_NNN.png), so the
crop is cut from the segment. --project-root replaces everything before "data".

Usage:
    python -m eval.make_ocr_gt_sheet --runs out/run25/chapter_full_results.json ... \
        --project-root /path/to/ComicGlot --per-class 40 --out ocr_gt_round1
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
from pathlib import Path


def local_path(src: str, project_root: Path) -> Path:
    parts = [p for p in re.split(r"[\\/]", src) if p]
    if "data" in parts:
        parts = parts[parts.index("data"):]
    return project_root.joinpath(*parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--margin", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show-readings", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from PIL import Image

    rng = random.Random(args.seed)
    by_cls = collections.defaultdict(list)
    for p in args.runs:
        run = p.parent.name
        data = json.load(open(p, encoding="utf-8"))
        for frame in data.get("frames", []):
            for b in frame.get("blocks", []):
                uid = f"{run}/{b.get('frame_id', frame.get('id'))}/{b.get('id')}"
                by_cls[b.get("class_name", "unknown")].append((uid, b, frame.get("image_path")))
    chosen = []
    for cls, items in sorted(by_cls.items()):
        rng.shuffle(items)
        chosen += items[: args.per_class]
    rng.shuffle(chosen)

    (args.out / "crops").mkdir(parents=True, exist_ok=True)
    key, rows = {}, []
    for n, (uid, b, frame_image) in enumerate(chosen, 1):
        src = local_path(frame_image or b["source_image"], args.project_root)
        if not src.exists():
            raise SystemExit(f"missing image {src}; segments are only saved with SAVE_SEGMENTS=True")
        im = Image.open(src).convert("RGB")
        x0, y0, x1, y1 = map(int, b["box"])
        m = args.margin
        crop = im.crop((max(0, x0 - m), max(0, y0 - m), min(im.width, x1 + m), min(im.height, y1 + m)))
        crop.save(args.out / "crops" / f"{n:04d}.png")
        key[n] = {"block_uid": uid, "class_name": b.get("class_name"), "box": b["box"], "source": str(src)}
        row = {"item": n, "block_uid": uid, "class_name": b.get("class_name"), "gt_text": "", "notes": ""}
        if args.show_readings:
            row.update({"reading_a": b.get("text", ""), "reading_b": b.get("secondary_text", "")})
        rows.append(row)
    with open(args.out / "sheet.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    json.dump(key, open(args.out / "key.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(rows)} blocks -> {args.out}/sheet.csv")
    print(collections.Counter(r["class_name"] for r in rows))


if __name__ == "__main__":
    main()
