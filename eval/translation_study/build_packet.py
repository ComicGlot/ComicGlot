"""Turn condition outputs into a blind rating workbook for translators.

Reads calls.jsonl from run_conditions.py and writes:
  <out>/packet_<rater>.xlsx   one workbook per rater, same items, item order and
                              letter order shuffled per rater
  <out>/key.json              rater -> item -> letter -> condition (keep private)

Each row shows the block crop, the previous lines of the scene, the Korean
source and one candidate translation labelled A, B, C ... Raters score every
candidate on the study's rating scales (adequacy, fluency, tone). Identical candidates are kept as
separate rows so raters cannot infer conditions from duplicates.

Usage:
    python -m eval.translation_study.build_packet --calls study_round1/calls.jsonl \
        --runs RUN.json ... --project-root /path/to/ComicGlot --raters r1 r2 r3 --out study_round1/packets
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import string
from pathlib import Path

from eval.make_ocr_gt_sheet import local_path
from eval.translation_study.run_conditions import load_blocks

SCALES = ["adequacy_1to5", "tone_fit_1to5", "fluency_1to5", "major_errors", "minor_errors", "notes"]


def crop_for(block, project_root: Path, max_w: int = 360):
    from PIL import Image
    src = local_path(block["_frame_image"] or block["source_image"], project_root)
    im = Image.open(src).convert("RGB")
    x0, y0, x1, y1 = map(int, block["box"])
    c = im.crop((max(0, x0 - 6), max(0, y0 - 6), min(im.width, x1 + 6), min(im.height, y1 + 6)))
    if c.width > max_w:
        c = c.resize((max_w, int(c.height * max_w / c.width)))
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calls", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--project-root", type=Path)
    ap.add_argument("--raters", nargs="+", default=["r1", "r2", "r3"])
    ap.add_argument("--context", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Alignment, Font

    calls = [json.loads(l) for l in open(args.calls, encoding="utf-8")]
    by_item = collections.defaultdict(dict)
    for c in calls:
        by_item[c["block_uid"]][c["condition"]] = c
    order, blocks = load_blocks(args.runs)
    items = sorted(by_item)
    args.out.mkdir(parents=True, exist_ok=True)
    crops_dir = args.out / "_crops"
    crops_dir.mkdir(exist_ok=True)
    key = {}
    for r_i, rater in enumerate(args.raters):
        rng = random.Random(f"{args.seed}-{rater}")
        its = items[:]
        rng.shuffle(its)
        wb = Workbook()
        ws = wb.active
        ws.title = "rate"
        head = ["item", "image", "previous lines", "Korean source", "candidate", "translation"] + SCALES
        ws.append(head)
        for cell in ws[1]:
            cell.font = Font(bold=True)
        key[rater] = {}
        row = 2
        for n, uid in enumerate(its, 1):
            conds = list(by_item[uid])
            rng.shuffle(conds)
            letters = dict(zip(string.ascii_uppercase, conds))
            key[rater][n] = {"block_uid": uid, "letters": letters}
            i = order.index(uid)
            prev = [blocks[u] for u in order[max(0, i - args.context):i] if blocks[u]["_run"] == blocks[uid]["_run"]]
            prev_txt = "\n".join((p.get("corrected_text") or p.get("text") or "") for p in prev)
            src = next(iter(by_item[uid].values()))["source_text"]
            first = row
            for letter, cond in letters.items():
                ws.append([n, "", prev_txt, src, letter, by_item[uid][cond]["translation"]] + [""] * len(SCALES))
                for col in (3, 4, 6):
                    ws.cell(row=row, column=col).alignment = Alignment(wrap_text=True, vertical="top")
                row += 1
            if args.project_root:
                img_path = crops_dir / f"{hashlib.md5(uid.encode()).hexdigest()[:12]}.png"
                if not img_path.exists():
                    crop_for(blocks[uid], args.project_root).save(img_path)
                img = XLImage(str(img_path))
                img.anchor = f"B{first}"
                ws.add_image(img)
                ws.row_dimensions[first].height = min(400, img.height * 0.75)
        for col, width in zip("ABCDEFGHIJKL", [6, 48, 40, 40, 10, 50, 12, 12, 12, 10, 10, 40]):
            ws.column_dimensions[col].width = width
        wb.save(args.out / f"packet_{rater}.xlsx")
    json.dump(key, open(args.out / "key.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(items)} items x {len(args.raters)} raters -> {args.out}")


if __name__ == "__main__":
    main()
