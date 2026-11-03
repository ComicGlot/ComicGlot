"""Blind class-annotation workbooks for the inter-annotator agreement study.

Takes a crop sample from `research.carrier_probe sample` (balanced, one crop
per image per class) and writes one workbook per annotator. Each row shows the
crop with some page context and the target box outlined in red, a dropdown
of the 13 classes, a second choice and a confidence score. The dataset label
is never shown; it goes to key.json with the item order of every annotator.

    python -m research.carrier_probe sample --dataset DS10K --work out/annot --per-class 46 --seed 7
    python -m eval.make_class_sheet --dataset DS10K --sample out/annot/sample.json \
        --annotators a1 a2 a3 --out out/annot/sheets

Score the filled sheets with eval.translation_study.analyse_ratings-style
alpha (nominal) or any agreement tool; key.json holds the dataset labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from research.common import CLASS_NAMES, DISPLAY_NAMES

CHOICES = [DISPLAY_NAMES[c] for c in CLASS_NAMES] + ["unsure"]


def context_crop(image_path: Path, box, margin: float = 0.5, max_side: int = 420):
    from PIL import Image, ImageDraw
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    xc, yc, w, h = box
    x0, y0, x1, y1 = (xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    cx0, cy0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
    cx1, cy1 = min(W, int(x1 + mx)), min(H, int(y1 + my))
    crop = im.crop((cx0, cy0, cx1, cy1))
    ImageDraw.Draw(crop).rectangle((x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0), outline=(230, 30, 30), width=3)
    scale = min(1.0, max_side / max(crop.size))
    if scale < 1:
        crop = crop.resize((int(crop.width * scale), int(crop.height * scale)))
    return crop


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--annotators", nargs="+", default=["a1", "a2", "a3"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Font
    from openpyxl.worksheet.datavalidation import DataValidation

    sample = json.load(open(args.sample))
    args.out.mkdir(parents=True, exist_ok=True)
    crops = args.out / "_crops"
    crops.mkdir(exist_ok=True)
    items = []
    for s in sample:
        uid = f"{s['split']}/{s['label']}/{s['cls']}/{','.join(f'{v:.4f}' for v in s['box'])}"
        name = hashlib.md5(uid.encode()).hexdigest()[:12] + ".png"
        if not (crops / name).exists():
            context_crop(args.dataset / s["split"] / "images" / s["image"], s["box"]).save(crops / name)
        items.append({"uid": uid, "crop": name, "dataset_class": CLASS_NAMES[s["cls"]], **s})
    key = {"items": items, "order": {}}
    for ann in args.annotators:
        order = list(range(len(items)))
        random.Random(f"{args.seed}-{ann}").shuffle(order)
        key["order"][ann] = order
        wb = Workbook()
        ws = wb.active
        ws.title = "classes"
        ws.append(["row", "crop (red box = the text to classify)", "class", "second choice", "confidence 1-3", "notes"])
        for c in ws[1]:
            c.font = Font(bold=True)
        dv = DataValidation(type="list", formula1='"' + ",".join(CHOICES) + '"', allow_blank=True)
        ws.add_data_validation(dv)
        for r, i in enumerate(order, start=2):
            ws.append([r - 1, "", "", "", "", ""])
            img = XLImage(str(crops / items[i]["crop"]))
            img.anchor = f"B{r}"
            ws.add_image(img)
            ws.row_dimensions[r].height = min(320, img.height * 0.75)
            dv.add(f"C{r}")
            dv.add(f"D{r}")
        for col, width in zip("ABCDEF", [6, 60, 20, 20, 14, 30]):
            ws.column_dimensions[col].width = width
        wb.save(args.out / f"classes_{ann}.xlsx")
    json.dump(key, open(args.out / "key.json", "w", encoding="utf-8"), indent=1)
    print(f"{len(items)} crops x {len(args.annotators)} annotators -> {args.out}")


if __name__ == "__main__":
    main()
