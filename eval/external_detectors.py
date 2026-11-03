"""Predictions from non-YOLO detectors in the cache format of eval/detect_eval.py.

Two kinds of model were compared with the deployed YOLO11m in the paper (Section 6.1):

  detr     the coursework's DETR ResNet-50 checkpoint (models/DETR-RESNET-50/...), trained
           with Roboflow COCO ids, where id 0 is the unused superclass and id k is YOLO class k-1
  rtdetr   ogkalu/comic-text-and-bubble-detector (RT-DETR-v2, classes bubble / text_bubble /
           text_free), the detector of the comic-translate project; class ids are kept as is

Both are NMS-free, so score them with `detect_eval score --nms 0.7`. The `families`
subcommand compares any cache with the 13-class labels at the level of the three
families the open detector can express (type ignored).

Usage:
    python -m eval.external_detectors detr --model models/DETR-RESNET-50/detr-resnet-50-bubbles-50ep \
        --images list.txt --out preds_detr.jsonl
    python -m eval.external_detectors rtdetr --images list.txt --out preds_ogkalu.jsonl
    python -m eval.detect_eval score --preds preds_detr.jsonl --labels-dir DS/test/labels --nms 0.7
    python -m eval.external_detectors families --preds preds_ogkalu.jsonl --labels-dir DS/test/labels \
        --map ogkalu --nms 0.7

Needs torch and transformers >= 4.49 (RT-DETR-v2); `pip install timm` for the DETR backbone.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from eval.detect_eval import apply_nms, box_iou, load_gt, match, summarise  # noqa: F401
from research.common import CLASS_NAMES

RTDETR_REPO = "ogkalu/comic-text-and-bubble-detector"
BALLOONS = {"speech_bubble", "think_bubble", "cloud_bubble", "wiggle_bubble", "scream_bubble",
            "spiked_bubble", "hexagon_bubble"}
FREE = {"standalone_text", "background_text", "title_panel"}


def _predict(model, proc, images, out: Path, label_shift: int, threshold: float):
    import torch
    from PIL import Image
    done = {json.loads(l)["f"] for l in open(out)} if out.exists() else set()
    with open(out, "a") as fh, torch.no_grad():
        for img_path in images:
            label = Path(img_path).with_suffix(".txt").name
            if label in done:
                continue
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            t0 = time.time()
            res = proc.post_process_object_detection(model(**proc(images=img, return_tensors="pt")),
                                                     threshold=threshold, target_sizes=[(h, w)])[0]
            ms = (time.time() - t0) * 1000
            keep = res["labels"] >= label_shift
            boxes = res["boxes"][keep].tolist()
            fh.write(json.dumps({
                "f": label,
                "xyxyn": [[min(1, max(0, b[0] / w)), min(1, max(0, b[1] / h)), min(1, max(0, b[2] / w)),
                           min(1, max(0, b[3] / h))] for b in boxes],
                "cls": (res["labels"][keep] - label_shift).tolist(),
                "conf": res["scores"][keep].tolist(), "ms": round(ms, 1)}) + "\n")


def cmd_detr(a):
    from transformers import DetrForObjectDetection, DetrImageProcessor
    model = DetrForObjectDetection.from_pretrained(a.model).eval()
    images = [l.strip() for l in open(a.images) if l.strip()]
    _predict(model, DetrImageProcessor(), images, a.out, label_shift=1, threshold=0.0)


def cmd_rtdetr(a):
    from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection
    model = RTDetrV2ForObjectDetection.from_pretrained(a.repo).eval()
    images = [l.strip() for l in open(a.images) if l.strip()]
    _predict(model, AutoImageProcessor.from_pretrained(a.repo), images, a.out, label_shift=0, threshold=0.001)


def family_maps(kind: str):
    """(gt_keep, pred_keep) predicates for the three comparisons reported in the paper."""
    if kind == "ogkalu":
        return {"outlined_balloons": (BALLOONS, {0}),
                "all_but_sfx": (set(CLASS_NAMES) - {"sfx_sound"}, {0, 2}),
                "sfx": ({"sfx_sound"}, {2})}
    return {"outlined_balloons": (BALLOONS, {CLASS_NAMES.index(c) for c in BALLOONS}),
            "all_but_sfx": (set(CLASS_NAMES) - {"sfx_sound"}, {i for i, c in enumerate(CLASS_NAMES) if c != "sfx_sound"}),
            "sfx": ({"sfx_sound"}, {CLASS_NAMES.index("sfx_sound")})}


def cmd_families(a):
    preds = [json.loads(l) for l in open(a.preds)]
    if a.nms:
        preds = apply_nms(preds, a.nms)
    out = {}
    for name, (gt_keep, pred_keep) in family_maps(a.map).items():
        per = []
        for p in preds:
            gx, gc = load_gt(a.labels_dir, p["f"])
            sel = np.array([CLASS_NAMES[c] in gt_keep for c in gc], bool)
            gx = gx[sel] if len(gx) else gx
            idx = [i for i, c in enumerate(p["cls"]) if c in pred_keep]
            px = np.array([p["xyxyn"][i] for i in idx]).reshape(-1, 4)
            cf = np.array([p["conf"][i] for i in idx], float)
            z = np.zeros(len(idx), int)
            per.append((match(px, z, gx, np.zeros(len(gx), int), True), cf, z, np.zeros(len(gx), int)))
        s = summarise(per, 1)
        out[name] = {"n": s["per_class"][0]["n"], "ap50": round(s["map50"], 3), "ap50_95": round(s["map"], 3)}
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detr")
    d.add_argument("--model", required=True)
    r = sub.add_parser("rtdetr")
    r.add_argument("--repo", default=RTDETR_REPO)
    for x in (d, r):
        x.add_argument("--images", type=Path, required=True, help="text file, one image path per line")
        x.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("families")
    f.add_argument("--preds", type=Path, required=True)
    f.add_argument("--labels-dir", type=Path, required=True)
    f.add_argument("--map", choices=["ogkalu", "comicglot"], required=True)
    f.add_argument("--nms", type=float, default=0.0)
    a = ap.parse_args()
    {"detr": cmd_detr, "rtdetr": cmd_rtdetr, "families": cmd_families}[a.cmd](a)


if __name__ == "__main__":
    main()
