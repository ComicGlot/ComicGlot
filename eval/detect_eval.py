"""Detection evaluation with explicit image lists and honest per-class counts.

Ultralytics' val reports scores on whatever split it is given; for the paper
we need (a) a test list that no training run has seen and that was not used to
pick epochs, (b) per-class AP with the number of instances behind it, and (c)
intervals. This script does the matching itself, the same way Ultralytics val
does (predictions at conf 0.001, NMS IoU 0.7, ten IoU thresholds 0.50:0.95,
101-point interpolated AP), on a list of images you choose.

Steps:
  predict   run one or more weights over the image list, caching predictions
  score     AP50, AP50-95, precision and recall at the F1-optimal confidence,
            per class with instance counts, plus a class-agnostic AP for
            comparison with 1- to 3-class comic detectors; 95% intervals by
            bootstrapping images

Usage:
    python -m eval.detect_eval predict --weights best.pt --images list.txt --labels-dir DIR/labels --imgsz 800 --out preds.jsonl
    python -m eval.detect_eval score --preds preds.jsonl --labels-dir DIR/labels [--boot 500]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from research.common import CLASS_NAMES, DISPLAY_NAMES, read_label_file

IOUS = np.linspace(0.5, 0.95, 10)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), 2)
    area_a = np.prod(a[:, 2:] - a[:, :2], 1)
    area_b = np.prod(b[:, 2:] - b[:, :2], 1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(pred_xyxy, pred_cls, gt_xyxy, gt_cls, agnostic=False) -> np.ndarray:
    """Ultralytics-style greedy matching. Returns tp[n_pred, 10]."""
    tp = np.zeros((len(pred_xyxy), len(IOUS)), bool)
    if not len(pred_xyxy) or not len(gt_xyxy):
        return tp
    iou = box_iou(gt_xyxy, pred_xyxy)
    same = np.ones_like(iou, bool) if agnostic else gt_cls[:, None] == pred_cls[None, :]
    iou = iou * same
    for k, t in enumerate(IOUS):
        gi, pi = np.nonzero(iou >= t)
        if not len(gi):
            continue
        vals = iou[gi, pi]
        order = np.argsort(-vals)
        gi, pi = gi[order], pi[order]
        _, first_p = np.unique(pi, return_index=True)
        gi, pi = gi[first_p], pi[first_p]
        order = np.argsort(-iou[gi, pi])
        gi, pi = gi[order], pi[order]
        _, first_g = np.unique(gi, return_index=True)
        tp[pi[first_g], k] = True
    return tp


AP_STYLE = "u83"  # "u83": Ultralytics 8.3 (used for all ComicGlot runs); "u84": 8.4, no recall extrapolation


def ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    if AP_STYLE == "u84":
        mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    else:
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(np.interp(x, mrec, mpre), x))


def per_class_ap(tp, conf, pcls, gcls, n_classes):
    order = np.argsort(-conf)
    tp, conf, pcls = tp[order], conf[order], pcls[order]
    out = {}
    for c in range(n_classes):
        n_gt = int((gcls == c).sum())
        sel = pcls == c
        if n_gt == 0:
            continue
        if not sel.any():
            out[c] = {"n": n_gt, "ap50": 0.0, "ap": 0.0, "p": 0.0, "r": 0.0}
            continue
        tpc = np.cumsum(tp[sel], 0)
        fpc = np.cumsum(1 - tp[sel], 0)
        rec = tpc / n_gt
        pre = tpc / (tpc + fpc)
        aps = [ap_101(rec[:, k], pre[:, k]) for k in range(len(IOUS))]
        f1 = 2 * pre[:, 0] * rec[:, 0] / (pre[:, 0] + rec[:, 0] + 1e-9)
        i = int(np.argmax(f1))
        out[c] = {"n": n_gt, "ap50": aps[0], "ap": float(np.mean(aps)), "p": float(pre[i, 0]), "r": float(rec[i, 0])}
    return out


def nms_classwise(xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray, iou_thr: float = 0.7, max_det: int = 300):
    """Greedy class-wise non-maximum suppression; returns kept indices in descending confidence.

    Ultralytics applies NMS (IoU 0.7) to YOLO outputs before scoring. DETR-style
    models are NMS-free and emit many low-score duplicates; under IoU-greedy
    matching a duplicate can take a ground-truth box away from the confident
    prediction, so their predictions must be suppressed the same way before scoring.
    """
    order = np.argsort(-conf)
    keep = []
    for c in np.unique(cls):
        idx = order[cls[order] == c]
        while len(idx):
            i = idx[0]
            keep.append(i)
            if len(idx) == 1:
                break
            iou = box_iou(xyxy[i:i + 1], xyxy[idx[1:]])[0]
            idx = idx[1:][iou <= iou_thr]
    keep = np.array(sorted(keep, key=lambda i: -conf[i]), int)
    return keep[:max_det]


def apply_nms(preds: List[Dict], iou_thr: float) -> List[Dict]:
    out = []
    for p in preds:
        if p["cls"]:
            k = nms_classwise(np.array(p["xyxyn"], float).reshape(-1, 4), np.array(p["conf"], float),
                              np.array(p["cls"], int), iou_thr)
            p = {**p, "xyxyn": [p["xyxyn"][i] for i in k], "cls": [p["cls"][i] for i in k],
                 "conf": [p["conf"][i] for i in k]}
        out.append(p)
    return out


def load_gt(labels_dir: Path, name: str):
    boxes = read_label_file(labels_dir / name) if (labels_dir / name).exists() else []
    xyxy = np.array([[b.xc - b.w / 2, b.yc - b.h / 2, b.xc + b.w / 2, b.yc + b.h / 2] for b in boxes]).reshape(-1, 4)
    return xyxy, np.array([b.cls for b in boxes], int)


def evaluate(preds: List[Dict], labels_dir: Path, agnostic=False):
    per_img = []
    for p in preds:
        gx, gc = load_gt(labels_dir, p["f"])
        px = np.array(p["xyxyn"]).reshape(-1, 4)
        pc = np.array(p["cls"], int)
        cf = np.array(p["conf"], float)
        tp = match(px, pc, gx, gc, agnostic)
        per_img.append((tp, cf, np.zeros_like(pc) if agnostic else pc, np.zeros_like(gc) if agnostic else gc))
    return per_img


def summarise(per_img, n_classes):
    tp = np.concatenate([x[0] for x in per_img]) if per_img else np.zeros((0, 10))
    cf = np.concatenate([x[1] for x in per_img])
    pc = np.concatenate([x[2] for x in per_img])
    gc = np.concatenate([x[3] for x in per_img])
    pcs = per_class_ap(tp, cf, pc, gc, n_classes)
    if not pcs:
        return {"map50": 0.0, "map": 0.0, "per_class": {}}
    return {"map50": float(np.mean([v["ap50"] for v in pcs.values()])),
            "map": float(np.mean([v["ap"] for v in pcs.values()])),
            "precision": float(np.mean([v["p"] for v in pcs.values()])),
            "recall": float(np.mean([v["r"] for v in pcs.values()])),
            "per_class": pcs}


def cmd_predict(args):
    from ultralytics import YOLO
    names = [l.strip() for l in open(args.images) if l.strip()]
    done = set()
    if args.out.exists():
        done = {json.loads(l)["f"] for l in open(args.out)}
    model = YOLO(args.weights)
    with open(args.out, "a") as fh:
        for img in names:
            label = Path(img).with_suffix(".txt").name
            if label in done:
                continue
            r = model.predict(img, imgsz=args.imgsz, conf=0.001, iou=0.7, max_det=300, verbose=False)[0]
            fh.write(json.dumps({"f": label, "xyxyn": r.boxes.xyxyn.tolist(), "cls": r.boxes.cls.int().tolist(),
                                 "conf": r.boxes.conf.tolist()}) + "\n")


def cmd_score(args):
    global AP_STYLE
    AP_STYLE = args.ap_style
    preds = [json.loads(l) for l in open(args.preds)]
    if args.only:
        keep = {l.strip() for l in open(args.only) if l.strip()}
        preds = [p for p in preds if p["f"] in keep]
    if args.nms:
        preds = apply_nms(preds, args.nms)
    per_img = evaluate(preds, args.labels_dir)
    res = summarise(per_img, len(CLASS_NAMES))
    agn = summarise(evaluate(preds, args.labels_dir, agnostic=True), 1)
    rng = np.random.default_rng(0)
    boots = [summarise([per_img[i] for i in rng.integers(0, len(per_img), len(per_img))], len(CLASS_NAMES))
             for _ in range(args.boot)]
    ci = lambda key: (float(np.percentile([b[key] for b in boots], 2.5)), float(np.percentile([b[key] for b in boots], 97.5))) if boots else None
    print(f"images {len(preds)}  mAP50 {res['map50']:.3f} {ci('map50')}  mAP50-95 {res['map']:.3f} {ci('map')}  "
          f"P {res['precision']:.3f} R {res['recall']:.3f}  class-agnostic AP50 {agn['map50']:.3f}")
    for c, v in sorted(res["per_class"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {DISPLAY_NAMES[CLASS_NAMES[c]]:16s} n={v['n']:5d}  AP50 {v['ap50']:.3f}  AP50-95 {v['ap']:.3f}  P {v['p']:.2f} R {v['r']:.2f}")
    if args.out:
        json.dump({"images": len(preds), **res, "ci95": {"map50": ci("map50"), "map": ci("map")},
                   "class_agnostic_ap50": agn["map50"]}, open(args.out, "w"), indent=1)


def cmd_seedcheck(args):
    """How many ground-truth boxes are near-copies of this model's predictions.

    Model-assisted labelling leaves a trace: boxes accepted from a pre-label
    match the seeding model's output almost exactly (IoU >= 0.98), while boxes
    drawn by hand rarely do. A high share means the labels are not independent
    of that model and should not be used to test it or its descendants.
    """
    preds = {json.loads(l)["f"]: json.loads(l) for l in open(args.preds)}
    if args.only:
        keep = {l.strip() for l in open(args.only) if l.strip()}
        preds = {k: v for k, v in preds.items() if k in keep}
    best = []
    for f, p in preds.items():
        gx, gc = load_gt(args.labels_dir, f)
        if not len(gx):
            continue
        conf = np.array(p["conf"], float)
        px = np.array(p["xyxyn"]).reshape(-1, 4)[conf >= args.conf]
        pc = np.array(p["cls"], int)[conf >= args.conf]
        iou = box_iou(gx, px)
        for i in range(len(gx)):
            same = iou[i][pc == gc[i]] if len(px) else np.array([])
            best.append(float(same.max()) if len(same) else 0.0)
    b = np.array(best)
    out = {"gt_boxes": int(len(b)), **{f"share_iou_ge_{t}": float((b >= t).mean()) for t in (0.5, 0.9, 0.95, 0.98, 0.99)}}
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("seedcheck", help="share of labels that are near-copies of a model's predictions")
    c.add_argument("--preds", type=Path, required=True)
    c.add_argument("--labels-dir", type=Path, required=True)
    c.add_argument("--only", type=Path)
    c.add_argument("--conf", type=float, default=0.25)
    p = sub.add_parser("predict")
    p.add_argument("--weights", required=True)
    p.add_argument("--images", type=Path, required=True, help="text file, one image path per line")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--preds", type=Path, required=True)
    s.add_argument("--labels-dir", type=Path, required=True)
    s.add_argument("--only", type=Path, help="restrict to these label filenames")
    s.add_argument("--boot", type=int, default=500)
    s.add_argument("--ap-style", choices=["u83", "u84"], default="u83",
                   help="u83 matches Ultralytics 8.3.x, which trained and validated every ComicGlot model")
    s.add_argument("--nms", type=float, default=0.0,
                   help="class-wise NMS IoU applied before scoring; use 0.7 for NMS-free models (DETR, RT-DETR)")
    s.add_argument("--out", type=Path)
    args = ap.parse_args()
    {"predict": cmd_predict, "score": cmd_score, "seedcheck": cmd_seedcheck}[args.cmd](args)


if __name__ == "__main__":
    main()
