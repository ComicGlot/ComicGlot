"""Compare open comic text detectors with ComicGlot on a labelled test split.

The systems are run from ONNX files only, so no third-party model code is executed:

  ctd       comic-text-detector (dmMaze), the detector of BallonsTranslator and one of the
            detectors of manga-image-translator. File: comictextdetector.pt.onnx from the
            manga-image-translator releases (beta-0.3). Only its text-block head is needed;
            `python -m eval.compare_systems blkonly ctd.onnx ctd_blk.onnx` cuts the segmentation
            and line decoders away, which makes it about 300 times faster on a CPU with the same
            block output.
  ogkalu    ogkalu/comic-text-and-bubble-detector (RT-DETR-v2), the detector of comic-translate.
            File: detector.onnx from that Hugging Face repository.
  bubble    kitsumed/yolov8m_seg-speech-bubble, a speech-bubble model credited by MangaTranslator.
            File: model_dynamic.onnx.
  manga109  deepghs/manga109_yolo, YOLO11m trained on Manga109 (body, face, frame, text), credited
            by MangaTranslator. File: v2023.12.07_m_yv11/model.onnx. Only the text class is used.
  ppocr     PP-OCRv4 text detector, a general OCR baseline. File: ch_PP-OCRv4_det_infer.onnx
            (SWHL/RapidOCR).

ComicGlot's own predictions come from the caches of eval/detect_eval.py.

Every system writes the same cache format as eval/detect_eval.py: one JSON line per image with
"f" (label file name), "wh", "xyxyn" (normalised boxes), "cls" and "conf". The boxes of the
different systems follow different conventions (text lines, text blocks, balloons), so the
comparison uses coverage, which does not depend on how tightly a box is drawn: a labelled block
counts as found when its centre lies inside a predicted box. Precision is the share of predicted
boxes whose centre lies inside some labelled block.

Usage:
    python -m eval.compare_systems predict --system ctd --model ctd.onnx --images list.txt --out preds_ctd.jsonl
    python -m eval.compare_systems table --labels-dir DS/test/labels \
        --run "ComicGlot=preds_y11m.jsonl::0.25" --run "comic-text-detector=preds_ctd.jsonl::0.4" \
        --run "Manga109 YOLO=preds_m109.jsonl:3:0.25" --out compare_systems.json

In --run, the three fields after "=" are the cache, the class ids to keep (empty for all) and
the confidence threshold (each system's own default is used in the paper).
"""
from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np

CLASS_NAMES = ["background_text", "cloud_bubble", "hexagon_bubble", "scream_bubble", "sfx_sound",
               "speech_bubble", "spiked_bubble", "sqare_bubble", "standalone_text", "system_bubble",
               "think_bubble", "title_panel", "wiggle_bubble"]
BALLOONS = {"speech_bubble", "think_bubble", "cloud_bubble", "wiggle_bubble", "scream_bubble",
            "spiked_bubble", "hexagon_bubble"}
GROUPS = {
    "outlined_balloons": lambda c: c in BALLOONS,
    "other_non_sfx": lambda c: c not in BALLOONS and c != "sfx_sound",
    "sfx": lambda c: c == "sfx_sound",
    "all": lambda c: True,
}


# ---------------------------------------------------------------- image helpers
def read_rgb(path: str) -> np.ndarray:
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def letterbox(img: np.ndarray, new: tuple[int, int], stride: int = 32, square: bool = True):
    """Resize keeping the aspect ratio and pad. Returns padded image, scale and (dx, dy)."""
    import cv2
    h, w = img.shape[:2]
    r = min(new[0] / h, new[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if square:
        H, W = new
    else:
        H, W = int(np.ceil(nh / stride) * stride), int(np.ceil(nw / stride) * stride)
    dy, dx = (H - nh) // 2, (W - nw) // 2
    out = np.full((H, W, 3), 114, np.uint8)
    out[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, (dx, dy)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        order = order[1:][inter / (a + b - inter + 1e-9) <= iou]
    return keep


def classwise_nms(boxes, scores, cls, iou):
    keep = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        keep += [int(idx[k]) for k in nms(boxes[idx], scores[idx], iou)]
    return sorted(keep)


# ---------------------------------------------------------------- detectors
class Ultralytics:
    """YOLOv8/YOLO11 detect or segment model exported to ONNX without NMS."""

    def __init__(self, model, imgsz=640, nms_iou=0.7, min_conf=0.05):
        import onnxruntime as ort
        self.s = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        meta = self.s.get_modelmeta().custom_metadata_map
        self.nc = len(ast.literal_eval(meta.get("names", "{0: 'x'}")))
        self.imgsz, self.iou, self.min_conf = imgsz, nms_iou, min_conf

    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        x, r, (dx, dy) = letterbox(rgb, (self.imgsz, self.imgsz), square=False)
        x = x.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.s.run(None, {self.s.get_inputs()[0].name: x})[0][0].T  # (N, 4 + nc [+ 32])
        sc = out[:, 4:4 + self.nc]
        cls, conf = sc.argmax(1), sc.max(1)
        m = conf >= self.min_conf
        b, cls, conf = out[m, :4], cls[m], conf[m]
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        keep = classwise_nms(xyxy, conf, cls, self.iou)
        xyxy = (xyxy[keep] - [dx, dy, dx, dy]) / r
        return xyxy, cls[keep], conf[keep], (w, h)


class CTD:
    """comic-text-detector: YOLOv5-style text-block head at 1024 x 1024."""

    def __init__(self, model, nms_iou=0.35, min_conf=0.05):
        import onnxruntime as ort
        self.s = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.iou, self.min_conf = nms_iou, min_conf

    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        x, r, (dx, dy) = letterbox(rgb, (1024, 1024), square=True)
        x = x.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        blk = self.s.run(["blk"], {"images": x})[0][0]  # (N, 7): cx cy w h obj c0 c1
        conf = blk[:, 4] * blk[:, 5:].max(1)
        cls = blk[:, 5:].argmax(1)
        m = conf >= self.min_conf
        b, conf, cls = blk[m, :4], conf[m], cls[m]
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        keep = nms(xyxy, conf, self.iou)
        xyxy = (xyxy[keep] - [dx, dy, dx, dy]) / r
        return xyxy, cls[keep], conf[keep], (w, h)


class OgkaluONNX:
    """comic-translate's RT-DETR-v2 export: 640 x 640 resize, boxes returned in original pixels."""

    def __init__(self, model, size=640):
        import onnxruntime as ort
        self.s = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.size = size

    def __call__(self, rgb):
        import cv2
        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, (self.size, self.size)).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        labels, boxes, scores = self.s.run(None, {"images": x, "orig_target_sizes": np.array([[w, h]], np.int64)})
        return boxes[0].astype(float), labels[0].astype(int), scores[0].astype(float), (w, h)


class PPOCRDet:
    """PP-OCRv4 DB text detector with PaddleOCR's default post-processing."""

    def __init__(self, model, limit=960, thresh=0.3, box_thresh=0.6, unclip=1.5):
        import onnxruntime as ort
        self.s = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.limit, self.thresh, self.box_thresh, self.unclip = limit, thresh, box_thresh, unclip

    def __call__(self, rgb):
        import cv2
        import pyclipper
        h, w = rgb.shape[:2]
        r = min(1.0, self.limit / max(h, w))
        nh, nw = max(32, int(round(h * r / 32)) * 32), max(32, int(round(w * r / 32)) * 32)
        bgr = cv2.cvtColor(cv2.resize(rgb, (nw, nh)), cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0
        x = ((bgr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
        prob = self.s.run(None, {self.s.get_inputs()[0].name: x})[0][0, 0]
        cnts, _ = cv2.findContours((prob > self.thresh).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        out, scores = [], []
        for c in cnts:
            if len(c) < 4:
                continue
            mask = np.zeros_like(prob, np.uint8)
            cv2.fillPoly(mask, [c], 1)
            score = float(prob[mask.astype(bool)].mean())
            if score < self.box_thresh:
                continue
            rect = cv2.minAreaRect(c)
            poly = cv2.boxPoints(rect)
            area, peri = cv2.contourArea(poly), cv2.arcLength(poly, True)
            if peri == 0:
                continue
            off = pyclipper.PyclipperOffset()
            off.AddPath(poly.astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            exp = off.Execute(area * self.unclip / peri)
            if not exp:
                continue
            p = np.array(exp[0])
            x1, y1, x2, y2 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
            out.append([x1 * w / nw, y1 * h / nh, x2 * w / nw, y2 * h / nh])
            scores.append(score)
        b = np.array(out, float).reshape(-1, 4)
        return b, np.zeros(len(b), int), np.array(scores, float), (w, h)


def cmd_blkonly(a):
    import onnx.utils
    onnx.utils.extract_model(str(a.src), str(a.dst), ["images"], ["blk"])


SYSTEMS = {"ctd": CTD, "ogkalu": OgkaluONNX, "bubble": Ultralytics, "manga109": Ultralytics, "ppocr": PPOCRDet}


def cmd_predict(a):
    det = SYSTEMS[a.system](a.model)
    images = [l.strip() for l in open(a.images) if l.strip()]
    done = {json.loads(l)["f"] for l in open(a.out)} if a.out.exists() else set()
    start = time.time()
    with open(a.out, "a") as fh:
        for p in images:
            label = Path(p).with_suffix(".txt").name
            if label in done:
                continue
            if a.budget and time.time() - start > a.budget:
                break
            rgb = read_rgb(p)
            t0 = time.time()
            xyxy, cls, conf, (w, h) = det(rgb)
            ms = (time.time() - t0) * 1000
            xyxyn = [[min(1, max(0, b[0] / w)), min(1, max(0, b[1] / h)), min(1, max(0, b[2] / w)), min(1, max(0, b[3] / h))]
                     for b in np.asarray(xyxy, float).reshape(-1, 4)]
            fh.write(json.dumps({"f": label, "wh": [w, h], "xyxyn": xyxyn, "cls": [int(c) for c in cls],
                                 "conf": [round(float(c), 5) for c in conf], "ms": round(ms, 1)}) + "\n")
            fh.flush()


# ---------------------------------------------------------------- scoring
def load_labels(labels_dir: Path, f: str):
    p = labels_dir / f
    rows = [l.split() for l in open(p)] if p.exists() else []
    rows = [r for r in rows if len(r) >= 5]
    cls = [CLASS_NAMES[int(r[0])] for r in rows]
    cxcywh = np.array([[float(v) for v in r[1:5]] for r in rows]).reshape(-1, 4)
    return cls, cxcywh


def coverage(preds: list[dict], labels_dir: Path, keep_cls: set[int] | None, conf: float):
    found = {g: 0 for g in GROUPS}
    total = {g: 0 for g in GROUPS}
    n_pred = n_pred_in = 0
    for p in preds:
        cls, gt = load_labels(labels_dir, p["f"])
        pb = np.array([b for b, c, s in zip(p["xyxyn"], p["cls"], p["conf"])
                       if s >= conf and (keep_cls is None or c in keep_cls)]).reshape(-1, 4)
        for name, (cx, cy, bw, bh) in zip(cls, gt):
            hit = bool(len(pb)) and bool(((pb[:, 0] <= cx) & (cx <= pb[:, 2]) & (pb[:, 1] <= cy) & (cy <= pb[:, 3])).any())
            for g, test in GROUPS.items():
                if test(name):
                    total[g] += 1
                    found[g] += hit
        if len(pb):
            pcx, pcy = (pb[:, 0] + pb[:, 2]) / 2, (pb[:, 1] + pb[:, 3]) / 2
            if len(gt):
                x1, y1 = gt[:, 0] - gt[:, 2] / 2, gt[:, 1] - gt[:, 3] / 2
                x2, y2 = gt[:, 0] + gt[:, 2] / 2, gt[:, 1] + gt[:, 3] / 2
                inside = ((x1[None] <= pcx[:, None]) & (pcx[:, None] <= x2[None]) &
                          (y1[None] <= pcy[:, None]) & (pcy[:, None] <= y2[None])).any(1)
                n_pred_in += int(inside.sum())
            n_pred += len(pb)
    res = {g: {"found": found[g], "n": total[g], "coverage": round(found[g] / total[g], 3) if total[g] else None}
           for g in GROUPS}
    res["boxes"] = n_pred
    res["precision"] = round(n_pred_in / n_pred, 3) if n_pred else None
    res["boxes_per_page"] = round(n_pred / max(1, len(preds)), 2)
    ms = [p.get("ms") for p in preds if p.get("ms") is not None]
    res["median_ms"] = round(float(np.median(ms)), 1) if ms else None
    return res


def cmd_table(a):
    out = {"_about": "coverage = labelled blocks whose centre lies in a predicted box at the given confidence; "
                     "precision = predicted boxes whose centre lies in a labelled block", "pages": None, "systems": {}}
    for spec in a.run:
        name, rest = spec.split("=", 1)
        path, cl, cf = (rest.split(":") + ["", ""])[:3]
        preds = [json.loads(l) for l in open(path)]
        if a.only:
            keep_f = {l.strip() for l in open(a.only) if l.strip()}
            preds = [p for p in preds if p["f"] in keep_f]
        out["pages"] = len(preds)
        keep = {int(c) for c in cl.split(",")} if cl else None
        out["systems"][name] = coverage(preds, a.labels_dir, keep, float(cf or 0.25))
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True, help="text file, one image path per line")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--budget", type=float, default=0, help="stop after this many seconds; rerun to resume")
    t = sub.add_parser("table")
    t.add_argument("--labels-dir", type=Path, required=True)
    t.add_argument("--run", action="append", required=True, help="name=cache.jsonl:class_ids:conf")
    t.add_argument("--only", type=Path, help="optional list of label file names to restrict to")
    t.add_argument("--out", type=Path)
    b = sub.add_parser("blkonly")
    b.add_argument("src", type=Path)
    b.add_argument("dst", type=Path)
    a = ap.parse_args()
    {"predict": cmd_predict, "table": cmd_table, "blkonly": cmd_blkonly}[a.cmd](a)


if __name__ == "__main__":
    main()
