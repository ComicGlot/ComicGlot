"""General OCR systems on the same blocks as the pipeline's readers, for the recognition table.

Crops every block of --blocks (block_uid column; blocks.csv of eval/sample_frames.py) from its frame,
with a few pixels of margin, and reads it with each system given:

  easyocr    EasyOCR, Korean and English, line detection inside the crop, lines joined top to bottom
  paddleocr  PaddleOCR (Korean), the same way, when the package is installed

Writes a CSV with block_uid and one column per system: the --readings file of eval/ocr_eval.py.

    python -m eval.ocr_baselines --runs data/output/<runs>/*/chapter_full_results.json \
        --blocks data/study/sample/blocks.csv --systems easyocr --out data/study/ocr_easyocr.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Callable, Dict, List

import cv2
import numpy as np

from eval.sample_frames import local_path


def crop(image: np.ndarray, box, pad: int = 4) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    return image[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]


def join_lines(items: List[tuple]) -> str:
    """items: (y_centre, x_centre, text) per detected line; reading order top to bottom, then left to right."""
    return " ".join(t for _, _, t in sorted(items, key=lambda r: (round(r[0] / 12), r[1])) if t).strip()


def easyocr_reader(gpu: bool) -> Callable[[np.ndarray], str]:
    import easyocr
    reader = easyocr.Reader(["ko", "en"], gpu=gpu, verbose=False)

    def read(img: np.ndarray) -> str:
        out = reader.readtext(img, detail=1, paragraph=False)
        return join_lines([((min(p[1] for p in b) + max(p[1] for p in b)) / 2, (min(p[0] for p in b) + max(p[0] for p in b)) / 2, t)
                           for b, t, _ in out])
    return read


def paddle_reader(gpu: bool) -> Callable[[np.ndarray], str]:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="korean", use_angle_cls=False, show_log=False, use_gpu=gpu)

    def read(img: np.ndarray) -> str:
        res = ocr.ocr(img, cls=False) or []
        lines = res[0] or [] if res else []
        return join_lines([((min(p[1] for p in b) + max(p[1] for p in b)) / 2, (min(p[0] for p in b) + max(p[0] for p in b)) / 2, t[0])
                           for b, t in lines])
    return read


SYSTEMS = {"easyocr": easyocr_reader, "paddleocr": paddle_reader}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--blocks", type=Path, required=True, help="CSV with a block_uid column")
    ap.add_argument("--systems", nargs="+", default=["easyocr"], choices=sorted(SYSTEMS))
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    want = [r["block_uid"] for r in csv.DictReader(open(a.blocks, encoding="utf-8-sig"))]
    where: Dict[str, tuple] = {}
    for f in a.runs:
        run = f.parent.name
        for fr in json.loads(f.read_text(encoding="utf-8")).get("frames", []):
            for b in fr.get("blocks", []):
                where[f"{run}/{b.get('frame_id', fr.get('id'))}/{b.get('id')}"] = (fr["image_path"], b["box"])
    missing = [u for u in want if u not in where]
    if missing:
        raise SystemExit(f"{len(missing)} blocks are not in --runs, e.g. {missing[:3]}")

    out: Dict[str, Dict[str, str]] = {u: {} for u in want}
    images: Dict[str, np.ndarray] = {}
    for name in a.systems:
        read = SYSTEMS[name](gpu=not a.cpu)
        t0 = time.time()
        for u in want:
            path, box = where[u]
            if path not in images:
                images[path] = cv2.imread(str(local_path(path, a.project_root)))
            try:
                out[u][name] = read(crop(images[path], box))
            except Exception as e:  # one unreadable crop must not stop the run
                out[u][name] = ""
                print(f"{name} failed on {u}: {e}")
        print(f"{name}: {len(want)} blocks in {time.time() - t0:.0f} s")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["block_uid", *a.systems])
        for u in want:
            w.writerow([u, *(out[u].get(s, "") for s in a.systems)])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
