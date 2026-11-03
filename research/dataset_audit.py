"""Audit one or more YOLO-format Roboflow exports of the bubble dataset.

Reports, per dataset: files per split, unique source images (augmentation
copies merged), empty pages, boxes per class, images containing each class,
median box area, objects and classes per image, double-labelled objects, and
leakage between splits. With several datasets it also reports how many
sources each pair shares (for example v7 test inside the 10k train split).

Read-only: nothing is written next to the datasets.

Usage:
    python -m research.dataset_audit DATASET_DIR [DATASET_DIR ...] --out reports/audit
"""
from __future__ import annotations

import argparse
import collections
import statistics
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

from research.common import (CLASS_NAMES, DISPLAY_NAMES, available_splits, iter_split,
                             save_json, unique_sources)


def iou_xywh(a, b) -> float:
    ax0, ay0, ax1, ay1 = a.xc - a.w / 2, a.yc - a.h / 2, a.xc + a.w / 2, a.yc + a.h / 2
    bx0, by0, bx1, by1 = b.xc - b.w / 2, b.yc - b.h / 2, b.xc + b.w / 2, b.yc + b.h / 2
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def audit_dataset(root: Path, double_iou: float = 0.9) -> Dict:
    report: Dict = {"dataset": str(root), "splits": {}}
    all_keys: Dict[str, set] = {}
    total_sources: Dict[Tuple, object] = {}
    for split in available_splits(root):
        files = list(iter_split(root, split))
        groups, empty = unique_sources(files)
        all_keys[split] = set(groups)
        boxes = [b for copies in groups.values() for b in copies[0].boxes]
        per_cls = collections.Counter(b.cls for b in boxes)
        polygons = sum(1 for b in boxes if b.polygon_points)
        report["splits"][split] = {
            "files": len(files),
            "empty_files": len(empty),
            "unique_sources": len(groups),
            "copies_per_source": dict(collections.Counter(len(v) for v in groups.values())),
            "boxes": len(boxes),
            "stored_as_polygon": polygons,
            "boxes_per_class": {CLASS_NAMES[k]: per_cls.get(k, 0) for k in range(len(CLASS_NAMES))},
        }
        for k, copies in groups.items():
            total_sources.setdefault(k, copies[0])

    # whole-dataset statistics over unique sources
    srcs = list(total_sources.values())
    per_cls = collections.Counter(b.cls for s in srcs for b in s.boxes)
    imgs_with = collections.Counter(c for s in srcs for c in {b.cls for b in s.boxes})
    areas = collections.defaultdict(list)
    for s in srcs:
        for b in s.boxes:
            areas[b.cls].append(b.w * b.h)
    doubles = []
    for s in srcs:
        for a, b in combinations(s.boxes, 2):
            if a.cls != b.cls and iou_xywh(a, b) >= double_iou:
                doubles.append((CLASS_NAMES[min(a.cls, b.cls)], CLASS_NAMES[max(a.cls, b.cls)]))
    report["overall"] = {
        "unique_sources_with_boxes": len(srcs),
        "boxes": sum(per_cls.values()),
        "per_class": [
            {
                "class": CLASS_NAMES[k],
                "display": DISPLAY_NAMES[CLASS_NAMES[k]],
                "boxes": per_cls.get(k, 0),
                "share": per_cls.get(k, 0) / max(1, sum(per_cls.values())),
                "images_containing": imgs_with.get(k, 0),
                "median_area_frac": statistics.median(areas[k]) if areas[k] else None,
            }
            for k in range(len(CLASS_NAMES))
        ],
        "objects_per_image": dict(sorted(collections.Counter(len(s.boxes) for s in srcs).items())),
        "classes_per_image": dict(sorted(collections.Counter(len({b.cls for b in s.boxes}) for s in srcs).items())),
        "double_labelled_objects": len(doubles),
        "double_labelled_pairs": collections.Counter(f"{a}+{b}" for a, b in doubles).most_common(),
    }
    report["leakage_between_splits"] = {
        f"{a}&{b}": len(all_keys[a] & all_keys[b]) for a, b in combinations(sorted(all_keys), 2)
    }
    report["_keys"] = {s: list(map(list, k)) for s, k in all_keys.items()}  # used for cross-dataset checks
    return report


def cross_dataset(reports: List[Dict]) -> List[Dict]:
    out = []
    keysets = []
    for r in reports:
        ks = {}
        for split, keys in r["_keys"].items():
            ks[split] = {tuple(tuple(x) for x in k) for k in keys}
        keysets.append(ks)
    for (i, a), (j, b) in combinations(enumerate(keysets), 2):
        for sa, ka in a.items():
            for sb, kb in b.items():
                n = len(ka & kb)
                if n:
                    out.append({"a": reports[i]["dataset"], "a_split": sa,
                                "b": reports[j]["dataset"], "b_split": sb, "shared_sources": n})
    return out


def to_markdown(reports: List[Dict], cross: List[Dict]) -> str:
    lines = ["# Dataset audit", ""]
    for r in reports:
        lines += [f"## {r['dataset']}", "", "| Split | Files | Empty | Unique sources | Boxes |", "| --- | --- | --- | --- | --- |"]
        for s, v in r["splits"].items():
            lines.append(f"| {s} | {v['files']} | {v['empty_files']} | {v['unique_sources']} | {v['boxes']} |")
        lines += ["", "| Class | Boxes | Share | Images containing | Median area |", "| --- | --- | --- | --- | --- |"]
        for c in sorted(r["overall"]["per_class"], key=lambda x: -x["boxes"]):
            area = f"{100 * c['median_area_frac']:.1f}%" if c["median_area_frac"] is not None else ""
            lines.append(f"| {c['display']} | {c['boxes']} | {100 * c['share']:.1f}% | {c['images_containing']} | {area} |")
        lines += ["", f"Double-labelled objects (IoU >= 0.9, two classes): {r['overall']['double_labelled_objects']}",
                  f"Leakage between splits (shared sources): {r['leakage_between_splits']}", ""]
    if cross:
        lines += ["## Sources shared across datasets", "", "| A | split | B | split | shared |", "| --- | --- | --- | --- | --- |"]
        for c in cross:
            lines.append(f"| {Path(c['a']).name} | {c['a_split']} | {Path(c['b']).name} | {c['b_split']} | {c['shared_sources']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/audit"))
    ap.add_argument("--double-iou", type=float, default=0.9)
    args = ap.parse_args()
    reports = [audit_dataset(d, args.double_iou) for d in args.datasets]
    cross = cross_dataset(reports)
    for r in reports:
        r.pop("_keys", None)
    save_json({"datasets": reports, "cross": cross}, args.out / "audit.json")
    (args.out / "audit.md").write_text(to_markdown(reports, cross), encoding="utf-8")
    print(to_markdown(reports, cross))


if __name__ == "__main__":
    main()
