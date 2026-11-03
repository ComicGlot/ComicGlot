"""Shared helpers for the ComicGlot research scripts.

Everything here is read-only with respect to datasets: functions parse label
files, read image headers and group Roboflow augmentation copies, but never
write into a dataset folder.
"""
from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

# Class order used by every Roboflow export of the project (v2 to v9).
# "sqare_bubble" is the spelling stored in the datasets; keep it for indexing
# and use DISPLAY_NAMES for anything a reader sees.
CLASS_NAMES: List[str] = [
    "background_text", "cloud_bubble", "hexagon_bubble", "scream_bubble",
    "sfx_sound", "speech_bubble", "spiked_bubble", "sqare_bubble",
    "standalone_text", "system_bubble", "think_bubble", "title_panel",
    "wiggle_bubble",
]
DISPLAY_NAMES: Dict[str, str] = {n: n for n in CLASS_NAMES}
DISPLAY_NAMES["sqare_bubble"] = "square_bubble"

# Type groups found by hierarchical clustering of the per-type average image features.
FAMILIES: Dict[str, str] = {
    "speech_bubble": "outlined_balloon", "think_bubble": "outlined_balloon",
    "cloud_bubble": "outlined_balloon", "wiggle_bubble": "outlined_balloon",
    "scream_bubble": "outlined_balloon", "hexagon_bubble": "outlined_balloon",
    "spiked_bubble": "outlined_balloon",
    "sqare_bubble": "box_or_free_text", "standalone_text": "box_or_free_text",
    "background_text": "box_or_free_text", "title_panel": "box_or_free_text",
    "system_bubble": "box_or_free_text",
    "sfx_sound": "sound_effect",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@dataclass
class Box:
    cls: int
    xc: float
    yc: float
    w: float
    h: float
    polygon_points: int = 0  # 0 = stored as a plain YOLO box

    @property
    def name(self) -> str:
        return CLASS_NAMES[self.cls] if 0 <= self.cls < len(CLASS_NAMES) else f"class_{self.cls}"

    def xyxy(self, width: int, height: int) -> Tuple[float, float, float, float]:
        return ((self.xc - self.w / 2) * width, (self.yc - self.h / 2) * height,
                (self.xc + self.w / 2) * width, (self.yc + self.h / 2) * height)


@dataclass
class LabelFile:
    split: str
    label_path: Path
    image_path: Optional[Path]
    boxes: List[Box] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return self.label_path.stem

    @property
    def roboflow_base(self) -> str:
        """Original filename before Roboflow's '.rf.<hash>' suffix."""
        return self.label_path.name.split(".rf.")[0]

    def box_key(self, ndigits: int = 3) -> Tuple:
        """Key shared by all augmentation copies of one source image.

        Roboflow's photometric augmentations keep geometry, so the three copies
        of a source carry the same boxes; rounding absorbs the polygon/box
        serialisation differences between copies.
        """
        return tuple(sorted((b.cls, round(b.xc, ndigits), round(b.yc, ndigits),
                             round(b.w, ndigits), round(b.h, ndigits)) for b in self.boxes))


def parse_label_line(line: str) -> Optional[Box]:
    parts = line.split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    vals = [float(v) for v in parts[1:]]
    if len(vals) == 4:
        return Box(cls, *vals)
    xs, ys = vals[0::2], vals[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return Box(cls, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, polygon_points=len(xs))


def read_label_file(path: Path) -> List[Box]:
    boxes = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            b = parse_label_line(line)
            if b is not None:
                boxes.append(b)
    return boxes


def find_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def iter_split(dataset_dir: Path, split: str) -> Iterator[LabelFile]:
    labels_dir = dataset_dir / split / "labels"
    images_dir = dataset_dir / split / "images"
    if not labels_dir.is_dir():
        return
    for name in sorted(os.listdir(labels_dir)):
        if not name.endswith(".txt"):
            continue
        lp = labels_dir / name
        yield LabelFile(split, lp, find_image(images_dir, lp.stem), read_label_file(lp))


def available_splits(dataset_dir: Path) -> List[str]:
    return [s for s in ("train", "valid", "val", "test") if (dataset_dir / s / "labels").is_dir()]


def unique_sources(files: Iterable[LabelFile]) -> Tuple[Dict[Tuple, List[LabelFile]], List[LabelFile]]:
    """Group augmentation copies. Returns ({box_key: copies}, empty_label_files)."""
    groups: Dict[Tuple, List[LabelFile]] = {}
    empty: List[LabelFile] = []
    for lf in files:
        if not lf.boxes:
            empty.append(lf)
            continue
        groups.setdefault(lf.box_key(), []).append(lf)
    return groups, empty


def image_size(path: Path, max_bytes: int = 65536) -> Tuple[Optional[int], Optional[int]]:
    """Width and height from a JPEG or PNG header without decoding the image."""
    with open(path, "rb") as fh:
        head = fh.read(max_bytes)
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", head[16:24])
        return w, h
    i = 2
    while i < len(head) - 9:
        if head[i] != 0xFF:
            i += 1
            continue
        marker = head[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack(">HH", head[i + 5:i + 9])
            return w, h
        seg_len = struct.unpack(">H", head[i + 2:i + 4])[0]
        i += 2 + seg_len
    return None, None


FILENAME_PATTERNS = [
    # autogen export: img_007401_png -> sequential index, no chapter information
    ("autogen", re.compile(r"^img_(\d+)_(png|jpe?g|webp)$")),
    # named chapter folders flattened by Roboflow: <chapter-id>_<page>_png
    ("named", re.compile(r"^(.+)_(\d+)_(png|jpe?g|webp)$")),
    # bare page numbers from the first datasets: 01_png
    ("bare", re.compile(r"^(\d+)_(png|jpe?g|webp)$")),
]


def filename_group(roboflow_base: str) -> Tuple[str, Optional[str], Optional[int]]:
    """(kind, chapter_key, page) parsed from a Roboflow base name."""
    for kind, pat in FILENAME_PATTERNS:
        m = pat.match(roboflow_base)
        if not m:
            continue
        if kind == "autogen":
            return kind, None, int(m.group(1))
        if kind == "named":
            return kind, m.group(1), int(m.group(2))
        return kind, None, int(m.group(1))
    return "unknown", None, None


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
