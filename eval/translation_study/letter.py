"""Letter an English line into its block on the page, for judging a translation the way a reader meets it.

The Korean is painted over with the block's own background colour (the commonest colour inside the box), in an
ellipse for balloons and a rectangle for boxes, windows, signs and titles, and the English is set centred in the
largest font that fits. Sound effects are drawn over the art without painting it out, in bold with an outline, as
English editions often do. The result is a crop of the page around the block, with room for the scene.

Fonts: CC Wild Words (the usual font of English manga and webtoon lettering) when installed, else a fallback.
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIRS = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts", Path("/usr/share/fonts/truetype"), Path.home() / ".fonts"]
FONTS = {
    "text": ["CCWILDWORDSLOWER-REGULAR.TTF", "CCWildWordsLower-Regular.ttf", "COMIC_0.TTF", "comic.ttf", "DejaVuSans.ttf"],
    "sfx": ["CCWILDWORDS-BOLD.TTF", "CCWildWords-Bold.ttf", "COMICBD_0.TTF", "comicbd.ttf", "DejaVuSans-Bold.ttf"],
}
ROUND = {"speech_bubble", "think_bubble", "cloud_bubble", "wiggle_bubble", "scream_bubble", "spiked_bubble"}


def font_path(kind: str) -> str | None:
    for name in FONTS[kind]:
        for d in FONT_DIRS:
            for p in (d / name, *d.glob(f"**/{name}")) if d.exists() else ():
                if p.exists():
                    return str(p)
    return None


_CACHE: dict = {}


def font(kind: str, size: int):
    key = (kind, size)
    if key not in _CACHE:
        p = font_path(kind)
        _CACHE[key] = ImageFont.truetype(p, size) if p else ImageFont.load_default()
    return _CACHE[key]


def wrap(draw, text: str, f, width: float) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        t = f"{cur} {w}".strip()
        if draw.textlength(t, font=f) <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def fit(draw, text: str, kind: str, w: float, h: float):
    """Largest font size whose wrapped text fits in w x h (no word wider than w)."""
    best = None
    for size in range(8, 96):
        f = font(kind, size)
        lines = wrap(draw, text, f, w)
        lh = size * 1.12
        if len(lines) * lh > h or any(draw.textlength(l, font=f) > w for l in lines):
            break
        best = (f, lines, lh)
    return best or (font(kind, 8), wrap(draw, text, font(kind, 8), w), 9)


def letter(image_path: Path, box, text: str, class_name: str, max_side: int = 640, margin: float = 0.6) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    x0, y0, x1, y1 = (int(v) for v in box)
    bw, bh = x1 - x0, y1 - y0
    draw = ImageDraw.Draw(img)
    sfx = class_name == "sfx_sound"
    if not sfx:
        region = np.asarray(img.crop((x0, y0, x1, y1))).reshape(-1, 3) // 16 * 16 + 8
        vals, counts = np.unique(region, axis=0, return_counts=True)
        fill = tuple(int(v) for v in vals[counts.argmax()])
        ink = (0, 0, 0) if sum(fill) > 380 else (255, 255, 255)
        if class_name in ROUND:
            ix, iy = bw * 0.06, bh * 0.06
            draw.ellipse((x0 + ix, y0 + iy, x1 - ix, y1 - iy), fill=fill)
            tw, th = bw * 0.70, bh * 0.66  # the largest box inside the ellipse, less a little air
        else:
            ix, iy = bw * 0.04, bh * 0.06
            draw.rectangle((x0 + ix, y0 + iy, x1 - ix, y1 - iy), fill=fill)
            tw, th = bw * 0.86, bh * 0.80
        f, lines, lh = fit(draw, text, "text", tw, th)
        stroke = 0
    else:
        ink, tw, th = (255, 255, 255), bw * 1.1, bh * 0.95
        f, lines, lh = fit(draw, text.upper(), "sfx", tw, th)
        stroke = max(2, int(lh / 9))
    cy = (y0 + y1) / 2 - len(lines) * lh / 2
    for i, line in enumerate(lines):
        w = draw.textlength(line, font=f)
        draw.text(((x0 + x1) / 2 - w / 2, cy + i * lh), line, font=f, fill=ink, stroke_width=stroke, stroke_fill=(0, 0, 0))
    W, H = img.size
    mx, my = bw * margin + 40, bh * margin + 40
    crop = img.crop((max(0, int(x0 - mx)), max(0, int(y0 - my)), min(W, int(x1 + mx)), min(H, int(y1 + my))))
    sc = min(1.0, max_side / max(crop.size))
    if sc < 1:
        crop = crop.resize((round(crop.width * sc), round(crop.height * sc)), Image.LANCZOS)
    return crop


def jpeg_b64(img: Image.Image, quality: int = 82) -> str:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()
