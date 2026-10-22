"""Thumbnail: the episode's key image, one bright focal creature, with 2-4 large bold words in the channel serif,
then three gates.

- Legibility: the thumbnail is shrunk to phone-feed size (320x180) and Tesseract OCR must read every word back.
- Contrast: the text needs at least 4.5:1 contrast with the image behind it.
- Brightness, a floor with its ceiling: the finished thumbnail's mean luma (Rec. 601, 0-255) must sit between
  LUMA_FLOOR and LUMA_CEIL. A dark sleep frame vanishes in the feed; a blown-out one has no creature left.
  It is measured on the composed picture at feed size, so it reads what a viewer sees, not the encoder.

The key image is first eased toward a bright mid-tone (brighten), then failing layouts are retried with other
shades and sizes. Because brighten can lift any frame over the luma floor, two more gates read the art itself:

- Focal light, on the key image before brighten: its brightest 5% must reach FOCAL_MIN (a bright creature or
  light exists); a uniformly dark key is rejected however far it could be lifted (cli.thumb_source then tries the
  next image).
- Tonal spread, a floor with its ceiling, on the finished thumbnail: p90-p10 luma within SPREAD, so a gamma lift
  that flattens the shadows into grey fails, as does a picture that is all clipped black and white.
"""
from __future__ import annotations

import difflib
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from . import plates, render

W, H = 1280, 720
MAX_WORDS = 4
LUMA_FLOOR, LUMA_CEIL = 70.0, 200.0
LUMA_TARGET = (115.0, 185.0)  # brighten() eases the key image into this band before the words go on
CONTRAST_MIN = 4.5
FOCAL_MIN = 100.0  # 95th-percentile luma of the key image before brighten
SPREAD = (40.0, 230.0)  # p90 - p10 luma of the finished thumbnail
WEIGHT = 700
WIDTH = 0.58  # the words may span this share of the width
STOP = {"the", "of", "a", "an", "and", "in", "on", "to", "for", "at", "with"}


def _lum(rgb: np.ndarray) -> np.ndarray:
    c = rgb / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def contrast(text_rgb: tuple, background: np.ndarray) -> float:
    lt = float(_lum(np.array(text_rgb, np.float32)[None, None])[0, 0])
    lb = float(np.percentile(_lum(background.astype(np.float32)), 90))
    return (max(lt, lb) + 0.05) / (min(lt, lb) + 0.05)


def mean_luma(img: Image.Image) -> float:
    """Mean Rec. 601 luma (0-255) at phone-feed size."""
    a = np.asarray(img.convert("RGB").resize((320, 180), Image.BILINEAR), np.float32)
    return float((0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).mean())


def luma_problems(value: float) -> list[str]:
    if value < LUMA_FLOOR:
        return [f"thumbnail too dark (mean luma {value:.0f} < {LUMA_FLOOR:.0f})"]
    if value > LUMA_CEIL:
        return [f"thumbnail too bright (mean luma {value:.0f} > {LUMA_CEIL:.0f})"]
    return []


def _luma_feed(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGB").resize((320, 180), Image.BILINEAR), np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def source_problems(img: Image.Image) -> list[str]:
    """The key image as generated, before any lift: is there a bright focal subject at all?"""
    focal = float(np.percentile(_luma_feed(img), 95))
    return [] if focal >= FOCAL_MIN else [f"key image has no bright focal subject (95th-percentile luma {focal:.0f} "
                                          f"< {FOCAL_MIN:.0f}): a dark key lifted by brighten() is still flat"]


def spread(img: Image.Image) -> float:
    lo, hi = np.percentile(_luma_feed(img), (10, 90))
    return float(hi - lo)


def spread_problems(value: float) -> list[str]:
    if value < SPREAD[0]:
        return [f"thumbnail is flat (p90-p10 luma {value:.0f} < {SPREAD[0]:.0f})"]
    if value > SPREAD[1]:
        return [f"thumbnail is harsh (p90-p10 luma {value:.0f} > {SPREAD[1]:.0f})"]
    return []


def brighten(base: Image.Image) -> Image.Image:
    """Lift a dark key image with a gamma curve (capped, so shadows do not turn to grey mud) or scale down a
    blown-out one, aiming at the LUMA_TARGET band."""
    lo, hi = LUMA_TARGET
    m = mean_luma(base)
    a = np.asarray(base.convert("RGB"), np.float32) / 255.0
    if m < lo:
        g_lo, g_hi = 0.45, 1.0  # gamma below 1 lifts; 0.45 is the cap
        for _ in range(12):
            g = (g_lo + g_hi) / 2
            if mean_luma(Image.fromarray((a ** g * 255).astype(np.uint8))) < lo:
                g_hi = g
            else:
                g_lo = g
        a = a ** g_lo
    elif m > hi:
        a = a * (hi / m)
    else:
        return base.convert("RGB")
    return Image.fromarray(np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8))


def fit_words(words: list[str]) -> list[str]:
    """At most MAX_WORDS large words: drop small joining words first, then cut the end."""
    words = [w for w in words if w.strip()]
    while len(words) > MAX_WORDS and any(w.lower() in STOP for w in words[1:]):
        k = max(i for i, w in enumerate(words) if i > 0 and w.lower() in STOP)
        words = words[:k] + words[k + 1:]
    return words[:MAX_WORDS]


def ocr_words(img: Image.Image, psm: int = 6) -> list[str]:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.png"
        img.save(p)
        out = subprocess.run(["tesseract", str(p), "stdout", "--psm", str(psm)], capture_output=True,
                             text=True).stdout
    return re.findall(r"[a-z0-9']+", out.lower())


def legible(img: Image.Image, words: list[str], box: tuple | None = None) -> tuple[bool, list[str]]:
    """Every word must read back at phone-feed size (320x180). With the text's box, only that region is read:
    Tesseract's block mode gives up on the whole frame when the picture beside the words is busy."""
    small = img.resize((320, 180), Image.LANCZOS)
    if box:
        sx, sy, m = 320 / img.width, 180 / img.height, 6
        region = small.crop((max(0, int(box[0] * sx) - m), max(0, int(box[1] * sy) - m),
                             min(320, int(box[2] * sx) + m), min(180, int(box[3] * sy) + m)))
        seen = []
        for k in (3, 4, 5):
            up = ImageOps.autocontrast(region.resize((region.width * k, region.height * k), Image.LANCZOS).convert("L"))
            up = ImageOps.expand(up, border=32, fill=up.getpixel((0, 0)))
            for view in (up, ImageOps.invert(up)):
                seen += ocr_words(view, 6) + ocr_words(view, 11)
    else:
        seen = ocr_words(small.resize((640, 360), Image.LANCZOS))
    missing = [w for w in (re.sub(r"[^a-z0-9']", "", x.lower()) for x in words)
               if w and not any(difflib.SequenceMatcher(None, w, s).ratio() >= 0.8 for s in seen)]
    return not missing, seen


def _shaded(base: Image.Image, shade: float) -> Image.Image:
    """Darken the left side, where the words go."""
    grad = np.clip(1 - np.linspace(0, 1.25, W), 0, 1) ** 1.3 * shade
    return Image.fromarray((np.asarray(base, np.float32) * (1 - grad[None, :, None])).astype(np.uint8))


def _font(size: int):
    return render.font(size, weight=WEIGHT)


def _layout(base: Image.Image, words: list[str], size: int, shade: float,
            panel: bool = False) -> tuple[Image.Image, tuple, Image.Image]:
    """The thumbnail, the text's bounding box, and the background behind the text (for the contrast test)."""
    img = _shaded(base, shade)
    lines = [" ".join(words[:2]), " ".join(words[2:])] if len(words) > 2 else [" ".join(words)]
    lines = [ln.upper() for ln in lines if ln]
    f = _font(size)
    while max(f.getlength(ln) for ln in lines) > W * WIDTH and size > 72:
        size -= 6
        f = _font(size)
    asc, desc = f.getmetrics()
    block_h = len(lines) * (asc + desc) - desc // 2
    y0 = (H - block_h) // 2
    if panel:
        width = int(max(f.getlength(ln) for ln in lines))
        backing = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(backing).rectangle((40, y0 - 30, 100 + width, y0 + block_h + 30), fill=(8, 10, 14, 225))
        img = img.convert("RGBA")
        img.alpha_composite(backing)
        img = img.convert("RGB")
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for i, ln in enumerate(lines):
        d.text((60, y0 + i * (asc + desc)), ln, font=f, fill=render.INK + (255,), stroke_width=max(2, size // 40),
               stroke_fill=render.INK + (255,))
    halo = layer.split()[3].filter(ImageFilter.GaussianBlur(16)).point(lambda v: min(255, int(v * 1.7)))
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow.putalpha(halo)
    out = img.convert("RGBA")
    out.alpha_composite(shadow)
    out.alpha_composite(layer)
    box = layer.getbbox() or (0, 0, W // 2, H)
    return out.convert("RGB"), box, img


ATTEMPTS = ((190, 0.45, False), (190, 0.65, False), (170, 0.8, False), (150, 0.9, False),
            (190, 0.35, True), (170, 0.35, True))


def build(image_png: Path, words: list[str], out: Path, upscale: bool = True) -> dict:
    words = fit_words(words)
    if not 2 <= len(words) <= MAX_WORDS:
        raise ValueError(f"thumbnail needs 2-{MAX_WORDS} words, got {len(words)}")
    src = Image.open(image_png).convert("RGB")
    key_problems = source_problems(src)
    base = brighten(plates.upscale(src, (W, H)) if upscale else plates.cover(src, (W, H)))
    attempts, best = [], None
    for size, shade, panel in ATTEMPTS:
        img, box, plain = _layout(base, words, size, shade, panel)
        behind = np.asarray(plain)[box[1]:box[3], box[0]:box[2]]
        ratio = contrast(render.INK, behind)
        luma, tone = mean_luma(img), spread(img)
        ok, seen = legible(img, words, box)
        problems = ([] if ok else ["OCR could not read every word"]) + \
                   ([] if ratio >= CONTRAST_MIN else [f"text contrast {ratio:.1f} < {CONTRAST_MIN}"]) + \
                   luma_problems(luma) + spread_problems(tone) + key_problems
        attempts.append({"size": size, "shade": shade, "panel": panel, "contrast": round(ratio, 2),
                         "luma": round(luma, 1), "spread": round(tone, 1), "ocr": seen, "ok": not problems,
                         "problems": problems})
        if best is None or len(problems) < len(best[1]):
            best = (img, problems, luma)
        if not problems:
            break
    img, problems, luma = best
    img.save(out, quality=92)
    return {"path": str(out), "words": words, "attempts": attempts, "ok": not problems, "luma": round(luma, 1),
            "problems": problems}
