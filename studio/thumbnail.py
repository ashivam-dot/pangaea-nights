"""Thumbnail: the episode's key image with 2-5 words set in the channel serif, then a legibility test.

The test shrinks the thumbnail to phone-feed size (320x180) and requires Tesseract OCR to read every word back,
and the text to have at least 4.5:1 contrast with the image behind it. Failing layouts are retried darker/larger.
"""
from __future__ import annotations

import difflib
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import plates, render

W, H = 1280, 720


def _lum(rgb: np.ndarray) -> np.ndarray:
    c = rgb / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def contrast(text_rgb: tuple, background: np.ndarray) -> float:
    lt = float(_lum(np.array(text_rgb, np.float32)[None, None])[0, 0])
    lb = float(np.percentile(_lum(background.astype(np.float32)), 90))
    return (max(lt, lb) + 0.05) / (min(lt, lb) + 0.05)


def ocr_words(img: Image.Image) -> list[str]:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.png"
        img.save(p)
        out = subprocess.run(["tesseract", str(p), "stdout", "--psm", "6"], capture_output=True, text=True).stdout
    return re.findall(r"[a-z0-9']+", out.lower())


def legible(img: Image.Image, words: list[str]) -> tuple[bool, list[str]]:
    small = img.resize((320, 180), Image.LANCZOS).resize((640, 360), Image.LANCZOS)
    seen = ocr_words(small)
    missing = [w for w in (re.sub(r"[^a-z0-9']", "", x.lower()) for x in words)
               if w and not any(difflib.SequenceMatcher(None, w, s).ratio() >= 0.8 for s in seen)]
    return not missing, seen


def _shaded(base: Image.Image, shade: float) -> Image.Image:
    """Darken the left side, where the words go."""
    grad = np.clip(1 - np.linspace(0, 1.25, W), 0, 1) ** 1.3 * shade
    return Image.fromarray((np.asarray(base, np.float32) * (1 - grad[None, :, None])).astype(np.uint8))


def _layout(base: Image.Image, words: list[str], size: int, shade: float) -> tuple[Image.Image, tuple]:
    img = _shaded(base, shade)
    lines = [" ".join(words[:2]), " ".join(words[2:])] if len(words) > 2 else [" ".join(words)]
    lines = [ln.upper() for ln in lines if ln]
    f = render.font(size, weight=650)
    while max(f.getlength(ln) for ln in lines) > W * 0.52 and size > 60:
        size -= 6
        f = render.font(size, weight=650)
    asc, desc = f.getmetrics()
    block_h = len(lines) * (asc + desc) - desc // 2
    y0 = (H - block_h) // 2
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for i, ln in enumerate(lines):
        d.text((70, y0 + i * (asc + desc)), ln, font=f, fill=render.INK + (255,))
    halo = layer.split()[3].filter(ImageFilter.GaussianBlur(14)).point(lambda v: min(255, int(v * 1.6)))
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow.putalpha(halo)
    out = img.convert("RGBA")
    out.alpha_composite(shadow)
    out.alpha_composite(layer)
    box = layer.getbbox() or (0, 0, W // 2, H)
    return out.convert("RGB"), box


def build(image_png: Path, words: list[str], out: Path, upscale: bool = True) -> dict:
    if not 2 <= len(words) <= 5:
        raise ValueError(f"thumbnail needs 2-5 words, got {len(words)}")
    src = Image.open(image_png).convert("RGB")
    base = plates.upscale(src, (W, H)) if upscale else plates.cover(src, (W, H))
    attempts = []
    for size, shade in ((150, 0.55), (150, 0.75), (128, 0.85), (112, 0.92)):
        img, box = _layout(base, words, size, shade)
        behind = np.asarray(_shaded(base, shade))[box[1]:box[3], box[0]:box[2]]
        ratio = contrast(render.INK, behind)
        ok, seen = legible(img, words)
        attempts.append({"size": size, "shade": shade, "contrast": round(ratio, 2), "ocr": seen, "ok": ok})
        if ok and ratio >= 4.5:
            img.save(out, quality=92)
            return {"path": str(out), "words": words, "attempts": attempts, "ok": True}
    img.save(out, quality=92)
    return {"path": str(out), "words": words, "attempts": attempts, "ok": False}
