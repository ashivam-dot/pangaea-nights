"""The on-screen graphics of a ladder Short, drawn with PIL in the repo's fonts.

Everything sits inside the safe area: never in the bottom 18% (title, channel name and buttons of the Shorts player)
or the right 12% (the like/comment/share column). The banner is on screen from frame 0, 10-28% of the frame height,
on a dark panel so its text keeps a contrast of at least 4.5:1 over any picture. The look is adult and plain:
serif type, dark glass panels, one amber accent, no cartoon shapes and no mascot.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import render

SAFE = (0.04, 0.03, 0.88, 0.82)  # x0, y0, x1, y1 as fractions of the frame
BANNER_H = (0.10, 0.28)
MIN_CONTRAST = 4.5
INK = render.INK
AMBER = (240, 172, 74)
PANEL = (8, 11, 18)
PANEL_ALPHA = 0.74
METER = [(70, 168, 160), (96, 176, 140), (150, 184, 110), (206, 186, 84), (240, 172, 74), (240, 132, 62),
         (228, 92, 52), (210, 52, 40)]


@dataclass
class Layer:
    rgb: np.ndarray
    alpha: np.ndarray
    x: int
    y: int
    box: tuple[int, int, int, int]  # the drawn (non-transparent) pixels, in frame coordinates

    def draw(self, frame: np.ndarray, opacity: float = 1.0) -> None:
        render.blit(frame, (self.rgb, self.alpha), self.x, self.y, opacity)


def safe_box(w: int, h: int) -> tuple[int, int, int, int]:
    return round(SAFE[0] * w), round(SAFE[1] * h), round(SAFE[2] * w), round(SAFE[3] * h)


def inside(box: tuple, w: int, h: int) -> bool:
    x0, y0, x1, y1 = safe_box(w, h)
    return box[0] >= x0 and box[1] >= y0 and box[2] <= x1 and box[3] <= y1


def _layer(img: Image.Image, x: int, y: int) -> Layer:
    rgb, alpha = render.layer_from(img)
    bb = img.getchannel("A").getbbox() or (0, 0, 0, 0)
    return Layer(rgb, alpha, x, y, (x + bb[0], y + bb[1], x + bb[2], y + bb[3]))


def _font(size: int, weight: int = 600, italic: bool = False) -> ImageFont.FreeTypeFont:
    return render.font(max(8, int(size)), italic, weight)


def _wrap(text: str, f: ImageFont.FreeTypeFont, width: int, max_lines: int) -> list[str] | None:
    for cols in range(max(8, len(text)), 4, -1):
        lines = textwrap.wrap(text, cols)
        if len(lines) > max_lines:
            return None
        if all(f.getlength(ln) <= width for ln in lines):
            return lines
    return None


def _fit(text: str, size: int, width: int, max_lines: int, weight: int = 600, italic: bool = False):
    """The largest font (down from `size`) at which text wraps into max_lines lines of at most width pixels."""
    while size > 10:
        f = _font(size, weight, italic)
        lines = _wrap(text, f, width, max_lines)
        if lines:
            return f, lines
        size -= 2
    f = _font(10, weight, italic)
    return f, textwrap.wrap(text, 12)[:max_lines]


def _panel(w: int, h: int, radius: int, alpha: float = PANEL_ALPHA) -> Image.Image:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle((0, 0, w - 1, h - 1), radius, fill=PANEL + (round(255 * alpha),))
    return img


def _text_lines(img: Image.Image, lines: list[str], f, top: int, fill=INK, cx: int | None = None,
                left: int | None = None, gap: float = 0.18) -> int:
    ink = ImageDraw.Draw(img)
    asc, desc = f.getmetrics()
    y = top
    for ln in lines:
        b = f.getbbox(ln)
        x = left if left is not None else (cx if cx is not None else img.width // 2) - (b[2] - b[0]) // 2 - b[0]
        ink.text((x, y), ln, font=f, fill=fill)
        y += round((asc + desc) * (1 + gap))
    return y


# ---------------------------------------------------------------------------------------------------- layers


def mark(w: int, h: int) -> Layer:
    """A small "PN" monogram, top left: the only branding on screen."""
    f = _font(h * 0.022, 700)
    b = f.getbbox("PN")
    img = Image.new("RGBA", (b[2] + 8, b[3] + 8), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((4 - b[0], 4), "PN", font=f, fill=INK + (200,))
    x0, y0 = safe_box(w, h)[:2]
    return _layer(img, x0, y0)


def banner(text: str, w: int, h: int) -> tuple[Layer, dict]:
    """The question, on a dark panel at the top, from frame 0. Returns the layer and its geometry."""
    x0, y0, x1, _ = safe_box(w, h)
    width = x1 - x0
    pad_x, pad_y = round(w * 0.045), round(h * 0.022)
    f, lines = _fit(text.upper(), round(h * 0.040), width - 2 * pad_x, 2, weight=700)
    asc, desc = f.getmetrics()
    text_h = round((asc + desc) * (1 + 0.18) * len(lines) - (asc + desc) * 0.18)
    ph = min(max(text_h + 2 * pad_y, round(BANNER_H[0] * h) + 2), round(BANNER_H[1] * h) - 2)
    img = _panel(width, ph, round(h * 0.012))
    top = (ph - text_h) // 2
    _text_lines(img, lines, f, top, fill=INK)
    ImageDraw.Draw(img).rectangle((pad_x, ph - max(2, round(h * 0.003)) - 2, width - pad_x, ph - 3),
                                  fill=AMBER + (220,))
    y = y0 + round(h * 0.03)
    layer = _layer(img, x0, y)
    geo = {"box": (x0, y, x0 + width, y + ph), "text_top": y + top, "height_frac": ph / h, "lines": lines}
    return layer, geo


def badge(n: int, w: int, h: int) -> Layer:
    """The option number, large, in a dark disc with an amber ring."""
    d = round(w * 0.17)
    img = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    ink = ImageDraw.Draw(img)
    ring = max(3, round(d * 0.045))
    ink.ellipse((0, 0, d - 1, d - 1), fill=AMBER + (235,))
    ink.ellipse((ring, ring, d - 1 - ring, d - 1 - ring), fill=PANEL + (225,))
    f = _font(d * 0.62, 700)
    b = f.getbbox(str(n))
    ink.text(((d - (b[2] - b[0])) // 2 - b[0], (d - (b[3] - b[1])) // 2 - b[1]), str(n), font=f, fill=INK)
    x0 = safe_box(w, h)[0]
    return _layer(img, x0 + round(w * 0.01), round(h * 0.25))


def label(text: str, w: int, h: int) -> Layer:
    """The option's name, beside the badge."""
    x0, _, x1, _ = safe_box(w, h)
    left = x0 + round(w * 0.01) + round(w * 0.17) + round(w * 0.03)
    width = x1 - left
    f, lines = _fit(text, round(h * 0.040), width - round(w * 0.05), 2, weight=700)
    asc, desc = f.getmetrics()
    text_h = round((asc + desc) * 1.18 * len(lines))
    pad = round(h * 0.012)
    img = _panel(width, text_h + 2 * pad, round(h * 0.010), 0.66)
    _text_lines(img, lines, f, pad, left=round(w * 0.025))
    top = round(h * 0.25) + (round(w * 0.17) - img.height) // 2
    return _layer(img, left, top)


def meter(level: float, w: int, h: int) -> Layer:
    """The danger meter: eight segments, filled to `level` (fractions fill the next segment part way)."""
    x0, _, x1, _ = safe_box(w, h)
    width = x1 - x0
    seg_h = round(h * 0.018)
    cap = _font(h * 0.016, 700)
    cap_h = sum(cap.getmetrics())
    pad = round(h * 0.010)
    img = _panel(width, cap_h + seg_h + 3 * pad, round(h * 0.008), 0.62)
    ink = ImageDraw.Draw(img)
    ink.text((round(w * 0.025), pad), "DANGER", font=cap, fill=INK + (215,))
    gap = round(w * 0.008)
    left, right = round(w * 0.025), width - round(w * 0.025)
    seg_w = (right - left - 7 * gap) / 8
    top = cap_h + 2 * pad
    for i in range(8):
        a = left + i * (seg_w + gap)
        ink.rectangle((round(a), top, round(a + seg_w), top + seg_h), fill=(255, 255, 255, 46))
        fill = min(max(level - i, 0.0), 1.0)
        if fill > 0:
            ink.rectangle((round(a), top, round(a + seg_w * fill), top + seg_h), fill=METER[i] + (245,))
    return _layer(img, x0, round(h * 0.715))


def caption(words: list[str], w: int, h: int) -> Layer:
    """The words spoken so far in the current caption line, centred above the player's controls."""
    x0, _, x1, y1 = safe_box(w, h)
    text = " ".join(words)
    f = render.fit_font(text, round(h * 0.036), x1 - x0 - round(w * 0.04), italic=False)
    img = render.text_block([(text, f, 1.0)], x1 - x0, shadow=1.0)
    top = round(h * 0.640) - img.height // 2
    return _layer(img, x0, top)


def verdict(n: int, option: str, source: str, w: int, h: int) -> Layer:
    x0, _, x1, _ = safe_box(w, h)
    width = x1 - x0
    pad = round(h * 0.024)
    small = _font(h * 0.019, 700)
    f, lines = _fit(f"No. {n}: {option}", round(h * 0.046), width - 2 * pad, 2, weight=700)
    src_f, src_lines = _fit(f"Source: Wikipedia, {source}", round(h * 0.017), width - 2 * pad, 2, weight=500,
                            italic=True)
    hs, hb, hr = sum(small.getmetrics()), sum(f.getmetrics()), sum(src_f.getmetrics())
    ph = pad + hs + pad // 2 + round(hb * 1.18 * len(lines)) + pad // 2 + round(hr * 1.18 * len(src_lines)) + pad
    img = _panel(width, ph, round(h * 0.014), 0.80)
    y = _text_lines(img, ["REAL ANSWER"], small, pad, fill=AMBER)
    y = _text_lines(img, lines, f, y + pad // 2 - round(hs * 0.18))
    _text_lines(img, src_lines, src_f, y + pad // 2 - round(hb * 0.18), fill=INK + (205,))
    return _layer(img, x0, round(h * 0.27))


def cta(w: int, h: int) -> Layer:
    x0, _, x1, _ = safe_box(w, h)
    width = x1 - x0
    pad = round(h * 0.022)
    f1, l1 = _fit("Which number did you pick?", round(h * 0.034), width - 2 * pad, 2, weight=600, italic=True)
    f2, l2 = _fit("COMMENT YOUR NUMBER", round(h * 0.040), width - 2 * pad, 1, weight=700)
    ph = pad + round(sum(f1.getmetrics()) * 1.18 * len(l1)) + round(sum(f2.getmetrics()) * 1.1) + pad
    img = _panel(width, ph, round(h * 0.014), 0.80)
    y = _text_lines(img, l1, f1, pad)
    _text_lines(img, l2, f2, y, fill=AMBER)
    return _layer(img, x0, round(h * 0.43))


# ---------------------------------------------------------------------------------------------------- checks


def banner_contrast(frame: np.ndarray, geo: dict) -> float:
    """WCAG contrast of the banner's ink against the brightest (90th percentile) panel pixels around the text,
    measured on the rendered frame, so the picture behind the panel counts."""
    from . import thumbnail

    x0, y0, x1, y1 = geo["box"]
    strip = frame[y0 + 2:max(y0 + 3, geo["text_top"] - 2), x0 + 4:x1 - 4]
    if strip.size == 0:
        strip = frame[y0:y1, x0:x1]
    return thumbnail.contrast(INK, strip.astype(np.float32))
