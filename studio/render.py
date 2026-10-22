"""Picture: slow depth-parallax camera moves over each plate, long crossfades, a quiet grade with film grain, and
serif title cards. Frames are piped to ffmpeg; each runner job renders one contiguous run of shots and the runs
are joined with a stream copy.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import config

FPS = 24
FONTS = config.ASSETS / "fonts"
INK = (244, 238, 226)
# Pans and parallax are fractions of the plate's spare margin, so the same moves work in 16:9 and 9:16.
MOTIONS = [
    {"name": "push in", "z": (1.000, 1.055), "dz": (0.000, 0.030), "pan": (0, 0, 0, -0.15), "par": (0, 0)},
    {"name": "drift right", "z": (1.050, 1.050), "dz": (0.012, 0.012), "pan": (-0.45, 0.45, 0, 0), "par": (0.15, 0)},
    {"name": "pull back", "z": (1.065, 1.005), "dz": (0.030, 0.004), "pan": (0, 0, -0.10, 0.10), "par": (0, 0)},
    {"name": "drift left", "z": (1.050, 1.050), "dz": (0.012, 0.012), "pan": (0.45, -0.45, 0, 0), "par": (-0.15, 0)},
    {"name": "rise", "z": (1.045, 1.060), "dz": (0.010, 0.020), "pan": (0, 0, 0.35, -0.35), "par": (0, 0.12)},
    {"name": "settle", "z": (1.060, 1.040), "dz": (0.020, 0.010), "pan": (0, 0, -0.35, 0.35), "par": (0, -0.12)},
]
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI",
         "XVII", "XVIII", "XIX", "XX"]


def ease(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def font(size: int, italic: bool = False, weight: int = 500) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONTS / ("CormorantGaramond-Italic.ttf" if italic else "CormorantGaramond.ttf")), size)
    try:
        f.set_variation_by_axes([weight])
    except (OSError, ValueError):
        pass
    return f


class Plate:
    def __init__(self, jpg: Path, depth_png: Path):
        self.rgb = cv2.cvtColor(cv2.imread(str(jpg), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        self.ph, self.pw = self.rgb.shape[:2]
        d = cv2.imread(str(depth_png), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
        d = cv2.resize(d, (self.pw, self.ph), interpolation=cv2.INTER_LINEAR)
        self.depth = cv2.GaussianBlur(d, (0, 0), 4) - 0.5


class Camera:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32) - w / 2, np.arange(h, dtype=np.float32) - h / 2)
        self.gx, self.gy = gx, gy

    def shoot(self, plate: Plate, u: float, motion: dict) -> np.ndarray:
        e = ease(u)
        mx, my = (plate.pw - self.w) / 2, (plate.ph - self.h) / 2
        z = motion["z"][0] + (motion["z"][1] - motion["z"][0]) * e
        dz = motion["dz"][0] + (motion["dz"][1] - motion["dz"][0]) * e
        px0, px1, py0, py1 = motion["pan"]
        cx = plate.pw / 2 + (px0 + (px1 - px0) * e) * mx
        cy = plate.ph / 2 + (py0 + (py1 - py0) * e) * my
        parx, pary = motion["par"][0] * mx * (e - 0.5), motion["par"][1] * my * (e - 0.5)
        d = cv2.remap(plate.depth, cx + self.gx / z, cy + self.gy / z, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)
        zz = 1 + dz * d * 2
        bx = cx + self.gx / (z * zz) - parx * d * 2
        by = cy + self.gy / (z * zz) - pary * d * 2
        return cv2.remap(plate.rgb, bx, by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class Grade:
    """Soft vignette, a touch darker overall, and fine grain (which also hides banding in dark gradients)."""

    def __init__(self, w: int, h: int, seed: int = 1, darken: float = 0.94):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w * 0.62)) ** 2 + ((yy - h / 2) / (h * 0.62)) ** 2)
        self.mask = (np.clip(1.04 - 0.30 * r ** 2.4, 0.62, 1.0) * darken)[..., None].astype(np.float32)
        rng = np.random.default_rng(seed)
        self.grain = [cv2.resize(rng.normal(0, 2.2, (h // 2, w // 2)).astype(np.float32), (w, h),
                                 interpolation=cv2.INTER_LINEAR)[..., None] for _ in range(8)]

    def apply(self, frame: np.ndarray, i: int, fade: float = 1.0) -> np.ndarray:
        f = frame.astype(np.float32) * (self.mask * fade) + self.grain[i % 8] * max(fade, 0.25)
        return np.clip(f, 0, 255).astype(np.uint8)


def layer_from(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(img, dtype=np.float32)
    return a[..., :3], a[..., 3:4] / 255.0


def blit(frame: np.ndarray, layer: tuple[np.ndarray, np.ndarray], x: int, y: int, alpha: float = 1.0) -> None:
    rgb, a = layer
    h, w = a.shape[:2]
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    sub = frame[y0:y1, x0:x1].astype(np.float32)
    la = a[y0 - y:y1 - y, x0 - x:x1 - x] * alpha
    frame[y0:y1, x0:x1] = (sub * (1 - la) + rgb[y0 - y:y1 - y, x0 - x:x1 - x] * la).astype(np.uint8)


def text_block(lines: list[tuple[str, ImageFont.FreeTypeFont, float]], width: int, gap: int = 14,
               tracking: float = 0.0, shadow: float = 0.75) -> Image.Image:
    """Centred lines (text, font, opacity) with a soft dark halo behind them so they read over any image."""
    def measure(t, f):
        if not tracking:
            b = f.getbbox(t)
            return b[2] - b[0], b
        return int(sum(f.getlength(c) for c in t) + tracking * f.size * (len(t) - 1)), f.getbbox(t)

    rows = [(t, f, o, *measure(t, f)) for t, f, o in lines]
    asc = [f.getmetrics() for _, f, _ in lines]
    height = sum(a + d for a, d in asc) + gap * (len(lines) - 1)
    pad = 60
    img = Image.new("RGBA", (width, height + 2 * pad), (0, 0, 0, 0))
    ink = ImageDraw.Draw(img)
    y = pad
    for (t, f, o, w, b), (a, d) in zip(rows, asc):
        x = (width - w) // 2
        if tracking:
            for c in t:
                ink.text((x, y), c, font=f, fill=INK + (int(255 * o),))
                x += f.getlength(c) + tracking * f.size
        else:
            ink.text((x - b[0], y), t, font=f, fill=INK + (int(255 * o),))
        y += a + d + gap
    halo = img.split()[3].filter(ImageFilter.GaussianBlur(18)).point(lambda v: min(255, int(v * 2.2 * shadow)))
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.putalpha(halo)
    out.alpha_composite(img)
    return out


def fit_font(text: str, size: int, max_w: int, italic: bool = True) -> ImageFont.FreeTypeFont:
    f = font(size, italic)
    while f.getlength(text) > max_w and size > 24:
        size -= 2
        f = font(size, italic)
    return f


def card_layer(card: dict, w: int, h: int) -> tuple[tuple[np.ndarray, np.ndarray], int]:
    """Returns the layer and its top y."""
    s = w / 1920 if w > h else w / 1080 * 0.8
    if card["kind"] == "intro":
        lines = [(config.CHANNEL_NAME.upper(), font(round(30 * s), weight=600), 0.82),
                 (card["title"], fit_font(card["title"], round(86 * s), int(w * 0.84)), 1.0)]
        img = text_block(lines, w, gap=round(22 * s), tracking=0)
        return layer_from(img), round(h * 0.46 - img.height / 2)
    label = f"CHAPTER {ROMAN[card['n']] if card['n'] < len(ROMAN) else card['n']}"
    lines = [(label, font(round(28 * s), weight=600), 0.8),
             (card["title"], fit_font(card["title"], round(70 * s), int(w * 0.84)), 1.0)]
    img = text_block(lines, w, gap=round(16 * s))
    return layer_from(img), round(h * 0.70 - img.height / 2)


def card_alpha(card: dict, t: float) -> float:
    u = t - card["start"]
    if u < 0 or u > card["seconds"]:
        return 0.0
    return ease(u / 2.0) * ease((card["seconds"] - u) / 2.5)


class Encoder:
    def __init__(self, out: Path, w: int, h: int, crf: int = 21, preset: str = "medium", maxrate: str = "5M",
                 fps: int = FPS):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-aq-mode", "3",
             "-maxrate", maxrate, "-bufsize", "10M",
             "-x264-params", "stitchable=1:keyint=240:min-keyint=24", "-pix_fmt", "yuv420p", "-an",
             "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed to encode")


class Plates:
    """Keeps the few plates the current frames need."""

    def __init__(self, image_dir: Path, keep: int = 3):
        self.dir, self.keep, self.cache = image_dir, keep, {}

    def __call__(self, image_id: str) -> Plate:
        if image_id not in self.cache:
            if len(self.cache) >= self.keep:
                self.cache.pop(next(iter(self.cache)))
            self.cache[image_id] = Plate(self.dir / f"{image_id}.jpg", self.dir / f"{image_id}.depth.png")
        return self.cache[image_id]


def segment(tl: dict, k: int, image_dir: Path, out: Path, stills: Path, size=(1920, 1080)) -> dict:
    """Render frames [f0, f1) of render shard k to out (H.264, no audio)."""
    w, h = size
    shard = tl["render_shards"][k]
    shots = tl["shots"]
    cf = tl["crossfade"]
    total = tl["seconds"]
    cam, grade, plates = Camera(w, h), Grade(w, h, seed=k + 1), Plates(image_dir)
    cards = [(c, *card_layer(c, w, h)) for c in tl["cards"]]
    stills.mkdir(parents=True, exist_ok=True)
    want_stills = {(s["f0"] + s["f1"]) // 2 for s in shots[shard["first"]:shard["last"] + 1]}
    want_stills |= {int((c["start"] + c["seconds"] * 0.45) * FPS) for c in tl["cards"]}
    enc = Encoder(out, w, h)
    i = shard["first"]
    started = time.monotonic()

    def view(j: int, t: float) -> np.ndarray:
        s = shots[j]
        travel = (s["f1"] - s["f0"]) / FPS + cf
        return cam.shoot(plates(s["image"]), (t - s["f0"] / FPS) / travel, MOTIONS[s["motion"]])

    for f in range(shard["f0"], shard["f1"]):
        while f >= shots[i]["f1"]:
            i += 1
        t = f / FPS
        frame = view(i, t)
        since = t - shots[i]["f0"] / FPS
        if i > 0 and since < cf:
            a = ease(since / cf)
            frame = cv2.addWeighted(frame, a, view(i - 1, t), 1 - a, 0)
        fade = min(ease(t / 3.0), ease((total - t) / tl["tail_fade"]) if t > total - tl["tail_fade"] else 1.0)
        frame = grade.apply(frame, f, fade)
        for c, layer, y in cards:
            a = card_alpha(c, t)
            if a > 0:
                blit(frame, layer, 0, y, a)
        enc.write(frame)
        if f in want_stills:
            Image.fromarray(frame).save(stills / f"f{f:07d}.jpg", quality=88)
    enc.close()
    secs = time.monotonic() - started
    frames = shard["f1"] - shard["f0"]
    return {"k": k, "frames": frames, "seconds": round(secs, 1), "fps": round(frames / max(secs, 1e-6), 2)}


def concat(segments: list[Path], out: Path) -> None:
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in segments))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c", "copy", str(out)], check=True)


def mux(video: Path, audio: Path, out: Path, chapters: list[dict] | None = None) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio)]
    if chapters:
        meta = out.with_suffix(".chapters.txt")
        rows = [";FFMETADATA1"]
        for c in chapters:
            rows += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(c['start'] * 1000)}", f"END={int(c['end'] * 1000)}",
                     "title=" + c["title"].replace("=", "\\=").replace(";", "\\;")]
        meta.write_text("\n".join(rows) + "\n")
        cmd += ["-i", str(meta), "-map_metadata", "2", "-map_chapters", "2"]
    acodec = ["-c:a", "copy"] if audio.suffix == ".m4a" else ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    cmd += ["-map", "0:v", "-map", "1:a", "-c:v", "copy", *acodec, "-shortest", "-movflags", "+faststart",
            str(out)]
    subprocess.run(cmd, check=True)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=1))
