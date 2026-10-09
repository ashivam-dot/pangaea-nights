"""Turn a generated still into a render plate: upscaled past 1080p (so the camera can move) plus a depth map.

Upscaling: Real-ESRGAN realesr-general-x4v3 (BSD-3-Clause), a compact network that runs in seconds on CPU,
loaded with spandrel (MIT). Depth: Depth Anything V2 Small (Apache-2.0).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
MARGIN = 1.12
PLATE = (round(1920 * MARGIN / 2) * 2, round(1080 * MARGIN / 2) * 2)  # 16:9 plate
PLATE_V = (round(1080 * MARGIN / 2) * 2, round(1920 * MARGIN / 2) * 2)  # 9:16 plate for Shorts

_sr = None
_depth = None


def _upscaler():
    global _sr
    if _sr is None:
        import torch
        from spandrel import ModelLoader

        path = Path(torch.hub.get_dir()) / "realesr-general-x4v3.pth"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(ESRGAN_URL, str(path), progress=False)
        _sr = ModelLoader().load_from_file(str(path)).eval()
    return _sr


def upscale(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """4x super-resolution, then a Lanczos resize to exactly `size` (cover crop if the aspect differs)."""
    import torch

    model = _upscaler()
    x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    with torch.no_grad():
        y = model(x).clamp(0, 1)[0].permute(1, 2, 0).numpy()
    big = Image.fromarray((y * 255 + 0.5).astype(np.uint8))
    return cover(big, size)


def cover(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    w, h = size
    scale = max(w / img.width, h / img.height)
    rs = img.resize((max(w, round(img.width * scale)), max(h, round(img.height * scale))), Image.LANCZOS)
    left, top = (rs.width - w) // 2, (rs.height - h) // 2
    return rs.crop((left, top, left + w, top + h))


def depth(img: Image.Image) -> np.ndarray:
    """Relative depth in [0, 1] (1 = nearest) at half the image size."""
    global _depth
    if _depth is None:
        from transformers import pipeline

        _depth = pipeline("depth-estimation", model=DEPTH_MODEL, device="cpu")
    small = img.convert("RGB").copy()
    small.thumbnail((1024, 1024))
    out = np.asarray(_depth(small)["predicted_depth"].squeeze(), dtype=np.float32)
    half = (img.width // 2, img.height // 2)
    out = np.asarray(Image.fromarray(out).resize(half, Image.BICUBIC), dtype=np.float32)
    lo, hi = np.percentile(out, 2), np.percentile(out, 98)
    return np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1)


def save_plate(img: Image.Image, out: Path, vertical: bool = False) -> dict:
    """out.jpg (plate) and out.depth.png (16-bit depth); returns their paths."""
    plate = upscale(img, PLATE_V if vertical else PLATE)
    jpg = out.with_suffix(".jpg")
    plate.save(jpg, quality=94, subsampling=0)
    d = depth(plate)
    dpng = out.with_suffix(".depth.png")
    Image.fromarray((d * 65535).astype(np.uint16)).save(dpng)
    return {"plate": jpg.name, "depth": dpng.name, "size": list(plate.size)}
