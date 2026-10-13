"""Join the rendered segments with the mix, then check the result before anything may be published."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from . import mix, render

MAX_SWAPPED = 0.15


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels:"
                          "format=duration,size", "-of", "json", str(path)], check=True, capture_output=True,
                         text=True).stdout
    return json.loads(out)


def frame_at(path: Path, t: float, size=(480, 270)) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                          "-vf", f"scale={size[0]}:{size[1]}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(size[1], size[0], 3)


def check_frames(path: Path, tl: dict, out_dir: Path, n: int = 14) -> dict:
    """Sample shot midpoints: frames must not be black or blank, and the camera must be moving."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shots = tl["shots"]
    picks = [shots[round(i * (len(shots) - 1) / max(1, n - 1))] for i in range(min(n, len(shots)))]
    rows, problems = [], []
    for s in picks:
        mid = (s["f0"] + s["f1"]) / 2 / tl["fps"]
        a, b = frame_at(path, mid), frame_at(path, mid + 2.0)
        lum = float(a.mean())
        spread = float(a.std())
        motion = float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
        rows.append({"t": round(mid, 1), "image": s["image"], "mean": round(lum, 1), "std": round(spread, 1),
                     "motion": round(motion, 2)})
        Image.fromarray(a).save(out_dir / f"check-{int(mid):05d}.jpg", quality=88)
        in_fade = mid > tl["seconds"] - tl["tail_fade"]
        if not in_fade and (lum < 6 or spread < 4):
            problems.append(f"blank or black frame at {mid:.0f}s")
        if motion < 0.05:
            problems.append(f"frozen picture at {mid:.0f}s")
    return {"frames": rows, "problems": problems}


def limited_mix(mix_path: Path) -> Path:
    """The mix as it is, or, when its true peak is too hot (mixes made before the limiter), a limited copy."""
    if mix.integrated(mix_path)["true_peak"] <= mix.LIMIT_TP + 0.5:
        return mix_path
    flac = mix_path.with_name("mix-limited.flac")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mix_path), "-af", mix.peak_limiter(),
                    "-c:a", "flac", str(flac)], check=True)
    out = mix.to_aac(flac, mix_path.with_name("mix-limited.m4a"))
    flac.unlink()
    return out


def assemble(tl: dict, seg_dir: Path, mix_path: Path, out: Path) -> Path:
    segs = sorted(seg_dir.rglob("seg*.mp4"))
    if len(segs) != len(tl["render_shards"]):
        raise RuntimeError(f"expected {len(tl['render_shards'])} segments, found {len(segs)}")
    video = out.with_name("video.mp4")
    render.concat(segs, video)
    render.mux(video, mix_path, out, chapters=tl["chapters"])
    video.unlink()
    return out


def qa(episode: dict, tl: dict, final: Path, out_dir: Path, thumb: dict, shorts: list[dict]) -> dict:
    problems = []
    info = probe(final)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next(s for s in info["streams"] if s["codec_type"] == "audio")
    duration = float(info["format"]["duration"])
    if (v["codec_name"], v["width"], v["height"], v["r_frame_rate"]) != ("h264", 1920, 1080, "24/1"):
        problems.append(f"video stream is {v['codec_name']} {v['width']}x{v['height']} @ {v['r_frame_rate']}")
    if a["codec_name"] != "aac":
        problems.append(f"audio is {a['codec_name']}")
    if abs(duration - tl["seconds"]) > 2.0:
        problems.append(f"duration {duration:.1f}s, timeline {tl['seconds']:.1f}s")
    loud = mix.integrated(final)
    loud["momentary_over_integrated"] = round(loud["max_momentary"] - loud["I"], 2)
    if abs(loud["I"] - mix.TARGET_I) > 1.0:
        problems.append(f"integrated loudness {loud['I']} LUFS")
    if loud["true_peak"] > mix.TARGET_TP + 0.5:
        problems.append(f"true peak {loud['true_peak']} dBFS")
    if loud["momentary_over_integrated"] > mix.MAX_MOMENTARY_OVER:
        problems.append(f"a loud moment {loud['momentary_over_integrated']} LU above the average")
    frames = check_frames(final, tl, out_dir)
    problems += frames["problems"]
    swapped = len(tl["swapped"]) / max(1, len(tl["shots"]))
    if swapped > MAX_SWAPPED:
        problems.append(f"{swapped:.0%} of shots reuse a neighbour image")
    if not thumb.get("ok"):
        problems.append("thumbnail failed the legibility test")
    hard = [w for c in episode["chapters"] for w in c.get("warnings", [])
            if w.startswith(("unsupported", "name not", "exclamation", "alarm", "call to"))]
    if hard:
        problems.append(f"script gate problems remain: {hard[:3]}")
    return {"ok": not problems, "problems": problems, "duration": round(duration, 2), "loudness": loud,
            "video": {k: v[k] for k in ("codec_name", "width", "height", "r_frame_rate")},
            "size_mb": round(int(info["format"]["size"]) / 1e6, 1), "frames": frames["frames"],
            "swapped_share": round(swapped, 3), "shorts": [{k: s[k] for k in ("k", "ok") if k in s} for s in shorts]}


def excerpt(final: Path, out: Path, start: float = 0.0, seconds: float = 100.0) -> Path:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.2f}", "-i", str(final), "-t", f"{seconds:.2f}",
                    "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    return out
