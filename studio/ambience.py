"""Nature ambience under the narration: CC0 recordings (licence re-checked at download) layered on a synthetic bed.

Clips with sudden loud events (a momentary level far above the clip's median) are rejected, so nothing jolts a
sleeping listener. The result is a seamless ten-minute loop that ffmpeg repeats under the whole episode.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

from . import config

SR = 48000
LOOP_SECONDS = 600
XF = 6.0
SPIKE_DB = 9.0
UA = {"User-Agent": "PangaeaNightsStudio/1.0 (https://github.com/ashivam-dot/pangaea-nights)"}
SYNTH = {"ocean": "waves", "rain": "rain", "wind": "wind", "underwater": "rumble"}


def catalog() -> dict:
    return json.loads((config.ASSETS / "ambience" / "catalog.json").read_text(encoding="utf-8"))


def _licence_ok(item: dict) -> tuple[bool, str]:
    """Re-check the licence at the source; returns (ok, direct download url)."""
    try:
        if item["source"] == "freesound":
            page = requests.get(item["page"], headers=UA, timeout=30).text
            return "creativecommons.org/publicdomain/zero/1.0" in page, item["url"]
        if item["source"] == "commons":
            d = requests.get("https://commons.wikimedia.org/w/api.php", headers=UA, timeout=30, params={
                "action": "query", "titles": item["id"], "prop": "imageinfo", "iiprop": "url|extmetadata",
                "format": "json", "formatversion": 2}).json()
            ii = d["query"]["pages"][0]["imageinfo"][0]
            lic = ii["extmetadata"].get("LicenseShortName", {}).get("value", "").lower()
            return ("cc0" in lic or "public domain" in lic), ii["url"]
    except (requests.RequestException, KeyError, IndexError, ValueError):
        pass
    return False, ""


def _decode(src: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).copy()


def spikes_db(x: np.ndarray) -> float:
    """How far the loudest 400 ms window rises above the clip's median window, in dB."""
    mono = x.mean(axis=1)
    n = int(0.4 * SR)
    k = len(mono) // n
    if k < 5:
        return 99.0
    rms = np.sqrt((mono[: k * n].reshape(k, n) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    return float(db.max() - np.median(db))


def fetch(key: str, cache: Path) -> list[dict]:
    """Download, verify and screen the recordings for an ambience key. Returns accepted clips with credits."""
    cache.mkdir(parents=True, exist_ok=True)
    accepted = []
    for item in catalog().get(key, []):
        ok, url = _licence_ok(item)
        if not ok or not url:
            print(f"ambience: skipped {item['title']} (licence not confirmed)", flush=True)
            continue
        name = re.sub(r"[^A-Za-z0-9]+", "_", str(item["id"])) + Path(url.split("?")[0]).suffix
        path = cache / name
        if not path.exists():
            r = requests.get(url, headers=UA, timeout=120)
            r.raise_for_status()
            path.write_bytes(r.content)
        x = _decode(path)
        spike = spikes_db(x)
        if spike > SPIKE_DB or len(x) < 60 * SR:
            print(f"ambience: rejected {item['title']} (spike {spike:.1f} dB, {len(x) / SR:.0f}s)", flush=True)
            continue
        accepted.append({**item, "path": str(path), "spike_db": round(spike, 1), "seconds": round(len(x) / SR, 1)})
    return accepted


def _filtered_noise(n: int, rng, lo: float, hi: float, tilt: float = 1.0) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.maximum(np.fft.rfftfreq(n, 1 / SR), 1.0)
    shape = f ** (-0.5 * tilt) * (f / lo) ** 2 / (1 + (f / lo) ** 2) / (1 + (f / hi) ** 4)
    out = np.fft.irfft(spec * shape, n)
    return out / (np.abs(out).max() + 1e-9)


def synth(kind: str, seconds: float = LOOP_SECONDS, seed: int = 7) -> np.ndarray:
    """Seamless synthetic beds: 'waves' (slow surf), 'wind', 'rumble' (deep water), 'rain' (soft hiss)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    chans = []
    for c in range(2):
        if kind == "waves":
            dark = _filtered_noise(n, rng, 30, 700, 1.4)
            bright = _filtered_noise(n, rng, 200, 5000, 1.0)
            period = seconds / round(seconds / 9.5)
            env = (0.5 - 0.5 * np.cos(2 * np.pi * (t / period + 0.17 * c))) ** 2.2
            sig = dark * (0.35 + 0.65 * env) + 0.35 * bright * env ** 1.5
        elif kind == "wind":
            base = _filtered_noise(n, rng, 60, 1800, 1.6)
            gust = 0.55 + 0.45 * np.sin(2 * np.pi * t / (seconds / 40) + c) * np.sin(2 * np.pi * t / (seconds / 13))
            sig = base * gust
        elif kind == "rumble":
            sig = _filtered_noise(n, rng, 20, 260, 1.8) * (0.8 + 0.2 * np.sin(2 * np.pi * t / (seconds / 20) + c))
        else:
            sig = _filtered_noise(n, rng, 400, 9000, 0.6) * 0.6 + _filtered_noise(n, rng, 60, 500, 1.2) * 0.4
        chans.append(sig)
    x = np.stack(chans, axis=1)
    return (x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05).astype(np.float32)


def _rms_norm(x: np.ndarray, rms: float = 0.05) -> np.ndarray:
    return x / (np.sqrt((x ** 2).mean()) + 1e-9) * rms


def _xfade_concat(clips: list[np.ndarray], seconds: float) -> np.ndarray:
    xf = int(XF * SR)
    ramp = np.sin(np.linspace(0, np.pi / 2, xf))[:, None]
    out = clips[0][int(2 * SR):-int(2 * SR)]
    i = 1
    while len(out) < seconds * SR + xf:
        nxt = clips[i % len(clips)][int(2 * SR):-int(2 * SR)]
        out = np.concatenate([out[:-xf], out[-xf:] * ramp[::-1] + nxt[:xf] * ramp, nxt[xf:]])
        i += 1
    out = out[: int(seconds * SR) + xf]
    head = out[:xf] * ramp + out[-xf:] * ramp[::-1]  # the loop's end flows into its start
    return np.concatenate([head, out[xf:-xf]])


def build_loop(key: str, out: Path, cache: Path) -> dict:
    clips = fetch(key, cache)
    bed = synth(SYNTH.get(key, "waves"))
    if clips:
        rec = _xfade_concat([_rms_norm(_decode(Path(c["path"]))) for c in clips], LOOP_SECONDS)
        loop = rec + 0.45 * bed[: len(rec)]
    else:
        loop = bed
    loop = _rms_norm(loop, 0.06).astype(np.float32)
    sf.write(str(out), loop, SR, subtype="PCM_16")
    credits = [{"title": c["title"], "author": c["author"], "page": c["page"], "license": c["license"]} for c in clips]
    return {"key": key, "recordings": credits, "synthetic": SYNTH.get(key, "waves"), "seconds": LOOP_SECONDS}
