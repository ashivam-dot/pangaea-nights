"""The sound of a ladder Short, all generated in code (nothing to license, nothing to claim):

- a four-note motif per series, heard under the hook and again under the verdict: the series' sound signature;
- a low pulse bed that quickens as the ladder climbs;
- a whoosh on every cut and a riser into the verdict.

The narration sits on top; the mix is brought to -14 LUFS integrated (+-1) with true peaks at or below -1.5 dBTP,
measured on the encoded AAC, not on the master.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SR = 48000
TARGET_I = -14.0
LIMIT_DB = -2.4  # the master's ceiling: AAC adds up to ~1 dB of inter-sample overshoot on top
MOTIFS = {  # MIDI notes; minor and open intervals, low register: night, not nursery
    "survive_night": [57, 60, 64, 62],
    "level_1_8": [52, 55, 59, 62],
    "countdown": [50, 53, 57, 56],
    "how_deep": [45, 52, 50, 48],
}
LEVELS = {"motif": 0.16, "bed": 0.22, "whoosh": 0.20, "riser": 0.16}


def _hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / SR
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-t / decay)


def note(m: float, seconds: float) -> np.ndarray:
    """A soft struck tone: fundamental with a few decaying partials (a low bell or a felt piano, not a chime)."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f = _hz(m)
    tone = sum(a * np.sin(2 * np.pi * f * k * t) * np.exp(-t * k * 0.9) for k, a in ((1, 1.0), (2, 0.35), (3, 0.12),
                                                                                    (4.01, 0.06)))
    return (tone * _env(n, 0.012, seconds / 2.2)).astype(np.float32)


def motif(series: str, step: float = 0.32) -> np.ndarray:
    notes = MOTIFS.get(series, MOTIFS["survive_night"])
    out = np.zeros(int((step * len(notes) + 1.6) * SR), np.float32)
    for i, m in enumerate(notes):
        a = note(m, 1.6)
        s = int(i * step * SR)
        out[s:s + len(a)] += a
    return out / max(np.abs(out).max(), 1e-6)


def pulse_bed(seconds: float, climb: list[tuple[float, float]], seed: int = 3) -> np.ndarray:
    """A low two-beat pulse (a muffled heartbeat at ~55 Hz) with a dark noise floor. climb: (time, bpm) points."""
    n = int(seconds * SR)
    out = np.zeros(n, np.float32)
    times, bpms = zip(*climb) if climb else ((0.0,), (70.0,))
    t = 0.0
    kick_n = int(0.22 * SR)
    kt = np.arange(kick_n) / SR
    kick = (np.sin(2 * np.pi * (48 + 30 * np.exp(-kt * 30)) * kt) * np.exp(-kt * 16)).astype(np.float32)
    while t < seconds:
        bpm = float(np.interp(t, times, bpms))
        for off, amp in ((0.0, 1.0), (0.24, 0.6)):
            s = int((t + off * 60 / bpm) * SR)
            if s < n:
                e = min(n, s + kick_n)
                out[s:e] += amp * kick[:e - s]
        t += 60.0 / bpm
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, n).astype(np.float32)
    k = int(SR / 180)  # a moving average: a very dark rumble
    noise = np.convolve(noise, np.ones(k, np.float32) / k, mode="same") * 0.6
    return (out + noise) / max(np.abs(out + noise).max(), 1e-6)


def whoosh(seconds: float = 0.45, seed: int = 1) -> np.ndarray:
    n = int(seconds * SR)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, n).astype(np.float32)
    out = np.zeros(n, np.float32)
    # a sweeping low-pass: the moving-average window shrinks as the air rushes past
    for i, k in enumerate(np.linspace(60, 6, 6).astype(int)):
        a, b = i * n // 6, (i + 1) * n // 6
        out[a:b] = np.convolve(noise, np.ones(k) / k, mode="same")[a:b]
    t = np.linspace(0, 1, n)
    env = np.sin(np.pi * t ** 0.6) ** 2
    return (out * env / max(np.abs(out * env).max(), 1e-6)).astype(np.float32)


def riser(seconds: float = 1.4, seed: int = 2) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    sweep = np.sin(2 * np.pi * np.cumsum(np.linspace(110, 440, n)) / SR)
    noise = np.random.default_rng(seed).normal(0, 0.4, n)
    env = (t / seconds) ** 2
    out = (sweep * 0.6 + noise * 0.4) * env
    return (out / max(np.abs(out).max(), 1e-6)).astype(np.float32)


def _place(track: np.ndarray, clip: np.ndarray, at: float, gain: float) -> None:
    s = int(max(at, 0.0) * SR)
    e = min(len(track), s + len(clip))
    if e > s:
        track[s:e] += gain * clip[:e - s]


def music(tl: dict, series: str) -> np.ndarray:
    """Everything but the voice, as a mono float track the length of the Short."""
    total = tl["seconds"]
    out = np.zeros(int(total * SR) + SR, np.float32)
    opts = [s for s in tl["shots"] if s["kind"] == "option"]
    climb = [(0.0, 64.0)] + [(s["start"] if s["n"] > 1 else tl["hook_end"], 64.0 + 7.0 * (s["danger"] - 1))
                             for s in opts] + [(tl["verdict_start"], 58.0)]
    bed = pulse_bed(total, climb)
    fade = np.minimum(1.0, np.arange(len(bed)) / (0.25 * SR))
    out[:len(bed)] += LEVELS["bed"] * bed * fade
    m = motif(series)
    _place(out, m, 0.0, LEVELS["motif"])
    _place(out, m, tl["verdict_start"] + 0.2, LEVELS["motif"])
    for i, t in enumerate(tl["whoosh"]):
        w = whoosh(seed=i + 1)
        _place(out, w, t - 0.18, LEVELS["whoosh"])
    _place(out, riser(), tl["riser"], LEVELS["riser"])
    return out[:int(total * SR)]


def voice_track(tl: dict, voice: dict, voice_dir: Path) -> np.ndarray:
    import soundfile as sf

    out = np.zeros(int(tl["seconds"] * SR) + SR, np.float32)
    for line in tl["lines"]:
        a, sr = sf.read(str(voice_dir / f"{line['id']}.wav"), dtype="float32")
        if sr != SR:
            idx = np.arange(0, len(a), sr / SR)
            a = np.interp(idx, np.arange(len(a)), a).astype(np.float32)
        _place(out, a, line["start"], 1.0)
    return out[:int(tl["seconds"] * SR)]


def build(tl: dict, voice: dict, voice_dir: Path, series: str, out: Path) -> Path:
    """voice + music -> a 48 kHz stereo WAV at about -14 LUFS, limited below LIMIT_DB true peak."""
    import soundfile as sf

    from . import mix

    v = voice_track(tl, voice, voice_dir)
    m = music(tl, series)
    # ducking: the music sits 9 dB lower while a line is spoken
    duck = np.ones_like(m)
    for line in tl["lines"]:
        d = voice["segments"][line["id"]]["duration"]
        s, e = int(line["start"] * SR), int((line["start"] + d) * SR)
        duck[s:e] = 0.35
    duck = np.convolve(duck, np.ones(int(0.08 * SR)) / int(0.08 * SR), mode="same").astype(np.float32)
    raw = v / max(np.abs(v).max(), 1e-6) * 0.7 + m * duck
    pre = out.with_suffix(".pre.wav")
    sf.write(str(pre), np.stack([raw, raw], axis=1), SR, subtype="FLOAT")
    gain = mix._gain_to(pre, TARGET_I)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(pre), "-af",
                    f"volume={gain:.2f}dB,{mix.peak_limiter(LIMIT_DB)}", "-ar", str(SR), "-c:a", "pcm_s24le",
                    str(out)], check=True)
    # the limiter takes a little loudness off the peaks: one correction pass, limited again
    fix = TARGET_I - mix.integrated(out)["I"]
    if abs(fix) > 0.3:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(pre), "-af",
                        f"volume={gain + fix:.2f}dB,{mix.peak_limiter(LIMIT_DB)}", "-ar", str(SR), "-c:a",
                        "pcm_s24le", str(out)], check=True)
    pre.unlink()
    return out
