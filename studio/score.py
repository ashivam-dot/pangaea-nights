"""A soft ambient score written in code: sustained pads and a low drone, one slow chord per chapter.

No melody and no rhythm: every voice is a few detuned sines with very slow swells, and chords change only at
chapter boundaries through a twenty-second crossfade. The key comes from the episode id, so episodes differ.
"""
from __future__ import annotations

import hashlib

import numpy as np
import soundfile as sf

SR = 24000
ROOTS = [43, 45, 46, 48, 50, 41]  # G2 A2 Bb2 C3 D3 F2 (MIDI)
# Chords as semitone offsets from the root; all consonant, slowly wandering around the home chord.
CHORDS = [(0, 7, 12, 16), (0, 7, 12, 15), (-3, 4, 9, 12), (-5, 2, 7, 11), (0, 7, 14, 16), (-7, 0, 5, 9),
          (0, 5, 12, 19), (-2, 5, 10, 14)]
PATH = [0, 2, 5, 0, 3, 1, 6, 0, 2, 7, 4, 0, 5, 1, 3, 0]
XFADE = 20.0


def _hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def _voices(root: int, chord: tuple) -> list[tuple[float, float]]:
    """(frequency, gain) for a chord: a sub drone, the chord tones, a faint octave shimmer."""
    out = [(_hz(root - 12), 0.55), (_hz(root), 0.45)]
    out += [(_hz(root + c), 0.32 / (1 + i * 0.35)) for i, c in enumerate(chord)]
    out.append((_hz(root + chord[-1] + 12), 0.05))
    return out


def _block(t: np.ndarray, voices: list, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = np.zeros_like(t)
    right = np.zeros_like(t)
    for k, (f, g) in enumerate(voices):
        for d in (-0.07, 0.0, 0.06):
            ff = f * 2 ** (d / 12)
            rate = rng.uniform(0.015, 0.05)
            phase = rng.uniform(0, 2 * np.pi)
            swell = 0.65 + 0.35 * np.sin(2 * np.pi * rate * t + phase)
            tone = np.sin(2 * np.pi * ff * t + k) + 0.18 * np.sin(4 * np.pi * ff * t + 2 * k)
            pan = 0.5 + 0.35 * np.sin(k + d * 10)
            left += g * swell * tone * (1 - pan)
            right += g * swell * tone * pan
    return np.stack([left, right], axis=1) / 6.0


def plan(n_chapters: int, episode_id: str) -> tuple[int, list[tuple]]:
    seed = int(hashlib.sha1(episode_id.encode()).hexdigest()[:8], 16)
    root = ROOTS[seed % len(ROOTS)]
    start = seed % len(PATH)
    return root, [CHORDS[PATH[(start + i) % len(PATH)]] for i in range(n_chapters + 1)]


def render(path, boundaries: list[float], total: float, episode_id: str, block: float = 30.0) -> dict:
    """Write the score to `path` (24 kHz stereo). boundaries = chapter start times; the last chord holds through
    the tail. Peak about -12 dBFS before mixing."""
    root, chords = plan(len(boundaries), episode_id)
    starts = list(boundaries) + [total + 1]
    seed = int(hashlib.sha1(episode_id.encode()).hexdigest()[8:16], 16)
    with sf.SoundFile(str(path), "w", SR, 2, "PCM_16") as fh:
        t0 = 0.0
        while t0 < total:
            t1 = min(total, t0 + block)
            t = np.arange(int(round(t0 * SR)), int(round(t1 * SR))) / SR
            idx = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(chords) - 1)
            out = np.zeros((len(t), 2))
            for c in np.unique(idx):
                c = int(c)
                mask = idx == c
                tm = t[mask]
                if c == 0:
                    out[mask] = _block(tm, _voices(root, chords[0]), seed)
                    continue
                u = np.clip((tm - starts[c]) / XFADE, 0, 1) * np.pi / 2
                sig = _block(tm, _voices(root, chords[c]), seed + c) * np.sin(u)[:, None]
                fading = u < np.pi / 2
                if fading.any():
                    old = _block(tm[fading], _voices(root, chords[c - 1]), seed + c - 1)
                    sig[fading] += old * np.cos(u[fading])[:, None]
                out[mask] = sig
            fade_in = np.clip(t / 8.0, 0, 1)[:, None]
            fade_out = np.clip((total - t) / 40.0, 0, 1)[:, None]
            fh.write((out * fade_in * fade_out * 0.25).astype(np.float32))
            t0 = t1
    return {"root_midi": root, "chords": [list(c) for c in chords]}
