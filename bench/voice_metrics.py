"""Naturalness (UTMOS22 strong, via SpeechMOS), calmness (pitch spread, pace) and intelligibility (Whisper)."""
from __future__ import annotations

import re

import numpy as np

_utmos = None
_whisper = None


def utmos(audio: np.ndarray, sr: int) -> float:
    """Mean UTMOS over ~8 s windows (the model was trained on utterance-length clips)."""
    global _utmos
    import librosa
    import torch

    if _utmos is None:
        _utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    wav = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
    win = 8 * 16000
    scores = []
    for i in range(0, max(1, len(wav) - win // 2), win):
        seg = wav[i:i + win]
        if len(seg) < 2 * 16000 or np.abs(seg).max() < 1e-3:
            continue
        with torch.no_grad():
            scores.append(float(_utmos(torch.from_numpy(seg)[None], 16000)[0]))
    return float(np.mean(scores)) if scores else 0.0


def pitch(audio: np.ndarray, sr: int) -> dict:
    import librosa

    wav = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
    f0, voiced, _ = librosa.pyin(wav, fmin=55, fmax=400, sr=16000, frame_length=1024)
    f0 = f0[voiced & ~np.isnan(f0)]
    if not len(f0):
        return {"f0_median_hz": 0.0, "f0_std_semitones": 0.0}
    st = 12 * np.log2(f0 / np.median(f0))
    return {"f0_median_hz": round(float(np.median(f0)), 1), "f0_std_semitones": round(float(st.std()), 2)}


def pace(text: str, audio: np.ndarray, sr: int) -> dict:
    words = len(re.findall(r"[A-Za-z0-9']+", text))
    total = len(audio) / sr
    frame = int(0.03 * sr)
    n = len(audio) // frame
    rms = np.sqrt((audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    speaking = (rms > 0.02 * rms.max()).sum() * 0.03
    return {"seconds": round(total, 2), "wpm": round(words / total * 60, 1),
            "articulation_wpm": round(words / max(speaking, 1e-6) * 60, 1),
            "pause_share": round(1 - speaking / total, 3)}


def wer(text: str, audio: np.ndarray, sr: int) -> float:
    global _whisper
    import librosa
    from faster_whisper import WhisperModel

    if _whisper is None:
        _whisper = WhisperModel("base.en", device="cpu", compute_type="int8")
    wav = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
    segs, _ = _whisper.transcribe(wav, language="en", beam_size=5)
    hyp = re.findall(r"[a-z']+", " ".join(s.text for s in segs).lower())
    ref = re.findall(r"[a-z']+", text.lower())
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return round(d[-1] / max(len(ref), 1), 3)


def all_metrics(text: str, audio: np.ndarray, sr: int) -> dict:
    return {"utmos": round(utmos(audio, sr), 3), **pitch(audio, sr), **pace(text, audio, sr),
            "wer": wer(text, audio, sr)}
