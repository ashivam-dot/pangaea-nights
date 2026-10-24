"""The narrator of a ladder Short: Kokoro am_michael by default, about 165 words a minute, every line checked back
with Whisper (word error rate at most 8%). Word times come from Kokoro's own token timestamps, for the captions.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from . import ladder

VOICE = os.environ.get("PANGAEA_LADDER_VOICE", "am_michael")
TARGET_WPM = 165.0
PACE_TOLERANCE = 3.0  # wpm: corrections continue until the pace is this close to TARGET_WPM
PACE_TAKES = 3
SPEED = (0.85, 1.30)
MAX_WER = 0.08


def line_id(segment: dict) -> str:
    """hook, o1..o8, verdict, cta: the name of one narration line (and of its wav)."""
    return segment["kind"] if segment["kind"] in ("hook", "verdict", "cta") else f"o{segment['n']}"


def wer(expected: str, heard: str) -> float:
    """Word error rate: word-level edit distance over the expected length, on spoken forms ('3' = 'three')."""
    from .voice import _norm_words

    # Whisper writes American unit spellings and sometimes hyphenates option/quantity pairs ("4-50").
    # Those transcript characters are punctuation, not an extra spoken "to". Preserve scripted ranges by
    # expanding the expected text first; never equate a wrong number, unit scale, negation, or homophone.
    from .speech import speakable

    def canonical(text):
        text = re.sub(r"\bsuper[ -]heated\b", "superheated", text, flags=re.I)
        units = {"metre": "meter", "metres": "meter", "meters": "meter",
                 "kilometre": "kilometer", "kilometres": "kilometer", "kilometers": "kilometer"}
        return [units.get(w, w) for w in _norm_words(text)]

    ref = canonical(speakable(expected))
    hyp = canonical(re.sub(r"(?<=\d)[-–—](?=\d)", " ", heard))
    if not ref:
        return 0.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def wpm(segments: dict) -> float:
    words = sum(len(re.findall(r"[A-Za-z0-9']+", s["text"])) for s in segments.values())
    seconds = sum(s["duration"] for s in segments.values())
    return 60.0 * words / max(seconds, 1e-6)


def _transcribe(audio: np.ndarray, sr: int) -> str:
    import librosa

    from .voice import whisper

    wav = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    segs, _ = whisper().transcribe(wav, language="en", beam_size=1, vad_filter=False,
                                   condition_on_previous_text=False)
    return " ".join(s.text for s in segs)


def _take(lines: list[dict], speed: float) -> dict:
    from .voice import SR, _sentence

    out = {}
    for line in lines:
        take = _sentence(line["text"], VOICE, speed)
        out[line["id"]] = {"text": line["text"], "audio": take.audio, "words": take.words,
                           "duration": round(len(take.audio) / SR, 3)}
    return out


def narrate(p: dict, out: Path) -> dict:
    """voice/<line>.wav for hook, o1..o8, verdict and cta, and voice/voice.json with durations, word times, pace
    and the Whisper check."""
    import soundfile as sf

    from .voice import SR

    out.mkdir(parents=True, exist_ok=True)
    lines = [{"id": line_id(s), "text": s["text"]} for s in ladder.segments(p)]
    speed = 1.0
    segs = _take(lines, speed)
    pace = wpm(segs)
    # Corrections toward the target pace (Kokoro's pace is close to linear in speed). The bank check
    # (shorts_ladder.fit_problems) assumes no slower than the wpm gate's floor, so a slow read must not be accepted.
    for _ in range(PACE_TAKES - 1):
        if abs(pace - TARGET_WPM) <= PACE_TOLERANCE:
            break
        new = min(max(speed * TARGET_WPM / pace, SPEED[0]), SPEED[1])
        if new == speed:
            break
        speed = new
        segs = _take(lines, speed)
        pace = wpm(segs)
    for attempt in range(2):
        for s in segs.values():
            s["heard"] = _transcribe(s["audio"], SR).strip()
            s["wer"] = round(wer(s["text"], s["heard"]), 3)
        bad = [k for k, s in segs.items() if s["wer"] > MAX_WER]
        if not bad or attempt:
            break
        retake = _take([ln for ln in lines if ln["id"] in bad], speed * 0.96)
        for k, s in retake.items():
            segs[k] = s
    words = sum(len(re.findall(r"[a-z0-9']+", s["text"].lower())) for s in segs.values())
    errors = sum(s["wer"] * len(re.findall(r"[a-z0-9']+", s["text"].lower())) for s in segs.values())
    for k, s in segs.items():
        sf.write(str(out / f"{k}.wav"), s.pop("audio"), SR)
    report = {"voice": VOICE, "speed": round(speed, 3), "wpm": round(wpm(segs), 1), "wer": round(errors / words, 3),
              "segments": segs, "durations": {k: s["duration"] for k, s in segs.items()}}
    (out / "voice.json").write_text(json.dumps(report, indent=1))
    return report
