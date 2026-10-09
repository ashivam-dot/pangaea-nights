"""Narration with Kokoro-82M (Apache-2.0): one sentence at a time, slow, with paragraph and chapter pauses.

Each paragraph is checked with faster-whisper (MIT): if words are missing it is re-read sentence by sentence.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

from .speech import speakable, split_sentences

SR = 24000
VOICE = os.environ.get("PANGAEA_VOICE", "af_heart")
SPEED = float(os.environ.get("PANGAEA_SPEED", "0.8"))
TARGET_WPM = 120.0
SENTENCE_GAP = 1.05  # starting pauses; narrate.py rescales them per chapter to reach TARGET_WPM
PARAGRAPH_GAP = 2.3
MIN_HEARD = 0.86
_pipe = None
_whisper = None


@dataclass
class Take:
    audio: np.ndarray
    words: list[dict]  # {"text", "start", "end"} relative to the take
    heard: float = 1.0
    parts: list = None  # per sentence: (audio, words relative to the sentence, pause weight after it)


def kokoro(voice: str = VOICE):
    """British voices (b*) use the British G2P, American voices (a*) the American one."""
    global _pipe
    lang = "b" if voice[:1] == "b" else "a"
    if _pipe is None:
        _pipe = {}
    if lang not in _pipe:
        from kokoro import KPipeline

        _pipe[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
    return _pipe[lang]


def _sentence(text: str, voice: str, speed: float) -> Take:
    pieces, words, t = [], [], 0.0
    for result in kokoro(voice)(speakable(text), voice=voice, speed=speed, split_pattern=None):
        audio = result.audio.numpy().astype(np.float32)
        for tok in result.tokens or []:
            if tok.start_ts is not None and tok.end_ts is not None and tok.text.strip():
                words.append({"text": tok.text.strip(), "start": t + tok.start_ts, "end": t + tok.end_ts})
        pieces.append(audio)
        t += len(audio) / SR
    audio = np.concatenate(pieces) if pieces else np.zeros(1, np.float32)
    return Take(_trim(audio), words)


def _trim(audio: np.ndarray, floor: float = 0.008) -> np.ndarray:
    loud = np.flatnonzero(np.abs(audio) > floor * max(np.abs(audio).max(initial=0.0), 1e-6))
    if not loud.size:
        return audio
    keep = int(0.04 * SR)
    return audio[max(0, loud[0] - keep): loud[-1] + keep]


def whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel

        _whisper = WhisperModel(os.environ.get("PANGAEA_WHISPER", "base.en"), device="cpu", compute_type="int8")
    return _whisper


def _norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", speakable(text).lower().replace("-", " "))


def heard_fraction(expected: str, audio: np.ndarray) -> float:
    """Share of the expected words that Whisper hears, in order (a longest-common-subsequence match)."""
    import librosa

    wav = librosa.resample(audio, orig_sr=SR, target_sr=16000)
    segs, _ = whisper().transcribe(wav, language="en", beam_size=1, vad_filter=False,
                                   condition_on_previous_text=False)
    heard = _norm_words(" ".join(s.text for s in segs))
    want = _norm_words(expected)
    if not want:
        return 1.0
    prev = [0] * (len(heard) + 1)
    for w in want:
        cur = [0]
        for j, h in enumerate(heard):
            cur.append(prev[j] + 1 if w == h or (len(w) > 4 and w[:5] == h[:5]) else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1] / len(want)


def paragraph(text: str, voice: str = VOICE, speed: float = SPEED, check: bool = True) -> Take:
    """One paragraph: sentences joined by calm gaps; re-read once if Whisper misses words."""
    sentences = split_sentences(text)
    for attempt in range(2):
        parts = []
        for s in sentences:
            take = _sentence(s, voice, speed * (1 if attempt == 0 else 0.97))
            parts.append((take.audio, take.words, 1.15 if s.endswith((";", ",")) else 1.0))
        joined, words = join(parts, SENTENCE_GAP)
        heard = heard_fraction(text, joined) if check else 1.0
        if heard >= MIN_HEARD:
            break
    return Take(joined, words, heard, parts)


def join(parts: list, gap: float) -> tuple[np.ndarray, list[dict]]:
    """Sentences joined by pauses of gap x their weight; word times shifted to the joined audio."""
    audio, words, t = [], [], 0.0
    for i, (a, ws, weight) in enumerate(parts):
        words += [{**w, "start": w["start"] + t, "end": w["end"] + t} for w in ws]
        audio.append(a)
        t += len(a) / SR
        if i < len(parts) - 1:
            n = int(gap * weight * SR)
            audio.append(np.zeros(n, np.float32))
            t += n / SR
    return (np.concatenate(audio) if audio else np.zeros(1, np.float32)), words
