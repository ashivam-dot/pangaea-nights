"""Voice an episode across parallel jobs: contiguous chapter groups balanced by word count.

Writes ch<NN>.flac + ch<NN>.json (paragraph spans, word timings, Whisper heard-fractions) and, on shard 0,
trailer<K>.flac + .json for the Shorts captions.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import voice


def chapter_shards(episode: dict, n: int) -> list[list[int]]:
    chapters = episode["chapters"]
    words = [sum(len(p.split()) for p in c["paragraphs"]) for c in chapters]
    total, groups, cur, acc = sum(words), [], [], 0
    for ch, w in zip(chapters, words):
        cur.append(ch["n"])
        acc += w
        if acc >= total * (len(groups) + 1) / n and len(groups) < n - 1:
            groups.append(cur)
            cur = []
    groups.append(cur)
    return groups + [[] for _ in range(n - len(groups))]


MIN_PAUSE_SCALE, MAX_PAUSE_SCALE = 0.75, 2.2


def pause_scale(words: int, speech_s: float, sentence_weights: float, paragraph_gaps: int) -> float:
    """How much to stretch the default pauses so the block reads at TARGET_WPM (bounded, so pauses stay natural)."""
    base = sentence_weights * voice.SENTENCE_GAP + paragraph_gaps * voice.PARAGRAPH_GAP
    if base <= 0:
        return 1.0
    want = words / voice.TARGET_WPM * 60
    return min(MAX_PAUSE_SCALE, max(MIN_PAUSE_SCALE, (want - speech_s) / base))


def _voice_block(paragraphs: list[str], voice_name: str, speed: float) -> tuple[np.ndarray, dict]:
    takes = [voice.paragraph(text, voice_name, speed) for text in paragraphs]
    speech = sum(len(a) for t in takes for a, _, _ in t.parts) / voice.SR
    weights = sum(w for t in takes for _, _, w in t.parts[:-1])
    n_words = sum(len(p.split()) for p in paragraphs)
    scale = pause_scale(n_words, speech, weights, len(takes) - 1)
    audio, words, spans, t = [], [], [], 0.0
    for i, take in enumerate(takes):
        para, para_words = voice.join(take.parts, voice.SENTENCE_GAP * scale)
        spans.append({"start": round(t, 3), "end": round(t + len(para) / voice.SR, 3),
                      "heard": round(take.heard, 3)})
        words += [{"text": w["text"], "start": round(w["start"] + t, 3), "end": round(w["end"] + t, 3)}
                  for w in para_words]
        audio.append(para)
        t += len(para) / voice.SR
        if i < len(takes) - 1:
            n = int(voice.PARAGRAPH_GAP * scale * voice.SR)
            audio.append(np.zeros(n, np.float32))
            t += n / voice.SR
    joined = np.concatenate(audio) if audio else np.zeros(1, np.float32)
    peak = float(np.abs(joined).max(initial=0.0))
    if peak > 0.98:
        joined = joined * (0.98 / peak)
    duration = len(joined) / voice.SR
    return joined, {"duration": round(duration, 3), "paragraphs": spans, "words": words,
                    "pause_scale": round(scale, 3), "wpm": round(n_words / duration * 60, 1)}


def narrate(episode: dict, out: Path, k: int, n: int, voice_name: str = voice.VOICE,
            speed: float = voice.SPEED) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    mine = chapter_shards(episode, n)[k]
    report = {"shard": k, "voice": voice_name, "speed": speed, "chapters": [], "trailers": []}
    start = time.monotonic()
    for ch in episode["chapters"]:
        if ch["n"] not in mine:
            continue
        t = time.monotonic()
        audio, info = _voice_block(ch["paragraphs"], voice_name, speed)
        sf.write(str(out / f"ch{ch['n']:02d}.flac"), audio, voice.SR, subtype="PCM_16")
        (out / f"ch{ch['n']:02d}.json").write_text(json.dumps(info))
        words = sum(len(p.split()) for p in ch["paragraphs"])
        row = {"n": ch["n"], "seconds": info["duration"], "synth_s": round(time.monotonic() - t, 1),
               "wpm": round(words / info["duration"] * 60, 1),
               "min_heard": min(p["heard"] for p in info["paragraphs"])}
        report["chapters"].append(row)
        print(f"voice ch{ch['n']:02d}: {row}", flush=True)
    if k == 0:
        for i, tr in enumerate(episode["trailers"]):
            audio, info = _voice_block([tr["narration"]], voice_name, speed)
            sf.write(str(out / f"trailer{i}.flac"), audio, voice.SR, subtype="PCM_16")
            (out / f"trailer{i}.json").write_text(json.dumps(info))
            report["trailers"].append({"k": i, "seconds": info["duration"],
                                       "heard": info["paragraphs"][0]["heard"]})
    report["seconds"] = round(time.monotonic() - start, 1)
    (out / f"voice-{k:02d}.json").write_text(json.dumps(report, indent=1))
    return report
