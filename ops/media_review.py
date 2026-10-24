"""Semantic review of the actual rendered pictures AND audio, bound to the original MP4.

The reviewer receives the complete clip and complete mixed soundtrack. This is an AI editorial review,
not a claim that a person listened. Missing media, unavailable quota or an incomplete verdict holds release.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import requests

API = "https://generativelanguage.googleapis.com/v1beta/models"
# Free-tier video/audio input and output verified in the official pricing table on 24 October 2026.
# Do not silently adopt a newly listed paid-only model or change billing when quota is unavailable.
FREE_MODELS = ("gemini-3.8-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite")
CHECKS = ("speech_clear", "pronunciation_ok", "music_balance_ok", "captions_readable", "visuals_match",
          "continuity_ok", "opening_clear", "complete_payoff", "no_material_factual_contradiction")
SCHEMA = {
    "type": "object", "properties": {
        "decision": {"type": "string", "enum": ["pass", "hold"]},
        "audio_listened": {"type": "boolean"}, "video_viewed": {"type": "boolean"},
        "heard_opening": {"type": "string"}, "heard_ending": {"type": "string"},
        "checks": {"type": "object", "properties": {k: {"type": "boolean"} for k in CHECKS},
                   "required": list(CHECKS)},
        "segments": {"type": "array", "items": {"type": "object", "properties": {
            "part": {"type": "string", "enum": ["opening", "middle", "ending"]},
            "seconds": {"type": "number"}, "seen": {"type": "string"}, "heard": {"type": "string"}},
            "required": ["part", "seconds", "seen", "heard"]}},
        "issues": {"type": "array", "items": {"type": "object", "properties": {
            "seconds": {"type": "number"}, "severity": {"type": "string", "enum": ["minor", "blocker"]},
            "problem": {"type": "string"}, "repair": {"type": "string"}},
            "required": ["seconds", "severity", "problem", "repair"]}}},
    "required": ["decision", "audio_listened", "video_viewed", "heard_opening", "heard_ending", "checks",
                 "segments", "issues"]}


class ReviewHold(RuntimeError):
    pass


def digest(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def brief_digest(brief: dict) -> str:
    return hashlib.sha256(json.dumps(brief, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def verdict_problems(verdict: dict, duration: float) -> list[str]:
    problems = []
    if verdict.get("decision") != "pass" or verdict.get("audio_listened") is not True \
            or verdict.get("video_viewed") is not True:
        problems.append("the audiovisual reviewer did not approve both media streams")
    for key in CHECKS:
        if verdict.get("checks", {}).get(key) is not True:
            problems.append(f"review: {key} is not confirmed")
    if any(not str(verdict.get(k, "")).strip() for k in ("heard_opening", "heard_ending")):
        problems.append("missing words actually heard at the opening or ending")
    segments = {s.get("part"): s for s in verdict.get("segments", [])}
    for part, lo, hi in (("opening", 0, min(5, duration / 4)),
                         ("middle", duration / 4, duration * 0.8),
                         ("ending", duration * 0.8, duration)):
        row = segments.get(part, {})
        t = row.get("seconds")
        if not isinstance(t, (int, float)) or not lo <= t <= hi or not row.get("seen") or not row.get("heard"):
            problems.append(f"missing rendered {part} observation")
    problems.extend(f"{i.get('seconds')}s: {i.get('problem')}" for i in verdict.get("issues", [])
                    if i.get("severity") != "minor")
    return problems


def receipt_problems(receipt: dict, mp4: Path, brief: dict) -> list[str]:
    if receipt.get("schema") != 1 or receipt.get("video_sha256") != digest(mp4) \
            or receipt.get("brief_sha256") != brief_digest(brief):
        return ["the review receipt is missing or belongs to different video/brief bytes"]
    if not receipt.get("model") or not receipt.get("review_video_sha256") or not receipt.get("review_audio_sha256"):
        return ["the review receipt lacks model or media provenance"]
    return verdict_problems(receipt.get("verdict", {}), receipt.get("duration", 0))


def _models(key: str) -> list[str]:
    try:
        r = requests.get(API, headers={"x-goog-api-key": key}, params={"pageSize": 1000}, timeout=30)
    except requests.RequestException as exc:
        raise ReviewHold("review model discovery network unavailable") from exc
    if r.status_code != 200:
        raise ReviewHold(f"review model discovery unavailable (HTTP {r.status_code})")
    try:
        found = [m["name"].split("/")[-1] for m in r.json().get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", [])
                 and m["name"].split("/")[-1] in FREE_MODELS]
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ReviewHold("review model discovery returned an unreadable response") from exc
    def rank(name):
        v = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        return ("lite" in name, "latest" in name, -float(v.group(1)) if v else 0, "preview" in name)
    return sorted(set(found), key=rank)


def review(mp4: Path, brief: dict, folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "av-review.json"
    if target.exists():
        old = json.loads(target.read_text())
        if old.get("schema") == 1 and old.get("video_sha256") == digest(mp4) \
                and old.get("brief_sha256") == brief_digest(brief) and old.get("model") \
                and old.get("review_video_sha256") and old.get("review_audio_sha256"):
            # Preserve a refusal for unchanged bytes too. The caller applies the verdict; repairing
            # the actual video/brief is the only way to request a new editorial decision.
            return old
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ReviewHold("no configured audiovisual reviewer; release remains held")
    info = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(mp4)],
                          check=True, capture_output=True, text=True)
    duration = float(json.loads(info.stdout)["format"]["duration"])
    if not 10 <= duration <= 180:
        raise ReviewHold("this reviewer accepts complete Shorts only (10–180 seconds)")
    video, audio = folder / "review-video.mp4", folder / "review-audio.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp4), "-vf", "scale=540:-2,fps=4", "-c:v",
                    "libx264", "-preset", "fast", "-crf", "27", "-c:a", "aac", "-b:a", "64k", str(video)],
                   check=True, timeout=180)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp4), "-vn", "-ac", "1", "-ar", "24000",
                    "-c:a", "pcm_s16le", str(audio)], check=True, timeout=60)
    if video.stat().st_size + audio.stat().st_size > 13_000_000:
        raise ReviewHold("review payload exceeds the bounded inline-media allowance")
    prompt = ("You are the release editor. Watch the COMPLETE attached rendered Short and listen to its COMPLETE "
              "mixed soundtrack (also supplied separately). Inspect opening, representative middle beats and ending. "
              "Judge what is ACTUALLY visible and audible, not merely the script or prior gate summaries. "
              "Give timestamps and concrete heard/seen observations. Transcribe the first and last spoken sentence "
              "in the spoken language. Check natural pronunciation, no clipped/rushed/garbled words, voice/music "
              "balance, captions at phone size, beat-to-visual alignment, object/character continuity, an immediately "
              "clear opening and a complete payoff. For science reject contradictions between claimed habitat/depth "
              "and rendered lighting/animals. Respect uncertainty and distinguish imagined scenarios from facts. "
              "For fiction respect the supplied world and stable props. Minor aesthetic preferences alone do not "
              "block; any substantive defect does. If a media stream cannot be examined, HOLD. Never claim human "
              "review. Return the required JSON. Editorial brief and pinned facts:\n" +
              json.dumps(brief, ensure_ascii=False))
    parts = [{"text": prompt},
             {"inlineData": {"mimeType": "video/mp4", "data": base64.b64encode(video.read_bytes()).decode()}},
             {"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(audio.read_bytes()).decode()}}]
    body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {
        "temperature": 0, "maxOutputTokens": 8192, "responseMimeType": "application/json",
        "responseJsonSchema": SCHEMA}}
    errors = []
    for model in _models(key)[:5]:
        try:
            r = requests.post(f"{API}/{model}:generateContent", headers={"x-goog-api-key": key}, json=body,
                              timeout=240)
            if r.status_code != 200:
                errors.append(f"{model}: HTTP {r.status_code}")
                if r.status_code in (401, 403):
                    break
                continue
            candidate = r.json().get("candidates", [{}])[0]
            if candidate.get("finishReason") != "STOP":
                errors.append(f"{model}: incomplete review")
                continue
            answer = json.loads("".join(p.get("text", "") for p in candidate.get("content", {}).get("parts", [])
                                        if not p.get("thought")))
            receipt = {"schema": 1, "model": model, "duration": duration, "video_sha256": digest(mp4),
                       "brief_sha256": brief_digest(brief), "review_video_sha256": digest(video),
                       "review_audio_sha256": digest(audio), "verdict": answer}
            target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
            # A negative editorial verdict is final for these exact bytes; never shop among models for a pass.
            return receipt
        except (requests.RequestException, ValueError, IndexError, TypeError):
            errors.append(f"{model}: unavailable or malformed review")
    raise ReviewHold("audiovisual review unavailable: " + "; ".join(errors))
