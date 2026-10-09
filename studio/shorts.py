"""Vertical trailers (Shorts) for an episode: 30-55 s, word-timed captions, a hook, and an end card that points
to the full episode."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import ambience, config, render, score

W, H = 1080, 1920
LEAD = 0.8
END_CARD = 3.5
MIN_S, MAX_S = 30.0, 55.0


def _chunks(words: list[dict], max_words: int = 4, max_chars: int = 26) -> list[list[dict]]:
    out, cur = [], []
    for w in words:
        if not w["text"].strip() or not any(ch.isalnum() for ch in w["text"]):
            if cur and w["text"].strip() in ".,;:?!":
                cur[-1] = {**cur[-1], "text": cur[-1]["text"] + w["text"].strip()}
            continue
        cur.append(w)
        if len(cur) >= max_words or len(" ".join(x["text"] for x in cur)) >= max_chars or \
                w["text"][-1:] in ".,;:?!":
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _caption_layers(words: list[dict]) -> list[tuple[float, float, tuple]]:
    rows = []
    for chunk in _chunks(words):
        text = " ".join(w["text"] for w in chunk)
        img = render.text_block([(text, render.fit_font(text, 70, W - 120, italic=False), 1.0)], W, shadow=1.0)
        rows.append((chunk[0]["start"] + LEAD, chunk[-1]["end"] + LEAD + 0.15, render.layer_from(img)))
    return rows


def _end_card(title_core: str) -> tuple:
    img = render.text_block([("Full episode on", render.font(50, weight=500), 0.85),
                             (config.CHANNEL_NAME, render.font(84, weight=600), 1.0),
                             (title_core, render.fit_font(title_core, 56, W - 140), 0.9)], W, gap=22, shadow=1.0)
    return render.layer_from(img)


def _mix(narration: Path, total: float, episode_id: str, amb_key: str, work: Path) -> Path:
    score_path = work / "score.wav"
    score.render(score_path, [0.0], total, episode_id + "-short")
    loop = work / "amb.wav"
    import soundfile as sf
    sf.write(str(loop), ambience.synth(ambience.SYNTH.get(amb_key, "waves"), seconds=60), ambience.SR)
    out = work / "mix.wav"
    graph = (f"[0:a]aresample=48000,adelay={int(LEAD * 1000)}:all=1,aformat=channel_layouts=stereo,apad[v];"
             "[1:a]aresample=48000,volume=0.35[s];[2:a]volume=0.25[a];"
             f"[v][s][a]amix=inputs=3:normalize=0:duration=shortest,atrim=0:{total:.3f},"
             f"afade=t=out:st={total - 1.5:.3f}:d=1.5,loudnorm=I=-16:TP=-2:LRA=7,aresample=48000[m]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(narration), "-i", str(score_path),
                    "-stream_loop", "-1", "-i", str(loop), "-filter_complex", graph, "-map", "[m]", "-ac", "2",
                    str(out)], check=True)
    return out


def build(episode: dict, k: int, voice_dir: Path, image_dir: Path, work: Path, amb_key: str) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    tr = episode["trailers"][k]
    info = json.loads((voice_dir / f"trailer{k}.json").read_text())
    total = LEAD + info["duration"] + 0.6 + END_CARD
    if total < MIN_S:
        total = MIN_S
    if total > MAX_S + 2:
        return {"k": k, "ok": False, "reason": f"trailer narration too long ({info['duration']:.0f}s)"}
    ids = [f"t{k}s{j:02d}" for j in range(len(tr["scenes"]))]
    ids = [i for i in ids if (image_dir / f"{i}.jpg").exists()]
    if not ids:
        return {"k": k, "ok": False, "reason": "no usable vertical images"}
    cam, grade, plates = render.Camera(W, H), render.Grade(W, H, seed=40 + k, darken=0.97), render.Plates(image_dir)
    captions = _caption_layers(info["words"])
    hook = render.layer_from(render.text_block(
        [(tr["hook"], render.fit_font(tr["hook"], 92, W - 120), 1.0)], W, shadow=1.0))
    end = _end_card(episode["title_core"])
    card_start = total - END_CARD
    span = card_start / len(ids)
    video = work / f"short{k}.video.mp4"
    enc = render.Encoder(video, W, H, crf=19)
    frames = int(round(total * render.FPS))
    stills = []
    for f in range(frames):
        t = f / render.FPS
        j = min(int(t // span), len(ids) - 1)
        shot = lambda jj: cam.shoot(plates(ids[jj]), (t - jj * span) / (span + 1.2), render.MOTIONS[(jj + k) % 6])  # noqa: E731
        frame = shot(j)
        since = t - j * span
        if j > 0 and since < 1.2:
            a = render.ease(since / 1.2)
            frame = cv2.addWeighted(frame, a, shot(j - 1), 1 - a, 0)
        if t >= card_start:
            frame = (frame.astype(np.float32) * (1 - 0.55 * render.ease((t - card_start) / 0.8))).astype(np.uint8)
        frame = grade.apply(frame, f, min(1.0, render.ease(t / 0.4)))
        if t < 4.0:
            render.blit(frame, hook, 0, int(H * 0.16), render.ease((4.0 - t) / 0.6))
        if t >= card_start:
            render.blit(frame, end, 0, int(H * 0.40), render.ease((t - card_start) / 0.8))
        else:
            for s0, s1, layer in captions:
                if s0 <= t < s1:
                    render.blit(frame, layer, 0, int(H * 0.68))
                    break
        enc.write(frame)
        if f in (int(1.5 * render.FPS), frames // 2, frames - render.FPS):
            stills.append(frame.copy())
    enc.close()
    audio = _mix(voice_dir / f"trailer{k}.flac", total, episode["id"], amb_key, work)
    out = work / f"short{k}.mp4"
    render.mux(video, audio, out)
    video.unlink()
    from PIL import Image
    for i, s in enumerate(stills):
        Image.fromarray(s).save(work / f"short{k}-still{i}.jpg", quality=88)
    return {"k": k, "ok": True, "path": str(out), "seconds": round(total, 2), "hook": tr["hook"],
            "chapter": tr.get("chapter")}
