"""End-to-end smoke test of the media stages without any model: synthetic plates and a synthetic 'voice'.

Exercises ambience, score, mix, timeline, render segments, concat/mux, QA, thumbnail and a Short in a few minutes.
Run in CI (tests.yml); needs ffmpeg and tesseract.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from studio import ambience, finish, mix, plates, render, score, shorts, thumbnail, timeline  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "smoke-out")


def fake_plate(path: Path, seed: int, size) -> None:
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([30 + 40 * np.sin(xx / 170 + seed), 50 + 30 * np.cos(yy / 120), 80 + 40 * np.sin((xx + yy) / 260)],
                    axis=2)
    for _ in range(25):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(20, 160)
        base += (np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)) * rng.uniform(20, 90))[..., None]
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    img.save(path.with_suffix(".jpg"), quality=92)
    depth = ((yy / h) * 0.7 + 0.3 * np.exp(-((xx - w / 2) ** 2) / (2 * (w / 5) ** 2))).astype(np.float32)
    Image.fromarray((depth[::2, ::2] * 65535).astype(np.uint16)).save(path.with_suffix(".depth.png"))
    if path.name == "thumb":
        img.resize((1024, 576)).save(path.with_suffix(".png"))


def fake_voice(seconds: float, sr: int = 24000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    syll = (np.sin(2 * np.pi * 3.2 * t) > 0.2) * (np.sin(2 * np.pi * 0.25 * t) > -0.6)
    tone = np.sin(2 * np.pi * (110 + 15 * np.sin(2 * np.pi * 0.3 * t)) * t) * 0.3
    return (tone * syll).astype(np.float32)


def main() -> None:
    images, voice, reps = OUT / "images", OUT / "voice", OUT / "reports"
    for d in (images, voice, reps):
        d.mkdir(parents=True, exist_ok=True)
    ep = {"id": "smoke-1", "series": "deep-sea", "title_core": "The Deep Sea at Night", "summary": "A test.",
          "chapters": [{"n": n, "title": f"The Quiet Water {n}", "scenes": ["s"] * 3, "warnings": []}
                       for n in (1, 2)],
          "tail_scenes": ["t"], "thumbnail": {"words": "Deep Sea Night"}, "writer": "stub", "publishable": False,
          "trailers": [{"hook": "The quiet deep", "narration": "x", "scenes": ["a", "b"], "chapter": 1}]}
    ids = [f"c{n:02d}s{j:02d}" for n in (1, 2) for j in range(3)] + ["tail00", "thumb"]
    for i, name in enumerate(ids):
        fake_plate(images / name, i, plates.PLATE)
    for j in range(2):
        fake_plate(images / f"t0s{j:02d}", 50 + j, plates.PLATE_V)
    (reps / "shard-00.json").write_text(json.dumps({"images": [{"id": i, "plate": f"{i}.jpg"} for i in ids]}))
    for n in (1, 2):
        sf.write(str(voice / f"ch{n:02d}.flac"), fake_voice(40), 24000)
        (voice / f"ch{n:02d}.json").write_text(json.dumps({"duration": 40.0}))
    sf.write(str(voice / "trailer0.flac"), fake_voice(24), 24000)
    words = [{"text": w, "start": 0.4 * i, "end": 0.4 * i + 0.35} for i, w in
             enumerate("In the deep water the light fades slowly and the animals drift in the dark.".split())]
    (voice / "trailer0.json").write_text(json.dumps({"duration": 24.0, "words": words, "paragraphs": [{"heard": 1}]}))

    tl = timeline.build(ep, voice, reps, tail_minutes=1.0, render_shards=3)
    (OUT / "timeline.json").write_text(json.dumps(tl, indent=1))
    amb = ambience.build_loop("underwater", OUT / "ambience.wav", OUT / "amb-cache")
    print("ambience:", json.dumps(amb))
    score.render(OUT / "score.wav", [c["start"] for c in tl["chapters"]], tl["seconds"], ep["id"])
    nar = mix.narration_track(tl["narration"], voice, tl["seconds"], OUT / "narration.flac")
    end = tl["chapters"][-1]["narration_start"] + tl["chapters"][-1]["narration_seconds"]
    result = mix.mix(nar, OUT / "score.wav", OUT / "ambience.wav", tl["seconds"], end, OUT / "mix.flac")
    print("mix:", json.dumps(result))
    mix.to_aac(OUT / "mix.flac", OUT / "mix.m4a")
    segs = OUT / "segments"
    segs.mkdir(exist_ok=True)
    for k in range(len(tl["render_shards"])):
        print("render:", json.dumps(render.segment(tl, k, images, segs / f"seg{k:02d}.mp4", OUT / "stills")))
    final = finish.assemble(tl, segs, OUT / "mix.m4a", OUT / "episode.mp4")
    thumb = thumbnail.build(images / "thumb.png", ep["thumbnail"]["words"].split(), OUT / "thumbnail.jpg",
                            upscale=False)
    print("thumbnail:", json.dumps(thumb))
    made = [shorts.build(ep, 0, voice, images, OUT, "underwater")]
    print("short:", json.dumps(made))
    qa = finish.qa(ep, tl, final, OUT / "checks", thumb, made)
    print("qa:", json.dumps(qa, indent=1))
    problems = [p for p in qa["problems"] if "thumbnail" not in p] if not thumb["ok"] else qa["problems"]
    assert not problems, problems
    assert made[0]["ok"], made
    info = finish.probe(OUT / "short0.mp4")
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1080, 1920), v
    assert 30 <= float(info["format"]["duration"]) <= 56
    print("smoke ok")


if __name__ == "__main__":
    main()
