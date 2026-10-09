"""Voice bake-off on a CPU runner: the same calm passage read by each candidate, scored and timed.

python bench/voices.py kokoro <out_dir>
python bench/voices.py chatterbox <out_dir> <reference_dir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

PASSAGE = [
    "Let the room grow quiet, and let your breathing slow. Tonight we drift back to a sea that existed long before "
    "the first fish, when the water was warm and green, and the sea floor was a soft carpet of microbial mats.",
    "Imagine yourself floating just above that floor. Light comes down in faint, wavering threads. Here and there "
    "a small animal moves across the sand, unhurried, leaving a thin trail behind it. Scientists think these "
    "trails are some of the oldest signs of animals moving with purpose.",
    "There is no rush here. The tide breathes in, and the tide breathes out. A curtain of tiny drifting creatures "
    "glows for a moment, and then fades, like a thought you no longer need to hold.",
]
KOKORO = ["bm_george", "bm_fable", "bm_lewis", "bm_daniel", "bf_emma", "bf_isabella", "bf_alice", "am_michael",
          "af_heart", "af_nicole"]
SPEEDS = [0.85, 0.92]
REFERENCES = ["bm_george", "bf_emma"]


def score(name: str, audio: np.ndarray, sr: int, seconds: float, out: Path, extra: dict) -> dict:
    import voice_metrics

    sf.write(out / f"{name}.wav", audio, sr, subtype="PCM_16")
    m = voice_metrics.all_metrics(" ".join(PASSAGE), audio, sr)
    row = {"name": name, **extra, **m, "synth_s": round(seconds, 1), "rtf": round(seconds / m["seconds"], 3)}
    print(json.dumps(row), flush=True)
    return row


def kokoro(out: Path) -> list[dict]:
    from studio import voice

    rows = []
    for v in KOKORO:
        for speed in SPEEDS:
            t = time.monotonic()
            parts = []
            for p in PASSAGE:
                parts += [voice.paragraph(p, v, speed, check=False).audio,
                          np.zeros(int(voice.PARAGRAPH_GAP * voice.SR), np.float32)]
            audio = np.concatenate(parts[:-1])
            rows.append(score(f"kokoro-{v}-{speed}", audio, voice.SR, time.monotonic() - t, out,
                              {"engine": "kokoro", "voice": v, "speed": speed}))
    refs = out / "refs"
    refs.mkdir(exist_ok=True)
    for v in REFERENCES:
        take = voice.paragraph(PASSAGE[1], v, 0.92, check=False)
        sf.write(refs / f"{v}.wav", take.audio[: 12 * voice.SR], voice.SR, subtype="PCM_16")
    return rows


def chatterbox(out: Path, refs: Path) -> list[dict]:
    import torch
    from chatterbox.tts import ChatterboxTTS

    torch.set_num_threads(4)
    model = ChatterboxTTS.from_pretrained(device="cpu")
    rows = []
    for ref in REFERENCES:
        for exaggeration, cfg in [(0.3, 0.3), (0.45, 0.4)]:
            t = time.monotonic()
            parts = []
            for p in PASSAGE:
                wav = model.generate(p, audio_prompt_path=str(refs / f"{ref}.wav"), exaggeration=exaggeration,
                                     cfg_weight=cfg)
                parts += [wav.squeeze(0).numpy().astype(np.float32), np.zeros(int(1.7 * model.sr), np.float32)]
            audio = np.concatenate(parts[:-1])
            rows.append(score(f"chatterbox-{ref}-{exaggeration}", audio, model.sr, time.monotonic() - t, out,
                              {"engine": "chatterbox", "reference": f"kokoro {ref}", "exaggeration": exaggeration,
                               "cfg_weight": cfg}))
    return rows


def main() -> None:
    mode, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows = kokoro(out) if mode == "kokoro" else chatterbox(out, Path(sys.argv[3]))
    (out / f"{mode}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
