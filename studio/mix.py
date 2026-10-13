"""The sound of an episode: narration over a quiet code-written score and a nature ambience loop.

Each stem is first set to a known loudness, then placed: the bed sits well under the voice, swells gently once the
narration ends, and the last minute fades out. A two-pass linear loudnorm brings the whole to -20 LUFS integrated
with -3 dBTP peaks, and an EBU R128 scan confirms nothing jumps out (max momentary within a few LU of integrated).
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

TARGET_I = -20.0
TARGET_TP = -3.0
# AAC encoding adds up to ~1 dB of inter-sample overshoot, so the FLAC master is limited well under TARGET_TP.
LIMIT_TP = -4.5
LEVELS = {"narration": -20.0, "score": -39.0, "ambience": -36.0}
TAIL_LIFT_DB = 6.0
MAX_MOMENTARY_OVER = 8.0


def _run(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stderr


def integrated(path: Path, extra: list[str] | None = None) -> dict:
    """EBU R128 summary: integrated (I), loudness range (LRA), true peak and the loudest momentary value."""
    err = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), *(extra or []), "-af",
                "ebur128=peak=true:framelog=info", "-f", "null", "-"])
    moments = [float(m) for m in re.findall(r"\sM:\s*(-?[\d.]+)", err)]
    summary = err[err.rfind("Summary:"):]
    get = lambda key: float(re.search(key + r":\s*(-?[\d.]+)", summary).group(1))  # noqa: E731
    if not moments:
        raise RuntimeError("ebur128 printed no momentary loudness values")
    return {"I": get("I"), "LRA": get("LRA"), "true_peak": get("Peak"), "max_momentary": max(moments)}


def _gain_to(path: Path, target: float) -> float:
    err = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128", "-f", "null", "-"])
    i = float(re.search(r"I:\s*(-?[\d.]+) LUFS", err[err.rfind("Summary:"):]).group(1))
    return target - i


def narration_track(placements: list[dict], voice_dir: Path, total: float, out: Path) -> Path:
    """All chapters on one 48 kHz mono track at their timeline offsets."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    parts = []
    for i, p in enumerate(placements):
        cmd += ["-i", str(voice_dir / p["file"])]
        ms = int(round(p["at"] * 1000))
        parts.append(f"[{i}:a]aresample=48000,highpass=f=60,adelay={ms}:all=1[n{i}]")
    graph = ";".join(parts) + ";" + "".join(f"[n{i}]" for i in range(len(placements)))
    graph += f"amix=inputs={len(placements)}:normalize=0:duration=longest,apad,atrim=0:{total:.3f}[out]"
    subprocess.run(cmd + ["-filter_complex", graph, "-map", "[out]", "-ac", "1", "-c:a", "flac", str(out)],
                   check=True)
    return out


def mix(narration: Path, score: Path, ambience_loop: Path, total: float, narration_end: float, out: Path) -> dict:
    gains = {"narration": _gain_to(narration, LEVELS["narration"]), "score": _gain_to(score, LEVELS["score"]),
             "ambience": _gain_to(ambience_loop, LEVELS["ambience"])}
    lift = 10 ** (TAIL_LIFT_DB / 20)
    swell = f"if(lt(t,{narration_end:.2f}),1,1+({lift - 1:.4f})*min(1,(t-{narration_end:.2f})/25))"
    fade = f"afade=t=out:st={max(0.0, total - 60):.2f}:d=60"
    graph = (
        f"[0:a]volume={gains['narration']:.2f}dB,acompressor=threshold=-26dB:ratio=2:attack=15:release=250,"
        f"aformat=channel_layouts=stereo[v];"
        f"[1:a]aresample=48000,volume={gains['score']:.2f}dB,volume='{swell}':eval=frame[s];"
        f"[2:a]aresample=48000,volume={gains['ambience']:.2f}dB,volume='{swell}':eval=frame,"
        f"atrim=0:{total:.3f}[a];"
        f"[v][s][a]amix=inputs=3:normalize=0:duration=first,afade=t=in:d=4,{fade}[m]"
    )
    pre = out.with_name("premix.flac")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(narration), "-i", str(score),
                    "-stream_loop", "-1", "-i", str(ambience_loop), "-filter_complex", graph, "-map", "[m]",
                    "-c:a", "flac", str(pre)], check=True)
    measure = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(pre), "-af",
                    f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA=11:print_format=json", "-f", "null", "-"])
    m = json.loads(measure[measure.rfind("{"):measure.rfind("}") + 1])
    norm = (f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
            f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:"
            f"linear=true")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(pre), "-af", norm + "," + peak_limiter(),
                    "-c:a", "flac", str(out)], check=True)
    pre.unlink()
    stats = integrated(out)
    stats["momentary_over_integrated"] = round(stats["max_momentary"] - stats["I"], 2)
    stats["ok"] = (abs(stats["I"] - TARGET_I) <= 1.0 and stats["true_peak"] <= TARGET_TP + 0.5
                   and stats["momentary_over_integrated"] <= MAX_MOMENTARY_OVER)
    return {"gains_db": {k: round(v, 2) for k, v in gains.items()}, "loudness": stats}


def peak_limiter(limit_db: float = LIMIT_TP) -> str:
    """A brickwall limiter run at 4x the sample rate, so it catches inter-sample (true) peaks, then back to 48 kHz."""
    return (f"aresample=192000,alimiter=limit={10 ** (limit_db / 20):.4f}:attack=1:release=80:level=disabled,"
            "aresample=48000")


def to_aac(src: Path, out: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:a", "aac", "-b:a", "192k",
                    "-ar", "48000", str(out)], check=True)
    return out
