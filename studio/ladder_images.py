"""The nine vertical stills of a ladder Short (options 1-8 and the end plate), one runner job per image.

Each image gets up to four takes. A take is kept only if CLIP sees no person (qa.figure) and no text, the picture matches its
scene (prompt similarity at least 0.25) and its brightness sits in the job's band (option 1 is frame 0, so its band
is the tightest). Up to two passing takes are kept, each upscaled (Real-ESRGAN) with a depth map, so the render can
swap in the second take when two options look too alike.
"""
from __future__ import annotations

import hashlib
import json
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from PIL import Image

from . import ladder, ladder_replay, plates, qa, sdcpp

TAKES = 4
KEEP = 2
MIN_SIMILARITY = 0.25
HARD = ("person", "text", "broken")
# Exposure grade: CPU klein renders night scenes underexposed (raw mean luma 24-61 in the first dry run), so each take
# is graded with one gamma towards its band's centre before review. The grade can only move midtones, so two raw
# checks keep it honest: the take must already hold a real highlight (a focal light), and the graded picture must keep
# a real tonal spread (a flat, muddy frame lifted to the band still fails).
GAMMA_RANGE = (0.3, 1.6)  # 0.4 left frame-0 takes (raw mean 35-43) at luma 82-86, under option 1's band
RAW_P99_FLOOR = 100.0
SPREAD = (40.0, 235.0)
# A misty aerial scene is soft by nature (the test upload's Quetzalcoatlus-over-mist takes scored sharpness 6.5-7.5
# against qa's floor of 8; the median over 73 ladder takes is 15.9), and one missing image sinks the whole Short. So
# when no take passes, the best take whose ONLY problem is mild softness is kept (Real-ESRGAN restores edge detail);
# a take with any other problem (person, text, broken, luma, exposure, off-prompt) is never used.
SOFT_SHARP_FLOOR = 6.0


def shard(jobs: list[dict], k: int, n: int) -> list[dict]:
    return jobs[k::n]


def _seed(key: str, job_id: str, take: int) -> int:
    return int(hashlib.sha1(f"{key}:{job_id}:{take}".encode()).hexdigest()[:8], 16) % 2_000_000_000


def _luma(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)


def expose(img: Image.Image, target: float) -> tuple[Image.Image, dict]:
    """One gamma that moves the mean luma to `target` (bounded by GAMMA_RANGE), plus the raw/graded stats."""
    raw = _luma(img)
    mean = max(float(raw.mean()), 1.0)
    gamma = float(np.clip(np.log(target / 255.0) / np.log(mean / 255.0), *GAMMA_RANGE))
    lut = [round(255.0 * (i / 255.0) ** gamma) for i in range(256)]
    graded = img.convert("RGB").point(lut * 3)
    g = _luma(graded)
    return graded, {"raw_mean": round(mean, 1), "raw_p99": round(float(np.percentile(raw, 99)), 1),
                    "gamma": round(gamma, 3),
                    "spread": round(float(np.percentile(g, 90) - np.percentile(g, 10)), 1)}


def exposure_problems(stats: dict) -> list[str]:
    out = []
    if stats["raw_p99"] < RAW_P99_FLOOR:
        out.append(f"no focal light: raw p99 {stats['raw_p99']:.0f} < {RAW_P99_FLOOR:.0f}")
    lo, hi = SPREAD
    if not lo <= stats["spread"] <= hi:
        out.append(f"tonal spread {stats['spread']:.0f} outside {lo:.0f}-{hi:.0f}")
    return out


def take_problems(verdict, job: dict, figure: float) -> list[str]:
    """`figure` is qa.figure(); it replaces the sleep-doc "person" label, which misreads bright ladder scenes."""
    out = [r for r in verdict.reasons
           if (r.split()[0] in HARD and r.split()[0] != "person") or r.startswith(("too ", "flat", "mushy", "aesth"))]
    if figure > qa.FIGURE_LIMIT:
        out.append(f"person {figure:.2f}")
    if verdict.similarity < MIN_SIMILARITY:
        out.append(f"prompt similarity {verdict.similarity:.3f} < {MIN_SIMILARITY}")
    lo, hi = job["luma"]
    if not lo <= verdict.brightness <= hi:
        out.append(f"luma {verdict.brightness:.0f} outside {lo:.0f}-{hi:.0f}")
    return out


def take_score(verdict, figure: float) -> float:
    """qa's take score with the figure contrast in place of the sleep-doc person label (which misreads these scenes)."""
    return verdict.score() + 4 * (verdict.person - figure)


def soft_only(problems: list[str], sharpness: float) -> bool:
    """True when a refused take failed on nothing but mild softness (the fallback above)."""
    return bool(problems) and all(q.startswith("mushy") for q in problems) and sharpness >= SOFT_SHARP_FLOOR


def generate(p: dict, key: str, out: Path, k: int, n: int, model: str = sdcpp.DEFAULT,
             budget_s: float = 110 * 60, replay: dict | None = None) -> dict:
    jobs = shard(ladder.image_jobs(p), k, n)
    out.mkdir(parents=True, exist_ok=True)
    report = {"shard": k, "of": n, "model": model, "premise": p["id"], "images": []}
    cand = sdcpp.CANDIDATES[model]
    w, h = cand.height, cand.width  # vertical
    start = time.monotonic()
    if replay and replay["premise"] != p["id"]:
        raise ValueError("image replay belongs to a different premise")
    cached_only = bool(replay and replay.get("cached_only"))
    report["cached_only"] = cached_only
    with ExitStack() as stack:
        server = None
        report["load_s"] = 0.0
        for job in jobs:
            entry = {"id": job["id"], "takes": [], "kept": []}
            # Only option images need alternatives for pairwise distinctness; the end is outside that gate.
            wanted = 1 if job["id"] == "end" else KEEP
            soft = []  # (score, take, img, verdict) of soft-only refusals

            def keep(img, verdict, fig, take, fallback=False, cached_from=None):
                stem = f"{job['id']}" if not entry["kept"] else f"{job['id']}-alt{len(entry['kept'])}"
                img.save(out / f"{stem}.png")
                files = plates.save_plate(img, out / stem, vertical=True)
                entry["kept"].append({"plate": files["plate"], "depth": files["depth"], "take": take,
                                      "similarity": round(verdict.similarity, 4),
                                      "luma": round(verdict.brightness, 1), "embedding": qa.pack(qa.embed(img)),
                                      "score": round(take_score(verdict, fig), 3), "fallback": fallback})
                if cached_from:
                    entry["kept"][-1]["cached_from"] = cached_from
            cached = ladder_replay.candidates(replay, job, model, (w, h))
            if cached_only and not cached:
                raise ValueError(f"cache-only replay lacks exact inputs for {job['id']}")
            for img, exposure, take, provenance in cached:
                verdict = qa.review(img, job["scene"])
                fig = qa.figure(img)
                probs = exposure_problems(exposure) + take_problems(verdict, job, fig)
                if provenance.get("editorial_refusal"):
                    probs.append("direct editorial refusal: " + provenance["editorial_refusal"]["reason"])
                entry["takes"].append({"seconds": 0, "problems": probs, "exposure": exposure,
                                       "figure": round(fig, 4), "cached_from": provenance, **verdict.as_dict()})
                if not probs:
                    keep(img, verdict, fig, take, cached_from=provenance)
                elif soft_only(probs, verdict.sharpness):
                    # Reused raw pictures follow the same established mild-softness fallback as new
                    # takes. Retain their exact source instead of repainting already exhausted seeds.
                    soft.append((take_score(verdict, fig), take, img, verdict, fig, provenance))
                if len(entry["kept"]) >= wanted:
                    break
            exhausted = ladder_replay.exhausted(replay, job, model, key, TAKES)
            for take in range(TAKES):
                if len(entry["kept"]) >= wanted or exhausted or cached_only:
                    break
                if replay and replay["source"]["key"] == key and any(
                        r.get("cached_from", {}).get("take") == take
                        and r["cached_from"].get("source_image") == job["id"] for r in entry["takes"]):
                    continue  # never count a cached seed and a deterministic repaint of that seed as two takes
                if take and time.monotonic() - start > budget_s:
                    break
                if server is None:
                    server = stack.enter_context(sdcpp.Server(cand))
                    report["load_s"] = round(server.load_seconds, 1)
                t = time.monotonic()
                raw = sdcpp.to_pil(server.generate(job["prompt"], _seed(key, job["id"], take), w, h))
                img, exposure = expose(raw, sum(job["luma"]) / 2)
                verdict = qa.review(img, job["scene"])
                fig = qa.figure(img)
                probs = exposure_problems(exposure) + take_problems(verdict, job, fig)
                entry["takes"].append({"seconds": round(time.monotonic() - t, 1), "problems": probs,
                                       "exposure": exposure, "figure": round(fig, 4), **verdict.as_dict()})
                print(f"{job['id']} take {take + 1}: {'ok' if not probs else probs}", flush=True)
                if probs:  # keep a small preview of every refused take so a failing run can be diagnosed
                    (out / "refused").mkdir(exist_ok=True)
                    img.resize((216, 384)).save(out / "refused" / f"{job['id']}-t{take + 1}.jpg", quality=80)
                    if soft_only(probs, verdict.sharpness):
                        soft.append((take_score(verdict, fig), take, img, verdict, fig, None))
                    continue
                keep(img, verdict, fig, take)
                if len(entry["kept"]) >= wanted:
                    break
            if not entry["kept"] and soft:
                _, take, img, verdict, fig, provenance = max(soft, key=lambda c: c[0])
                print(f"{job['id']}: keeping soft take {take + 1} (sharpness {verdict.sharpness:.1f})", flush=True)
                keep(img, verdict, fig, take, fallback=True, cached_from=provenance)
            entry["ok"] = bool(entry["kept"])
            report["images"].append(entry)
            (out / f"shard-{k:02d}.json").write_text(json.dumps(report, indent=1))
    report["seconds"] = round(time.monotonic() - start, 1)
    (out / f"shard-{k:02d}.json").write_text(json.dumps(report, indent=1))
    return report
