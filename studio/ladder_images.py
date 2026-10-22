"""The nine vertical stills of a ladder Short (options 1-8 and the end plate), one runner job per image.

Each image gets up to four takes. A take is kept only if CLIP sees no person and no text, the picture matches its
scene (prompt similarity at least 0.25) and its brightness sits in the job's band (option 1 is frame 0, so its band
is the tightest). Up to two passing takes are kept, each upscaled (Real-ESRGAN) with a depth map, so the render can
swap in the second take when two options look too alike.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import ladder, plates, qa, sdcpp

TAKES = 4
KEEP = 2
MIN_SIMILARITY = 0.25
HARD = ("person", "text", "broken")


def shard(jobs: list[dict], k: int, n: int) -> list[dict]:
    return jobs[k::n]


def _seed(key: str, job_id: str, take: int) -> int:
    return int(hashlib.sha1(f"{key}:{job_id}:{take}".encode()).hexdigest()[:8], 16) % 2_000_000_000


def take_problems(verdict, job: dict) -> list[str]:
    out = [r for r in verdict.reasons if r.split()[0] in HARD or r.startswith(("too ", "flat", "mushy", "aesth"))]
    if verdict.similarity < MIN_SIMILARITY:
        out.append(f"prompt similarity {verdict.similarity:.3f} < {MIN_SIMILARITY}")
    lo, hi = job["luma"]
    if not lo <= verdict.brightness <= hi:
        out.append(f"luma {verdict.brightness:.0f} outside {lo:.0f}-{hi:.0f}")
    return out


def generate(p: dict, key: str, out: Path, k: int, n: int, model: str = sdcpp.DEFAULT,
             budget_s: float = 110 * 60) -> dict:
    jobs = shard(ladder.image_jobs(p), k, n)
    out.mkdir(parents=True, exist_ok=True)
    report = {"shard": k, "of": n, "model": model, "premise": p["id"], "images": []}
    cand = sdcpp.CANDIDATES[model]
    w, h = cand.height, cand.width  # vertical
    start = time.monotonic()
    with sdcpp.Server(cand) as server:
        report["load_s"] = round(server.load_seconds, 1)
        for job in jobs:
            entry = {"id": job["id"], "takes": [], "kept": []}
            for take in range(TAKES):
                if take and time.monotonic() - start > budget_s:
                    break
                t = time.monotonic()
                img = sdcpp.to_pil(server.generate(job["prompt"], _seed(key, job["id"], take), w, h))
                verdict = qa.review(img, job["scene"])
                probs = take_problems(verdict, job)
                entry["takes"].append({"seconds": round(time.monotonic() - t, 1), "problems": probs,
                                       **verdict.as_dict()})
                print(f"{job['id']} take {take + 1}: {'ok' if not probs else probs}", flush=True)
                if probs:
                    continue
                stem = f"{job['id']}" if not entry["kept"] else f"{job['id']}-alt{len(entry['kept'])}"
                img.save(out / f"{stem}.png")
                files = plates.save_plate(img, out / stem, vertical=True)
                entry["kept"].append({"plate": files["plate"], "depth": files["depth"], "take": take,
                                      "similarity": round(verdict.similarity, 4),
                                      "luma": round(verdict.brightness, 1), "embedding": qa.pack(qa.embed(img)),
                                      "score": round(verdict.score(), 3)})
                if len(entry["kept"]) >= KEEP:
                    break
            entry["ok"] = bool(entry["kept"])
            report["images"].append(entry)
            (out / f"shard-{k:02d}.json").write_text(json.dumps(report, indent=1))
    report["seconds"] = round(time.monotonic() - start, 1)
    (out / f"shard-{k:02d}.json").write_text(json.dumps(report, indent=1))
    return report
