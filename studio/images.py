"""Every image an episode needs, split across parallel runner jobs.

Each image gets up to three takes (new seeds); every take is reviewed automatically (qa.py) and the best passing
take becomes a render plate. A take that shows people or text is never used: the slot is marked and the render
reuses its neighbour with a different camera move.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import plates, qa, sdcpp, writer

HARD = ("person", "text")


def manifest(episode: dict) -> list[dict]:
    art = episode["art_direction"]
    jobs = []
    for ch in episode["chapters"]:
        for j, scene in enumerate(ch["scenes"]):
            jobs.append({"id": f"c{ch['n']:02d}s{j:02d}", "kind": "scene", "chapter": ch["n"], "scene": scene,
                         "prompt": writer.image_prompt(scene, art), "vertical": False})
    for j, scene in enumerate(episode["tail_scenes"]):
        jobs.append({"id": f"tail{j:02d}", "kind": "tail", "scene": scene,
                     "prompt": writer.image_prompt(scene, art + ", very dark night, deep shadows, faint light"),
                     "vertical": False})
    for k, tr in enumerate(episode["trailers"]):
        for j, scene in enumerate(tr["scenes"]):
            jobs.append({"id": f"t{k}s{j:02d}", "kind": "trailer", "trailer": k, "scene": scene,
                         "prompt": writer.image_prompt(scene, art + ", tall vertical composition"), "vertical": True})
    th = episode["thumbnail"]
    jobs.append({"id": "thumb", "kind": "thumb", "scene": th["prompt"],
                 "prompt": writer.image_prompt(th["prompt"], art + ", subject large on the right, calm dark empty "
                                                                  "space on the left"), "vertical": False})
    return jobs


def shard(jobs: list[dict], k: int, n: int) -> list[dict]:
    """Round-robin so every shard gets a similar mix of horizontal and vertical images."""
    horizontal = [j for j in jobs if not j["vertical"]]
    vertical = [j for j in jobs if j["vertical"]]
    mine = horizontal[k::n]
    offset = len(horizontal) % n
    mine += [j for i, j in enumerate(vertical) if (i + offset) % n == k]
    return mine


def _seed(job_id: str, take: int, episode_id: str) -> int:
    return int(hashlib.sha1(f"{episode_id}:{job_id}:{take}".encode()).hexdigest()[:8], 16) % 2_000_000_000


def generate(episode: dict, out: Path, k: int, n: int, budget_s: float = 72 * 60,
             model: str = sdcpp.DEFAULT) -> dict:
    jobs = shard(manifest(episode), k, n)
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    report = {"shard": k, "of": n, "model": model, "images": []}
    cand = sdcpp.CANDIDATES[model]
    with sdcpp.Server(cand) as server:
        report["load_s"] = round(server.load_seconds, 1)
        for job in jobs:
            w, h = (cand.height, cand.width) if job["vertical"] else (cand.width, cand.height)
            takes = []
            for take in range(3):
                if take and time.monotonic() - start > budget_s:
                    break
                t = time.monotonic()
                img = sdcpp.to_pil(server.generate(job["prompt"], _seed(job["id"], take, episode["id"]), w, h))
                gen_s = time.monotonic() - t
                verdict = qa.review(img, job["scene"])
                takes.append((verdict, img, round(gen_s, 1)))
                print(f"image {job['id']} take {take + 1}: {gen_s:.0f}s ok={verdict.ok} {verdict.reasons}",
                      flush=True)
                if verdict.ok:
                    break
            usable = [x for x in takes if not any(r.split()[0] in HARD for r in x[0].reasons)]
            pool = [x for x in usable if x[0].ok] or usable
            entry = {"id": job["id"], "kind": job["kind"], "takes": [
                {"seconds": s, **v.as_dict()} for v, _, s in takes]}
            if pool:
                verdict, img, _ = max(pool, key=lambda x: x[0].score())
                img.save(out / f"{job['id']}.png")
                if job["kind"] != "thumb":
                    entry.update(plates.save_plate(img, out / job["id"], vertical=job["vertical"]))
                entry["ok"] = verdict.ok
            else:
                entry["ok"] = False
                entry["unusable"] = True
            report["images"].append(entry)
            (out / f"shard-{k:02d}.json").write_text(json.dumps(report, indent=1))
    report["seconds"] = round(time.monotonic() - start, 1)
    (out / f"shard-{k:02d}.json").write_text(json.dumps(report, indent=1))
    return report
