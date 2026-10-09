"""Where everything sits in time: chapter starts, narration offsets, title cards, shots, the ambience tail.

Times are seconds; shot boundaries are snapped to frames so parallel render shards join exactly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

FPS = 24
INTRO = 10.0  # quiet opening with the title card before the first words
GAP = 7.0  # breath + chapter card between chapters
TAIL_FADE = 45.0
CROSSFADE = 4.0
N_MOTIONS = 6


def _frames(t: float) -> int:
    return int(round(t * FPS))


def usable_images(report_dir: Path) -> dict[str, dict]:
    """Images with a plate, according to the image shard reports (the plates themselves may be elsewhere)."""
    found = {}
    for rep in sorted(report_dir.rglob("shard-*.json")):
        for img in json.loads(rep.read_text())["images"]:
            if img.get("plate"):
                found[img["id"]] = img
    return found


def _motions(episode_id: str, n: int) -> list[int]:
    seq, prev = [], -1
    h = hashlib.sha1(episode_id.encode()).digest()
    for i in range(n):
        m = (h[i % len(h)] + i) % N_MOTIONS
        if m == prev:
            m = (m + 1) % N_MOTIONS
        seq.append(m)
        prev = m
    return seq


def _assign(ids: list[str], usable: dict, fallback: list[str]) -> list[tuple[str, bool]]:
    """Each slot gets its own image or, if that one failed review, the nearest usable neighbour."""
    good = [i for i in ids if i in usable]
    pool = good or [f for f in fallback if f in usable]
    if not pool:
        raise RuntimeError(f"no usable images for {ids[0][:3]}")
    out = []
    for j, i in enumerate(ids):
        if i in usable:
            out.append((i, False))
        else:
            out.append((min(pool, key=lambda g: abs(ids.index(g) - j) if g in ids else 99), True))
    return out


def build(episode: dict, voice_dir: Path, image_dir: Path, tail_minutes: float, render_shards: int) -> dict:
    usable = usable_images(image_dir)
    t = 0.0
    chapters, cards, shots, narration = [], [], [], []
    all_ids = [f"c{c['n']:02d}s{j:02d}" for c in episode["chapters"] for j in range(len(c["scenes"]))]
    for idx, ch in enumerate(episode["chapters"]):
        info = json.loads((voice_dir / f"ch{ch['n']:02d}.json").read_text())
        lead = INTRO if idx == 0 else GAP
        start = t
        narr_start = start + lead
        end = narr_start + info["duration"] + 1.5
        chapters.append({"n": ch["n"], "title": ch["title"], "start": round(start, 3), "end": round(end, 3),
                         "narration_start": round(narr_start, 3), "narration_seconds": info["duration"]})
        narration.append({"file": f"ch{ch['n']:02d}.flac", "at": round(narr_start, 3)})
        if idx == 0:
            cards.append({"kind": "intro", "start": 1.5, "seconds": 8.0, "title": episode["title_core"]})
            cards.append({"kind": "chapter", "start": 11.0, "seconds": 6.0, "n": ch["n"], "title": ch["title"]})
        else:
            cards.append({"kind": "chapter", "start": round(start + 0.5, 3), "seconds": 7.0, "n": ch["n"],
                          "title": ch["title"]})
        ids = [f"c{ch['n']:02d}s{j:02d}" for j in range(len(ch["scenes"]))]
        span = (end - start) / len(ids)
        for j, (img, swapped) in enumerate(_assign(ids, usable, all_ids)):
            shots.append({"image": img, "slot": ids[j], "swapped": swapped, "start": start + j * span})
        t = end
    tail_start = t
    total = tail_start + tail_minutes * 60
    tail_ids = [f"tail{j:02d}" for j in range(len(episode["tail_scenes"]))]
    if tail_ids:
        span = (total - tail_start) / len(tail_ids)
        for j, (img, swapped) in enumerate(_assign(tail_ids, usable, all_ids[::-1])):
            shots.append({"image": img, "slot": tail_ids[j], "swapped": swapped, "start": tail_start + j * span})
    motions = _motions(episode["id"], len(shots))
    total_f = _frames(total)
    for i, s in enumerate(shots):
        s["f0"] = _frames(s["start"])
        s["f1"] = _frames(shots[i + 1]["start"]) if i + 1 < len(shots) else total_f
        s["motion"] = motions[i] if not s["swapped"] else (motions[i] + 3) % N_MOTIONS
        s["start"] = round(s["start"], 3)
    return {
        "fps": FPS, "frames": total_f, "seconds": round(total_f / FPS, 3), "crossfade": CROSSFADE,
        "tail_start": round(tail_start, 3), "tail_fade": TAIL_FADE, "chapters": chapters, "cards": cards,
        "narration": narration, "shots": shots, "render_shards": split(shots, render_shards),
        "swapped": [s["slot"] for s in shots if s["swapped"]],
    }


def split(shots: list[dict], n: int) -> list[dict]:
    """Contiguous runs of shots with similar total length; each run starts on a shot boundary."""
    n = max(1, min(n, len(shots)))
    total = shots[-1]["f1"]
    bounds = [0]
    for k in range(1, n):
        target = total * k / n
        bounds.append(min(range(bounds[-1] + 1, len(shots) - (n - k) + 1),
                          key=lambda i: abs(shots[i]["f0"] - target)))
    bounds.append(len(shots))
    return [{"k": k, "first": bounds[k], "last": bounds[k + 1] - 1, "f0": shots[bounds[k]]["f0"],
             "f1": shots[bounds[k + 1] - 1]["f1"]} for k in range(n)]


def youtube_chapters(tl: dict, tail_label: str) -> list[tuple[str, str]]:
    def stamp(s: float) -> str:
        s = int(s)
        return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if tl["seconds"] >= 3600 else f"{s // 60}:{s % 60:02d}"

    rows = [(stamp(0 if i == 0 else c["start"]), c["title"]) for i, c in enumerate(tl["chapters"])]
    if tl["seconds"] - tl["tail_start"] >= 60:
        rows.append((stamp(tl["tail_start"]), tail_label))
    return rows
