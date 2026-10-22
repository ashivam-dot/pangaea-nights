"""A ladder Short, start to finish: the timeline, the frames, the sound and the gates that decide if it may publish.

    0.00-1.6 s   option 1 full frame with the banner (the question); the hook narration starts at 0.15 s
    ladder       eight options, 3.5-4.0 s each: badge, label and a climbing danger meter, a whoosh on every cut
    verdict      back on the right answer: "Real answer: number N, because ..." with the sourced fact on a card
    call         "Comment your number." over the end plate
    loop         the last half second is frame 0 again (option 1, banner, same framing): a match cut, so the
                 Short loops without a seam

1080x1920 at 30 fps, 30-45 s. Frames are rendered in Python and piped to ffmpeg (no -loop 1 inputs, no overlay
filters), and every measurement is taken on the rendered frames, before the encoder, so the gates measure the
picture and not the codec. Each floor has a paired ceiling.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from . import ladder

W, H, FPS = 1080, 1920, 30
HOOK_VO_AT = 0.15
HOOK_MIN = 1.6
HOLD = (3.5, 4.0)
OPTION_LEAD = 0.20  # cut, then the option's line
OPTION_TAIL = 0.30
VERDICT_LEAD, VERDICT_TAIL = 0.35, 0.55
CTA_LEAD, CTA_TAIL, CTA_MIN = 0.15, 0.35, 1.8
LOOP = 0.5
TOTAL = (30.0, 45.0)
FADE = 0.22  # overlays fade in over this long

GATES = {
    "duration_s": (30.0, 45.0), "hook_vo_start_s": (0.0, 0.30), "hook_hold_s": (1.6, 4.2), "option_hold_s": HOLD,
    "parallax_frac": (0.005, 0.04), "banner_height_frac": (0.10, 0.28), "banner_contrast": (4.5, 18.0),
    "banner_ocr": (0.75, 1.0), "frame0_luma": (90.0, 170.0), "luma": (60.0, 200.0), "luma_drop": (0.0, 0.35),
    "motion": (0.15, 9.0), "loudness_lufs": (-15.0, -13.0), "true_peak_db": (-12.0, -1.5), "wer": (0.0, 0.08),
    "wpm": (140.0, 190.0), "prompt_similarity": (0.25, 1.0), "option_pair_similarity": (0.30, 0.90),
    "frame0_vs_history": (0.0, 0.93), "novelty_4gram": (0.0, ladder.NOVELTY_MAX), "fps": (FPS, FPS),
}
# Ladder camera moves (fractions of the plate margin, as in render.MOTIONS): steadier than the sleep moves, but with
# enough depth parallax to feel alive on a phone.
MOTIONS = [
    {"name": "drift right", "z": (1.030, 1.050), "dz": (0.012, 0.012), "pan": (-0.30, 0.30, 0, 0), "par": (0.22, 0)},
    {"name": "rise", "z": (1.030, 1.060), "dz": (0.010, 0.014), "pan": (0, 0, 0.30, -0.30), "par": (0.05, 0.14)},
    {"name": "drift left", "z": (1.050, 1.030), "dz": (0.012, 0.012), "pan": (0.30, -0.30, 0, 0), "par": (-0.22, 0)},
    {"name": "push", "z": (1.000, 1.060), "dz": (0.006, 0.016), "pan": (0, 0, 0, -0.15), "par": (0.12, 0.06)},
]


def gate(name: str, value) -> dict:
    lo, hi = GATES[name]
    ok = value is not None and lo <= value <= hi
    return {"value": None if value is None else round(float(value), 4), "floor": lo, "ceiling": hi, "ok": ok}


# ---------------------------------------------------------------------------------------------------- timeline


def _hold(vo: float) -> float:
    return min(max(OPTION_LEAD + vo + OPTION_TAIL, HOLD[0]), HOLD[1])


def timeline(p: dict, durations: dict[str, float], fps: int = FPS) -> dict:
    """Shots, narration placements and cuts from the measured line lengths ({'hook', 'o1'..'o8', 'verdict',
    'cta'}: seconds). Pure: no media is touched, so every timing rule is testable."""
    problems = []
    shots, lines = [], []
    hook_end = max(HOOK_MIN, HOOK_VO_AT + durations["hook"] + 0.15)
    lines.append({"id": "hook", "start": HOOK_VO_AT})
    t = hook_end
    first_motion = 0
    for o in p["options"]:
        n = o["n"]
        vo = durations[f"o{n}"]
        hold = _hold(vo)
        if OPTION_LEAD + vo + 0.10 > HOLD[1]:
            problems.append(f"option {n} narration is {vo:.2f}s: it cannot fit a {HOLD[1]}s hold")
        start = 0.0 if n == 1 else t  # option 1 is already on screen under the hook: same shot, no cut
        shots.append({"kind": "option", "n": n, "image": f"o{n}", "start": start, "end": t + hold,
                      "hold": hold, "motion": (first_motion + n - 1) % len(MOTIONS), "danger": o["danger"],
                      "label": o["label"], "overlay_from": t})
        lines.append({"id": f"o{n}", "start": t + OPTION_LEAD})
        t += hold
    verdict = VERDICT_LEAD + durations["verdict"] + VERDICT_TAIL
    shots.append({"kind": "verdict", "n": p["answer_n"], "image": f"o{p['answer_n']}", "start": t, "end": t + verdict,
                  "motion": 3, "overlay_from": t})
    lines.append({"id": "verdict", "start": t + VERDICT_LEAD})
    t += verdict
    cta = max(CTA_MIN, CTA_LEAD + durations["cta"] + CTA_TAIL)
    shots.append({"kind": "cta", "n": 0, "image": "end", "start": t, "end": t + cta, "motion": 1, "overlay_from": t})
    lines.append({"id": "cta", "start": t + CTA_LEAD})
    t += cta
    shots.append({"kind": "loop", "n": 1, "image": "o1", "start": t, "end": t + LOOP, "motion": shots[0]["motion"],
                  "overlay_from": t})
    t += LOOP
    frames = int(round(t * fps))
    for s in shots:
        s["f0"], s["f1"] = int(round(s["start"] * fps)), int(round(s["end"] * fps))
    shots[-1]["f1"] = frames
    cuts = [s["f0"] for s in shots[1:] if s["kind"] != "option" or s["n"] > 1]
    whoosh = [round(s["start"], 3) for s in shots if (s["kind"] == "option" and s["n"] > 1) or s["kind"] in
              ("verdict", "loop")]
    if not TOTAL[0] <= t <= TOTAL[1]:
        problems.append(f"total {t:.2f}s is outside {TOTAL[0]:.0f}-{TOTAL[1]:.0f}s")
    return {"fps": fps, "seconds": round(t, 3), "frames": frames, "hook_end": round(hook_end, 3), "shots": shots,
            "lines": lines, "cuts": cuts, "whoosh": whoosh, "riser": round(shots[-3]["start"] - 1.4, 3),
            "verdict_start": round(shots[-3]["start"], 3), "problems": problems}


def shot_at(tl: dict, f: int) -> dict:
    # Option 1 starts at frame 0 but option 2 starts later: the last shot whose span holds f, from the end.
    for s in reversed(tl["shots"]):
        if s["f0"] <= f < s["f1"]:
            return s
    return tl["shots"][-1]


def shot_u(s: dict, f: int, fps: int = FPS) -> float:
    """Camera progress through a shot. The loop shot is frame 0's framing, held still."""
    if s["kind"] == "loop":
        return 0.0
    return (f / fps - s["start"]) / max(s["end"] - s["start"], 1e-6)


def parallax_fraction(motion: dict, plate_w: int, plate_h: int, w: int = W, h: int = H) -> float:
    """How far near ground moves against far ground over one shot, as a fraction of the frame width: the depth
    offsets of render.Camera (par x margin, at depth +-0.5) plus the depth-dependent zoom."""
    mx, my = (plate_w - w) / 2, (plate_h - h) / 2
    lateral = 2 * float(np.hypot(motion["par"][0] * mx, motion["par"][1] * my))
    zoom = abs(motion["dz"][1] - motion["dz"][0]) * w
    return (lateral + zoom) / w


def captions(tl: dict, voice: dict) -> list[dict]:
    """Word-by-word captions: each line of at most four words; each word appears when it is spoken."""
    from .shorts import _chunks

    out = []
    for line in tl["lines"]:
        words = [{**w, "start": w["start"] + line["start"], "end": w["end"] + line["start"]}
                 for w in voice["segments"][line["id"]]["words"]]
        for chunk in _chunks(words):
            end = chunk[-1]["end"] + 0.25
            for i, wd in enumerate(chunk):
                nxt = chunk[i + 1]["start"] if i + 1 < len(chunk) else end
                out.append({"words": [x["text"] for x in chunk[:i + 1]], "start": wd["start"], "end": nxt})
    for a, b in zip(out, out[1:]):
        a["end"] = min(a["end"], b["start"])
    return out


# ---------------------------------------------------------------------------------------------------- frames


def frame_luma(frame: np.ndarray) -> float:
    s = frame[::6, ::6].astype(np.float32)
    return float((0.299 * s[..., 0] + 0.587 * s[..., 1] + 0.114 * s[..., 2]).mean())


def luma_report(lumas: list[float], diffs: list[float], cuts: list[int]) -> dict:
    """Per-frame luma (60-200), the largest frame-to-frame drop away from cuts (at most 35%), and how much the
    picture moves at all (median mean-absolute frame difference: a still image fails, so does strobing)."""
    cut = set(cuts)
    drops = [(lumas[i - 1] - lumas[i]) / max(lumas[i - 1], 1.0) for i in range(1, len(lumas)) if i not in cut]
    motion = [d for i, d in enumerate(diffs, start=1) if i not in cut]
    return {"luma_min": min(lumas), "luma_max": max(lumas), "luma_drop": max(drops, default=0.0),
            "motion": float(np.median(motion)) if motion else 0.0}


def _pick_images(reports: list[dict]) -> tuple[dict[str, str], dict]:
    """Choose one take per image so no two options look alike (pairwise CLIP at most 0.90). Each image job keeps up
    to two passing takes; start from the best and swap in a second take where that breaks a too-close pair."""
    from . import qa

    takes = {}
    for rep in reports:
        for img in rep["images"]:
            if img.get("kept"):
                takes[img["id"]] = img["kept"]
    choice = {k: 0 for k in takes}
    opts = sorted(k for k in takes if k.startswith("o"))
    emb = {(k, i): qa.unpack(t["embedding"]) for k in takes for i, t in enumerate(takes[k])}

    def worst():
        pairs = [(qa.cosine(emb[(a, choice[a])], emb[(b, choice[b])]), a, b) for i, a in enumerate(opts)
                 for b in opts[i + 1:]]
        return max(pairs, default=(0.0, None, None)), pairs

    for _ in range(12):
        (sim, a, b), _ = worst()
        if sim <= GATES["option_pair_similarity"][1] or a is None:
            break
        best = None
        for k in (a, b):
            for alt in range(len(takes[k])):
                if alt == choice[k]:
                    continue
                old, choice[k] = choice[k], alt
                s = worst()[0][0]
                choice[k] = old
                if s < sim and (best is None or s < best[0]):
                    best = (s, k, alt)
        if not best:
            break
        choice[best[1]] = best[2]
    (sim, _, _), pairs = worst()
    files = {k: takes[k][choice[k]]["plate"] for k in takes}
    stats = {"missing": [k for k in ladder.IMAGE_IDS if k not in takes], "pair_max": sim,
             "pair_min": min((p[0] for p in pairs), default=sim),
             "prompt_similarity_min": min((takes[k][choice[k]]["similarity"] for k in opts), default=None),
             "choice": choice}
    return files, stats


def render(p: dict, tl: dict, voice: dict, images: Path, files: dict[str, str], out: Path,
           size: tuple[int, int] = (W, H)) -> dict:
    """Frames to out/video.mp4 (no audio), plus frame0.jpg and review stills; returns the frame measurements."""
    import cv2
    from PIL import Image

    from . import overlay
    from . import render as rd

    w, h = size
    fps = tl["fps"]
    out.mkdir(parents=True, exist_ok=True)
    cam, grade = rd.Camera(w, h), rd.Grade(w, h, seed=7, darken=1.0)
    plates = {}

    def plate(image_id: str):
        if image_id not in plates:
            stem = Path(files[image_id]).with_suffix("")
            plates[image_id] = rd.Plate(images / f"{stem.name}.jpg", images / f"{stem.name}.depth.png")
        return plates[image_id]

    banner, geo = overlay.banner(p["banner"], w, h)
    mark = overlay.mark(w, h)
    badges = {o["n"]: overlay.badge(o["n"], w, h) for o in p["options"]}
    labels = {o["n"]: overlay.label(o["label"], w, h) for o in p["options"]}
    answer = next(o for o in p["options"] if o["n"] == p["answer_n"])
    verdict = overlay.verdict(p["answer_n"], answer["label"], p["fact"]["wiki_title"], w, h)
    cta = overlay.cta(w, h)
    caps = captions(tl, voice)
    cap_cache: dict[tuple, overlay.Layer] = {}
    layers = [banner, mark, verdict, cta, *badges.values(), *labels.values(), overlay.meter(8, w, h)]
    enc = rd.Encoder(out / "video.mp4", w, h, crf=19, maxrate="9M", fps=fps)
    lumas, diffs, prev = [], [], None
    stills = {s["f0"] + (s["f1"] - s["f0"]) // 2: s["image"] for s in tl["shots"]}
    parallax = {}
    ci = 0
    for f in range(tl["frames"]):
        t = f / fps
        s = shot_at(tl, f)
        pl = plate(s["image"])
        motion = MOTIONS[s["motion"]]
        if s["kind"] == "option":
            parallax[s["n"]] = parallax_fraction(motion, pl.pw, pl.ph, w, h)
        frame = grade.apply(cam.shoot(pl, shot_u(s, f, fps), motion), f, 1.0)
        a = min(1.0, max(0.0, (t - s["overlay_from"]) / FADE)) if s["kind"] != "loop" else 0.0
        banner.draw(frame)
        mark.draw(frame)
        if s["kind"] == "option" and t >= s["overlay_from"]:
            badges[s["n"]].draw(frame, a)
            labels[s["n"]].draw(frame, a)
            before = p["options"][s["n"] - 2]["danger"] if s["n"] > 1 else 0
            level = before + (s["danger"] - before) * rd.ease((t - s["overlay_from"]) / 0.6)
            overlay.meter(level, w, h).draw(frame, a if s["n"] == 1 else 1.0)
        elif s["kind"] == "verdict":
            verdict.draw(frame, a)
        elif s["kind"] == "cta":
            cta.draw(frame, a)
        while ci < len(caps) and caps[ci]["end"] <= t:
            ci += 1
        if s["kind"] != "loop" and ci < len(caps) and caps[ci]["start"] <= t:
            key = tuple(caps[ci]["words"])
            if key not in cap_cache:
                cap_cache[key] = overlay.caption(list(key), w, h)
                layers.append(cap_cache[key])
            cap_cache[key].draw(frame)
        lumas.append(frame_luma(frame))
        small = cv2.resize(frame, (w // 8, h // 8), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.abs(small - prev).mean()))
        prev = small
        enc.write(frame)
        if f == 0:
            Image.fromarray(frame).save(out / "frame0.jpg", quality=92)
            frame0 = frame.copy()
        if f in stills:
            Image.fromarray(frame).save(out / f"still-{f:04d}-{stills[f]}.jpg", quality=85)
    enc.close()
    rep = luma_report(lumas, diffs, tl["cuts"])
    rep.update({"frame0_luma": lumas[0], "loop_match": float(np.abs(lumas[0] - lumas[-1])),
                "banner_height_frac": geo["height_frac"], "banner_contrast": overlay.banner_contrast(frame0, geo),
                "banner_ocr": banner_ocr(frame0, geo, p["banner"]),
                "parallax": parallax, "safe": all(overlay.inside(x.box, w, h) for x in layers)})
    return rep


def banner_ocr(frame: np.ndarray, geo: dict, text: str) -> float | None:
    """Share of the banner's words Tesseract reads back from frame 0 (None when Tesseract is not installed)."""
    import difflib
    import re
    import shutil

    from PIL import Image, ImageOps

    from . import thumbnail

    if not shutil.which("tesseract"):
        return None
    x0, y0, x1, y1 = geo["box"]
    crop = Image.fromarray(frame[y0:y1, x0:x1]).convert("L")
    if crop.height < 120:
        crop = crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS)
    seen = []
    for view in (crop, ImageOps.invert(crop)):
        seen += thumbnail.ocr_words(ImageOps.autocontrast(view), 6)
    want = [re.sub(r"[^a-z0-9']", "", x.lower()) for x in text.split()]
    want = [x for x in want if x]
    hit = [x for x in want if any(difflib.SequenceMatcher(None, x, s).ratio() >= 0.8 for s in seen)]
    return len(hit) / max(len(want), 1)


def count_frames(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(path)], check=True, capture_output=True,
                         text=True).stdout.strip()
    return int(out.split(",")[0])


# ---------------------------------------------------------------------------------------------------- gates


def gates(tl: dict, frames: dict, media: dict, voice: dict, images: dict, plan: dict,
          frame0_history: float | None) -> dict:
    """Every publish gate, as {name: {value, floor, ceiling, ok}}; the Short may publish only if all are ok."""
    holds = [s["hold"] for s in tl["shots"] if s["kind"] == "option"]
    par = list(frames.get("parallax", {}).values())
    g = {
        "timeline": {"ok": not tl["problems"], "problems": tl["problems"]},
        "duration_s": gate("duration_s", media.get("duration")),
        "frames_counted": {"value": media.get("frames"), "want": tl["frames"],
                           "ok": media.get("frames") == tl["frames"]},
        "fps": gate("fps", media.get("fps")),
        "size": {"value": media.get("size"), "ok": media.get("size") == [W, H]},
        "hook_vo_start_s": gate("hook_vo_start_s", tl["lines"][0]["start"]),
        "hook_hold_s": gate("hook_hold_s", tl["hook_end"]),
        "option_hold_min_s": gate("option_hold_s", min(holds)),
        "option_hold_max_s": gate("option_hold_s", max(holds)),
        "parallax_min": gate("parallax_frac", min(par) if par else None),
        "parallax_max": gate("parallax_frac", max(par) if par else None),
        "banner_height_frac": gate("banner_height_frac", frames.get("banner_height_frac")),
        "banner_contrast": gate("banner_contrast", frames.get("banner_contrast")),
        "banner_ocr": gate("banner_ocr", frames.get("banner_ocr")),
        "safe_zones": {"ok": bool(frames.get("safe"))},
        "frame0_luma": gate("frame0_luma", frames.get("frame0_luma")),
        "luma_min": gate("luma", frames.get("luma_min")),
        "luma_max": gate("luma", frames.get("luma_max")),
        "luma_drop": gate("luma_drop", frames.get("luma_drop")),
        "motion": gate("motion", frames.get("motion")),
        "loudness_lufs": gate("loudness_lufs", media.get("I")),
        "true_peak_db": gate("true_peak_db", media.get("true_peak")),
        "wer": gate("wer", voice.get("wer")),
        "wpm": gate("wpm", voice.get("wpm")),
        "images_clean": {"ok": images.get("missing") == [], "missing": images.get("missing")},
        "prompt_similarity": gate("prompt_similarity", images.get("prompt_similarity_min")),
        "option_pair_max": gate("option_pair_similarity", images.get("pair_max")),
        "option_pair_min": gate("option_pair_similarity", images.get("pair_min")),
        "frame0_vs_history": gate("frame0_vs_history", frame0_history if frame0_history is not None else 0.0),
        "novelty_4gram": gate("novelty_4gram", plan["gates"]["novelty_4gram"]["value"]),
        "premise": {"ok": plan["gates"]["premise"]["ok"] and plan["gates"]["reuse"]["ok"]},
    }
    return g


def all_ok(g: dict) -> bool:
    return all(v.get("ok") for v in g.values())


def failures(g: dict) -> list[str]:
    return [f"{k}: {json.dumps({x: v[x] for x in v if x != 'ok'})}" for k, v in g.items() if not v.get("ok")]
