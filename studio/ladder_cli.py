"""Command line for the ladder Shorts; .github/workflows/shorts.yml calls one stage per job.

    python -m studio.cli ladder-check                      every bank premise passes its gates
    python -m studio.cli ladder-ramp                       which slots may publish today, and why
    python -m studio.cli ladder-plan --mode dry            choose premises, write ladder/<slot>/plan.json
    python -m studio.cli ladder-images --slot A --shard 3/9
    python -m studio.cli ladder-voice --slot A
    python -m studio.cli ladder-render --slot A            frames, sound, mux, every gate -> gates.json
    python -m studio.cli ladder-publish --slot A --mode publish
    python -m studio.cli ladder-record                     builds, Shorts, alerts into the ledger
    python -m studio.cli ladder-learn                      24 h / 72 h numbers -> weights, queue, summary
    python -m studio.cli ladder-health --hours 30          a Short uploaded in the last 30 h (after launch)

Modes: 'publish' schedules a public release at the slot time; 'test' uploads a private video with no release time
(and consumes no premise); 'dry' builds and gates everything but uploads nothing.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config, ladder, ledger

MODES = ("publish", "test", "dry")
PEAK_JOBS = 19  # image + voice runners at once, leaving one of the free plan's 20 for an episode plan job (fanout)
IMAGE_JOBS = PEAK_JOBS - 2  # with two slots; image_budget() for any count


def image_budget(slots: int) -> int:
    """Image runner jobs a run may fan out to: the peak less one voice job per slot (three a day: 16)."""
    return PEAK_JOBS - max(slots, 1)
PLAN_TRIES = 8  # premises tried per slot before the slot is given up
# The one private test upload the launch checklist asks for happens while PAUSED is still true; it needs this
# separately named switch (shorts.yml input allow_test_while_paused), so a plain test dispatch never uploads.
TEST_WHILE_PAUSED = "SHORTS_TEST_WHILE_PAUSED"


def _dir(slot: str) -> Path:
    return config.WORK / "ladder" / slot


def _load(slot: str, name: str) -> dict:
    return json.loads((_dir(slot) / name).read_text())


def _save(slot: str, name: str, data: dict) -> None:
    p = _dir(slot) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1, ensure_ascii=False))


def paused() -> bool:
    """The channel's PAUSED switch (paused unless explicitly false) or the Shorts-only SHORTS_PAUSED=true."""
    from .cli import _paused

    return _paused() or os.environ.get("SHORTS_PAUSED", "").strip().lower() == "true"


def upload_refusal(mode: str) -> str | None:
    """Why a run in this mode must not write to the channel (None = it may). Every writing mode honours the pause."""
    if mode == "dry":
        return "dry run: nothing is uploaded"
    if not paused():
        return None
    if mode == "test" and os.environ.get(TEST_WHILE_PAUSED, "").strip().lower() == "true":
        return None
    return "paused (PAUSED or SHORTS_PAUSED)" + (f": a test upload needs {TEST_WHILE_PAUSED}=true" if mode == "test"
                                                 else "")


def _plan_alerts(rows: list[dict]) -> None:
    """Problems found while planning, for ladder-record to file (ledger alerts + the alert issue)."""
    root = config.WORK / "ladder"
    root.mkdir(parents=True, exist_ok=True)
    (root / "plan_alerts.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))


def choose(banks: dict, data: dict, today, planned: list[dict], mode: str, override: str | None):
    """A premise and its plan gates. Without an override, a premise whose premise, reuse, novelty or rotation gate
    already fails is skipped for the next candidate rather than planned to certain failure. Raises BankError."""
    skipped: set[str] = set()
    for _ in range(PLAN_TRIES):
        p = ladder.pick(banks, data, today, planned, override=override, exclude=skipped)
        g = ladder.plan_problems(p, data, today, planned)
        hard = [k for k in ("premise", "reuse", "novelty_4gram", "rotation") if not g[k]["ok"]]
        if override or not hard or (mode != "publish" and hard == ["rotation"]):
            return p, g, sorted(skipped)
        print(f"skipping {p['id']}: {', '.join(hard)} not ok")
        skipped.add(p["id"])
    raise ladder.BankError(f"no premise passes its plan gates after {PLAN_TRIES} tries (skipped {sorted(skipped)})")


def _output(rows: dict) -> None:
    lines = [f"{k}={json.dumps(v)}" for k, v in rows.items()]
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _planned_slots() -> list[str]:
    root = config.WORK / "ladder"
    return sorted(p.parent.name for p in root.glob("*/plan.json")) if root.exists() else []


# ---------------------------------------------------------------------------------------------------- stages


def check(args) -> None:
    banks = ladder.load_banks(strict=False)
    problems = {}
    for path in sorted(ladder.BANKS.glob("*.json")):
        problems.update({f"{path.name} {k}": v for k, v in ladder.bank_problems(ladder.load_bank(path)).items()})
    # Proven premises live on the state branch and are re-gated at plan time; one failing there is left out, never a
    # reason to stop the day's Shorts, so they are counted here but do not fail the check.
    proven = sum(len(ps) for ps in ladder.with_proven({}, ledger.load()).values())
    print(json.dumps({"usable": {s: len(ps) for s, ps in banks.items()}, "proven": proven, "problems": problems},
                     indent=1))
    if problems:
        sys.exit("bank premises fail their gates")


def ramp(args) -> None:
    data = ledger.load()
    slots, why = ladder.slots_allowed(ladder.launch_day(os.environ.get("SHORTS_LAUNCH")), ladder.now_day(),
                                      ladder.builds(data))
    print(json.dumps({"slots": slots, "reason": why}, indent=1))


def plan(args) -> None:
    data = ledger.load()
    banks = ladder.with_proven(ladder.load_banks(strict=False), data)
    today = ladder.now_day()
    if args.mode == "publish":
        slots, why = ladder.slots_allowed(ladder.launch_day(os.environ.get("SHORTS_LAUNCH")), today,
                                          ladder.builds(data))
        if args.slot:
            slots = [s for s in slots if s == args.slot]
        if paused():
            slots, why = [], "paused (PAUSED or SHORTS_PAUSED)"
    else:
        slots, why = [args.slot or "A"], f"{args.mode} run"
    run = os.environ.get("GITHUB_RUN_ID", "local")
    planned, made, alerts = [], [], []
    seen = {a.get("premise") for a in ladder.recent_alerts(data, "demoted", today)}
    for pid, n in sorted(ladder.demoted(data, today).items()):
        if pid not in seen:
            alerts.append({"kind": "demoted", "premise": pid, "run": run,
                           "problems": [f"{n} failed publish builds in {ladder.FAILED_DAYS} days: resting it; fix the "
                                        "premise (see the builds' failures in the ledger)"]})
    for slot in slots:
        try:
            p, g, skipped = choose(banks, data, today, planned, args.mode, args.premise or None)
        except ladder.BankError as exc:
            print(f"slot {slot}: {exc}")
            if args.mode == "publish":  # an allowed slot that plans nothing must be seen, not end a green run
                alerts.append({"kind": "plan", "slot": slot, "run": run, "problems": [str(exc)]})
            continue
        planned.append(p)
        _save(slot, "plan.json", {"slot": slot, "mode": args.mode, "day": today.isoformat(), "run": run,
                                  "key": f"{today.isoformat()}-{p['id']}", "premise": p, "gates": g,
                                  "script": ladder.script_text(p), "created": ledger.now()})
        made.append(slot)
        print(json.dumps({"slot": slot, "premise": p["id"], "title": p["title"],
                          "gates": {k: v["ok"] for k, v in g.items()}}))
    print(f"slots: {made} ({why})")
    _plan_alerts(alerts)
    if alerts:
        from . import alert

        alert.send("Pangaea Nights Shorts plan:\n" + "\n".join(
            f"- {a.get('slot') or a.get('premise')}: {'; '.join(a['problems'])}" for a in alerts))
    _output({"slots": made, "images": image_matrix(made, args.shards), "any": bool(made)})
    if any(a["kind"] == "plan" for a in alerts) and not made:
        sys.exit("an allowed publish slot planned no premise")


def image_matrix(slots: list[str], shards: int, budget: int | None = None) -> list[dict]:
    """The image jobs: every slot gets its own shards, and all of them fit in one wave of `budget` runners, so slot
    A's render never waits on a second wave for a later slot (a later slot takes fewer shards, two images on some)."""
    out, left = [], image_budget(len(slots)) if budget is None else budget
    for i, slot in enumerate(slots):
        n = max(1, min(shards, -(-left // (len(slots) - i))))
        out += [{"slot": slot, "shard": f"{k}/{n}"} for k in range(n)]
        left -= n
    return out


def images(args) -> None:
    from . import ladder_images

    k, n = (int(x) for x in args.shard.split("/"))
    pl = _load(args.slot, "plan.json")
    rep = ladder_images.generate(pl["premise"], pl["key"], _dir(args.slot) / "images", k, n)
    print(json.dumps({k: rep.get(k) for k in ("shard", "of", "model", "load_s", "seconds")}))
    print(json.dumps({i["id"]: len(i["kept"]) for i in rep["images"]}))


def voice(args) -> None:
    from . import ladder_voice

    pl = _load(args.slot, "plan.json")
    rep = ladder_voice.narrate(pl["premise"], _dir(args.slot) / "voice")
    print(json.dumps({k: rep[k] for k in ("voice", "speed", "wpm", "wer")}))


def _media(path: Path) -> dict:
    from . import finish, mix, shorts_ladder

    info = finish.probe(path)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = (int(x) for x in v["r_frame_rate"].split("/"))
    loud = mix.integrated(path)
    return {"duration": float(info["format"]["duration"]), "fps": num / den, "size": [v["width"], v["height"]],
            "frames": shorts_ladder.count_frames(path), "I": loud["I"], "true_peak": loud["true_peak"],
            "size_mb": round(int(info["format"]["size"]) / 1e6, 2)}


def _frame0_history(data: dict, frame0: Path) -> tuple[float | None, str]:
    from PIL import Image

    from . import qa

    vec = qa.embed(Image.open(frame0).convert("RGB"))
    past = [qa.unpack(s["frame0"]) for s in data.get("shorts", [])[-ladder.HISTORY:] if s.get("frame0")]
    return (max((qa.cosine(vec, p) for p in past), default=None), qa.pack(vec))


def _hook_onset(wav: Path) -> float | None:
    """When the hook's voice is first heard in the Short: its placement plus the onset measured in its wav."""
    import soundfile as sf

    from . import shorts_ladder

    if not wav.exists():
        return None
    audio, sr = sf.read(str(wav), dtype="float32")
    return shorts_ladder.HOOK_VO_AT + shorts_ladder.onset_s(audio, sr)


def render(args) -> None:
    import subprocess

    from . import ladder_audio, shorts_ladder

    d = _dir(args.slot)
    pl = _load(args.slot, "plan.json")
    p = pl["premise"]
    vo = _load(args.slot, "voice/voice.json")
    reports = [json.loads(f.read_text()) for f in sorted((d / "images").glob("shard-*.json"))]
    files, img_stats = shorts_ladder._pick_images(reports)
    tl = shorts_ladder.timeline(p, vo["durations"])
    _save(args.slot, "timeline.json", tl)
    size = tuple(int(x) for x in args.size.split("x")) if args.size else (shorts_ladder.W, shorts_ladder.H)
    out = d / "out"
    frames = {}
    if not img_stats["missing"]:
        frames = shorts_ladder.render(p, tl, vo, d / "images", files, out, size)
        ladder_audio.build(tl, vo, d / "voice", p["series"], out / "sound.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out / "video.mp4"), "-i",
                        str(out / "sound.wav"), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                        "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(out / "short.mp4")],
                       check=True)
        (out / "video.mp4").unlink()
        (out / "sound.wav").unlink()
        media = _media(out / "short.mp4")
        sim, packed = _frame0_history(ledger.load(), out / "frame0.jpg")
    else:
        media, sim, packed = {}, None, None
    g = shorts_ladder.gates(tl, frames, media, vo, img_stats, pl, sim, _hook_onset(d / "voice" / "hook.wav"))
    ok = shorts_ladder.all_ok(g)
    report = {"ok": ok, "failures": shorts_ladder.failures(g), "gates": g, "media": media,
              "frames": {k: v for k, v in frames.items() if k != "parallax"}, "images": img_stats,
              "files": files, "frame0": packed, "frame0_vs_history": sim}
    _save(args.slot, "out/gates.json", report)
    print(json.dumps({"ok": ok, "failures": report["failures"], "seconds": tl["seconds"]}, indent=1))


def publish(args) -> None:
    from . import alert, youtube

    pl, gates = _load(args.slot, "plan.json"), _load(args.slot, "out/gates.json")
    p = pl["premise"]
    why = []
    refusal = upload_refusal(args.mode)
    if refusal:
        why.append(refusal)
    if not gates.get("ok"):
        why.append("gates failed: " + "; ".join(gates.get("failures", [])[:6]))
    sibling = sibling_refusals(args.slot, gates)
    why += sibling
    when, api, channel, existing = None, None, None, None
    test_run = pl["run"] if args.mode == "test" else None
    marker = youtube.ladder_marker(p, test_run)
    if not why:
        api = youtube.client()
        channel = youtube.identity(api)
        # A re-run after an upload that crashed before publish.json: find it first, keep its release time and file it
        # under its slot, instead of refusing it as late and leaving a live Short out of the ledger.
        existing = youtube.find(api, channel, marker)
    if args.mode == "publish" and not why:
        if existing:
            when = youtube._instant(existing.get("status", {}).get("publishAt")) or \
                youtube.slot_publish_at(args.slot, datetime.fromisoformat(pl["day"]).date(), late_ok=True)
        else:
            when = youtube.slot_publish_at(args.slot, datetime.fromisoformat(pl["day"]).date())
            if not when:
                why.append(f"slot {args.slot} is less than 20 minutes away: skipped rather than late")
    if why:
        print("not publishing:\n- " + "\n- ".join(why))
        _save(args.slot, "out/publish.json", {"published": False, "refusals": why, "mode": args.mode})
        if args.mode == "publish":  # a planned publish slot that made nothing: fail so the alert and the doctor see it
            sys.exit("a planned publish slot was refused")
        if not gates.get("ok") or sibling:  # a dry or test run whose Short a publish run would refuse must not end green
            sys.exit("the Short failed its gates" if not gates.get("ok") else "the Short is too close to its sibling slot")
        return
    data = ledger.load()
    tag_list = youtube.ladder_tags(p, marker)
    episode = youtube.related_episode(data, p["related_topic_hint"])
    desc = youtube.ladder_description(p, episode)
    res = youtube.upload_ladder(api, channel, _dir(args.slot) / "out" / "short.mp4", p, desc, tag_list, when)
    try:  # the upload is done: nothing after it may abort before publish.json is written
        res["comment"] = youtube.first_comment(api, res["video_id"], ladder.FIRST_COMMENT)
    except Exception as exc:
        res["comment"] = f"pending: {type(exc).__name__}"
    res.update({"published": True, "mode": args.mode, "slot": args.slot, "related": (episode or {}).get("url")})
    _save(args.slot, "out/publish.json", res)
    if res["readback_problems"]:
        alert.send(f"Pangaea Nights Short {p['id']} ({res['url']}): read-back mismatch:\n- " +
                   "\n- ".join(res["readback_problems"]))
    print(json.dumps(res, indent=1))


def sibling_refusals(slot: str, gates: dict) -> list[str]:
    """A slot against the earlier slots of the same run (ladder.SLOTS order; the ledger's history cannot see a Short
    made in this run) whose gates are on disk: shorts.yml's publish job gives slot B slot A's, and slot C slot B's.
    Only the later slot is refused, and only when the earlier one's own gates passed, so one of them still goes out."""
    from . import shorts_ladder

    order = list(ladder.SLOTS)
    out = []
    for earlier in order[:order.index(slot)]:
        other = _dir(earlier) / "out" / "gates.json"
        if not other.exists():
            continue
        g = json.loads(other.read_text())
        if g.get("ok"):
            out += [f"too close to slot {earlier}: {x}" for x in shorts_ladder.sibling_problems(gates, g)]
    return out


def adopt_loose_downloads() -> None:
    """download-artifact unpacks a pattern that matches a single artifact straight into work/dl, with no
    ladder-out-<slot> or ladder-pub-<slot> folder, so shorts.yml's copy loop misses a one-slot day entirely."""
    dl = config.WORK / "dl"
    loose = [f for f in dl.iterdir() if f.is_file() and f.name != "short.mp4"] if dl.exists() else []
    if not loose:
        return
    slots = _planned_slots()
    pub = dl / "publish.json"
    slot = json.loads(pub.read_text()).get("slot") if pub.exists() else None
    if slot is None and len(slots) == 1:
        slot = slots[0]
    if slot not in slots:
        print(f"work/dl: {len(loose)} loose files and no single slot to file them under; left unread")
        return
    out = _dir(slot) / "out"
    out.mkdir(parents=True, exist_ok=True)
    for f in loose:
        if not (out / f.name).exists():
            (out / f.name).write_bytes(f.read_bytes())


def record(args) -> None:
    """Fold every planned slot's outcome into the ledger: a build entry always, a Short only when published."""
    adopt_loose_downloads()
    data = ledger.load()
    mismatches = []
    for slot in _planned_slots():
        pl = _load(slot, "plan.json")
        p = pl["premise"]
        d = _dir(slot) / "out"
        gates = json.loads((d / "gates.json").read_text()) if (d / "gates.json").exists() else {"ok": False}
        pub = json.loads((d / "publish.json").read_text()) if (d / "publish.json").exists() else {}
        ledger.ladder_log(data, "builds", {"mode": pl["mode"], "slot": slot, "premise": p["id"], "run": pl["run"],
                                           "ok": bool(gates.get("ok")), "failures": gates.get("failures", [])[:12],
                                           "published": bool(pub.get("published")),
                                           "refusals": pub.get("refusals")}, unique=("run", "slot"))
        if not pub.get("published"):
            continue
        if pub.get("readback_problems"):
            ledger.ladder_log(data, "alerts", {"kind": "readback", "premise": p["id"], "run": pl["run"], "slot": slot,
                                               "video_id": pub["video_id"], "problems": pub["readback_problems"]},
                              unique=("kind", "run", "slot"))
            mismatches.append(f"{p['id']} {pub['url']}:\n- " + "\n- ".join(pub["readback_problems"]))
        entry = {"premise": p["id"], "series": p["series"], "era": p["era"], "slot": slot,
                 "video_id": pub["video_id"], "url": pub["url"], "title": pub["title"],
                 "publish_at": pub.get("publish_at"), "uploaded_at": ledger.now(), "comment": pub.get("comment"),
                 "script": pl["script"], "frame0": gates.get("frame0"), "related": pub.get("related"),
                 "run": pl["run"], "checkpoints": {}}
        if pl["mode"] == "test":  # a test upload never consumes a premise or counts toward the learner
            ledger.ladder_log(data, "tests", {k: entry[k] for k in ("premise", "video_id", "url", "run", "slot")},
                              unique=("run", "slot"))
            continue
        ledger.record_short(data, entry)
        data["ladder"]["queue"] = [q for q in data["ladder"]["queue"] if q != p["id"]]
    plan_alerts = config.WORK / "ladder" / "plan_alerts.json"
    for a in json.loads(plan_alerts.read_text()) if plan_alerts.exists() else []:
        ledger.ladder_log(data, "alerts", a, unique=("kind", "run", "slot", "premise"))
        mismatches.append(f"{a['kind']} {a.get('slot') or a.get('premise')}:\n- " + "\n- ".join(a["problems"]))
    ledger.save(data)
    # shorts.yml opens an "alert" issue from this file, so the mismatch also reaches the doctor's digest.
    alerts = config.WORK / "ladder" / "alerts.txt"
    alerts.parent.mkdir(parents=True, exist_ok=True)
    alerts.write_text("\n\n".join(mismatches) + ("\n" if mismatches else ""))
    print(json.dumps(data["ladder"]["builds"][-3:], indent=1))


def health(args) -> None:
    if paused():
        print("Shorts paused: no upload expected")
        return
    why = ladder.stale(ledger.load(), ladder.launch_day(os.environ.get("SHORTS_LAUNCH")), datetime.now(timezone.utc),
                       args.hours)
    if why:
        sys.exit(why)
    print("Shorts are on schedule")


def learn(args) -> None:
    from . import alert, ladder_learn, youtube

    data = ledger.load()
    api = analytics = None
    if os.environ.get("YOUTUBE_OAUTH_JSON"):
        api = youtube.client()
        youtube.identity(api)
        try:
            analytics = youtube.analytics_client()
        except Exception as exc:
            print(f"analytics client: {type(exc).__name__}")
    if args.dry:
        api = None  # the learner posts pending first comments through the Data API: a dry run must not write
    sm = ladder_learn.run(data, ladder.with_proven(ladder.load_banks(strict=False), data), analytics, api,
                          datetime.now(timezone.utc))
    msg = ladder_learn.message(sm)
    print(msg)
    if args.dry:  # a dispatch dry run: compute and print, but neither save the ledger nor message anyone
        return
    ledger.save(data)
    alert.send(msg)


def register(sub) -> None:
    sub.add_parser("ladder-check").set_defaults(fn=check)
    sub.add_parser("ladder-ramp").set_defaults(fn=ramp)
    s = sub.add_parser("ladder-plan")
    s.add_argument("--mode", choices=MODES, default="dry")
    s.add_argument("--slot", choices=["", *ladder.SLOTS], default="", help="empty = every slot the ramp allows")
    s.add_argument("--premise", default="")
    s.add_argument("--shards", type=int, default=len(ladder.IMAGE_IDS))
    s.set_defaults(fn=plan)
    s = sub.add_parser("ladder-images")
    s.add_argument("--slot", choices=list(ladder.SLOTS), required=True)
    s.add_argument("--shard", required=True, help="k/n, e.g. 0/9")
    s.set_defaults(fn=images)
    for name, fn in (("ladder-voice", voice), ("ladder-render", render)):
        s = sub.add_parser(name)
        s.add_argument("--slot", choices=list(ladder.SLOTS), required=True)
        if name == "ladder-render":
            s.add_argument("--size", default="", help="WxH, only for a low-resolution smoke")
        s.set_defaults(fn=fn)
    s = sub.add_parser("ladder-publish")
    s.add_argument("--slot", choices=list(ladder.SLOTS), required=True)
    s.add_argument("--mode", choices=MODES, default="dry")
    s.set_defaults(fn=publish)
    sub.add_parser("ladder-record").set_defaults(fn=record)
    s = sub.add_parser("ladder-health")
    s.add_argument("--hours", type=float, default=30)
    s.set_defaults(fn=health)
    s = sub.add_parser("ladder-learn")
    s.add_argument("--dry", action="store_true")
    s.set_defaults(fn=learn)
