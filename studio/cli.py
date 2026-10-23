"""Command line for every pipeline stage; the workflows call these one job at a time.

    python -m studio.cli plan --mode dry --shape short --writer auto
    python -m studio.cli images 0 5        python -m studio.cli voice 0 5
    python -m studio.cli audio             python -m studio.cli render 3
    python -m studio.cli finish            python -m studio.cli publish
    python -m studio.cli record            python -m studio.cli health
    python -m studio.cli ladder-...        the choice-ladder Shorts (see studio/ladder_cli.py)
    python -m studio.cli topics-scan       the weekly proven-topics feed (see studio/topics.py)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config, ledger, planner

W = config.WORK


def _load(name: str) -> dict:
    return json.loads((W / name).read_text())


def _save(name: str, data: dict) -> None:
    p = W / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1, ensure_ascii=False))


def _paused() -> bool:
    """Paused unless the PAUSED variable is explicitly false: going live is a deliberate act."""
    return os.environ.get("PAUSED", "").strip().lower() not in ("false", "0", "no")


def plan(args) -> None:
    from . import wiki, writer

    shape = config.shape(args.shape)
    data = ledger.load()
    topic = planner.next_topic(data, override=args.topic or None)
    docs = wiki.fetch_all(topic["articles"], cache=W / "wiki")
    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    use = args.writer
    if use == "auto":
        use = "gemini" if has_key else "stub"
    if args.mode == "publish" and (use != "gemini" or args.shape != "full"):
        sys.exit("publish runs need the real writer (GEMINI_API_KEY) and the full shape")
    started = time.monotonic()
    script = writer.write(topic, docs, shape) if use == "gemini" else writer.write_stub(topic, docs, shape)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run = os.environ.get("GITHUB_RUN_ID", "local")
    ep_id = f"{day}-{topic['id']}" if args.mode == "publish" else f"{args.mode}-{day}-{topic['id']}-{run}"
    words = sum(len(p.split()) for c in script["chapters"] for p in c["paragraphs"])
    hours = max(1, round(words / config.WPM / 60 + shape.tail_minutes / 60))
    episode = {**script, "id": ep_id, "topic": topic["id"], "series": topic["series"], "shape": shape.name,
               "mode": args.mode, "created": ledger.now(), "words": words,
               "title": planner.title(topic, script["title_core"], hours, data),
               "docs": [{k: d[k] for k in ("title", "revid", "url", "page_url")} for d in docs],
               "timings": {"write_s": round(time.monotonic() - started, 1)}}
    _save("episode.json", episode)
    print(json.dumps({"id": ep_id, "title": episode["title"], "writer": episode["writer"], "words": words,
                      "chapters": len(episode["chapters"]),
                      "scenes": sum(len(c["scenes"]) for c in episode["chapters"])}, indent=1))


def images_cmd(args) -> None:
    from . import images

    rep = images.generate(_load("episode.json"), W / "images", args.k, args.n)
    print(json.dumps({k: rep[k] for k in ("shard", "model", "load_s", "seconds")}))


def voice_cmd(args) -> None:
    from . import narrate

    rep = narrate.narrate(_load("episode.json"), W / "voice", args.k, args.n)
    print(json.dumps({"shard": rep["shard"], "seconds": rep["seconds"], "chapters": len(rep["chapters"])}))


def audio(args) -> None:
    from . import ambience, mix, score, timeline

    ep = _load("episode.json")
    shape = config.shape(ep["shape"])
    started = time.monotonic()
    tl = timeline.build(ep, W / "voice", W / "reports", shape.tail_minutes, shape.render_shards)
    _save("timeline.json", tl)
    key = config.series()[ep["series"]]["ambience"]
    amb = ambience.build_loop(key, W / "ambience.wav", W / "ambience-cache")
    sc = score.render(W / "score.wav", [c["start"] for c in tl["chapters"]], tl["seconds"], ep["id"])
    nar = mix.narration_track(tl["narration"], W / "voice", tl["seconds"], W / "narration.flac")
    narration_end = tl["chapters"][-1]["narration_start"] + tl["chapters"][-1]["narration_seconds"]
    result = mix.mix(nar, W / "score.wav", W / "ambience.wav", tl["seconds"], narration_end, W / "mix.flac")
    mix.to_aac(W / "mix.flac", W / "mix.m4a")
    for p in ("score.wav", "narration.flac", "mix.flac"):
        (W / p).unlink()
    report = {"ambience": amb, "score": sc, **result, "seconds": round(time.monotonic() - started, 1),
              "timeline": {k: tl[k] for k in ("seconds", "frames", "tail_start", "swapped")},
              "render_shards": len(tl["render_shards"])}
    _save("audio.json", report)
    print(json.dumps(report["loudness"]), json.dumps(report["timeline"]))


def render_cmd(args) -> None:
    from . import render

    tl = _load("timeline.json")
    if args.k >= len(tl["render_shards"]):
        print(f"render shard {args.k}: nothing to do")
        return
    out = W / "segments" / f"seg{args.k:02d}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    rep = render.segment(tl, args.k, W / "images", out, W / "stills")
    _save(f"segments/seg{args.k:02d}.json", rep)
    print(json.dumps(rep))


def thumb_source(images: Path, usable=None) -> Path:
    """The thumbnail's own image, or when every take of it failed the gates, the earliest chapter plate (plates
    exist only for takes that passed the hard person/text checks). usable(path) -> bool: skip candidates that fail
    it (a key with no bright focal subject); when none passes, the first candidate, whose gates then report why."""
    own = images / "thumb.png"
    candidates = ([own] if own.exists() else []) + sorted(images.glob("c[0-9][0-9]s[0-9][0-9].jpg"))
    if not candidates:
        raise RuntimeError("no thumbnail image and no chapter plate to fall back on")
    return next((c for c in candidates if usable is None or usable(c)), candidates[0])


def finish(args) -> None:
    from PIL import Image

    from . import finish as fin
    from . import shorts, thumbnail

    ep, tl = _load("episode.json"), _load("timeline.json")
    out = W / "out"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    final = fin.assemble(tl, W / "segments", fin.limited_mix(W / "mix.m4a"), out / "episode.mp4")
    t_assemble = time.monotonic() - t0
    words = ep["thumbnail"]["words"].split()[:5]
    if len(words) < 2:
        words = ep["title_core"].split()[:4]
    key_image = thumb_source(W / "images", lambda c: not thumbnail.source_problems(Image.open(c)))
    thumb = thumbnail.build(key_image, words, out / "thumbnail.jpg")
    key = config.series()[ep["series"]]["ambience"]
    made = [shorts.build(ep, k, W / "voice", W / "images", out, key) for k in range(len(ep["trailers"]))]
    report = fin.qa(ep, tl, final, out / "checks", thumb, made)
    report["thumbnail"] = thumb
    report["timings"] = {"assemble_s": round(t_assemble, 1), "finish_s": round(time.monotonic() - t0, 1)}
    report["episode"] = {k: ep[k] for k in ("id", "title", "writer", "publishable", "mode", "shape", "words")}
    report["shorts_detail"] = made
    _save("out/qa.json", report)
    fin.excerpt(final, out / "excerpt-opening.mp4", 0, 100)
    if len(tl["chapters"]) > 1:
        fin.excerpt(final, out / "excerpt-chapter2.mp4", max(0, tl["chapters"][1]["start"] - 10), 40)
    print(json.dumps({k: report[k] for k in ("ok", "problems", "duration", "loudness", "size_mb")}, indent=1))


def _channel_stats() -> None:
    """Read the channel's numbers for the record step; never blocks a publish."""
    from . import learn, youtube

    if not os.environ.get("YOUTUBE_OAUTH_JSON"):
        return
    try:
        api = youtube.client()
        _save("out/channel_stats.json", learn.fetch(api, youtube.identity(api)))
    except Exception as exc:
        print(f"channel stats: {type(exc).__name__}: {exc}"[:300])


def publish(args) -> None:
    from . import timeline, youtube

    ep, tl, qa = _load("episode.json"), _load("timeline.json"), _load("out/qa.json")
    _channel_stats()
    why = youtube.refusals(ep, args.mode, _paused(), qa)
    if why:
        print("not publishing:\n- " + "\n- ".join(why))
        _save("out/publish.json", {"published": False, "refusals": why})
        if args.mode == "publish" and not _paused():
            sys.exit("a publish run made nothing publishable: failing the run so the alert and the doctor see it")
        return
    api = youtube.client()
    channel = youtube.identity(api)
    audio = _load("audio.json")
    rows = timeline.youtube_chapters(tl, f"{config.series()[ep['series']]['ambience'].title()} ambience for sleep")
    desc = youtube.description(ep, rows, ep["docs"], audio["ambience"]["recordings"],
                               _model_label(), os.environ.get("PANGAEA_VOICE", "bm_george"))
    out = W / "out"
    when = youtube.soon() if os.environ.get("PANGAEA_RELEASE") == "now" else youtube.episode_slot()
    main = youtube.upload(api, channel, out / "episode.mp4", ep["title"], desc, youtube.tags(ep), when,
                          out / "thumbnail.jpg")
    when = main.get("publish_at") or when
    result = {"published": True, "episode": main, "shorts": []}
    _save("out/publish.json", result)
    for s in _load("out/qa.json")["shorts_detail"]:
        if not s.get("ok"):
            continue
        tr = ep["trailers"][s["k"]]
        tag_list = [*youtube.tags(ep)[:-1][:8], f"{youtube.MARKER}{ep['id']}-short{s['k']}"]
        res = youtube.upload(api, channel, Path(s["path"]), youtube.short_title(tr["hook"]),
                             youtube.short_description(main["url"], ep["title"], ep["summary"]), tag_list,
                             youtube.after(when.replace(".000Z", "Z"), 14.5 + 24 * s["k"]))
        result["shorts"].append(res)
        _save("out/publish.json", result)
    problems = readback_alerts(result)
    if problems:
        from . import alert

        alert.send(f"Pangaea Nights episode {ep['id']}: read-back mismatch:\n" + "\n".join(problems))
    print(json.dumps(result, indent=1))


def readback_alerts(pub: dict) -> list[str]:
    """One line per uploaded video whose read-back (made for kids, privacy, publishAt) differs from what was asked."""
    rows = [pub["episode"], *pub.get("shorts", [])] if pub.get("published") else []
    return [f"- {r['url']}: " + "; ".join(r["readback_problems"]) for r in rows if r.get("readback_problems")]


def promote(args) -> None:
    """Turn a full-length dry run's episode into a publish episode, so its finished media can be published without
    a rebuild. The publish guard still applies: real writer, full shape, media QA, not paused."""
    ep = _load("episode.json")
    if ep.get("shape") not in config.LONG_SHAPES or ep.get("writer") != "gemini":
        sys.exit("only a full-length episode from the real writer can be promoted")
    if ep.get("mode") != "publish":
        ep["promoted_from"] = ep["id"]
        ep["id"] = f"{ep['created'][:10]}-{ep['topic']}"
        ep["mode"] = "publish"
        _save("episode.json", ep)
    print(json.dumps({k: ep.get(k) for k in ("id", "promoted_from", "title", "mode")}, indent=1))


def _model_label() -> str:
    from . import sdcpp

    for prefix, label in (("klein", "FLUX.2 klein 4B"), ("zimage", "Z-Image-Turbo"), ("qwen", "Qwen-Image-2512")):
        if sdcpp.DEFAULT.startswith(prefix):
            return label
    return sdcpp.DEFAULT


def record(args) -> None:
    ep = _load("episode.json")
    data = ledger.load()
    pub_path, qa_path = W / "out" / "publish.json", W / "out" / "qa.json"
    pub = json.loads(pub_path.read_text()) if pub_path.exists() else {"published": False}
    qa = json.loads(qa_path.read_text()) if qa_path.exists() else {}
    entry = {"id": ep["id"], "topic": ep["topic"], "series": ep["series"], "title": ep["title"],
             "writer": ep["writer"], "words": ep["words"], "shape": ep["shape"], "mode": ep["mode"],
             "created": ep["created"], "qa_ok": qa.get("ok"), "duration": qa.get("duration"),
             "run": os.environ.get("GITHUB_RUN_ID")}
    mismatches = readback_alerts(pub)
    if mismatches:
        ledger.ladder_log(data, "alerts", {"kind": "readback", "episode": ep["id"], "run": entry["run"],
                                           "problems": mismatches}, unique=("kind", "episode", "run"))
    # episode.yml opens an "alert" issue from this file
    (W / "out").mkdir(parents=True, exist_ok=True)
    (W / "out" / "alerts.txt").write_text(f"{ep['id']}:\n" + "\n".join(mismatches) + "\n" if mismatches else "")
    if pub.get("published"):
        main = pub["episode"]
        entry.update({"video_id": main["video_id"], "url": main["url"], "publish_at": main["publish_at"],
                      "uploaded_at": ledger.now(), "shorts": [s["video_id"] for s in pub["shorts"]]})
        ledger.record_episode(data, entry)
    else:
        entry["refusals"] = pub.get("refusals")
        ledger.record_dry_run(data, entry)
    stats_path = W / "out" / "channel_stats.json"
    if stats_path.exists():
        from . import learn

        try:  # the episode record above must be saved even if the numbers are malformed
            learn.update(data, json.loads(stats_path.read_text()))
        except Exception as exc:
            print(f"learn: {type(exc).__name__}: {exc}"[:300])
    ledger.save(data)
    print(json.dumps(entry, indent=1))


def health(args) -> None:
    data = ledger.load()
    if _paused():
        print("paused: no upload expected")
        return
    hours = ledger.hours_since_upload(data)
    if hours is None or hours > args.hours:
        sys.exit(f"no upload for {'ever' if hours is None else f'{hours:.0f} hours'} (limit {args.hours})")
    print(f"last upload {hours:.1f} hours ago")


def matrix(args) -> None:
    """GitHub Actions outputs: the shape and the shard lists for each parallel stage."""
    shape = config.shape(args.shape)
    rows = {"shape": shape.name, "images": list(range(shape.image_shards)),
            "voice": list(range(shape.voice_shards)), "render": list(range(shape.render_shards))}
    lines = [f"{k}={json.dumps(v) if isinstance(v, list) else v}" for k, v in rows.items()]
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="studio")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("matrix")
    s.add_argument("--shape", choices=list(config.SHAPES), default="full")
    s.set_defaults(fn=matrix)
    s = sub.add_parser("plan")
    s.add_argument("--mode", choices=["publish", "dry", "short"], default="dry")
    s.add_argument("--shape", choices=list(config.SHAPES), default="full")
    s.add_argument("--writer", choices=["auto", "gemini", "stub"], default="auto")
    s.add_argument("--topic", default="")
    s.set_defaults(fn=plan)
    for name, fn in (("images", images_cmd), ("voice", voice_cmd)):
        s = sub.add_parser(name)
        s.add_argument("k", type=int)
        s.add_argument("n", type=int)
        s.set_defaults(fn=fn)
    sub.add_parser("audio").set_defaults(fn=audio)
    s = sub.add_parser("render")
    s.add_argument("k", type=int)
    s.set_defaults(fn=render_cmd)
    sub.add_parser("finish").set_defaults(fn=finish)
    s = sub.add_parser("publish")
    s.add_argument("--mode", choices=["publish", "dry", "short"], default="dry")
    s.set_defaults(fn=publish)
    sub.add_parser("promote").set_defaults(fn=promote)
    sub.add_parser("record").set_defaults(fn=record)
    s = sub.add_parser("health")
    s.add_argument("--hours", type=float, default=30)
    s.set_defaults(fn=health)
    from . import branding, ladder_cli, topics

    branding.register(sub)
    ladder_cli.register(sub)
    topics.register(sub)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
