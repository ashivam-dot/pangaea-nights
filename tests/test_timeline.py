import json

from studio import timeline


def setup(tmp_path, chapters=3, scenes=4, missing=()):
    voice, reps = tmp_path / "voice", tmp_path / "reports"
    voice.mkdir()
    reps.mkdir()
    ep = {"id": "dry-x", "title_core": "The Cambrian Seas", "chapters": [], "tail_scenes": ["a", "b"]}
    images = []
    for n in range(1, chapters + 1):
        ep["chapters"].append({"n": n, "title": f"Chapter {n}", "scenes": ["s"] * scenes})
        (voice / f"ch{n:02d}.json").write_text(json.dumps({"duration": 300.0 + n}))
        images += [f"c{n:02d}s{j:02d}" for j in range(scenes)]
    images += ["tail00", "tail01"]
    rows = [{"id": i, "plate": f"{i}.jpg"} for i in images if i not in missing]
    (reps / "shard-00.json").write_text(json.dumps({"images": rows}))
    return ep, voice, reps


def test_shots_tile_the_whole_episode(tmp_path):
    ep, voice, reps = setup(tmp_path)
    tl = timeline.build(ep, voice, reps, tail_minutes=2, render_shards=4)
    shots = tl["shots"]
    assert shots[0]["f0"] == 0 and shots[-1]["f1"] == tl["frames"]
    assert all(a["f1"] == b["f0"] for a, b in zip(shots, shots[1:]))
    assert all(s["f1"] > s["f0"] for s in shots)
    assert tl["chapters"][0]["narration_start"] == timeline.INTRO
    assert abs(tl["seconds"] - (tl["tail_start"] + 120)) < 0.1


def test_render_shards_cover_every_frame_once(tmp_path):
    ep, voice, reps = setup(tmp_path, chapters=4, scenes=5)
    tl = timeline.build(ep, voice, reps, tail_minutes=1, render_shards=6)
    sh = tl["render_shards"]
    assert sh[0]["f0"] == 0 and sh[-1]["f1"] == tl["frames"]
    assert all(a["f1"] == b["f0"] and a["last"] + 1 == b["first"] for a, b in zip(sh, sh[1:]))
    sizes = [s["f1"] - s["f0"] for s in sh]
    assert max(sizes) < 2.2 * min(sizes)


def test_unusable_images_are_swapped_for_neighbours(tmp_path):
    ep, voice, reps = setup(tmp_path, missing=("c01s02", "tail01"))
    tl = timeline.build(ep, voice, reps, tail_minutes=1, render_shards=2)
    by_slot = {s["slot"]: s for s in tl["shots"]}
    assert by_slot["c01s02"]["swapped"] and by_slot["c01s02"]["image"] in ("c01s01", "c01s03")
    assert by_slot["tail01"]["image"] == "tail00"
    assert set(tl["swapped"]) == {"c01s02", "tail01"}


def test_youtube_chapters_start_at_zero(tmp_path):
    ep, voice, reps = setup(tmp_path)
    tl = timeline.build(ep, voice, reps, tail_minutes=2, render_shards=2)
    rows = timeline.youtube_chapters(tl, "Ocean ambience for sleep")
    assert rows[0] == ("0:00", "Chapter 1") and rows[-1][1] == "Ocean ambience for sleep" and len(rows) == 4
