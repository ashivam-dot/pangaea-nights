"""Regression tests for the review fixes on the ladder Shorts, the episode read-back and the thumbnail."""
import copy
import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest
from PIL import Image

from studio import cli, config, fanout, ladder, ladder_cli, ledger, qa, shorts_ladder, thumbnail, voice, youtube

BANKS = ladder.load_banks(strict=True)
ALL = [p for ps in BANKS.values() for p in ps]
DAY = date(2026, 11, 10)


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK", tmp_path / "work")
    monkeypatch.setattr(config, "STATE", tmp_path / "state")
    for k in ("GITHUB_OUTPUT", "GITHUB_RUN_ID", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", ladder_cli.TEST_WHILE_PAUSED):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


# ---------------------------------------------------------------------------------------------------- cli
def test_plan_accepts_the_empty_slot_the_workflow_passes(dirs, capsys):
    cli.main(["ladder-plan", "--mode", "dry", "--slot", "", "--premise", ""])
    assert (config.WORK / "ladder" / "A" / "plan.json").exists()
    assert "any=true" in capsys.readouterr().out


def test_image_matrix_fits_one_wave():
    for slots in (["A"], ["A", "B"]):
        m = ladder_cli.image_matrix(slots, len(ladder.IMAGE_IDS))
        assert len(m) <= ladder_cli.IMAGE_JOBS and {r["slot"] for r in m} == set(slots)
    m = ladder_cli.image_matrix(["A", "B"], 9)
    assert [r["shard"] for r in m if r["slot"] == "B"][-1] == "7/8"


def test_choose_falls_through_to_the_next_premise(monkeypatch):
    data = copy.deepcopy(ledger.EMPTY)
    first = ladder.pick(BANKS, data, DAY)
    real = ladder.plan_problems

    def fake(p, *a, **k):
        g = real(p, *a, **k)
        if p["id"] == first["id"]:
            g["novelty_4gram"]["ok"] = False
        return g

    monkeypatch.setattr(ladder, "plan_problems", fake)
    p, g, skipped = ladder_cli.choose(BANKS, data, DAY, [], "publish", None)
    assert p["id"] != first["id"] and skipped == [first["id"]] and g["novelty_4gram"]["ok"]
    p, _, _ = ladder_cli.choose(BANKS, data, DAY, [], "publish", first["id"])  # an override is taken as given
    assert p["id"] == first["id"]


def test_test_upload_while_paused_needs_its_own_switch(monkeypatch):
    monkeypatch.setenv("PAUSED", "true")
    monkeypatch.delenv(ladder_cli.TEST_WHILE_PAUSED, raising=False)
    assert ladder_cli.upload_refusal("test") and ladder_cli.upload_refusal("publish")
    monkeypatch.setenv(ladder_cli.TEST_WHILE_PAUSED, "true")
    assert ladder_cli.upload_refusal("test") is None
    assert ladder_cli.upload_refusal("publish") and ladder_cli.upload_refusal("dry")
    monkeypatch.setenv("PAUSED", "false")
    assert ladder_cli.upload_refusal("publish") is None


def _gates_json(frame0, embeddings, ok=True):
    return {"ok": ok, "frame0": qa.pack(np.asarray(frame0, np.float32)),
            "images": {"embeddings": {k: qa.pack(np.asarray(v, np.float32)) for k, v in embeddings.items()}}}


def test_slot_b_is_refused_when_too_close_to_slot_a(dirs):
    a = _gates_json([1, 0, 0], {"o1": [0, 1, 0]})
    (config.WORK / "ladder" / "A" / "out").mkdir(parents=True)
    (config.WORK / "ladder" / "A" / "out" / "gates.json").write_text(json.dumps(a))
    far = _gates_json([0, 0, 1], {"o1": [1, 0, 1]})
    same = _gates_json([1, 0, 0.01], {"o1": [0, 1, 0]})
    assert ladder_cli.sibling_refusals("B", far) == []
    why = ladder_cli.sibling_refusals("B", same)
    assert any("first frame" in x for x in why) and any("option image" in x for x in why)
    assert ladder_cli.sibling_refusals("A", same) == []
    (config.WORK / "ladder" / "A" / "out" / "gates.json").write_text(json.dumps({**a, "ok": False}))
    assert ladder_cli.sibling_refusals("B", same) == []  # A failed: B may still go out


def _plan(slot, premise, run="7", mode="publish"):
    d = config.WORK / "ladder" / slot
    (d / "out").mkdir(parents=True, exist_ok=True)
    (d / "plan.json").write_text(json.dumps({"slot": slot, "mode": mode, "run": run, "premise": premise,
                                             "script": ladder.script_text(premise)}))
    (d / "out" / "gates.json").write_text(json.dumps({"ok": True, "failures": []}))
    return d


def test_record_is_idempotent_and_files_readback_alerts(dirs):
    p = ALL[0]
    d = _plan("A", p)
    (d / "out" / "publish.json").write_text(json.dumps({
        "published": True, "video_id": "v1", "url": "u1", "title": "t", "publish_at": "2026-11-10T14:30:00Z",
        "readback_problems": ["madeForKids is True, not False"]}))
    (config.WORK / "ladder" / "plan_alerts.json").write_text(json.dumps(
        [{"kind": "plan", "slot": "B", "run": "7", "problems": ["no premise"]}]))
    for _ in range(2):  # a re-run of the record job
        ladder_cli.record(None)
    data = ledger.load()
    assert len(data["ladder"]["builds"]) == 1 and len(data["shorts"]) == 1
    assert sorted(a["kind"] for a in data["ladder"]["alerts"]) == ["plan", "readback"]
    text = (config.WORK / "ladder" / "alerts.txt").read_text()
    assert "madeForKids" in text and "no premise" in text


def test_record_reads_a_one_slot_day_unpacked_straight_into_dl(dirs):
    p = ALL[0]
    d = _plan("A", p)
    (d / "out" / "gates.json").unlink()
    dl = config.WORK / "dl"
    dl.mkdir(parents=True)
    (dl / "gates.json").write_text(json.dumps({"ok": True, "failures": []}))
    (dl / "publish.json").write_text(json.dumps({
        "published": True, "video_id": "v1", "url": "u1", "title": "t", "publish_at": "2026-11-10T14:30:00Z",
        "slot": "A"}))
    ladder_cli.record(None)
    data = ledger.load()
    assert data["ladder"]["builds"][0]["ok"] is True and data["ladder"]["builds"][0]["published"] is True
    assert [s["video_id"] for s in data["shorts"]] == ["v1"]


def test_ladder_log_unique_replaces_and_keeps_the_first_time():
    data = copy.deepcopy(ledger.EMPTY)
    ledger.ladder_log(data, "builds", {"run": "1", "slot": "A", "ok": False}, unique=("run", "slot"))
    first = data["ladder"]["builds"][0]["at"]
    ledger.ladder_log(data, "builds", {"run": "1", "slot": "A", "ok": True}, unique=("run", "slot"))
    ledger.ladder_log(data, "builds", {"run": "1", "slot": "B", "ok": True}, unique=("run", "slot"))
    b = data["ladder"]["builds"]
    assert len(b) == 2 and b[0]["ok"] is True and b[0]["at"] == first


# ---------------------------------------------------------------------------------------------------- picking
def test_a_premise_that_keeps_failing_rests():
    data = copy.deepcopy(ledger.EMPTY)
    first = ladder.pick(BANKS, data, DAY)
    for i in range(ladder.FAILED_MAX):
        data["ladder"]["builds"].append({"at": f"{(DAY - timedelta(days=i)).isoformat()}T08:00:00Z",
                                         "mode": "publish", "premise": first["id"], "ok": False,
                                         "published": False})
    assert ladder.demoted(data, DAY) == {first["id"]: ladder.FAILED_MAX}
    assert ladder.pick(BANKS, data, DAY)["id"] != first["id"]
    assert ladder.demoted(data, DAY + timedelta(days=ladder.FAILED_DAYS + 1)) == {}
    assert ladder.pick(BANKS, data, DAY, exclude={first["id"]})["id"] != first["id"]


def test_shorts_health_flags_a_stall():
    launch = date(2026, 11, 1)
    data = copy.deepcopy(ledger.EMPTY)
    now = datetime(2026, 11, 3, 12, tzinfo=timezone.utc)
    assert ladder.stale(data, None, now, 30) is None
    assert ladder.stale(data, date(2026, 11, 5), now, 30) is None
    assert "since the launch" in ladder.stale(data, launch, now, 30)
    data["shorts"].append({"video_id": "v", "uploaded_at": "2026-11-03T00:00:00Z"})
    assert ladder.stale(data, launch, now, 30) is None
    assert "for 36 hours" in ladder.stale(data, launch, now + timedelta(hours=24), 30)


# ---------------------------------------------------------------------------------------------------- bank checks
def test_banner_and_composed_prompts_hit_the_blocklists():
    p = copy.deepcopy(ALL[0])
    p["banner"] = p["banner"] + " Dogesh"
    assert any("banner hits the blocklist" in x for x in ladder.premise_problems(p, p["series"]))
    p = copy.deepcopy(ALL[0])
    p["art_direction"] = p["art_direction"].rstrip(".") + ", a lone backpack on the moss."
    assert any("prompt: mentions 'backpack'" in x for x in ladder.premise_problems(p, p["series"]))


def test_a_premise_too_long_for_the_timeline_fails_the_bank_check():
    p = copy.deepcopy(ALL[0])
    assert shorts_ladder.fit_problems(p) == []
    p["verdict_vo"] = p["verdict_vo"] + " And that is the whole story of this one, told slowly, word by word, for you."
    assert any("over 44.7s" in x for x in shorts_ladder.fit_problems(p))


# ---------------------------------------------------------------------------------------------------- render gates
def test_motion_is_zero_for_a_frozen_picture_and_positive_for_a_camera_move():
    rng = np.random.default_rng(0)
    plate = rng.integers(0, 255, (96, 64, 3)).astype(np.uint8)
    frozen = [shorts_ladder.motion_sample(plate) for _ in range(4)]
    assert np.median(shorts_ladder.motion_diffs(frozen)) == 0.0
    assert not shorts_ladder.gate("motion", 0.0)["ok"]
    x = np.linspace(0, 6 * np.pi, 256)
    waves = (127 + 100 * np.sin(x)[None, :, None] * np.ones((128, 1, 3))).astype(np.uint8)
    pan = [shorts_ladder.motion_sample(np.roll(waves, k, axis=1)) for k in range(4)]  # a slow pan
    assert shorts_ladder.gate("motion", float(np.median(shorts_ladder.motion_diffs(pan))))["ok"]
    strobe = [shorts_ladder.motion_sample(plate if k % 2 else 255 - plate) for k in range(4)]
    assert not shorts_ladder.gate("motion", float(np.median(shorts_ladder.motion_diffs(strobe))))["ok"]


def test_a_flat_depth_map_has_no_parallax():
    motion = shorts_ladder.MOTIONS[0]
    assert shorts_ladder.depth_spread(np.full((64, 64), 0.5, np.float32)) == 0.0
    assert shorts_ladder.parallax_fraction(motion, 1300, 2300, 0.0) == 0.0
    assert not shorts_ladder.gate("parallax_frac", 0.0)["ok"]
    ramp = np.tile(np.linspace(0, 1, 64, dtype=np.float32), (64, 1))
    assert shorts_ladder.depth_spread(ramp) > 0.7


def test_onset_measures_leading_silence():
    sr = 24000
    a = np.concatenate([np.zeros(int(0.5 * sr), np.float32), 0.5 * np.ones(sr, np.float32)])
    assert abs(shorts_ladder.onset_s(a, sr) - 0.5) < 0.01


def test_trim_moves_word_times_with_the_cut():
    sr = voice.SR
    a = np.concatenate([np.zeros(sr, np.float32), 0.5 * np.ones(sr, np.float32), np.zeros(sr // 2, np.float32)])
    trimmed, cut = voice._trim(a)
    assert cut == sr - int(0.04 * sr) and len(trimmed) < len(a)
    words = voice.shift_words([{"word": "hi", "start": 1.0, "end": 1.5}, {"word": "x", "start": 0.0, "end": 9.0}],
                              -cut / sr, len(trimmed) / sr)
    assert abs(words[0]["start"] - 0.04) < 1e-6
    assert words[1]["start"] == 0.0 and words[1]["end"] == len(trimmed) / sr


def _gates(voice_extra=None, mode="publish", rotation_ok=True, hook_onset=0.1):
    p = ALL[0]
    durs = {"hook": 1.4, **{f"o{n}": 2.4 for n in range(1, 9)}, "verdict": 6.0, "cta": 1.2}
    tl = shorts_ladder.timeline(p, durs)
    vo = {"durations": durs, "wer": 0.02, "wpm": 165.0,
          "segments": {"hook": {"wer": 0.0}, "verdict": {"wer": 0.05, "heard": f"Real answer, number {p['answer_n']}."}}}
    vo.update(voice_extra or {})
    plan = {"mode": mode, "premise": p, "gates": {"novelty_4gram": {"value": 0.1}, "premise": {"ok": True},
                                                  "reuse": {"ok": True},
                                                  "rotation": {"ok": rotation_ok, "problems": []}}}
    return shorts_ladder.gates(tl, {}, {}, vo, {}, plan, None, hook_onset)


def test_voice_gates_read_each_line_and_the_spoken_answer():
    g = _gates()
    assert g["wer_line_max"]["ok"] and g["verdict_number_heard"]["ok"] and g["hook_vo_onset_s"]["ok"]
    assert g["option_vo_min_s"]["ok"] and g["option_vo_max_s"]["ok"]
    bad = _gates({"segments": {"o3": {"wer": 0.5}, "verdict": {"wer": 0.0, "heard": "Real answer, number nine."}}})
    assert not bad["wer_line_max"]["ok"] and bad["wer_line_max"]["line"] == "o3"
    assert not bad["verdict_number_heard"]["ok"]
    assert not _gates(hook_onset=0.6)["hook_vo_onset_s"]["ok"]
    assert not _gates(hook_onset=None)["hook_vo_onset_s"]["ok"]
    assert not _gates({"wpm": 150.0})["wpm"]["ok"]


def test_rotation_holds_only_a_publish_build():
    assert not _gates(rotation_ok=False)["rotation"]["ok"]
    assert _gates(rotation_ok=False, mode="test")["rotation"]["ok"]


# ---------------------------------------------------------------------------------------------------- upload
WHEN = "2026-11-03T14:30:00Z"


def _video(**status):
    st = {"privacyStatus": "private", "publishAt": WHEN, "madeForKids": False, "selfDeclaredMadeForKids": False,
          **status}
    return {"snippet": {"channelId": config.CHANNEL_ID}, "status": st}


def test_readback_after_the_release_accepts_public():
    after = datetime(2026, 11, 3, 15, tzinfo=timezone.utc)
    before = datetime(2026, 11, 3, 14, tzinfo=timezone.utc)
    released = _video(privacyStatus="public", publishAt=None)
    assert youtube.readback_problems(released, WHEN, after) == []
    assert youtube.readback_problems(released, WHEN, before)
    assert youtube.readback_problems(_video(privacyStatus="unlisted"), WHEN, after)
    assert youtube.readback_problems(_video(privacyStatus="public", madeForKids=True), WHEN, after)


def test_an_existing_upload_keeps_its_late_slot():
    late = datetime(2026, 11, 3, 14, 20, tzinfo=timezone.utc)
    assert youtube.slot_publish_at("A", date(2026, 11, 3), late) is None
    assert youtube.slot_publish_at("A", date(2026, 11, 3), late, late_ok=True) == WHEN


def test_episode_readback_alerts():
    pub = {"published": True, "episode": {"url": "e", "readback_problems": ["publishAt is None"]},
           "shorts": [{"url": "s", "readback_problems": []}]}
    assert cli.readback_alerts(pub) == ["- e: publishAt is None"]
    assert cli.readback_alerts({"published": False}) == []


# ---------------------------------------------------------------------------------------------------- thumbnail
def test_a_dark_key_without_a_focal_light_is_rejected():
    dark = Image.fromarray(np.full((180, 320, 3), 30, np.uint8))
    lit = np.full((180, 320, 3), 30, np.uint8)
    lit[40:140, 200:300] = 200
    assert thumbnail.source_problems(dark) and thumbnail.source_problems(Image.fromarray(lit)) == []


def test_thumbnail_tonal_spread_has_a_floor_and_a_ceiling():
    assert thumbnail.spread_problems(10.0) and thumbnail.spread_problems(250.0)
    assert thumbnail.spread_problems(120.0) == []
    grey = Image.fromarray(np.full((180, 320, 3), 128, np.uint8))
    assert thumbnail.spread(grey) == 0.0


def test_thumb_source_skips_an_unusable_key(tmp_path):
    for name in ("thumb.png", "c01s00.jpg", "c01s01.jpg"):
        (tmp_path / name).write_bytes(b"x")
    assert cli.thumb_source(tmp_path, lambda c: c.name != "thumb.png").name == "c01s00.jpg"
    assert cli.thumb_source(tmp_path, lambda c: False).name == "thumb.png"  # none usable: the first, gates report


# ---------------------------------------------------------------------------------------------------- fan-out
def test_a_late_single_runner_stage_is_not_fanned_out():
    assert not fanout.fanned_out([{"name": "plan", "status": "completed"}, {"name": "finish", "status": "in_progress"}])
    assert fanout.fanned_out([{"name": "images (3)", "status": "queued"}])
    assert fanout.fanned_out([{"name": "render (A)", "status": "in_progress"}])


def test_exposure_grade_lifts_dark_take_with_a_focal_light():
    from PIL import Image
    from studio import ladder_images as li
    a = np.full((200, 120), 30, np.uint8)
    a[:, :60] = np.linspace(10, 90, 60, dtype=np.uint8)
    a[90:110, 50:70] = 230  # warm focal light
    graded, st = li.expose(Image.fromarray(a).convert("RGB"), 130.0)
    assert 110 <= np.asarray(graded.convert("L")).mean() <= 150
    assert li.exposure_problems(st) == []


def test_exposure_grade_cannot_rescue_a_flat_muddy_take():
    from PIL import Image
    from studio import ladder_images as li
    flat = Image.fromarray(np.full((200, 120), 35, np.uint8)).convert("RGB")
    _, st = li.expose(flat, 130.0)
    probs = li.exposure_problems(st)
    assert any("focal light" in p for p in probs) and any("tonal spread" in p for p in probs)


def _verdict(reasons, brightness=130.0, similarity=0.30):
    from types import SimpleNamespace
    return SimpleNamespace(reasons=reasons, brightness=brightness, similarity=similarity)


def test_ladder_person_gate_uses_figure_contrast_not_sleepdoc_label():
    from studio import ladder_images as li, qa
    job = {"luma": (92, 168)}
    # dry run 2: an empty fireball scene got "person 0.54" from the sleep-doc labels; the figure contrast clears it
    assert li.take_problems(_verdict(["person 0.54"]), job, figure=0.20) == []
    assert li.take_problems(_verdict([]), job, figure=qa.FIGURE_LIMIT + 0.1) == [f"person {qa.FIGURE_LIMIT + 0.1:.2f}"]
    # the other hard reasons still refuse a take
    assert li.take_problems(_verdict(["text 0.41", "broken 0.50"]), job, figure=0.1) == ["text 0.41", "broken 0.50"]


def test_frame0_dark_take_reaches_option1_band():
    from PIL import Image
    from studio import ladder_images as li
    a = np.full((200, 120), 34, np.uint8)
    a[:, :60] = np.linspace(8, 70, 60, dtype=np.uint8)
    a[90:110, 50:70] = 240
    graded, st = li.expose(Image.fromarray(a).convert("RGB"), 130.0)
    assert st["raw_mean"] < 45 and 92 <= np.asarray(graded.convert("L")).mean() <= 168


def test_soft_fallback_only_for_mild_softness():
    from studio import ladder_images as li
    assert li.soft_only(["mushy 7.2"], 7.2)
    assert not li.soft_only(["mushy 5.1"], 5.1)  # truly mushy stays refused
    assert not li.soft_only(["broken 0.49", "mushy 7.5"], 7.5)
    assert not li.soft_only(["mushy 7.0", "luma 80 outside 92-168"], 7.0)
    assert not li.soft_only(["person 0.62"], 12.0) and not li.soft_only([], 12.0)


def test_a_dry_run_refused_only_as_too_close_to_its_sibling_ends_red(dirs, monkeypatch):
    from argparse import Namespace
    monkeypatch.setenv("PAUSED", "false")
    d = _plan("B", ALL[0], mode="dry")
    (d / "out" / "gates.json").write_text(json.dumps(_gates_json([1, 0, 0.01], {"o1": [0, 1, 0]})))
    a = config.WORK / "ladder" / "A" / "out"
    a.mkdir(parents=True, exist_ok=True)
    (a / "gates.json").write_text(json.dumps(_gates_json([1, 0, 0], {"o1": [0, 1, 0]})))
    with pytest.raises(SystemExit, match="sibling"):
        ladder_cli.publish(Namespace(slot="B", mode="dry"))


def test_take_score_ranks_by_figure_not_sleepdoc_person():
    from studio import ladder_images as li
    v = qa.Verdict(True, [], 0.6, 0.0, 0.0, 0.30, 6.0, 120.0, 40.0, 12.0)
    assert abs(li.take_score(v, 0.05) - (6.0 + 12 * 0.30 - 4 * 0.05)) < 1e-9
