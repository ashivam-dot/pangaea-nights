"""The ladder Shorts: banks, cadence ramp, rotation, novelty, timeline, upload flags and read-back."""
import copy
from datetime import date, datetime, timedelta, timezone

import pytest

from studio import config, ladder, ledger, shorts_ladder, youtube

BANKS = ladder.load_banks(strict=True)
ALL = [p for ps in BANKS.values() for p in ps]


def test_every_bank_validates_strictly():
    assert sorted(BANKS) == sorted(ladder.SERIES)
    for path in sorted(ladder.BANKS.glob("*.json")):
        assert ladder.bank_problems(ladder.load_bank(path)) == {}, path.name
    assert all(len(ps) >= 16 for ps in BANKS.values()), {s: len(ps) for s, ps in BANKS.items()}


def test_premises_are_distinct():
    for key in ("id", "title", "art_direction", "banner", "hook_vo"):
        values = [p[key].strip().lower() for p in ALL]
        assert len(values) == len(set(values)), key
    prompts = [o["image_prompt"].strip().lower() for p in ALL for o in p["options"]]
    assert len(prompts) == len(set(prompts))


def test_bank_scripts_pass_the_novelty_gate_against_each_other():
    scripts = {p["id"]: ladder.script_text(p) for p in ALL}
    worst = max((ladder.overlap(s, [t for j, t in scripts.items() if j != i]), i) for i, s in scripts.items())
    assert worst[0] <= ladder.NOVELTY_MAX, worst


def test_no_mascot_and_a_spoken_verdict():
    for p in ALL:
        assert p["verdict_vo"].startswith(f"Real answer: number {p['answer_n']}"), p["id"]
        text = " ".join([p["verdict_vo"], p["hook_vo"], p["art_direction"], p["description"]])
        assert not ladder.MASCOT.search(text), p["id"]  # the plant is fine


def test_bank_failures_are_caught():
    p = copy.deepcopy(ALL[0])
    p["fact"]["text"] = p["fact"]["text"] + " It weighed 9999 tonnes."
    assert any("9999" in x for x in ladder.premise_problems(p, p["series"]))
    p = copy.deepcopy(ALL[0])
    p["options"][1]["image_prompt"] += " with a wooden paddle"
    assert ladder.premise_problems(p, p["series"])


def test_inland_shot_can_replace_coastal_setting_without_changing_other_cached_prompts():
    p = copy.deepcopy(ladder.find(BANKS, "countdown-02"))
    original = ladder.image_jobs(p)
    p["options"][5]["art_direction"] = "Violet storm sky over high inland conifer ridges, warm amber rim light."
    repaired = ladder.image_jobs(p)
    assert not ladder.premise_problems(p)
    assert "Gulf coast" not in repaired[5]["prompt"] and "inland conifer ridges" in repaired[5]["prompt"]
    assert all(a == b for i, (a, b) in enumerate(zip(original, repaired)) if i != 5)


@pytest.mark.parametrize("direction", ["", None, "A warm scene. Another scene.", "A person in a boat."])
def test_option_art_direction_cannot_bypass_composed_prompt_gates(direction):
    p = copy.deepcopy(ALL[0])
    p["options"][5]["art_direction"] = direction
    assert ladder.premise_problems(p)


def test_moonlit_refuge_preserves_cached_siblings_and_exposure_checks():
    p = copy.deepcopy(ladder.find(BANKS, "survive_night-05"))
    p.pop('end_image', None)
    before = ladder.image_jobs(p)
    p['options'][1].update(
        image_prompt='An empty pale sandbar under reflected amber moonlight, a narrow river and distant green coal forest beyond, with fine sand ripples visible in the near foreground.',
        art_direction='Moonlit green forest and pale sand under reflected amber lunar light.',
        lighting='moonlight')
    assert not ladder.premise_problems(p)
    after = ladder.image_jobs(p)
    assert all(a == b for i, (a, b) in enumerate(zip(before, after)) if i != 1)
    assert 'single orange flame' not in after[1]['prompt']
    assert 'glowing focal light' not in after[1]['prompt']
    assert after[1]['luma'] == before[1]['luma']


@pytest.mark.parametrize('lighting,scene', [
    ('unchecked', 'Amber moonlit sandbar'),
    ('moonlight', 'Amber pitch black deep water in a trench'),
    ('moonlight', 'A bright amber sandbar'),
    ('moonlight', 'A moonlit person in a boat with amber light'),
])
def test_lighting_override_cannot_bypass_scene_or_content_gates(lighting, scene):
    p = copy.deepcopy(ALL[0])
    p['options'][1].update(lighting=lighting, image_prompt=scene)
    assert ladder.premise_problems(p)


def test_overlap_measures_shared_4grams():
    a = "Number one: a deep burrow under the roots of the old forest at night."
    assert ladder.overlap(a, [a]) == 1.0
    assert ladder.overlap(a, ["Something else entirely about tidal flats and moonlight."]) == 0.0
    assert ladder.overlap(a, []) == 0.0


# ---------------------------------------------------------------------------------------------------- cadence
LAUNCH = date(2026, 11, 1)


def clean(n, ok=True, mode="publish"):
    return [{"mode": mode, "ok": ok} for _ in range(n)]


def test_ramp_needs_a_launch_date():
    assert ladder.slots_allowed(None, LAUNCH, [])[0] == []
    assert ladder.launch_day("") is None and ladder.launch_day("2026-13-01") is None
    assert ladder.launch_day("2026-11-01") == LAUNCH
    assert ladder.slots_allowed(LAUNCH, LAUNCH - timedelta(days=1), [])[0] == []


@pytest.mark.parametrize("day", [1, 2, 8, 30])
def test_three_slots_every_day_from_launch(day):
    builds = clean(3) + clean(2, ok=False)  # a refused Short never holds the next day's slots back
    assert ladder.slots_allowed(LAUNCH, LAUNCH + timedelta(days=day - 1), builds)[0] == ["A", "B", "C"]


def test_slot_times():
    day = date(2026, 11, 3)
    early = datetime(2026, 11, 3, 8, 0, tzinfo=timezone.utc)
    assert youtube.slot_publish_at("A", day, early) == "2026-11-03T14:30:00Z"
    assert youtube.slot_publish_at("C", day, early) == "2026-11-03T19:30:00Z"
    assert youtube.slot_publish_at("B", day, early) == "2026-11-04T00:30:00Z"
    assert youtube.slot_publish_at("A", day, datetime(2026, 11, 3, 14, 15, tzinfo=timezone.utc)) is None


# ---------------------------------------------------------------------------------------------------- rotation
def published(series_days):
    data = copy.deepcopy(ledger.EMPTY)
    for i, (d, s) in enumerate(series_days):
        p = BANKS[s][i % len(BANKS[s])]
        data["shorts"].append({"premise": p["id"], "series": s, "era": p["era"], "video_id": f"v{i}",
                               "publish_at": f"{d.isoformat()}T14:30:00Z", "script": ladder.script_text(p)})
    return data


def test_rotation_rules():
    d = date(2026, 11, 10)
    assert ladder.rotation_problems([(d, "countdown"), (d, "countdown")], "countdown", d)
    assert not ladder.rotation_problems([(d, "countdown"), (d, "how_deep")], "level_1_8", d)
    assert not ladder.rotation_problems([(d, "countdown")], "how_deep", d)  # too early to need 3 series
    # three Shorts in a week must span three series
    assert ladder.rotation_problems([(d, "countdown"), (d, "how_deep")], "countdown", d)
    old = d - timedelta(days=7)
    assert not ladder.rotation_problems([(old, "countdown"), (old, "how_deep"), (d, "level_1_8")], "countdown", d)


def test_pick_never_repeats_a_premise_and_respects_rotation():
    at = date(2026, 11, 10)
    data = copy.deepcopy(ledger.EMPTY)
    chosen = []
    for i in range(12):
        day = at + timedelta(days=i)
        p = ladder.pick(BANKS, data, day)
        assert p["id"] not in {c["id"] for c in chosen}
        past = [(ladder._day(s["publish_at"]), s["series"]) for s in data["shorts"]]
        assert not ladder.rotation_problems(past, p["series"], day)
        ledger.record_short(data, {"premise": p["id"], "series": p["series"], "era": p["era"], "video_id": f"v{i}",
                                   "publish_at": f"{day.isoformat()}T14:30:00Z", "script": ladder.script_text(p)})
        chosen.append(p)
    assert len({p["series"] for p in chosen}) == len(ladder.SERIES)


def test_pick_override_refuses_a_published_premise():
    p = ALL[0]
    data = published([(date(2026, 11, 1), p["series"])])
    data["shorts"][0]["premise"] = p["id"]
    with pytest.raises(ladder.BankError):
        ladder.pick(BANKS, data, date(2026, 11, 2), override=p["id"])
    with pytest.raises(ladder.BankError):
        ladder.pick(BANKS, data, date(2026, 11, 2), override="no-such-premise")


def test_plan_gates_flag_a_repeated_script():
    p = ALL[0]
    data = published([(date(2026, 11, 1), p["series"])])
    data["shorts"][0]["script"] = ladder.script_text(p)
    g = ladder.plan_problems(p, data, date(2026, 11, 2))
    assert not g["novelty_4gram"]["ok"]


def test_record_short_refuses_a_second_publish_of_a_premise():
    data = copy.deepcopy(ledger.EMPTY)
    ledger.record_short(data, {"premise": "x-01", "video_id": "a"})
    ledger.record_short(data, {"premise": "x-01", "video_id": "a", "title": "same video, updated"})
    assert len(data["shorts"]) == 1 and data["shorts"][0]["title"] == "same video, updated"
    with pytest.raises(ValueError):
        ledger.record_short(data, {"premise": "x-01", "video_id": "b"})


def test_ladder_log_rolls():
    data = copy.deepcopy(ledger.EMPTY)
    for i in range(5):
        ledger.ladder_log(data, "builds", {"i": i}, keep=3)
    assert [b["i"] for b in data["ladder"]["builds"]] == [2, 3, 4]


# ---------------------------------------------------------------------------------------------------- timeline
def durations(option=2.4, hook=1.4, verdict=6.0, cta=1.2):
    return {"hook": hook, **{f"o{n}": option for n in range(1, 9)}, "verdict": verdict, "cta": cta}


def test_timeline_fits_the_duration_window():
    tl = shorts_ladder.timeline(ALL[0], durations())
    lo, hi = shorts_ladder.GATES["duration_s"]
    assert lo <= tl["seconds"] <= hi and tl["problems"] == []
    assert tl["frames"] == round(tl["seconds"] * tl["fps"])
    assert tl["shots"][-1]["f1"] == tl["frames"]
    for s in tl["shots"]:
        if s["kind"] == "option":
            assert shorts_ladder.HOLD[0] <= s["hold"] <= shorts_ladder.HOLD[1]


def test_timeline_reports_overlong_narration():
    tl = shorts_ladder.timeline(ALL[0], durations(option=4.5))
    assert any("cannot fit" in x for x in tl["problems"])


def test_every_gate_has_a_floor_and_a_ceiling():
    for name, (lo, hi) in shorts_ladder.GATES.items():
        assert lo is not None and hi is not None and lo <= hi, name
    assert shorts_ladder.GATES["option_pair_similarity"][1] == 0.90
    assert shorts_ladder.GATES["frame0_vs_history"][1] == 0.93
    assert shorts_ladder.GATES["novelty_4gram"][1] == 0.35
    assert not shorts_ladder.gate("frame0_vs_history", 0.95)["ok"]
    assert shorts_ladder.gate("frame0_vs_history", 0.5)["ok"]
    assert not shorts_ladder.gate("luma", None)["ok"]


# ---------------------------------------------------------------------------------------------------- upload
WHEN = "2026-11-03T14:30:00Z"


def test_upload_body_declares_synthetic_media_and_not_for_kids():
    p = ALL[0]
    body = youtube.ladder_body(p, "desc", ["t"], WHEN)
    st = body["status"]
    assert st["containsSyntheticMedia"] is True and st["selfDeclaredMadeForKids"] is False
    assert st["privacyStatus"] == "private" and st["publishAt"] == WHEN
    assert "publishAt" not in youtube.ladder_body(p, "desc", ["t"], None)["status"]


def readback(**status):
    st = {"privacyStatus": "private", "publishAt": "2026-11-03T14:30:00.000Z", "madeForKids": False,
          "selfDeclaredMadeForKids": False, **status}
    return {"snippet": {"channelId": config.CHANNEL_ID}, "status": st}


def test_readback_matches():
    assert youtube.readback_problems(readback(), WHEN) == []


@pytest.mark.parametrize("change, word", [
    ({"madeForKids": True}, "madeForKids"),
    ({"selfDeclaredMadeForKids": None}, "selfDeclaredMadeForKids"),
    ({"privacyStatus": "public"}, "privacyStatus"),
    ({"publishAt": "2026-11-04T00:30:00Z"}, "publishAt"),
    ({"containsSyntheticMedia": False}, "containsSyntheticMedia"),
])
def test_readback_mismatches(change, word):
    assert any(word in x for x in youtube.readback_problems(readback(**change), WHEN))


def test_readback_other_channel():
    v = readback()
    v["snippet"]["channelId"] = "UCsomeoneelse"
    assert youtube.readback_problems(v, WHEN)


def test_description_cites_the_pinned_revision():
    p = ALL[0]
    d = youtube.ladder_description(p, None)
    assert f"oldid={p['fact']['wiki_revid']}" in d and youtube.LADDER_DISCLOSURE in d


def test_pterosaur_prompts_spell_out_membrane_wings():
    from studio import ladder
    p = {"art_direction": "a", "options": [{"n": 1, "image_prompt": "A giant Quetzalcoatlus soaring"},
                                           {"n": 2, "image_prompt": "A herd of Triceratops"}]}
    jobs = {j["id"]: j for j in ladder.image_jobs(p)}
    assert "no feathers" in jobs["o1"]["prompt"] and "no feathers" not in jobs["o2"]["prompt"]
    assert jobs["o1"]["scene"] == "A giant Quetzalcoatlus soaring"  # CLIP similarity still scores the plain scene


def test_jaekelopterus_has_its_aquatic_body_plan_without_overriding_other_eurypterids():
    from studio import ladder
    p = {"art_direction": "copper underwater light", "options": [
        {"n": 1, "image_prompt": "Jaekelopterus gliding over silt"},
        {"n": 2, "image_prompt": "A spiny-legged eurypterid sifting mud"}]}
    jobs = {j["id"]: j for j in ladder.image_jobs(p)}
    assert "fully submerged" in jobs["o1"]["prompt"]
    assert "triangular telson" in jobs["o1"]["prompt"]
    assert "cheliceral pincers" in jobs["o1"]["prompt"]
    assert "triangular telson" not in jobs["o2"]["prompt"]
    assert jobs["o1"]["scene"] == p["options"][0]["image_prompt"]


def test_readback_flags_a_failed_upload():
    assert youtube.readback_problems(readback(uploadStatus="uploaded"), WHEN) == []
    assert youtube.readback_problems(readback(uploadStatus="processed"), WHEN) == []
    bad = youtube.readback_problems(readback(uploadStatus="failed", failureReason="conversion"), WHEN)
    assert any("uploadStatus is 'failed' (conversion)" in x for x in bad)


def test_insert_resumes_after_a_dropped_connection(monkeypatch, tmp_path):
    import ssl
    import sys
    import types

    errors, http = types.ModuleType("googleapiclient.errors"), types.ModuleType("googleapiclient.http")
    errors.HttpError = type("HttpError", (Exception,), {})
    http.MediaFileUpload = lambda *a, **kw: None
    for name, mod in [("googleapiclient", types.ModuleType("googleapiclient")), ("googleapiclient.errors", errors),
                      ("googleapiclient.http", http)]:
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(youtube.time, "sleep", lambda s: None)
    calls = []

    class Request:
        def next_chunk(self, num_retries=0):
            calls.append(1)
            if len(calls) == 1:
                raise ssl.SSLEOFError("EOF occurred in violation of protocol")
            return None, {"id": "abc"}

    class Videos:
        def insert(self, **kw):
            return Request()

    class Api:
        def videos(self):
            return Videos()

    clip = tmp_path / "short.mp4"
    clip.write_bytes(b"\0" * 16)
    assert youtube._insert(Api(), clip, {}) == "abc"
    assert len(calls) == 2
