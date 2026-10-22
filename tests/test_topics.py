"""The proven-topics feed (studio/outliers.py + studio/topics.py), with fakes only: no network, no googleapiclient."""
import argparse
import copy
import json
from datetime import date, datetime, timezone

import pytest

from studio import config, ladder, ledger, outliers, planner, topics

BANKS = ladder.load_banks(strict=True)
AT = datetime(2026, 10, 26, 5, 0, tzinfo=timezone.utc)
CID = "UCfakechannel0000000000"
SECRET = "SECRET DESCRIPTION TEXT"


def model_premise():
    """A bank premise standing in for what Gemini writes; the banks the run sees do not contain it."""
    return BANKS["countdown"][0]


def banks_without(p):
    return {s: [q for q in ps if q["id"] != p["id"]] for s, ps in BANKS.items()}


# ---------------------------------------------------------------------------------------------------- fakes


def video(vid, title, views, duration="PT45S", channel=CID):
    return {"id": vid, "snippet": {"title": title, "publishedAt": "2026-10-01T10:00:00Z", "channelId": channel,
                                   "description": SECRET, "thumbnails": {"high": {"url": "https://i.ytimg/x.jpg"}}},
            "statistics": {"viewCount": str(views), "likeCount": "5"},
            "contentDetails": {"duration": duration, "caption": "true"}}


def world():
    """One competitor channel: ten Shorts around 50k and one at 1.2M; six long-form videos around 10k and one at
    200k whose title is about a backlog topic."""
    items = [video(f"s{i}", f"Short clip number {i}", 40_000 + 2_000 * i) for i in range(10)]
    items.append(video("sOUT", "I Survived 100 Days In A Frozen Wasteland", 1_200_000))
    items += [video(f"l{i}", f"Long video number {i}", 9_000 + 500 * i, "PT1H2M") for i in range(6)]
    items.append(video("lOUT", "What If Dinosaurs Never Went Extinct?", 200_000, "PT2H"))
    return items


class Call:
    def __init__(self, fn):
        self.fn = fn

    def execute(self):
        return self.fn()


class FakeApi:
    def __init__(self, items=None):
        self.items = world() if items is None else items
        self.calls = []

    def channels(self):
        def list_(**kw):
            self.calls.append(("channels", kw))
            ids = kw["id"].split(",")
            return Call(lambda: {"items": [{"id": i, "contentDetails": {"relatedPlaylists": {"uploads": "UU" + i}}}
                                           for i in ids]})
        return type("R", (), {"list": staticmethod(list_)})()

    def playlistItems(self):
        def list_(**kw):
            self.calls.append(("playlistItems", kw))
            cid = kw["playlistId"][2:]
            mine = [v for v in self.items if v["snippet"]["channelId"] == cid]
            return Call(lambda: {"items": [{"contentDetails": {"videoId": v["id"]}} for v in mine]})
        return type("R", (), {"list": staticmethod(list_)})()

    def videos(self):
        def list_(**kw):
            self.calls.append(("videos", kw))
            ids = set(kw["id"].split(","))
            return Call(lambda: {"items": [v for v in self.items if v["id"] in ids]})
        return type("R", (), {"list": staticmethod(list_)})()


class FakeLLM:
    def __init__(self, premise, drafts=None):
        self.premise = premise
        self.drafts = list(drafts or [])
        self.prompts = []

    def generate(self, system, prompt, schema=None, temperature=0.8):
        self.prompts.append(system + prompt)
        p = self.premise
        if schema is topics.MAP_SCHEMA:
            return {"ideas": [{"video_id": "sOUT", "series": "countdown", "era": p["era"],
                               "idea": "a deep-time catastrophe", "wiki_title": p["fact"]["wiki_title"], "reject": ""},
                              {"video_id": "s1", "series": "countdown", "era": "Jurassic", "idea": "x",
                               "wiki_title": "Y", "reject": "no honest deep-time version"},
                              {"video_id": "not-a-scanned-id", "series": "countdown", "era": "Jurassic",
                               "idea": "x", "wiki_title": "Y", "reject": ""}]}
        if self.drafts:
            return self.drafts.pop(0)
        raw = {k: copy.deepcopy(v) for k, v in p.items() if k not in ("id", "series")}
        raw["fact"] = {"text": p["fact"]["text"], "source_sentence": p["fact"]["source_sentence"],
                       "wiki_revid": 1, "wiki_title": "Something the model made up"}
        raw["unexpected_key"] = "dropped"
        return raw


def fake_fetch(premise):
    f = premise["fact"]

    def fetch(title, cache=None):
        return {"title": f["wiki_title"], "revid": f["wiki_revid"], "url": "u",
                "text": f"{f['wiki_title']} happened about 66 million years ago. {f['source_sentence']} More text."}
    return fetch


@pytest.fixture
def one_channel(monkeypatch):
    monkeypatch.setattr(outliers, "competitors", lambda path=outliers.CHANNELS: [{"id": CID, "name": "Fake"}])


def fresh():
    return json.loads(json.dumps(ledger.EMPTY))


# ---------------------------------------------------------------------------------------------------- outliers


def test_competitor_list_is_well_formed():
    doc = outliers.load()
    ids = [c["id"] for c in doc["channels"]]
    assert len(ids) == len(set(ids)) >= 10
    assert all(i.startswith("UC") and len(i) == 24 for i in ids)
    assert all(c["format"] in ("shorts", "long", "mixed") for c in doc["channels"])
    names = outliers.do_not_copy()
    assert "MonAI" in names and "el brayan" in names


@pytest.mark.parametrize("iso, secs", [("PT45S", 45), ("PT1M5S", 65), ("PT3M", 180), ("PT1H2M3S", 3723),
                                       ("P1DT1H", 90000), ("", 0), ("P0D", 0)])
def test_duration_parsing(iso, secs):
    assert outliers.duration_s(iso) == secs


def test_shorts_and_long_form_are_kept_apart():
    assert outliers.kind(180) == "short" and outliers.kind(181) == "long"


def test_outlier_maths():
    vids = [{"video_id": f"v{i}", "channel_id": "c", "title": "t", "views": v, "published": "", "duration_s": 30}
            for i, v in enumerate([100_000, 110_000, 120_000, 130_000, 140_000, 400_000, 2_000_000])]
    out = outliers.flag(vids)
    # median 130k: 400k is 3.08x (in), 2M is 15.4x (in), best ratio first
    assert [o["video_id"] for o in out] == ["v6", "v5"]
    assert out[0]["median"] == 130_000 and out[0]["ratio"] == pytest.approx(15.38, abs=0.01)
    assert out[0]["kind"] == "short"
    # under 3x is never an outlier
    assert not outliers.flag(vids[:5] + [{**vids[0], "video_id": "x", "views": 370_000}])  # median 125k: 2.96x


def test_outliers_need_the_floor_and_enough_samples():
    small = [{"video_id": f"v{i}", "channel_id": "c", "title": "t", "views": v, "published": "", "duration_s": 30}
             for i, v in enumerate([1000, 1100, 1200, 1300, 1400, 90_000])]
    assert not outliers.flag(small)  # 64x its median, but under the 100k Shorts floor
    assert [o["video_id"] for o in outliers.flag(small, floors={"short": 50_000, "long": 1})] == ["v5"]
    # four normal videos plus the spike is exactly MIN_SAMPLES; one fewer is not enough
    assert outliers.flag(small[:4] + small[5:], floors={"short": 1, "long": 1})
    assert not outliers.flag(small[:3] + small[5:], floors={"short": 1, "long": 1})


def test_medians_are_per_channel_and_per_kind():
    rows = []
    for ch, base in (("big", 1_000_000), ("small", 10_000)):
        rows += [{"video_id": f"{ch}{i}", "channel_id": ch, "title": "t", "views": base, "published": "",
                  "duration_s": 40} for i in range(5)]
    rows.append({"video_id": "smallOUT", "channel_id": "small", "title": "t", "views": 200_000, "published": "",
                 "duration_s": 40})
    rows += [{"video_id": f"L{i}", "channel_id": "small", "title": "t", "views": 30_000, "published": "",
              "duration_s": 3600} for i in range(5)]
    out = outliers.flag(rows)
    assert [o["video_id"] for o in out] == ["smallOUT"]  # 200k is nothing on the big channel; long-form is apart


def test_scan_keeps_public_metadata_only_and_stays_in_budget():
    api = FakeApi()
    scan = outliers.scan(api, [{"id": CID}])
    assert len(scan["videos"]) == len(world())
    for row in scan["videos"]:
        assert set(row) == set(outliers.KEEP)
    assert SECRET not in json.dumps(scan) and "ytimg" not in json.dumps(scan)
    for name, kw in api.calls:
        assert "fields" in kw  # partial responses: descriptions and thumbnails are never even sent to us
    video_fields = next(kw["fields"] for name, kw in api.calls if name == "videos")
    assert "description" not in video_fields and "thumbnails" not in video_fields


def test_full_scan_costs_under_300_units():
    chans = outliers.competitors()
    items = [video(f"{c['id']}-{i}", "t", 1000, channel=c["id"]) for c in chans for i in range(50)]
    scan = outliers.scan(FakeApi(items), chans)
    assert scan["units"] < outliers.QUOTA_MAX and scan["units"] == 1 + 2 * len(chans)
    with pytest.raises(outliers.QuotaExceeded):
        outliers.scan(FakeApi(items), chans, outliers.Budget(10))


def test_one_broken_channel_does_not_stop_the_scan():
    api = FakeApi()
    real = api.playlistItems

    def broken():
        r = real()
        orig = r.list

        def list_(**kw):
            if kw["playlistId"] == "UUbad":
                raise RuntimeError("gone")
            return orig(**kw)
        return type("R", (), {"list": staticmethod(list_)})()
    api.playlistItems = broken
    scan = outliers.scan(api, [{"id": "bad"}, {"id": CID}])
    assert scan["channels"]["bad"] == {"error": "RuntimeError"} and scan["channels"][CID]["read"] == len(world())


# ---------------------------------------------------------------------------------------------------- gates


NAMES = ["MonAI", "el brayan", "Tralalero"]


def test_bank_premises_pass_the_new_gates():
    names = outliers.do_not_copy()
    pool = [p for ps in BANKS.values() for p in ps]
    for p in pool:
        assert topics.niche_problems(p) == [], p["id"]
        assert topics.copy_problems(p, [], names) == [], p["id"]
        assert topics.dedupe_problems(p, pool, []) == [], p["id"]


def test_similarity_gate_rejects_a_verbatim_title():
    p = model_premise()
    probs = topics.copy_problems(p, [p["title"].upper() + "!!"], NAMES)
    assert any("verbatim" in x and x.startswith("title") for x in probs)


def test_similarity_gate_rejects_a_shared_four_word_run():
    p = copy.deepcopy(model_premise())
    p["hook_vo"] = "Which cave would you choose tonight?"
    probs = topics.copy_problems(p, ["WHICH CAVE would THEY pick?!"], NAMES)
    assert probs == []  # three shared words is fine
    probs = topics.copy_problems(p, ["I asked: which cave would you... pick"], NAMES)
    assert any(x.startswith("hook_vo shares 'which cave would you'") for x in probs)


def test_similarity_gate_rejects_competitor_names_anywhere():
    p = copy.deepcopy(model_premise())
    p["description"] += " Inspired by MonAI."
    assert any("MonAI" in x for x in topics.copy_problems(p, [], NAMES))
    p = copy.deepcopy(model_premise())
    p["options"][3]["image_prompt"] += " in the style of El Brayan"
    assert any("el brayan" in x for x in topics.copy_problems(p, [], NAMES))
    p = copy.deepcopy(model_premise())
    p["tags"].append("tralalero")
    assert any("Tralalero" in x for x in topics.copy_problems(p, [], NAMES))
    assert topics.copy_problems(model_premise(), [], NAMES) == []


def test_niche_gate_rejects_people_and_myth():
    p = copy.deepcopy(model_premise())
    p["options"][0]["vo"] = "Hide with the village elders."
    p2 = copy.deepcopy(model_premise())
    p2["banner"] = "Where would the dragon sleep?"
    p3 = copy.deepcopy(model_premise())
    p3["era"] = "Bronze Age"
    assert any("village" in x for x in topics.niche_problems(p))
    assert any("dragon" in x for x in topics.niche_problems(p2))
    assert any("geological" in x for x in topics.niche_problems(p3))


def test_source_must_be_in_the_pinned_revision():
    p = model_premise()
    doc = fake_fetch(p)("x")
    assert topics.source_problems(p, doc) == []
    assert topics.source_problems(p, {**doc, "revid": doc["revid"] + 1})
    assert topics.source_problems(p, {**doc, "text": "nothing here, 66 million years ago"})
    assert topics.article_problems({"title": "Somebody", "text": "Somebody (born 1950) is a writer."})
    assert topics.article_problems({"title": "Cats", "text": "Cats are pets."})
    assert topics.article_problems(doc) == []


def test_dedupe_against_banks_and_published():
    p = copy.deepcopy(model_premise())
    pool = [q for ps in BANKS.values() for q in ps]
    clone = {**p, "id": "countdown-101"}
    probs = topics.dedupe_problems(clone, pool, [])
    assert any("title duplicates" in x for x in probs) and any("4-grams" in x for x in probs)
    others = [q for q in pool if q["id"] != p["id"]]
    assert topics.dedupe_problems(clone, others, []) == []
    near = p["title"].replace("?", "!")
    assert any("too close" in x for x in topics.dedupe_problems(clone, others, [near]))
    assert topics.next_id("countdown", {"countdown-101", "countdown-102"}) == "countdown-103"


# ---------------------------------------------------------------------------------------------------- run


def test_run_accepts_a_grounded_premise_with_provenance(one_channel):
    p = model_premise()
    data, llm = fresh(), FakeLLM(p)
    sm = topics.run(data, FakeApi(), llm, fake_fetch(p), banks=banks_without(p), backlog=config.backlog(), at=AT)
    assert [a["id"] for a in sm["accepted"]] == ["countdown-101"]
    got = data["ladder"]["proven"][0]
    assert got["id"] == "countdown-101" and got["series"] == "countdown" and got["added_at"] == "2026-10-26T05:00:00Z"
    assert got["fact"]["wiki_revid"] == p["fact"]["wiki_revid"]  # from the fetch, not the model's made-up one
    assert got["fact"]["wiki_title"] == p["fact"]["wiki_title"] and "unexpected_key" not in got
    prov = got["provenance"]
    assert prov["channel_id"] == CID and prov["video_id"] == "sOUT" and prov["views"] == 1_200_000
    assert prov["ratio"] > 20 and prov["median"] > 0
    assert ladder.premise_problems(got, "countdown") == []
    # ideas for videos that were not outliers (s1) or were never scanned are ignored
    assert sm["rejected"] == []
    assert set(data["topics"]["tried"]) == {"sOUT"}
    # nothing of theirs but public metadata reached the ledger or the model
    assert SECRET not in json.dumps(data) and not any(SECRET in x for x in llm.prompts)
    assert not any("ytimg" in x for x in llm.prompts)
    # the long-form outlier boosted its backlog topic
    assert sm["boosts"] == [{"topic": "dinosaurs-never-extinct", "video_id": "lOUT",
                             "ratio": data["topics"]["boosts"]["dinosaurs-never-extinct"]["ratio"]}]
    assert sm["supply_days"] >= topics.SUPPLY_DAYS_MIN and not sm["low_supply"]


def test_run_rejects_on_any_gate_failure(one_channel):
    p = model_premise()
    bad = {k: copy.deepcopy(v) for k, v in p.items() if k not in ("id", "series")}
    bad["title"] = "I Survived 100 Days In A Frozen Wasteland"  # their title, verbatim
    bad2 = copy.deepcopy(bad)
    bad2["title"] = p["title"]
    bad2["description"] += " Like MonAI."
    data = fresh()
    sm = topics.run(data, FakeApi(), FakeLLM(p, drafts=[bad, bad2]), fake_fetch(p), banks=banks_without(p),
                    backlog=[], at=AT)
    assert sm["accepted"] == [] and data["ladder"]["proven"] == []
    why = next(r["why"] for r in sm["rejected"] if r["video_id"] == "sOUT")
    assert any("MonAI" in w for w in why)


def test_run_never_reuses_a_tried_outlier_and_honours_the_cap(one_channel):
    p = model_premise()
    data = fresh()
    topics.run(data, FakeApi(), FakeLLM(p), fake_fetch(p), banks=banks_without(p), backlog=[], at=AT)
    again = topics.run(data, FakeApi(), FakeLLM(p), fake_fetch(p), banks=banks_without(p), backlog=[], at=AT)
    assert again["accepted"] == [] and len(data["ladder"]["proven"]) == 1
    capped = fresh()
    sm = topics.run(capped, FakeApi(), FakeLLM(p), fake_fetch(p), banks=banks_without(p), backlog=[], at=AT, cap=0)
    assert sm["accepted"] == [] and any("cap" in n for n in sm["notes"])
    assert topics.added_since(data, AT) == 1


def test_run_dedupes_against_the_banks(one_channel):
    p = model_premise()
    data = fresh()
    sm = topics.run(data, FakeApi(), FakeLLM(p), fake_fetch(p), banks=BANKS, backlog=[], at=AT)
    assert sm["accepted"] == []
    assert any("duplicates" in w for r in sm["rejected"] for w in r["why"])


def test_proven_premises_go_ahead_of_the_banks(one_channel):
    p = model_premise()
    data = fresh()
    topics.run(data, FakeApi(), FakeLLM(p), fake_fetch(p), banks=banks_without(p), backlog=[], at=AT)
    banks = ladder.with_proven(banks_without(p), data)
    assert ladder.pick(banks, data, date(2026, 10, 27))["id"] == "countdown-101"
    # once published it is never picked again
    data["shorts"].append({"premise": "countdown-101", "series": "countdown", "era": p["era"], "video_id": "v",
                           "publish_at": "2026-10-27T14:30:00Z"})
    assert ladder.pick(banks, data, date(2026, 10, 28))["id"] != "countdown-101"
    # a proven premise that fails a gate tightened later is left out, not picked
    data["ladder"]["proven"][0]["title"] = "x"
    assert "countdown-101" not in {q["id"] for q in ladder.with_proven(banks_without(p), data)["countdown"]}


def test_planner_moves_a_boosted_topic_forward():
    series = {s: {"name": s} for s in ("a", "b", "c")}
    backlog = [{"id": f"{s}{i}", "series": s, "title": f"Topic {s}{i}"} for s in "abc" for i in range(3)]
    data = {"episodes": [], "dry_runs": [], "last_upload": None}
    assert planner.next_topic(data, backlog, series)["id"] == "a0"
    data["topics"] = {"boosts": {"b2": {"ratio": 6.0, "at": datetime.now(timezone.utc).isoformat()}}}
    assert planner.next_topic(data, backlog, series)["id"] == "b2"
    data["topics"]["boosts"]["b2"]["at"] = "2020-01-01T00:00:00Z"  # stale boosts expire
    assert planner.next_topic(data, backlog, series)["id"] == "a0"


def test_long_form_matching_invents_nothing():
    backlog = config.backlog()
    assert topics.match_topic("What If Dinosaurs Never Went Extinct?", backlog, set())[0]["id"] == \
        "dinosaurs-never-extinct"
    assert topics.match_topic("Cats compilation 10 hours", backlog, set()) is None
    assert topics.match_topic("What If Dinosaurs Never Went Extinct?", backlog, {"dinosaurs-never-extinct"}) is None


def test_low_supply_alerts():
    two = {"countdown": BANKS["countdown"][:1], "how_deep": BANKS["how_deep"][:1]}
    sm = topics.run(fresh(), None, None, None, banks=two, backlog=[], at=AT)
    assert sm["supply_days"] < topics.SUPPLY_DAYS_MIN and sm["low_supply"]
    assert "LOW SUPPLY" in topics.message(sm) and topics.low_supply_text(sm)
    full = topics.run(fresh(), None, None, None, banks=BANKS, backlog=[], at=AT)
    assert full["supply_days"] >= topics.SUPPLY_DAYS_MIN and topics.low_supply_text(full) == ""


def _cli(monkeypatch, tmp_path, p, sent):
    from studio import alert, wiki

    monkeypatch.setattr(config, "STATE", tmp_path / "state")
    monkeypatch.setattr(config, "WORK", tmp_path / "work")
    monkeypatch.setattr(topics, "_youtube", lambda: FakeApi())
    monkeypatch.setattr(topics, "_gemini", lambda: FakeLLM(p))
    monkeypatch.setattr(wiki, "fetch", fake_fetch(p))
    monkeypatch.setattr(ladder, "load_banks", lambda *a, **kw: banks_without(p))
    monkeypatch.setattr(alert, "send", lambda text: sent.append(text) or True)


def test_dry_run_saves_nothing(monkeypatch, tmp_path, one_channel):
    p, sent = model_premise(), []
    _cli(monkeypatch, tmp_path, p, sent)
    topics.scan_cmd(argparse.Namespace(dry=True))
    assert not ledger.path().exists() and sent == []
    sm = json.loads((tmp_path / "work" / "topics" / "summary.json").read_text())
    assert [a["id"] for a in sm["accepted"]] == ["countdown-101"]
    topics.scan_cmd(argparse.Namespace(dry=False))
    saved = ledger.load()
    assert [q["id"] for q in saved["ladder"]["proven"]] == ["countdown-101"] and len(sent) == 1
    assert (tmp_path / "work" / "topics" / "alerts.txt").read_text() == ""


def test_cli_registers_topics_scan():
    import studio.cli as cli

    with pytest.raises(SystemExit):
        cli.main(["topics-scan", "--help"])


def test_ledger_has_the_new_keys(tmp_path):
    (tmp_path / "ledger.json").write_text(json.dumps({"episodes": [], "ladder": {"weights": {}}}))
    d = ledger.load(tmp_path)
    assert d["ladder"]["proven"] == [] and d["topics"] == {"outliers": [], "tried": [], "boosts": {}, "scans": []}


def test_an_outlier_with_no_honest_deep_time_version_is_rejected(one_channel):
    p = model_premise()

    class Rejects(FakeLLM):
        def generate(self, system, prompt, schema=None, temperature=0.8):
            return {"ideas": [{"video_id": "sOUT", "series": "countdown", "era": "Jurassic", "idea": "x",
                               "wiki_title": "Y", "reject": "it is about a person"}]}
    data = fresh()
    sm = topics.run(data, FakeApi(), Rejects(p), fake_fetch(p), banks=banks_without(p), backlog=[], at=AT)
    assert sm["accepted"] == [] and [r["video_id"] for r in sm["rejected"]] == ["sOUT"]
    assert data["ladder"]["proven"] == [] and data["topics"]["tried"] == ["sOUT"]
