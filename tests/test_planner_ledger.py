from datetime import datetime, timedelta, timezone

import pytest

from studio import config, ledger, planner

SERIES = {s: {"name": s} for s in ("a", "b", "c")}
BACKLOG = [{"id": f"{s}{i}", "series": s, "title": f"Topic {s}{i}"} for s in "abc" for i in range(3)]


def data_with(*episodes):
    d = {"episodes": [], "dry_runs": [], "last_upload": None}
    for topic, series in episodes:
        d["episodes"].append({"id": topic, "topic": topic, "series": series, "title": f"T {topic}"})
    return d


def test_never_repeats_a_published_topic():
    d = data_with(("a0", "a"))
    picks = set()
    for _ in range(8):
        t = planner.next_topic(d, BACKLOG, SERIES)
        assert t["id"] not in ledger.used_topics(d)
        picks.add(t["id"])
        d["episodes"].append({"id": t["id"], "topic": t["id"], "series": t["series"], "title": t["id"]})
    assert len(picks) == 8


def test_rotates_series_away_from_the_last_two():
    d = data_with(("a0", "a"), ("b0", "b"))
    assert planner.next_topic(d, BACKLOG, SERIES)["series"] == "c"
    d = data_with(("c0", "c"), ("a0", "a"))
    assert planner.next_topic(d, BACKLOG, SERIES)["series"] == "b"


def test_exhausted_backlog_raises():
    d = data_with(*[(t["id"], t["series"]) for t in BACKLOG])
    with pytest.raises(planner.BacklogExhausted):
        planner.next_topic(d, BACKLOG, SERIES)


def test_override_and_unknown_topic():
    assert planner.next_topic(data_with(), BACKLOG, SERIES, override="b2")["id"] == "b2"
    with pytest.raises(ValueError):
        planner.next_topic(data_with(), BACKLOG, SERIES, override="nope")


TOPIC = {"id": "last-day", "series": "a", "title": "The Last Day of the Dinosaurs",
         "lead": "The Last Day of the Dinosaurs", "promise": "The Asteroid and the Long Dark"}


def test_titles_follow_the_famous_first_formula_and_rotate():
    d = data_with()
    t1 = planner.title(TOPIC, "ignored core", 2, d, SERIES)
    assert t1 == "The Last Day of the Dinosaurs: The Asteroid and the Long Dark | Documentary for Sleep"
    assert planner.title_problems(t1) == []
    d["episodes"].append({"id": "x", "topic": "x", "series": "a", "title": "Something Else | Sleep Documentary"})
    t2 = planner.title(TOPIC, "ignored core", 2, d, SERIES)
    assert t2 == "The Last Day of the Dinosaurs: The Asteroid and the Long Dark | Sleep Documentary"
    d["episodes"].append({"id": "y", "topic": "y", "series": "a", "title": t1})
    t3 = planner.title(TOPIC, "ignored core", 2, d, SERIES)
    assert t3 != t1 and len(t3) <= 100 and not planner.title_problems(t3)


def test_title_without_a_lead_uses_the_writers_core():
    t = planner.title({"id": "z", "series": "a", "title": "x"}, "Megalodon and the Whales", 2, data_with(), SERIES)
    assert t == "Megalodon and the Whales | Documentary for Sleep"


@pytest.mark.parametrize("bad, why", [
    ("Fall Asleep in the Garden of Ediacara | 2 Hours", "jargon"),
    ("The Hadean Earth: Earth's First Oceans | Documentary for Sleep", "jargon"),
    ("Life in the Carboniferous: Coal Forests | Sleep Documentary", "jargon"),
    ("Dinosaurs (As We Understand It) | Boring History For Sleep", "banned"),
    ("Megalodon: The Giant Shark | SELECTED SOURCES | Sleep Documentary", "banned"),
    ("What If Dinosaurs Never Went Extinct | Documentary for Sleep", "banned"),
    ("Megalodon: The Giant Shark of the Warm Ancient Ocean", "suffix"),
    ("A Quiet Evening Story: Calm Thoughts | Documentary for Sleep", "famous"),
    ("Dickinsonia: Soft Quilted Creatures | Documentary for Sleep", "uncommon"),
])
def test_title_gate_rejects_jargon_banned_phrases_and_obscure_leads(bad, why):
    problems = planner.title_problems(bad)
    assert problems
    key = {"jargon": "jargon", "banned": "banned", "suffix": "must end", "famous": "nothing famous",
           "uncommon": "uncommon"}[why]
    assert any(key in p for p in problems), problems


def test_every_backlog_title_passes_the_gate():
    data = data_with()
    for t in config.backlog():
        assert t["lead"] and t["promise"], t["id"]
        made = planner.title(t, t["lead"], 2, data, config.series())
        assert planner.title_problems(made) == [], (t["id"], made, planner.title_problems(made))
        assert made.startswith(t["lead"]), made


PUBLISHED = [("ediacaran-garden", "seas-before-fish"), ("coal-forests", "giant-insects"),
             ("mammoth-steppe", "ice-age-nights"), ("earth-forms", "before-the-continents"),
             ("hydrothermal-vents", "deep-sea"), ("ichthyosaurs", "ancient-giants-of-the-sea")]
FAMOUS_FIRST = ["last-day-dinosaurs", "dinosaurs-never-extinct", "pangaea", "megalodon", "last-glacial-maximum",
                "hell-creek", "plesiosaurs", "meganeura-wings", "siberian-traps", "snowball-earth"]


def test_famous_topics_come_first_in_order_from_the_real_ledger():
    backlog, series = config.backlog(), config.series()
    d = data_with(*PUBLISHED)
    picks = []
    for _ in range(len(FAMOUS_FIRST)):
        t = planner.next_topic(d, backlog, series)
        picks.append(t["id"])
        d["episodes"].append({"id": t["id"], "topic": t["id"], "series": t["series"], "title": t["title"]})
    assert picks == FAMOUS_FIRST


def test_ledger_refuses_a_second_publish_of_a_topic():
    d = data_with()
    ledger.record_episode(d, {"id": "e1", "topic": "a0", "video_id": "v1", "uploaded_at": "2026-10-10T00:30:00Z"})
    assert d["last_upload"] == "2026-10-10T00:30:00Z"
    ledger.record_episode(d, {"id": "e1", "topic": "a0", "title": "updated"})
    assert len(d["episodes"]) == 1 and d["episodes"][0]["title"] == "updated"
    with pytest.raises(ValueError):
        ledger.record_episode(d, {"id": "e2", "topic": "a0"})


def test_dry_runs_never_consume_topics_and_are_capped():
    d = data_with()
    for i in range(70):
        ledger.record_dry_run(d, {"id": f"dry{i}", "topic": "a0"})
    assert len(d["dry_runs"]) == 60 and "a0" not in ledger.used_topics(d)


def test_ledger_round_trip_and_hours(tmp_path):
    d = ledger.load(tmp_path)
    assert d == {"episodes": [], "dry_runs": [], "last_upload": None}
    d["last_upload"] = "2026-10-10T00:00:00Z"
    ledger.save(d, tmp_path)
    again = ledger.load(tmp_path)
    at = datetime(2026, 10, 11, 7, 0, tzinfo=timezone.utc)
    assert ledger.hours_since_upload(again, at) == pytest.approx(31.0)
    assert ledger.hours_since_upload(again, at - timedelta(hours=7)) == pytest.approx(24.0)


def test_backlog_is_large_unique_and_rotatable():
    backlog, series = config.backlog(), config.series()
    assert len(backlog) >= 60 and len(series) >= 5
    assert len({t["id"] for t in backlog}) == len(backlog)
    assert len({t["title"].lower() for t in backlog}) == len(backlog)
    for t in backlog:
        assert t["series"] in series and len(t["articles"]) >= 3
    d = data_with()
    for _ in range(len(backlog)):
        t = planner.next_topic(d, backlog, series)
        if d["episodes"]:
            assert t["series"] != d["episodes"][-1]["series"] or len({x["series"] for x in backlog
                                                                     if x["id"] not in ledger.used_topics(d)}) == 1
        d["episodes"].append({"id": t["id"], "topic": t["id"], "series": t["series"], "title": t["title"]})
    assert len(ledger.used_topics(d)) == len(backlog)
