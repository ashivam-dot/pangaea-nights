from datetime import datetime, timedelta, timezone

import pytest

from studio import config, ledger, planner

SERIES = {s: {"titles": ["Fall Asleep in {title} | {hours} Hours", "{title}: A Sleep Documentary",
                         "Drift Through {title} | Calm Deep-Time Story"]} for s in ("a", "b", "c")}
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


def test_titles_rotate_and_avoid_near_duplicates():
    d = data_with()
    t1 = planner.title(BACKLOG[0], "The Cambrian Seas", 2, d, SERIES)
    assert t1 == "Fall Asleep in the Cambrian Seas | 2 Hours"
    d["episodes"].append({"id": "x", "topic": "x", "series": "a", "title": t1})
    t2 = planner.title(BACKLOG[0], "The Cambrian Seas", 2, d, SERIES)
    assert t2 != t1 and len(t2) <= 100


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
