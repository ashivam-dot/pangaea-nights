from datetime import datetime, timedelta, timezone

import pytest

from studio import learn, planner

AT = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
SERIES = {s: {"titles": ["{title} | {hours} Hours"]} for s in ("a", "b", "c", "d")}
BACKLOG = [{"id": f"{s}{i}", "series": s, "title": f"Topic {s}{i}"} for s in "abcd" for i in range(6)]


def ep(i, series, views72=None, video="v"):
    return {"id": f"e{i}", "topic": f"t{i}", "series": series, "title": f"T {i}", "video_id": f"{video}{i}",
            "publish_at": (AT - timedelta(hours=100)).isoformat().replace("+00:00", "Z"), "views_at_72h": views72}


def test_views_at_interpolates_and_waits():
    assert learn.views_at([{"age_h": 48, "views": 100}, {"age_h": 96, "views": 300}], 72) == pytest.approx(200)
    assert learn.views_at([{"age_h": 10, "views": 5}], 72) is None


def test_lifts_need_samples_and_are_bounded():
    few = [ep(i, "a", 100) for i in range(3)]
    assert learn.lifts(few)["lifts"] == {}
    eps = [ep(i, "a", 5000) for i in range(4)] + [ep(10 + i, "b", 5) for i in range(4)]
    out = learn.lifts(eps)["lifts"]
    assert out["a"]["lift"] == learn.LIFT_RANGE[1] and out["b"]["lift"] == learn.LIFT_RANGE[0]


def test_update_snapshots_public_episodes_only():
    data = {"episodes": [ep(1, "a"), {**ep(2, "b"), "publish_at": (AT + timedelta(hours=5)).isoformat()}]}
    learn.update(data, {"v1": {"view": 40, "like": 2, "comment": 0}, "v2": {"view": 0, "like": 0, "comment": 0}},
                 at=AT)
    assert data["episodes"][0]["snapshots"][0]["views"] == 40
    assert data["episodes"][0]["views_at_72h"] == pytest.approx(40 * (72 / 100) ** 0.5)
    assert "snapshots" not in data["episodes"][1]
    assert data["learned"]["scored"] == 1


def test_planner_prefers_a_strong_series_among_the_most_rested_but_never_repeats_recent():
    history = [ep(i, s) for i, s in enumerate("abcd")]
    data = {"episodes": history, "dry_runs": [], "last_upload": None}
    assert planner.next_topic(data, BACKLOG, SERIES)["series"] == "a"
    data["learned"] = {"lifts": {"b": {"lift": 1.3}, "d": {"lift": 1.35}}}
    assert planner.next_topic(data, BACKLOG, SERIES)["series"] == "b"
