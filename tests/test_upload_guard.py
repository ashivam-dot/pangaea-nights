from datetime import datetime, timezone

import pytest

from studio import cli, config, youtube

GOOD = {"id": "2026-10-11-x", "writer": "gemini", "publishable": True, "summary": "Calm.", "tags": ["deep time"],
        "chapters": [{"n": 1, "warnings": []}]}
QA_OK = {"ok": True, "problems": []}


def test_good_episode_may_publish():
    assert youtube.refusals(GOOD, "publish", False, QA_OK) == []


@pytest.mark.parametrize("change, expect", [
    ({"writer": "stub", "publishable": False}, "real writer"),
    ({"publishable": False}, "real writer"),
    ({"chapters": [{"n": 1, "warnings": ["stub"]}]}, "stub text"),
])
def test_stub_scripts_are_refused(change, expect):
    assert any(expect in r for r in youtube.refusals({**GOOD, **change}, "publish", False, QA_OK))


def test_dry_paused_and_failed_qa_are_refused():
    assert any("mode" in r for r in youtube.refusals(GOOD, "dry", False, QA_OK))
    assert any("paused" in r for r in youtube.refusals(GOOD, "publish", True, QA_OK))
    assert any("QA" in r for r in youtube.refusals(GOOD, "publish", False, {"ok": False, "problems": ["x"]}))


def test_identity_must_be_exactly_the_channel():
    ok = youtube.check_identity({"items": [{"id": config.CHANNEL_ID}]})
    assert ok["id"] == "UCtVG78IUTqvP9rH-LLies0A"
    for bad in ({"items": []}, {"items": [{"id": "UCsomeoneelse"}]},
                {"items": [{"id": config.CHANNEL_ID}, {"id": config.CHANNEL_ID}]}, {}):
        with pytest.raises(youtube.Refused):
            youtube.check_identity(bad)


def test_paused_unless_explicitly_false(monkeypatch):
    monkeypatch.delenv("PAUSED", raising=False)
    assert cli._paused()
    for v, want in (("true", True), ("", True), ("FALSE", False), ("false", False)):
        monkeypatch.setenv("PAUSED", v)
        assert cli._paused() is want


def test_publish_time_is_the_next_slot_at_least_two_hours_away():
    at = datetime(2026, 10, 10, 18, 0, tzinfo=timezone.utc)
    assert youtube.publish_at(0, 0, 30, now=at) == "2026-10-11T00:30:00Z"
    late = datetime(2026, 10, 10, 23, 0, tzinfo=timezone.utc)
    assert youtube.publish_at(0, 0, 30, now=late) == "2026-10-12T00:30:00Z"
    assert youtube.after("2026-10-11T00:30:00Z", 14.5) == "2026-10-11T15:00:00Z"


def test_metadata_is_clean_and_bounded():
    ep = {**GOOD, "summary": "A <calm> night.", "tags": ["t" * 60] * 20}
    desc = youtube.description(ep, [("0:00", "One")], [{"title": "A", "page_url": "u"}], [], "FLUX.2 klein 4B",
                               "bm_george")
    assert "<" not in desc and ">" not in desc and len(desc) <= 4900
    assert "AI" in desc and "0:00 One" in desc
    tags = youtube.tags(ep)
    assert tags[-1] == youtube.MARKER + ep["id"] and sum(len(t) + 3 for t in tags) <= 500
