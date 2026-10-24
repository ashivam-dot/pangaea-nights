"""A future editorial refusal cannot freeze an earlier approved release or hide a booked release."""
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ops import cloud_batch as batch


@pytest.fixture
def scenario(monkeypatch):
    first = {"premise": "first", "stage": "reviewed", "run_id": 1, "target": "now-after-review"}
    held = {"premise": "second", "stage": "held", "run_id": 2, "problems": ["wrong animal anatomy"]}
    later = {"premise": "third", "stage": "reviewed", "run_id": 3, "target": "2030-01-01T14:30:00Z"}
    data = {"jobs": [first, held, later], "halted": "Repair held entry second: wrong animal anatomy"}
    calls = []
    monkeypatch.setattr(batch, "load", lambda: data)
    monkeypatch.setattr(batch, "materialize", lambda r, a, j: calls.append(("materialize", j["premise"])) or "folder")
    monkeypatch.setattr(batch, "restore_review", lambda *a: None)
    monkeypatch.setattr(batch.media_review, "review", lambda *a: {"verdict": {"decision": "pass"}})
    monkeypatch.setattr(batch.media_review, "receipt_problems", lambda *a: [])
    monkeypatch.setattr(batch, "archive_review", lambda *a: "saved-original-media")
    monkeypatch.setattr(batch, "builds", lambda *a: pytest.fail("dispatch through a sequence hold"))
    def publish(job, *a):
        calls.append(("publish", job["premise"]))
        return {"privacy": "public", "video_id": "retained-id"}
    adapter = SimpleNamespace(REPO="owner/repo", WORKFLOW="run.yml", CHANNEL_ID="channel",
                              paused=lambda: False, validate=lambda *a: ("media", {}),
                              taken=lambda: set(), release_ready=lambda *a: True, publish=publish)
    return data, adapter, calls


def test_future_hold_allows_only_the_earlier_approved_release(scenario):
    data, adapter, calls = scenario
    checkpoints = []
    batch.tick(adapter, "publish", save=lambda d: checkpoints.append(deepcopy(d)))
    assert calls == [("materialize", "first"), ("publish", "first")]
    assert any(d["jobs"][0]["stage"] == "upload_intent" for d in checkpoints)
    assert [j["stage"] for j in data["jobs"]] == ["public", "held", "reviewed"]
    assert data["halted"] == "Repair held entry second: wrong animal anatomy"


@pytest.mark.parametrize("halt", ["Uncertain dispatch; reconcile", "Delivery wrong-channel",
                                 "Repair held entry another: wrong animal anatomy"])
def test_unknown_or_mismatched_halt_still_blocks_all_writes(scenario, halt):
    data, adapter, calls = scenario
    data["halted"] = halt
    batch.tick(adapter, "publish", save=lambda d: pytest.fail("unrecognised halt mutated state"))
    assert not calls and data["halted"] == halt


def test_due_booked_release_is_polled_despite_a_later_hold(scenario, monkeypatch):
    data, adapter, calls = scenario
    first = data["jobs"][0]
    first.update(stage="scheduled", intended_publish_at="2026-10-25T00:30:00Z",
                 delivery={"video_id": "original-upload-id"})
    monkeypatch.setattr(batch, "now", lambda: datetime(2026, 10, 25, 1, tzinfo=timezone.utc))
    adapter.poll_delivery = lambda job: {"snippet": {"channelId": "channel"}, "status": {
        "privacyStatus": "public", "madeForKids": False, "selfDeclaredMadeForKids": False,
        "uploadStatus": "processed", "containsSyntheticMedia": True}}
    batch.tick(adapter, "publish", save=lambda d: None)
    assert first["stage"] == "public" and first["delivery"]["video_id"] == "original-upload-id"
    assert not calls


def test_due_delivery_mismatch_remains_a_global_stop(scenario, monkeypatch):
    data, adapter, calls = scenario
    data["jobs"][0].update(stage="scheduled", intended_publish_at="2026-10-25T00:30:00Z",
                            delivery={"video_id": "original-upload-id"})
    monkeypatch.setattr(batch, "now", lambda: datetime(2026, 10, 25, 1, tzinfo=timezone.utc))
    adapter.poll_delivery = lambda job: {"snippet": {"channelId": "wrong-channel"}, "status": {}}
    batch.tick(adapter, "publish", save=lambda d: None)
    assert data["halted"].startswith("Delivery original-upload-id:") and not calls


def test_direct_editorial_observation_cannot_be_overruled_by_green_model_review(scenario):
    data, adapter, calls = scenario
    data["halted"] = None
    job = data["jobs"][0]
    job.update(stage="building", editorial_holds=[{"reason": "exact base and alternate depict pillbugs"}])
    batch.tick(adapter, "publish", save=lambda d: None)
    assert job["stage"] == "held" and "pillbugs" in data["halted"] and not calls


def test_resolved_direct_observation_is_retained_as_history():
    job = {"editorial_holds": [{"reason": "old wrong image", "resolved_at": "2026-10-25T01:00:00Z"}]}
    assert batch.editorial_problems(job) == []
