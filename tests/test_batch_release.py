"""Release safety and resumption, rather than a green workflow alone, define batch completion."""
import io
import json
import zipfile
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ops import batch_state, cloud_batch as batch, media_review as review


def verdict():
    return {"decision": "pass", "audio_listened": True, "video_viewed": True,
            "heard_opening": "The actual opening", "heard_ending": "The actual closing",
            "checks": {k: True for k in review.CHECKS}, "issues": [],
            "segments": [{"part": k, "seconds": t, "seen": "Specific scene", "heard": "Spoken words"}
                         for k, t in (("opening", 1), ("middle", 20), ("ending", 39))]}


def test_a_pass_alone_is_not_an_audiovisual_review():
    assert review.verdict_problems({"decision": "pass"}, 40)
    assert not review.verdict_problems(verdict(), 40)
    v = verdict()
    v["segments"][-1]["seconds"] = 1
    assert "ending" in " ".join(review.verdict_problems(v, 40))
    v = verdict()
    v["issues"] = [{"severity": "blocker", "seconds": 10, "problem": "Wrong object"}]
    assert "Wrong object" in " ".join(review.verdict_problems(v, 40))


def test_review_cannot_be_reused_for_different_video_or_script(tmp_path):
    mp4 = tmp_path / "short.mp4"
    mp4.write_bytes(b"original media")
    brief = {"script": "original"}
    receipt = {"schema": 1, "model": "gemini-3.8-flash", "duration": 40, "video_sha256": review.digest(mp4),
               "brief_sha256": review.brief_digest(brief), "review_video_sha256": "a", "review_audio_sha256": "b",
               "verdict": verdict()}
    assert not review.receipt_problems(receipt, mp4, brief)
    assert review.receipt_problems(receipt, mp4, {"script": "changed"})
    mp4.write_bytes(b"different media")
    assert review.receipt_problems(receipt, mp4, brief)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "safe/../../escape", "safe\\escape"])
def test_artifact_paths_cannot_escape_the_work_directory(tmp_path, name):
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as z:
        z.writestr(name, "x")
    with pytest.raises(ValueError, match="unsafe"):
        batch.unpack(blob.getvalue(), tmp_path / "out")


def test_calendar_never_replaces_an_occupied_slot_or_silently_moves_a_late_one():
    at = datetime(2026, 10, 25, tzinfo=timezone.utc)
    assert batch.release_time({"target": "now-after-review"}, set(), at) is None
    job = {"target": "2026-10-25T07:30:00Z"}
    assert batch.release_time(job, set(), at) == job["target"]
    with pytest.raises(ValueError, match="occupied"):
        batch.release_time(job, {job["target"]}, at)
    with pytest.raises(ValueError, match="too close or past"):
        batch.release_time({"target": "2026-10-24T07:30:00Z"}, set(), at)


def setup_tick(monkeypatch, stage="building"):
    job = {"premise": "e01", "stage": stage, "run_id": 123, "source_sha": "abc", "target": "now-after-review"}
    data = {"schema": 1, "enabled": True, "jobs": [job], "halted": None}
    monkeypatch.setattr(batch, "load", lambda: data)
    monkeypatch.setattr(batch, "gh", lambda *a, **k: {"status": "completed", "conclusion": "success", "head_sha": "abc"})
    monkeypatch.setattr(batch, "materialize", lambda *a: "folder")
    monkeypatch.setattr(batch, "archive_review", lambda *a: "private-review")
    adapter = SimpleNamespace(REPO="owner/repo", WORKFLOW="run.yml", paused=lambda: False,
                              validate=lambda *a: ("media", {}), taken=lambda: set(),
                              release_ready=lambda *a: True,
                              publish=lambda *a: pytest.fail("unapproved media uploaded"))
    receipt = {"model": "gemini-3.8-flash", "video_sha256": "digest", "verdict": verdict(), "duration": 40}
    monkeypatch.setattr(review, "review", lambda *a: receipt)
    return job, data, adapter


def test_a_negative_review_never_reaches_youtube(monkeypatch):
    job, data, adapter = setup_tick(monkeypatch)
    monkeypatch.setattr(review, "receipt_problems", lambda *a: ["The spoken payoff is missing"])
    batch.tick(adapter, "publish", save=lambda d: None)
    assert job["stage"] == "held" and "payoff" in data["halted"]


def test_upload_failure_keeps_the_saved_intent_for_marker_recovery(monkeypatch):
    job, data, adapter = setup_tick(monkeypatch)
    monkeypatch.setattr(review, "receipt_problems", lambda *a: [])
    checkpoints = []
    def fail(*args):
        assert checkpoints[-1] == "upload_intent"
        raise ConnectionError("network lost after request")
    adapter.publish = fail
    with pytest.raises(ConnectionError):
        batch.tick(adapter, "publish", save=lambda d: checkpoints.append(job["stage"]))
    assert job["stage"] == "upload_intent"
    assert "intended_publish_at" in job and job["intended_publish_at"] is None


def test_an_ambiguous_dispatch_is_not_submitted_again(monkeypatch):
    job, data, adapter = setup_tick(monkeypatch, "dispatch_unresolved")
    job.update(before_ids=[], submitted_at=batch.stamp())
    monkeypatch.setattr(batch, "resolve", lambda *a: False)
    monkeypatch.setattr(batch, "gh", lambda *a, **k: pytest.fail("duplicate dispatch"))
    batch.tick(adapter, "publish", save=lambda d: None)
    assert job["stage"] == "dispatch_unresolved"


def test_ordinary_production_resumes_only_after_the_reserved_calendar(tmp_path):
    path = tmp_path / "batch-production.json"
    path.write_text(json.dumps({"schema": 1, "enabled": True, "complete": True,
                                "jobs": [{"target": "2026-11-03T13:30:00Z"}]}))
    assert batch_state.production_hold(tmp_path, datetime(2026, 11, 2, tzinfo=timezone.utc))
    assert batch_state.production_hold(tmp_path, datetime(2026, 11, 4, tzinfo=timezone.utc)) is None


def test_a_booked_release_must_actually_become_public():
    job = {"stage": "scheduled", "intended_publish_at": "2026-10-25T07:30:00Z", "delivery": {"video_id": "v1"}}
    data = {"jobs": [job], "halted": None}
    video = {"snippet": {"channelId": "UCa"}, "status": {"privacyStatus": "private", "madeForKids": False,
             "selfDeclaredMadeForKids": False, "uploadStatus": "processed"}}
    adapter = SimpleNamespace(CHANNEL_ID="UCa", poll_delivery=lambda j: video)
    batch.poll_due(data, adapter, datetime(2026, 10, 25, 7, 40, tzinfo=timezone.utc))
    assert data["halted"] is None and job["stage"] == "scheduled"
    batch.poll_due(data, adapter, datetime(2026, 10, 25, 7, 50, tzinfo=timezone.utc))
    assert "not public" in data["halted"]
    data["halted"] = None
    video["status"]["privacyStatus"] = "public"
    batch.poll_due(data, adapter, datetime(2026, 10, 25, 8, tzinfo=timezone.utc))
    assert job["stage"] == "public" and job["delivery"]["owner_readback"]["channel_id"] == "UCa"
