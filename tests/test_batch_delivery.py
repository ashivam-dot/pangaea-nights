"""An immediate batch release must remain private until processing and metadata are verified."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from studio import batch_release as release, config, ledger, youtube


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    p = {"id": "how_deep-01", "series": "how_deep", "era": "Deep sea", "title": "A sourced choice",
         "related_topic_hint": "ocean"}
    (tmp_path / "plan.json").write_text(json.dumps({"premise": p, "script": "The sourced script"}))
    (tmp_path / "gates.json").write_text(json.dumps({"ok": True, "frame0": "fingerprint"}))
    (tmp_path / "short.mp4").write_bytes(b"exact reviewed media")
    video = {"id": "existing-or-new-id", "snippet": {"channelId": config.CHANNEL_ID,
             "title": p["title"], "publishedAt": "2026-10-25T00:00:00Z"}, "status": {
             "privacyStatus": "private", "madeForKids": False, "selfDeclaredMadeForKids": False,
             "uploadStatus": "processed"}}
    writes = []
    class API:
        def videos(self):
            return self

        def list(self, **kwargs):
            return SimpleNamespace(execute=lambda: {"items": [deepcopy(video)]})

        def update(self, **kwargs):
            writes.append(kwargs["body"])
            video["status"] = {**video["status"], **kwargs["body"]["status"]}
            return SimpleNamespace(execute=lambda: {})

    api = API()
    monkeypatch.setattr(release, "validate", lambda *a: None)
    monkeypatch.setattr(youtube, "client", lambda: api)
    monkeypatch.setattr(youtube, "identity", lambda api: {"id": config.CHANNEL_ID})
    monkeypatch.setattr(youtube, "related_episode", lambda *a: None)
    monkeypatch.setattr(youtube, "ladder_description", lambda *a: "Sources and disclosure")
    monkeypatch.setattr(youtube, "ladder_tags", lambda *a: ["marker"])
    def upload(*args):
        when = args[-1]
        if when:
            video["status"]["publishAt"] = when
        return {"video_id": video["id"], "url": "https://youtube.com/shorts/" + video["id"]}
    monkeypatch.setattr(youtube, "upload_ladder", upload)
    monkeypatch.setattr(release.time, "sleep", lambda *a: None)
    data = deepcopy(ledger.EMPTY)
    monkeypatch.setattr(ledger, "load", lambda: deepcopy(data))
    monkeypatch.setattr(ledger, "save", lambda new: data.update(deepcopy(new)))
    job = {"premise": p["id"], "run_id": 123, "review": {"video_sha256": "approved-digest"}}
    return job, tmp_path, video, writes, data


def test_immediate_release_is_public_only_after_the_private_preflight(prepared):
    job, folder, video, writes, data = prepared
    result = release.publish(job, folder, None)
    assert [w["status"]["privacyStatus"] for w in writes] == ["public"]
    assert result["owner_readback"]["madeForKids"] is False
    assert result["privacy"] == "public" and data["shorts"][0]["video_id"] == video["id"]
    assert data["shorts"][0]["av_review_sha256"] == "approved-digest"


def test_a_wrong_audience_flag_never_becomes_public(prepared):
    job, folder, video, writes, data = prepared
    video["status"]["madeForKids"] = True
    with pytest.raises(RuntimeError, match="private pre-release readback"):
        release.publish(job, folder, None)
    assert writes == [] and data["shorts"] == []


def test_a_future_schedule_stays_private_and_keeps_its_exact_slot(prepared):
    job, folder, video, writes, data = prepared
    when = "2026-11-01T14:30:00Z"
    result = release.publish(job, folder, when)
    assert result["privacy"] == "private" and result["publish_at"] == when
    assert writes == [] and data["shorts"][0]["publish_at"] == when


def test_recovery_of_an_already_public_upload_preserves_its_id(prepared):
    job, folder, video, writes, data = prepared
    video["status"]["privacyStatus"] = "public"
    result = release.publish(job, folder, None)
    assert writes == [] and result["video_id"] == video["id"]
