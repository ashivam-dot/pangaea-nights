from email.utils import format_datetime
from datetime import datetime, timedelta, timezone

import pytest
import requests

from studio import sdcpp


class Response:
    def __init__(self, status=200, headers=None, chunks=()):
        self.status_code = status
        self.headers = headers or {}
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def iter_content(self, size):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


def test_rate_limit_honours_delay_then_installs_only_complete_download(tmp_path, monkeypatch):
    target = tmp_path / "model.gguf"
    partial = target.with_suffix(".gguf.part")
    partial.write_bytes(b"old incomplete download")
    replies = iter([Response(429, {"Retry-After": "120"}), Response(chunks=[b"model", b" weights"])])
    waits = []
    monkeypatch.setattr(sdcpp.requests, "get", lambda *a, **kw: next(replies))
    def wait(seconds):
        assert not target.exists() and not partial.exists()
        waits.append(seconds)
    monkeypatch.setattr(sdcpp.time, "sleep", wait)
    assert sdcpp._download("https://example.test/model", target) == target
    assert waits == [120]
    assert target.read_bytes() == b"model weights" and not partial.exists()


def test_interrupted_stream_never_becomes_a_cached_model(tmp_path, monkeypatch):
    target = tmp_path / "model.gguf"
    replies = iter([Response(chunks=[b"broken", requests.ConnectionError("disconnected")]),
                    Response(chunks=[b"complete"])])
    monkeypatch.setattr(sdcpp.requests, "get", lambda *a, **kw: next(replies))
    monkeypatch.setattr(sdcpp.time, "sleep", lambda seconds: None)
    sdcpp._download("https://example.test/model", target)
    assert target.read_bytes() == b"complete"


def test_permanent_rejection_does_not_repeat_requests(tmp_path, monkeypatch):
    calls = []
    def get(*a, **kw):
        calls.append(a)
        return Response(403)
    monkeypatch.setattr(sdcpp.requests, "get", get)
    monkeypatch.setattr(sdcpp.time, "sleep", lambda seconds: pytest.fail("permanent failure retried"))
    with pytest.raises(requests.HTTPError):
        sdcpp._download("https://example.test/model", tmp_path / "model.gguf")
    assert len(calls) == 1


def test_repeated_rate_limit_is_bounded_and_leaves_no_cached_file(tmp_path, monkeypatch):
    target = tmp_path / "model.gguf"
    waits = []
    monkeypatch.setattr(sdcpp.requests, "get", lambda *a, **kw: Response(429, {"Retry-After": "3600"}))
    monkeypatch.setattr(sdcpp.time, "sleep", waits.append)
    with pytest.raises(requests.HTTPError):
        sdcpp._download("https://example.test/model", target)
    assert waits == [180, 180, 180]
    assert not target.exists() and not target.with_suffix(".gguf.part").exists()


def test_retry_date_and_invalid_headers_have_bounded_fallback():
    future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=120), usegmt=True)
    assert 118 <= sdcpp._download_delay(Response(503, {"Retry-After": future}), 0) <= 120
    assert sdcpp._download_delay(Response(429, {"Retry-After": "invalid"}), 1) == 60
