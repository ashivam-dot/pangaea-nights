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


@pytest.mark.parametrize('blocked,mode,dispatch', [(1, 'true', True), (4, 'true', False),
                                                 (1, 'false', False)])
def test_known_editorial_hold_can_fill_only_a_bounded_dry_buffer(monkeypatch, blocked, mode, dispatch):
    jobs = [{'premise': str(i), 'stage': 'held', 'problems': ['wrong image']} for i in range(blocked)]
    jobs.append({'premise': 'next', 'stage': 'planned', 'inputs': {'dry_run': mode}})
    data = {'jobs': jobs, 'halted': 'Repair held entry 0: wrong image'}
    writes = []
    monkeypatch.setattr(batch, 'load', lambda: data)
    monkeypatch.setattr(batch, 'builds', lambda *a: [])
    def gh(*args, **kw):
        if kw.get('body'):
            assert jobs[-1]['stage'] == 'dispatch_unresolved'
            writes.append(kw['body'])
            return None
        return {'sha': 'pinned'}
    monkeypatch.setattr(batch, 'gh', gh)
    adapter = SimpleNamespace(REPO='owner/repo', WORKFLOW='run.yml', paused=lambda: False)
    batch.tick(adapter, 'publish', save=lambda d: None)
    assert bool(writes) is dispatch
    assert all(j['stage'] == 'held' for j in jobs[:-1])


def test_dispatch_behind_editorial_hold_is_reconciled_before_another_write(monkeypatch):
    pending = {'premise': 'next', 'stage': 'dispatch_unresolved', 'submitted_at': batch.stamp()}
    data = {'jobs': [{'premise': 'first', 'stage': 'held', 'problems': ['wrong image']}, pending],
            'halted': 'Repair held entry first: wrong image'}
    monkeypatch.setattr(batch, 'load', lambda: data)
    seen = []
    monkeypatch.setattr(batch, 'resolve', lambda r, w, j: seen.append(j['premise']) or False)
    monkeypatch.setattr(batch, 'gh', lambda *a, **k: pytest.fail('new write before reconciliation'))
    batch.tick(SimpleNamespace(REPO='owner/repo', WORKFLOW='run.yml', paused=lambda: False),
               'publish', save=lambda d: None)
    assert seen == ['next'] and pending['stage'] == 'dispatch_unresolved'


def test_completed_future_preview_waits_for_predecessor_history(monkeypatch):
    current = {'premise': 'first', 'stage': 'building', 'run_id': 1, 'source_sha': 'pinned'}
    future = {'premise': 'next', 'stage': 'building', 'run_id': 2, 'source_sha': 'pinned'}
    data = {'jobs': [current, future], 'halted': None}
    monkeypatch.setattr(batch, 'load', lambda: data)
    monkeypatch.setattr(batch, 'gh', lambda *a, **k: {'status': 'in_progress', 'head_sha': 'pinned'}
                        if args_run(a) == '1' else
                        {'status': 'completed', 'conclusion': 'success', 'head_sha': 'pinned'})
    monkeypatch.setattr(batch, 'materialize', lambda *a: pytest.fail('premature rotation review'))
    adapter = SimpleNamespace(REPO='owner/repo', WORKFLOW='run.yml', paused=lambda: False,
                              review_ready=lambda d, j: j is current)
    batch.tick(adapter, 'publish', save=lambda d: None)
    assert future['stage'] == 'building' and not data['halted']


def args_run(args):
    return args[1].rsplit('/', 1)[-1]


@pytest.mark.parametrize('with_hold', [False, True])
@pytest.mark.parametrize('active_ids,extra_buffer,dispatch', [([11], 0, True), ([11, 99], 0, False),
                                                            ([11], 2, False)])
def test_finished_future_preview_frees_producer_for_predecessor_repair(
        monkeypatch, active_ids, extra_buffer, dispatch, with_hold):
    repair = {'premise': 'repair', 'stage': 'planned', 'inputs': {'dry_run': 'true'}}
    running = {'premise': 'running', 'stage': 'building', 'run_id': 11, 'source_sha': 'abc'}
    finished = {'premise': 'finished', 'stage': 'building', 'run_id': 12, 'source_sha': 'abc'}
    data = {'jobs': [repair, running, finished] + [
        {'premise': f'buffer-{i}', 'stage': 'reviewed'} for i in range(extra_buffer)], 'halted': None}
    if with_hold:
        data['jobs'].insert(0, {'premise':'held', 'stage':'held', 'problems':['wrong image']})
        data['halted'] = 'Repair held entry held: wrong image'
    monkeypatch.setattr(batch, 'load', lambda: data)
    monkeypatch.setattr(batch, 'builds', lambda *a: [{'id': i, 'status': 'in_progress'} for i in active_ids])
    monkeypatch.setattr(batch, 'materialize', lambda *a: pytest.fail('future preview reviewed before repair'))
    writes, checkpoints = [], []
    def gh(*args, **kw):
        if kw.get('body'):
            assert next(j for j in checkpoints[-1]['jobs'] if j['premise']=='repair')['stage'] == 'dispatch_unresolved'
            writes.append(kw['body'])
            return None
        if args[1].endswith('/commits/main'):
            return {'sha': 'abc'}
        return {'head_sha': 'abc', 'status': 'in_progress'} if args_run(args) == '11' else {
            'head_sha': 'abc', 'status': 'completed', 'conclusion': 'success'}
    monkeypatch.setattr(batch, 'gh', gh)
    adapter = SimpleNamespace(REPO='owner/repo', WORKFLOW='run.yml', paused=lambda: False,
                              review_ready=lambda *a: False)
    batch.tick(adapter, 'publish', save=lambda d: checkpoints.append(deepcopy(d)))
    assert bool(writes) is dispatch
    assert repair['stage'] == ('dispatch_unresolved' if dispatch else 'planned')
    assert running['stage'] == finished['stage'] == 'building'
    assert finished['cloud_status'] == 'completed' and not finished.get('review')
    assert bool(data['halted']) is with_hold



@pytest.mark.parametrize('active_ids,expected', [([11], True), ([11, 99], False)])
def test_bounded_lookahead_overlaps_only_owned_dry_previews(monkeypatch, active_ids, expected):
    data = {'jobs': [{'premise': 'current', 'stage': 'building', 'run_id': 11, 'source_sha': 'abc'},
                     {'premise': 'next', 'stage': 'planned', 'inputs': {'dry_run': 'true'}, 'attempts': 0}],
            'halted': None}
    monkeypatch.setattr(batch, 'load', lambda: data)
    monkeypatch.setattr(batch, 'builds', lambda *a: [{'id': i, 'status': 'in_progress'} for i in active_ids])
    writes = []
    def gh(*args, **kw):
        if kw.get('body'):
            assert data['jobs'][1]['stage'] == 'dispatch_unresolved'
            writes.append(kw['body'])
            return None
        return {'sha': 'abc'} if args[1].endswith('/commits/main') else {
            'status': 'in_progress', 'head_sha': 'abc'}
    monkeypatch.setattr(batch, 'gh', gh)
    adapter = SimpleNamespace(REPO='owner/repo', WORKFLOW='run.yml', paused=lambda: False)
    batch.tick(adapter, 'publish', save=lambda d: None)
    assert bool(writes) is expected
    assert data['jobs'][1]['stage'] == ('dispatch_unresolved' if expected else 'planned')


def test_two_reserved_previews_never_dispatch_a_third(monkeypatch):
    data = {'jobs': [{'premise': str(i), 'stage': 'building', 'run_id': i, 'source_sha': 'abc'}
                     for i in (11, 12)] + [{'premise': 'later', 'stage': 'planned', 'inputs': {'dry_run': 'true'}}], 'halted': None}
    monkeypatch.setattr(batch, 'load', lambda: data)
    monkeypatch.setattr(batch, 'gh', lambda *a, **k: {'status': 'in_progress', 'head_sha': 'abc'})
    monkeypatch.setattr(batch, 'builds', lambda *a: pytest.fail('unbounded third production dispatch'))
    batch.tick(SimpleNamespace(REPO='owner/repo', WORKFLOW='run.yml', paused=lambda: False),
               'publish', save=lambda d: None)
    assert data['jobs'][2]['stage'] == 'planned'
