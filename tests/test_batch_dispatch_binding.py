"""A returned dispatch identity disambiguates concurrent repairs without a new POST."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from ops import cloud_batch as batch


@pytest.mark.parametrize('pinned,expected', [(None, False), (11, True), (13, False), (9, False)])
def test_resolve_uses_saved_dispatch_identity_and_preserves_old_safety(monkeypatch, pinned, expected):
    job = {'stage': 'dispatch_unresolved', 'before_ids': [9], 'source_sha': 'source'}
    if pinned:
        job['dispatch_run_id'] = pinned
    rows = [{'id': i, 'head_sha': 'source', 'event': 'workflow_dispatch'} for i in (9, 11, 12)]
    rows.append({'id': 13, 'head_sha': 'different', 'event': 'workflow_dispatch'})
    monkeypatch.setattr(batch, 'builds', lambda *a: rows)
    before = deepcopy(job)
    assert batch.resolve('owner/repo', 'shorts.yml', job) is expected
    if expected:
        assert job['stage'] == 'building' and job['run_id'] == 11
    else:
        assert job == before


def test_legacy_unique_dispatch_still_resolves(monkeypatch):
    job = {'stage': 'dispatch_unresolved', 'before_ids': [], 'source_sha': 'source'}
    monkeypatch.setattr(batch, 'builds', lambda *a: [
        {'id': 11, 'head_sha': 'source', 'event': 'workflow_dispatch'}])
    assert batch.resolve('owner/repo', 'shorts.yml', job)
    assert job['run_id'] == 11


def test_dispatch_response_is_checkpointed_while_intent_remains_durable(monkeypatch):
    pending = {'premise': 'next', 'stage': 'planned', 'inputs': {'dry_run': 'true'}}
    data = {'jobs': [pending], 'halted': None}
    checkpoints = []
    monkeypatch.setattr(batch, 'load', lambda: data)
    monkeypatch.setattr(batch, 'builds', lambda *a: [])
    def gh(*args, **kw):
        if kw.get('body'):
            assert checkpoints[-1]['jobs'][0]['stage'] == 'dispatch_unresolved'
            assert kw['body']['return_run_details'] is True
            return {'workflow_run_id': 11}
        return {'sha': 'source'}
    monkeypatch.setattr(batch, 'gh', gh)
    batch.tick(SimpleNamespace(REPO='owner/repo', WORKFLOW='shorts.yml', paused=lambda: False),
               'review', save=lambda d: checkpoints.append(deepcopy(d)))
    assert pending['dispatch_run_id'] == 11
    assert any(d['jobs'][0].get('dispatch_run_id') == 11 for d in checkpoints)
