"""A concurrent later repair must preserve an earlier booking and both live job references."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest
from ops import batch_checkpoint, cloud_batch


def state():
    return {'schema': 1, 'enabled': True, 'halted': None, 'jobs': [
        {'premise': 'e04', 'stage': 'building', 'run_id': 104},
        {'premise': 'e06', 'stage': 'building', 'run_id': 106}]}


def held(data, index):
    job = data['jobs'][index]
    job.update(stage='held', problems=['Exact bad image'], editorial_holds=[{'id': 'bad-pixels'}])
    data['halted'] = f"Repair held entry {job['premise']}: Exact bad image"


def test_merge_preserves_independent_delivery_hold_and_object_references():
    base = state()
    local, remote = deepcopy(base), deepcopy(base)
    local['jobs'][0].update(stage='scheduled', delivery={'video_id': 'existing-id'})
    held(remote, 1)
    first, later = local['jobs']
    batch_checkpoint.refresh_in_place(local, batch_checkpoint.merge(base, local, remote))
    assert local['jobs'][0] is first and local['jobs'][1] is later
    assert first['delivery']['video_id'] == 'existing-id' and later['stage'] == 'held'
    assert local['halted'].startswith('Repair held entry e06:')


def test_same_job_order_and_global_halt_conflicts_remain_blocking():
    base = state()
    local, remote = deepcopy(base), deepcopy(base)
    local['jobs'][0]['stage'] = 'upload_intent'
    held(remote, 0)
    with pytest.raises(batch_checkpoint.StateConflict):
        batch_checkpoint.merge(base, local, remote)
    remote = deepcopy(base)
    remote['jobs'].reverse()
    with pytest.raises(batch_checkpoint.StateConflict):
        batch_checkpoint.merge(base, local, remote)
    remote = deepcopy(base)
    remote['halted'] = 'Delivery identity mismatch'
    with pytest.raises(batch_checkpoint.StateConflict):
        batch_checkpoint.merge(base, local, remote)


@pytest.mark.parametrize('same_job', [False, True])
def test_real_git_concurrent_repair_is_preserved_or_stops_checkpoint(tmp_path, monkeypatch, same_job):
    def git(path, *args):
        return subprocess.check_output(['git', '-C', str(path), *args], stderr=subprocess.DEVNULL, text=True)

    remote = tmp_path / 'remote.git'
    subprocess.run(['git', 'init', '--bare', str(remote)], check=True, capture_output=True)
    writer = tmp_path / 'writer'
    subprocess.run(['git', 'clone', str(remote), str(writer)], check=True, capture_output=True)
    git(writer, 'checkout', '-b', 'state')
    git(writer, 'config', 'user.name', 'test')
    git(writer, 'config', 'user.email', 'test@example.invalid')
    filename = cloud_batch.STATE_FILE
    (writer / filename).write_text(json.dumps(state(), indent=2) + '\n')
    git(writer, 'add', filename)
    git(writer, 'commit', '-m', 'initial')
    git(writer, 'push', 'origin', 'state')
    checkout = tmp_path / 'checkout'
    subprocess.run(['git', 'clone', '--branch', 'state', str(remote), str(checkout)],
                   check=True, capture_output=True)
    local = json.loads((checkout / filename).read_text())
    first, later = local['jobs']
    first.update(stage='scheduled', delivery={'video_id': 'existing-id'})
    incoming = state()
    held(incoming, 0 if same_job else 1)
    (writer / filename).write_text(json.dumps(incoming, indent=2) + '\n')
    git(writer, 'add', filename)
    git(writer, 'commit', '-m', 'exact input hold')
    git(writer, 'push', 'origin', 'state')
    monkeypatch.setattr(cloud_batch, 'STATE', checkout)
    if same_job:
        with pytest.raises(cloud_batch.TransientOperation):
            cloud_batch.persist(local)
        saved = json.loads(git(remote, 'show', 'state:' + filename))
        assert saved['jobs'][0]['stage'] == 'held' and not saved['jobs'][0].get('delivery')
    else:
        cloud_batch.persist(local)
        # A subsequent checkpoint must retain the imported hold too.
        first['public_verified_at'] = 'later check'
        cloud_batch.persist(local)
        saved = json.loads(git(remote, 'show', 'state:' + filename))
        assert first is local['jobs'][0] and later is local['jobs'][1]
        assert saved['jobs'][0]['delivery']['video_id'] == 'existing-id'
        assert saved['jobs'][1]['editorial_holds'] == [{'id': 'bad-pixels'}]
