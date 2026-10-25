"""Merge independent batch job checkpoints without overruling a concurrent repair."""
from copy import deepcopy


class StateConflict(RuntimeError):
    pass


def merge(base, local, remote):
    """Only independent jobs may merge. Concurrent edits to one job require reconciliation."""
    keys = [[j['premise'] for j in d['jobs']] for d in (base, local, remote)]
    if keys[0] != keys[1] or keys[0] != keys[2]:
        raise StateConflict('batch order changed during checkpoint; reconcile before another write')

    def choose(before, ours, theirs, label):
        if ours == theirs or theirs == before:
            return ours if ours is missing else deepcopy(ours)
        if ours == before:
            return theirs if theirs is missing else deepcopy(theirs)
        raise StateConflict(f'concurrent batch edit to {label}; reconcile before another write')

    missing = object()
    result = {}
    derived = {'jobs', 'updated_at', 'health', 'complete', 'halted'}
    for key in (set(base) | set(local) | set(remote)) - derived:
        value = choose(base.get(key, missing), local.get(key, missing), remote.get(key, missing), key)
        if value is not missing:
            result[key] = value
    result['jobs'] = [choose(b, l, r, b['premise'])
                      for b, l, r in zip(base['jobs'], local['jobs'], remote['jobs'])]
    # A delivery/global halt is never cleared or silently merged into a sequence hold.
    for d in (local, remote):
        halt = d.get('halted')
        if halt and not halt.startswith('Repair held entry '):
            raise StateConflict('global batch halt changed during checkpoint; reconcile before another write')
    held = [j for j in result['jobs'] if j['stage'] == 'held']
    result['halted'] = ('Repair held entry ' + held[0]['premise'] + ': ' +
                        '; '.join(held[0].get('problems', []))) if held else None
    result['complete'] = all(j['stage'] in ('public', 'scheduled') for j in result['jobs'])
    if 'health' in local:
        result['health'] = {**local['health'],
                            'public': sum(j['stage'] == 'public' for j in result['jobs']),
                            'scheduled': sum(j['stage'] == 'scheduled' for j in result['jobs']),
                            'held': len(held)}
    result['updated_at'] = local.get('updated_at')
    return result


def refresh_in_place(data, merged):
    """Retain the coordinator's live job references while importing a later job's repair."""
    jobs = data['jobs']
    for current, updated in zip(jobs, merged['jobs']):
        current.clear()
        current.update(updated)
    data.clear()
    data.update({**merged, 'jobs': jobs})
