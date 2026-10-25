import copy
from datetime import date

import pytest

from studio import ladder, ladder_cli, ledger


def held_bank():
    banks = copy.deepcopy(ladder.load_banks(strict=True))
    held = banks['level_1_8'][0]
    held['production_hold'] = 'Exact scientific body plan failed direct review'
    data = copy.deepcopy(ledger.EMPTY)
    data['ladder']['queue'] = [held['id']]
    return banks, held, data


def test_ordinary_selection_skips_held_queued_premise():
    banks, held, data = held_bank()
    assert ladder.pick(banks, data, date(2026, 10, 26))['id'] != held['id']
    assert not ladder.plan_problems(held, data, date(2026, 10, 26))['premise']['ok']


@pytest.mark.parametrize('mode', ['dry', 'test', 'publish'])
def test_explicit_preview_or_publish_cannot_override_editorial_hold(mode):
    banks, held, data = held_bank()
    with pytest.raises(ladder.BankError, match='production hold'):
        ladder_cli.choose(banks, data, date(2026, 10, 26), [], mode, held['id'])


def test_all_held_supply_fails_closed_instead_of_planning_bad_art():
    banks, _, data = held_bank()
    for ps in banks.values():
        for premise in ps:
            premise['production_hold'] = 'Exact render needs repair'
    with pytest.raises(ladder.BankError, match='no premise satisfies'):
        ladder.pick(banks, data, date(2026, 10, 26))
