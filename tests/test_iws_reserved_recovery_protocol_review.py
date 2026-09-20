"""Independent synthetic tests of the post-access numerical recovery boundary."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from shiftwm.real_video_iws_reserved_recovery import protocol as p


def fixture_contract():
    baseline = {k: {'unchanged': k} for k in ('runs', 'tasks', 'evaluation', 'extraction', 'qualitative_selection')}
    baseline.update(expected_runs=36, dependencies={'old_source.py': 'a' * 64})
    recovery = {
        'command_gru_dispatch': 'one_native_row_per_call', 'same_weights': True,
        'same_normalization': True, 'same_tolerance': True, 'same_metrics': True,
        'same_examples': True, 'same_comparators': True, 'all36_rerun': True,
        'original_results_retained': True, 'future_targets_used_to_choose_backend': False,
        'accuracy_used_to_choose_backend': False, 'initial_reserved_access_precedes_this_revision': True,
    }
    registry = {**copy.deepcopy(baseline), 'schema': p.SCHEMA, 'status': p.STATE,
                'cache_registration_path': str(p.original.REGISTRATION_PATH), 'numerical_recovery': recovery}
    registry['dependencies']['new_source.py'] = 'b' * 64
    return baseline, registry


@pytest.mark.parametrize('field', ['runs', 'tasks', 'evaluation', 'extraction', 'qualitative_selection', 'expected_runs'])
def test_cannot_change_any_original_scientific_contract(field):
    base, registered = fixture_contract()
    assert p.validate_recovery(registered, base) == registered
    registered[field] = 'changed'
    with pytest.raises(ValueError, match='scientific contract'):
        p.validate_recovery(registered, base)


@pytest.mark.parametrize('field', ['same_tolerance', 'same_metrics', 'same_comparators', 'all36_rerun',
                                  'original_results_retained', 'initial_reserved_access_precedes_this_revision',
                                  'future_targets_used_to_choose_backend', 'accuracy_used_to_choose_backend'])
def test_cannot_relax_tolerance_mix_results_or_hide_postaccess(field):
    base, registered = fixture_contract()
    registered['numerical_recovery'][field] = not registered['numerical_recovery'][field]
    with pytest.raises(ValueError, match='Recovery scope'):
        p.validate_recovery(registered, base)


@pytest.mark.parametrize('kind', ['drop', 'mutate'])
def test_all_original_source_bindings_retained(kind):
    base, registered = fixture_contract()
    if kind == 'drop':
        del registered['dependencies']['old_source.py']
    else:
        registered['dependencies']['old_source.py'] = 'c' * 64
    with pytest.raises(ValueError, match='original frozen source'):
        p.validate_recovery(registered, base)


def test_review_checked_before_new_source_hashing(tmp_path, monkeypatch):
    base, registered = fixture_contract()
    registered['original_registration_sha256'] = 'a' * 64
    path = tmp_path / p.REGISTRATION_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(registered))
    monkeypatch.setattr(p.original, 'checked_registration', lambda root: base)
    original_read = p.original.read_json
    def read(name):
        if Path(name) == tmp_path / p.REVIEW_PATH:
            return {'status': 'pending'}
        return original_read(name)
    monkeypatch.setattr(p.original, 'read_json', read)
    def digest(name):
        assert Path(name) == tmp_path / p.original.REGISTRATION_PATH, 'New sources hashed before review'
        return 'a' * 64
    monkeypatch.setattr(p.original, 'sha', digest)
    with pytest.raises(ValueError, match='Independent numerical recovery review'):
        p.checked_registration(tmp_path)
