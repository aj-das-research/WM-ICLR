"""Synthetic-only validator and layout tests; no active paper outputs."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result); return result

renderer = load(ROOT / 'paper/scripts/render_iws_reserved_evidence.py', 'reserved_report_test')
fixtures = load(ROOT / 'tests/test_iws_reserved_recovery_finalization.py', 'reserved_report_synthetic')

@pytest.fixture(scope='module')
def complete():
    rows, populations = fixtures.fixture()
    arrays = fixtures.final.assemble(rows, populations)
    results = fixtures.final.numerical_report(*arrays)
    receipts = {}; identities = []
    for row in rows:
        name = f"{row['task']}_{row['mode']}_s{row['seed']}"
        identity = {k: row[k] for k in ('task', 'mode', 'seed')}
        identity.update(name=name, selected_epoch=7, package_kind='synthetic', checkpoint_sha256='a'*64)
        identities.append(identity)
        receipt = copy.deepcopy(row['receipt'])
        receipt.update(identity, schema='shiftwm_iws_reserved_recovery_evaluation_v2', status='passed', numerical_recovery=renderer.RECOVERY,
                       selected_checkpoint_sha256='a'*64, registration_sha256='b'*64, population=populations[row['task']])
        receipts[name] = receipt
    value = {'schema': 'shiftwm_iws_reserved_recovery_finalization_v2', 'status': 'passed', 'scope': 'reserved_upstream_validation',
             'numerical_recovery': renderer.RECOVERY,
             'expected_runs': 36, 'completed_runs': 36, 'per_run': identities, 'populations': populations,
             'registration_sha256': 'b'*64, 'total_unique_handles': 600, 'total_unique_trajectories': 30, 'results': results}
    return value, receipts


def test_recompute_matches_frozen_scientific_finalizer(complete):
    value, receipts = complete
    result = renderer.validate_numerics(value, receipts)
    assert result['signed_gain_intervals'] == 80 and result['h60_score_cells'] == 60
    assert result['public_binary_dependencies'] == 0


def test_pending_writes_no_pack_or_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(renderer, 'FINAL', tmp_path/'absent.json')
    result = renderer.render(if_ready=True, pack=tmp_path/'pack', output=tmp_path/'output')
    assert result['status'] == 'pending' and not result['outputs_written']
    assert list(tmp_path.iterdir()) == []


def test_finalization_without_independent_review_stays_pending(monkeypatch, tmp_path):
    final=tmp_path/'final.json';final.write_text('{}')
    monkeypatch.setattr(renderer,'FINAL',final)
    monkeypatch.setattr(renderer,'REVIEW',tmp_path/'missing-review.json')
    assert renderer.live_pack(tmp_path/'pack',if_ready=True) is None
    assert not (tmp_path/'pack').exists()
    assert renderer.render(if_ready=True,pack=tmp_path/'pack',output=tmp_path/'out')['reason']=='Independent complete-result review absent'


def test_stale_or_failed_independent_review_is_rejected():
    good={'status':'passed','completed_runs':36,'finalization_sha256':'a'*64,'registration_sha256':'b'*64}
    assert renderer.checked_result_review(good,'a'*64,'b'*64)==good
    for changed in ({'status':'failed'},{'completed_runs':35},{'finalization_sha256':'c'*64},{'registration_sha256':'c'*64}):
        with pytest.raises(ValueError):renderer.checked_result_review({**good,**changed},'a'*64,'b'*64)


def test_missing_model_rejected_before_bootstrap(complete):
    value, receipts = copy.deepcopy(complete); value['per_run'].pop()
    with pytest.raises(ValueError, match='Incomplete36'):
        renderer.validate_numerics(value, receipts)


def test_original_or_mixed_backend_cannot_enter_pack(complete):
    value, receipts = copy.deepcopy(complete)
    value['schema'] = 'shiftwm_iws_reserved_finalization_v1'
    with pytest.raises(ValueError, match='schema'):
        renderer.validate_numerics(value, receipts)
    value, receipts = copy.deepcopy(complete)
    next(iter(receipts.values()))['numerical_recovery'] = {}
    with pytest.raises(ValueError, match='backend'):
        renderer.validate_numerics(value, receipts)


def test_per_trajectory_tamper_rejected(complete):
    value, receipts = copy.deepcopy(complete)
    row = next(iter(receipts.values())); row['episodes'][0]['standardized_mse_by_offset'][0] += .1
    with pytest.raises(ValueError, match='trajectory means'):
        renderer.validate_numerics(value, receipts)


def test_unequal_handle_weights_not_replaced_with_equal_trajectory(complete):
    value, receipts = copy.deepcopy(complete)
    row = next(iter(receipts.values())); row['equal_handle']['standardized_mse'] = row['equal_trajectory']['standardized_mse']
    with pytest.raises(ValueError, match='Weighted handle'):
        renderer.validate_numerics(value, receipts)


def test_signed_plot_and_all_tables_keep_complete_roster(complete):
    value, _ = copy.deepcopy(complete)
    negative = value['results']['task_results']['pusht']['standardized_mse']['h60_comparisons']['bounded_vs_additive']
    negative['relative_error_reduction_percent'] = -4.5
    negative['paired95']['gain_percent']['percentile95'] = [-9.0, 1.0]
    figure, geometry = renderer.make_figure(value)
    assert len(geometry['marks']) == 15
    assert geometry['marks'][0]['gain_percent'] == -4.5
    assert geometry['shared_gain_axis_limits'][0] < -9
    assert geometry['minimum_font_pt'] == 8
    renderer.plt.close(figure)
    tables = renderer.tables(value)
    assert len(tables) == 6
    assert tables['scores.tex'].count('No tanh (ours, ablation)') == 3
    assert '-4.50' in tables['contrasts_standardized_mse.tex']
    assert r'\label{tab:iws-reserved-main-scores}' in tables['main_scores.tex']
    for task in renderer.TASKS:
        means=value['results']['task_results'][task]['standardized_mse']['equal_trajectory']['horizon_means']['60']
        row=next(line for line in tables['main_scores.tex'].splitlines() if line.startswith(renderer.NAMES[task]+' &'))
        for mode in renderer.MODES:
            assert f'{means[mode]:.5f}' in row


def test_undefined_interval_retains_defined_signed_point():
    assert '-1.23' in renderer.gain_cell(-1.23, None)
    assert 'undefined' in renderer.gain_cell(-1.23, None)
    assert 'undefined' in renderer.gain_cell(None, None)


def test_positive_highlight_is_point_only_and_retains_crossing_ci():
    value=renderer.gain_cell(1.23,[-2.0,4.0])
    assert r'\textcolor{gainpositive}{\mathbf{+1.23}}' in value
    assert '-2.00' in value and '+4.00' in value
    assert 'gainpositive' not in renderer.gain_cell(-1.23,[-4.,2.])
    assert 'gainpositive' not in renderer.gain_cell(0.,[-1.,1.])
