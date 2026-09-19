"""Presentation-only fixtures; never written to active experiment/paper outputs."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1]/'scripts/render_current_real_scorecards.py'
spec = importlib.util.spec_from_file_location('real_scorecard_test',SOURCE)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


@pytest.fixture
def synthetic_iws(monkeypatch,tmp_path):
    """All27 synthetic receipts, with distinct H60 values and no inference."""
    monkeypatch.setattr(m,'ROOT',tmp_path)
    objects = {}; rows = []; dependencies = {}
    for task in m.TASKS:
        n = 120 if task == 'pusht' else 121
        for mode_i,mode in enumerate(m.BASE):
            for seed in range(3):
                path = f'receipts/{task}_{mode}_{seed}.json'
                row = dict(task=task,mode=mode,seed=seed,completed_epochs=30,selected_epoch=30,
                           selected_checkpoint_sha256=f'ckpt-{task}-{mode}-{seed}',
                           evaluation_path=path,evaluation_sha256=f'receipt-{task}-{mode}-{seed}')
                dependencies[path] = row['evaluation_sha256']
                dependencies[f'weights/{task}_{mode}_{seed}'] = row['selected_checkpoint_sha256']
                rows.append(row)
                episodes = []
                for i in range(n):
                    ep = {'episode_id':str(i),'windows':28}
                    for j,metric in enumerate(m.METRICS):
                        values = np.linspace(.01,.2+mode_i*.02+seed*.001+j*.1,59).tolist()
                        ep[metric+'_by_offset'] = values
                        ep['persistence_'+metric+'_by_offset'] = [1.+j*.1]*59
                    episodes.append(ep)
                objects[path] = {**row,'status':'passed','scope':'internal_development',
                    'offsets':list(range(1,60)),'metrics':list(m.METRICS),'official_validation_payloads_read':0,
                    'eligible_trajectories':n,'total_windows':n*28,
                    'population_audit':{'split':'internal_development'},'episodes':episodes}
    objects[m.IWS_FINAL] = {'status':'passed','completed_runs':27,'expected_runs':27,
        'scope':'internal_development','official_validation_payloads_read':0,
        'per_run':rows,'source_dependencies':dependencies}
    def read(name,sources,expected=None):
        sources[name] = expected or 'synthetic-hash'
        return objects[name]
    monkeypatch.setattr(m,'read_bound',read)
    return objects


def test_all_four_metrics_h60_and_adverse_best(synthetic_iws):
    data = m.iws_scores({}); text = m.iws_table(data)
    assert not data['unbounded_included']
    assert 'No $\\tanh$' not in text and 'positivegain' not in text
    for task in m.TASKS:
        means = data['tasks'][task]['methods']['autoregressive']['mean_curves']
        assert means['standardized_mse'][58] == pytest.approx(.201)
        assert means['standardized_mse'][13] != pytest.approx(.201)
        for j,metric in enumerate(m.METRICS):
            assert means[metric][58] == pytest.approx(.201+j*.1)
    assert text.count(r'\textbf{0.201000}') == 3
    assert '0.241000' in text  # worse proposed method retained
    assert text.count('ShiftWM (ours)') == 3


@pytest.mark.parametrize('mutation',['partial','duplicate','reserved','bad_seed','persistence'])
def test_invalid_completed_receipts_rejected(synthetic_iws,mutation):
    final = synthetic_iws[m.IWS_FINAL]
    receipt = synthetic_iws[final['per_run'][0]['evaluation_path']]
    if mutation == 'partial': final['completed_runs'] = 26
    elif mutation == 'duplicate': final['per_run'][-1] = final['per_run'][0]
    elif mutation == 'reserved': receipt['scope'] = 'official_validation'
    elif mutation == 'bad_seed': receipt['seed'] = 2
    else: receipt['episodes'][0]['persistence_standardized_mae_by_offset'][58] += .1
    with pytest.raises(ValueError):m.iws_scores({})


def test_present_partial_unbounded_fails_closed(synthetic_iws):
    path = m.local(m.UNBOUNDED_FINAL);path.parent.mkdir(parents=True);path.write_text('{}')
    synthetic_iws[m.UNBOUNDED_FINAL] = {'status':'passed','completed_new_runs':3}
    with pytest.raises(ValueError,match='nine-plus-27'):m.iws_scores({})


def test_unbounded_complete_gate_keeps_all_comparators(synthetic_iws):
    base = synthetic_iws[m.IWS_FINAL]; rows = copy.deepcopy(base['per_run']); bindings = dict(base['source_dependencies'])
    for task in m.TASKS:
        for seed in range(3):
            old = next(r for r in base['per_run'] if (r['task'],r['mode'],r['seed'])==(task,'bounded_spatial_mix',seed))
            row = {**old,'mode':'unbounded_spatial_mix','evaluation_path':f'receipts/{task}_extra_{seed}.json',
                   'evaluation_sha256':f'extra-{task}-{seed}'}
            receipt = copy.deepcopy(synthetic_iws[old['evaluation_path']]);receipt.update(row)
            for ep in receipt['episodes']:
                for metric in m.METRICS: ep[metric+'_by_offset'] = [x*1.2 for x in ep[metric+'_by_offset']]
            synthetic_iws[row['evaluation_path']] = receipt;rows.append(row)
            bindings[row['evaluation_path']] = row['evaluation_sha256']
    extra = {'schema':'iws_unbounded_complete_comparison_v1','status':'passed','completed_new_runs':9,
        'completed_v1_comparator_runs':27,'official_validation_payloads_read':0,
        'scope':'exploratory_internal_development_after_v1','registration_sha256':m.UNBOUNDED_REG_SHA,
        'baseline_finalization_sha256':m.IWS_FINAL_SHA,'per_run':rows,'source_dependencies':bindings,
        'results':{'task_results':{}}}
    for task in m.TASKS:
        metrics = {}
        for metric in m.METRICS:
            curves = {}
            for mode in (*m.BASE,'unbounded_spatial_mix'):
                receipts = [synthetic_iws[r['evaluation_path']] for r in rows if r['task']==task and r['mode']==mode]
                curves[mode] = np.mean([[e[metric+'_by_offset'] for e in r['episodes']] for r in receipts],axis=(0,1)).tolist()
            curves['persistence'] = receipts[0]['episodes'][0]['persistence_'+metric+'_by_offset']
            metrics[metric] = {'means_all59_offsets':curves}
        extra['results']['task_results'][task] = metrics
    path = m.local(m.UNBOUNDED_FINAL);path.parent.mkdir(parents=True);path.write_text('{}')
    synthetic_iws[m.UNBOUNDED_FINAL] = extra
    data = m.iws_scores({});text = m.iws_table(data)
    assert data['unbounded_included'] and data['completion']['unbounded_runs']==9
    assert text.count('No $\\tanh$ (ours, ablation)')==3
    assert '0.289200' in text and r'\textbf{0.289200}' not in text


def test_pending_writes_nothing(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    assert m.render(if_ready=True)=={'status':'pending','outputs_written':False}
    assert list(tmp_path.iterdir())==[]


def test_changed_bound_json_rejected(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'ROOT',tmp_path); (tmp_path/'input.json').write_text('{}')
    with pytest.raises(ValueError,match='Changed'):m.read_bound('input.json',{},'wrong')
    with pytest.raises(ValueError,match='escapes'):m.local('../escape')


def test_portable_render_is_exact_idempotent_and_checks_pack(monkeypatch,tmp_path):
    """Current public artifact is real source evidence, copied only to tmp outputs."""
    original_root = m.ROOT
    pack = tmp_path/m.PACK;pack.mkdir(parents=True)
    for name in ('data.json','manifest.json'):
        (pack/name).write_bytes((original_root/m.PACK/name).read_bytes())
    monkeypatch.setattr(m,'ROOT',tmp_path)
    first = m.render(from_pack=True)
    before = {p.name:p.read_bytes() for p in (tmp_path/m.OUT).iterdir()}
    second = m.render(from_pack=True)
    assert first['outputs_written'] and not second['outputs_written']
    assert before=={p.name:p.read_bytes() for p in (tmp_path/m.OUT).iterdir()}
    (pack/'data.json').write_text('{}')
    with pytest.raises(ValueError,match='hash/schema'):m.render(from_pack=True)


def test_portable_partial_or_missing_method_never_gets_a_minimum():
    data = json.loads((m.ROOT/m.PACK/'data.json').read_text())
    broken = copy.deepcopy(data)
    del broken['iws']['tasks']['pusht']['methods']['autoregressive']
    with pytest.raises(ValueError,match='method'):m.validate_payload(broken)
    broken = copy.deepcopy(data);broken['iws']['unbounded_included'] = True
    with pytest.raises(ValueError,match='full-nine'):m.validate_payload(broken)
