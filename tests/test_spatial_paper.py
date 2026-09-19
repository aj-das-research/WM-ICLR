"""Reporting-contract tests; synthetic fixtures are not experimental results."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('spatial_paper_tests',ROOT/'scripts/real_video_spatial_reporting/finalize_paper.py')
paper=importlib.util.module_from_spec(spec);spec.loader.exec_module(paper)


def records(offset=0):
    return [{'seed':s,'episodes':[{'episode_id':f'e{i}','session_id':f's{i//2}','windows':i+1,
            'native_mse':[float(1+i/10+s/100+offset) for _ in range(10)]}
            for i in range(6)]} for s in range(3)]


def test_independent_paired_interval_matches_frozen_implementation():
    first,second=records(),records()
    for row in first:
        for i,e in enumerate(row['episodes']):
            e['native_mse']=[v-.02*(row['seed']+1)*(i+1) for v in e['native_mse']]
    for mode,rows in [('transport',first),('autoregressive',second)]:
        for row in rows:
            row['mode']=mode
            row['windows']=[{'episode_id':e['episode_id'],'session_id':e['session_id'],'window_start':i}
                            for e in row['episodes'] for i in range(e['windows'])]
            for e in row['episodes']:
                for metric in paper.METRICS: e[metric]=e['native_mse'][:]
    validator=paper.module('scripts/real_video_spatial/validate_ledger.py','spatial_reporting_reference')
    expected=validator.paired_intervals(first+second,draws=300,bootstrap_seed=173)['metrics']['native_mse'][9]
    actual=paper.independent_comparison(first,second,'native_mse',10,draws=300)
    np.testing.assert_allclose(actual['paired_95_percent_interval'],expected['ci95'],atol=1e-14)
    assert actual['method_minus_comparator']==pytest.approx(-.14)
    assert actual['relative_error_reduction_percent']>0
    assert not actual['interval_includes_zero']


@pytest.mark.parametrize('mutation',["seed","session","windows","nonfinite"])
def test_paired_population_corruption_rejected(mutation):
    first,second=records(-.1),records()
    if mutation=='seed': first[1]['seed']=0
    elif mutation=='session': second[0]['episodes'][0]['session_id']='other'
    elif mutation=='windows': second[0]['episodes'][0]['windows']=44
    else: first[0]['episodes'][0]['native_mse'][9]=float('nan')
    with pytest.raises(ValueError): paper.independent_comparison(first,second,'native_mse',10,draws=4)


def test_negative_and_zero_denominator_never_become_positive_claims():
    result=paper.independent_comparison(records(.1),records(),'native_mse',10,draws=40)
    assert result['relative_error_reduction_percent']<0
    assert result['paired_95_percent_interval'][0]>0
    zero=records()
    for row in zero:
        for e in row['episodes']: e['native_mse']=[0.]*10
    result=paper.independent_comparison(records(),zero,'native_mse',10,draws=4)
    assert result['relative_error_reduction_percent'] is None
    assert 'positivegain' not in paper.gain(-1)
    assert 'positivegain' not in paper.gain(None)
    assert 'dagger' in paper.gain(.1,True)


def test_incomplete_study_cannot_generate_summary():
    with pytest.raises(ValueError,match='fifteen'): paper.summarize([], {})


@pytest.mark.parametrize('mutation',[None,'stale_parity','altered_export','different_selected'])
def test_finalizer_parity_binds_exact_selected_package(tmp_path,monkeypatch,mutation):
    monkeypatch.setattr(paper,'ROOT',tmp_path)
    name='transport_s0';exported=tmp_path/'artifacts/releases/spatial_v1/models'/name
    selected=tmp_path/'selected';exported.mkdir(parents=True);selected.mkdir()
    for root in (exported,selected):
        (root/'model.pt').write_bytes(b'original verified checkpoint bytes')
        (root/'config.json').write_text('{"coordinate_layout":"shared channel"}')
    manifest={'format_version':1,'package_kind':'shiftwm_real_video_spatial_v1',
              'files':{name:paper.sha(exported/name) for name in ('model.pt','config.json')}}
    (exported/'package_manifest.json').write_text(json.dumps(manifest))
    parity={'name':name,'status':'passed','max_abs_error':0,'relocated_isolated_process':True,
            'package_manifest_sha256':paper.sha(exported/'package_manifest.json')}
    if mutation=='stale_parity': parity['package_manifest_sha256']='0'*64
    elif mutation=='altered_export': (exported/'model.pt').write_bytes(b'changed after finalizer')
    elif mutation=='different_selected': (selected/'model.pt').write_bytes(b'replaced best checkpoint')
    if mutation:
        with pytest.raises(ValueError): paper.bind_finalizer_parity(name,selected,parity)
    else:
        identities=paper.bind_finalizer_parity(name,selected,parity)
        assert len(identities)==3
        assert identities[str((exported/'model.pt').relative_to(tmp_path))]==paper.sha(selected/'model.pt')


def fixture_tables():
    summary={m:{k:{'mean':[.123456]*10,'seed_sd':[.001234]*10} for k in paper.METRICS} for m in paper.MODES}
    effects=[{'comparator':m,'metric':k,'horizon':h,'relative_error_reduction_percent':.5 if h==10 else -.1,
              'method_minus_comparator':-.000617 if h==10 else .000123,'paired_95_percent_interval':[-.001234,.000567],
              'interval_includes_zero':True} for m in paper.MODES if m!='transport' for k in ('native_mse','original_2x2_mse') for h in (5,10)]
    runs=[{'mode':m,'seed':s,'selected_epoch':10+s,'parameter_counts':{'total':1026305,'trainable':1007776}}
          for m in paper.MODES for s in range(3)]
    return paper.tables({'paired_effects':effects},runs,summary)


def test_all_signed_contrasts_and_checkpoints_are_rendered():
    files=fixture_tables()
    assert len(files)==5
    assert sum(v.count('Context-off &') for k,v in files.items() if 'intervals' in k)==4
    assert '-0.100' in files['native_intervals.tex']
    assert 'Spatial transport (ours)' in files['native.tex']
    assert '10/11/12' in files['checkpoints.tex']


def test_official_width_engineering_fixture(tmp_path):
    proof=paper.proof(tmp_path,fixture_tables())
    assert proof['status']=='passed'
