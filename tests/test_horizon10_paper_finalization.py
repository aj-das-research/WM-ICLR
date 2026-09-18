"""Integrity checks and explicitly artificial, unpublished table geometry proof."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('horizon10_paper_tests',ROOT/'scripts/real_video_development/finalize_horizon10_paper.py')
paper=importlib.util.module_from_spec(spec);spec.loader.exec_module(paper)


def test_rejects_partial_campaign_before_creating_manuscript_include(monkeypatch,tmp_path):
    registry=tmp_path/'registration.json';results=tmp_path/'results.json'
    registry.write_text('{}');results.write_text('{"status":"in_progress","completed_runs":11}')
    monkeypatch.setattr(paper,'REGISTRY',registry);monkeypatch.setattr(paper,'RESULTS',results)
    with pytest.raises(ValueError,match='12 full runs and 60'):
        paper.collect()


def test_independent_bootstrap_matches_registered_reference():
    sys.path.insert(0,str(ROOT/'scripts/real_video'))
    import evaluate
    values=np.array([[.01,-.02,.005,.03],[-.03,.01,.002,.02],[.02,-.015,.001,-.01]])
    sessions=['one','one','two','three']
    actual=paper.bootstrap(values,sessions)
    expected=evaluate.crossed_session_bootstrap(values,sessions,draws=10000,seed=5198010)
    assert actual['ci95']==expected['ci95']
    assert actual['mean_difference']==expected['mean_difference']
    assert actual['session_count']==3


def test_positive_color_does_not_silence_inconclusive_or_negative_results():
    assert paper.gain(1.234,[-.02,.01])==r'\positivegain{+1.23}$^{\dagger}$'
    assert paper.gain(-1.234,[.01,.02])=='-1.23'
    assert paper.gain(1.234,[-.02,-.01])==r'\positivegain{+1.23}'


def evaluation_fixture():
    metrics={'mean_standardized_mse':.1,'mean_raw_mse':.2,'mean_cosine_error':.3,
             'h10_standardized_mse':.4,'h10_raw_mse':.5,'h10_cosine_error':.6}
    methods={m:deepcopy(metrics) for m in ('model','persistence','constant_velocity','reversed_future_actions')}
    row={'name':'factorized_s0','mode':'factorized','seed':0}
    doc={'scope':'original validation development only','mode':'factorized','seed':0,'horizon':10,
         'kind':'h10_trained_standard_h10','registration_sha256':'registry','checkpoint_sha256':'checkpoint',
         'result':{'episodes':[{'episode_id':'a','session_id':'s','windows':2,'window_starts':[0,5],'errors':methods}],
                   'summary':deepcopy(methods),'episode_count':1,'session_count':1,'window_count':2}}
    pop={'a':{'session_id':'s','starts':[0,5]}}
    return doc,row,pop


@pytest.mark.parametrize('damage',['mean','window','scope','horizon','checkpoint','negative','missing_metric'])
def test_evaluation_audit_rejects_altered_science(damage):
    doc,row,pop=evaluation_fixture()
    paper.check_evaluation(doc,row,'validation_h10','checkpoint',pop,'registry')
    if damage=='mean':doc['result']['summary']['model']['h10_standardized_mse']+=.1
    elif damage=='window':doc['result']['episodes'][0]['window_starts']=[0,10]
    elif damage=='scope':doc['scope']='test'
    elif damage=='horizon':doc['horizon']=5
    elif damage=='checkpoint':doc['checkpoint_sha256']='other'
    elif damage=='negative':doc['result']['episodes'][0]['errors']['model']['h10_standardized_mse']=-.1
    else:doc['result']['episodes'][0]['errors']['model'].pop('mean_cosine_error')
    with pytest.raises(ValueError):paper.check_evaluation(doc,row,'validation_h10','checkpoint',pop,'registry')


def table_fixture():
    # Artificial format extremes for engineering layout only. Never result evidence.
    native=[];comparisons=[];records=[]
    for h in (5,10):
        for metric in (f'h{h}_standardized_mse','mean_standardized_mse'):
            methods={m:{'mean':.12345,'seed_sd':.01234,'gain_percent':(-1 if i%2 else 1)*12.34} for i,m in enumerate(paper.MODES)}
            native.append({'horizon':h,'metric':metric,'methods':methods})
            comparisons.append({'type':'matched_h10_training_modes','horizon':h,'metric':metric,
                'first_mean':.12345,'second_mean':.23456,'gain_percent':12.34,'paired':{'ci95':[-.01234,.00567]}})
    for mode in paper.MODES:
        for seed in (0,1,2):
            records.append({'mode':mode,'seed':seed,'training':{'best_epoch':30,'parameter_counts':{'total':1234567,'trainable':123456}},'elapsed_seconds':12345.+seed})
        for h in (5,10):
            for metric in (f'h{h}_standardized_mse','mean_standardized_mse'):
                comparisons.append({'type':'horizon_training_control','mode':mode,'horizon':h,'metric':metric,
                    'first_mean':.12345,'second_mean':.23456,'gain_percent':12.34,'paired':{'ci95':[-.01234,.00567]}})
    return records,native,comparisons


def test_all_methods_all20_contrasts_have_native_width_tex_proof(tmp_path):
    files=paper.tables(*table_fixture())
    assert len(files)==6
    for name,contents in files.items():
        assert 'ShiftWM (ours)' in contents or name=='ours_intervals.tex'
        # Explicitly label the engineering-only rendered artifact.
        files[name]=r'\noindent\textbf{ARTIFICIAL ENGINEERING LAYOUT FIXTURE; NOT RESULTS.}\par'+contents
    result=paper.proof(tmp_path,files)
    assert result['status']=='passed'
    assert 'pending' in result['visual_review']
    assert (tmp_path/'proof.pdf').is_file()
