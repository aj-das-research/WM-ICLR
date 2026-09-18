"""Adversarial fixtures for the independent spatial ledger finalization gate."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("spatial_ledger_tested",ROOT/"scripts/real_video_spatial/validate_ledger.py")
ledger=importlib.util.module_from_spec(spec);spec.loader.exec_module(ledger)


@pytest.fixture
def fixture(tmp_path):
    def write(name,value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value) if not isinstance(value,str) else value)
        return path
    def episode(eid,split,count,session):
        return {"episode_id":eid,"session_id":session,"split":split,"cameras":{"exterior_image_1_left":{"frames":count}}}
    episodes=[episode('train','train',13,'train-session'),episode('a','val',18,'session-a'),episode('b','val',13,'session-b'),episode('short','val',5,'session-c')]
    original=write('original/manifest.json',{'episodes':episodes+[episode('test','test',13,'test-session')]})
    cache=write('cache/manifest.json',{'status':'complete','feature_dim':6144,'action_dim':35,'episodes':episodes,
        'identity':{'original_2x2_cache_manifest_sha256':ledger.sha(original)}})
    normalization=write('original/training_statistics.json',{'fit_split':'train'})
    checkpoint=write('run/best/model.pt','fixture bytes; parent validator checks actual checkpoint')
    source=write('scripts/real_video_spatial/evaluate.py','fixture evaluator')
    counts={'total':100,'trainable':80}
    config={'mode':'transport','seed':0,'cache_root':'cache','original_cache':'original','output_dir':'run','model_config':{'history_length':3}}
    state={'epoch':4,'config':{'model_config':{'history_length':3,'mode':'transport'},'metadata':{'parameter_counts':counts}}}
    windows=[]
    for eid,session,start,value in [('a','session-a',0,1.),('a','session-a',5,3.),('b','session-b',0,8.)]:
        windows.append({'episode_id':eid,'session_id':session,'window_start':start,**{m:[value+i/10]*10 for i,m in enumerate(ledger.METRICS)}})
    aggregate=[]
    for eid,session,count,value in [('a','session-a',2,2.),('b','session-b',1,8.)]:
        aggregate.append({'episode_id':eid,'session_id':session,'windows':count,**{m:[value+i/10]*10 for i,m in enumerate(ledger.METRICS)}})
    result={'status':'complete','scope':'original_validation_development_only','aggregation':'mean windows within episode; equal episodes',
            'mode':'transport','seed':0,'selected_epoch':4,'parameter_counts':counts,'original_target_coordinates':ledger.TARGET_COORDINATES,
            'checkpoint_sha256':ledger.sha(checkpoint),'cache_manifest_sha256':ledger.sha(cache),
            'original_cache_manifest_sha256':ledger.sha(original),'original_normalization_sha256':ledger.sha(normalization),
            'source_sha256':ledger.sha(source),'windows':windows,'episodes':aggregate,
            'summary':{m:[5.+i/10]*10 for i,m in enumerate(ledger.METRICS)}}
    return result,config,tmp_path,state


def test_complete_unequal_window_population_reconstructs_equal_episode_mean(fixture):
    result=ledger.validate_ledger(*fixture)
    assert result['native_mse']==[5.]*10


@pytest.mark.parametrize('damage',['missing_window','duplicate_window','wrong_start','wrong_session','test_window',
    'missing_episode','duplicate_episode','episode_count','episode_mean','global_mean','nan','negative','nine_horizons',
    'boolean_error','checkpoint_hash','cache_hash','source_hash','normalization_hash','original_manifest_hash',
    'wrong_epoch','wrong_counts','wrong_scope','pooled_targets','wrong_mode','wrong_seed','wrong_architecture'])
def test_rejects_damaged_or_incomplete_evidence(fixture,damage):
    result,config,root,state=deepcopy(fixture)
    if damage=='missing_window':result['windows'].pop()
    elif damage=='duplicate_window':result['windows'].append(deepcopy(result['windows'][0]))
    elif damage=='wrong_start':result['windows'][0]['window_start']=1
    elif damage=='wrong_session':result['windows'][0]['session_id']='another'
    elif damage=='test_window':result['windows'][0]['episode_id']='test'
    elif damage=='missing_episode':result['episodes'].pop()
    elif damage=='duplicate_episode':result['episodes'].append(deepcopy(result['episodes'][0]))
    elif damage=='episode_count':result['episodes'][0]['windows']=99
    elif damage=='episode_mean':result['episodes'][0]['native_mse'][4]+=.01
    elif damage=='global_mean':result['summary']['native_mse'][4]+=.01
    elif damage=='nan':result['windows'][0]['native_mse'][0]=float('nan')
    elif damage=='negative':result['windows'][0]['native_mse'][0]=-.01
    elif damage=='nine_horizons':result['windows'][0]['native_mse'].pop()
    elif damage=='boolean_error':result['windows'][0]['native_mse'][0]=True
    elif damage.endswith('_hash'):
        field={'checkpoint_hash':'checkpoint_sha256','cache_hash':'cache_manifest_sha256','source_hash':'source_sha256',
               'normalization_hash':'original_normalization_sha256','original_manifest_hash':'original_cache_manifest_sha256'}[damage]
        result[field]='wrong'
    elif damage=='wrong_epoch':result['selected_epoch']=5
    elif damage=='wrong_counts':result['parameter_counts']={'total':101,'trainable':80}
    elif damage=='wrong_scope':result['scope']='test'
    elif damage=='pooled_targets':result['original_target_coordinates']='recomputed pooled target'
    elif damage=='wrong_mode':result['mode']='autoregressive'
    elif damage=='wrong_seed':result['seed']=2
    elif damage=='wrong_architecture':state['config']['model_config']['history_length']=8
    with pytest.raises(ValueError):ledger.validate_ledger(result,config,root,state)


def test_rejects_rebound_manifest_with_changed_original_population(fixture):
    result,config,root,state=fixture
    path=root/'original/manifest.json';original=json.loads(path.read_text());original['episodes'][1]['session_id']='another'
    path.write_text(json.dumps(original));result['original_cache_manifest_sha256']=ledger.sha(path)
    cache=root/'cache/manifest.json';manifest=json.loads(cache.read_text());manifest['identity']['original_2x2_cache_manifest_sha256']=ledger.sha(path)
    cache.write_text(json.dumps(manifest));result['cache_manifest_sha256']=ledger.sha(cache)
    with pytest.raises(ValueError,match='Original target population'):ledger.validate_ledger(result,config,root,state)


def test_paired_intervals_require_complete_three_seed_matched_populations(fixture):
    original=fixture[0];rows=[]
    for mode in ('transport','autoregressive'):
        for seed in (0,1,2):
            row=deepcopy(original);row.update(mode=mode,seed=seed)
            if mode=='autoregressive':
                for episode in row['episodes']:
                    for metric in ledger.METRICS:episode[metric]=[x+1 for x in episode[metric]]
            rows.append(row)
    result=ledger.paired_intervals(rows,draws=100)
    np.testing.assert_allclose(result['metrics']['native_mse'][4]['ci95'],[-1,-1])
    with pytest.raises(ValueError,match='three unique'):ledger.paired_intervals(rows[:-1],draws=10)
    rows[-1]['windows'][0]['window_start']=10
    with pytest.raises(ValueError,match='populations differ'):ledger.paired_intervals(rows,draws=10)
