"""Synthetic gating, tensor algebra and averaging; no actual data or inference."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('closest_replay_test',HERE/'replay.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
c=sys.modules['core']


def group():
 rows=[]
 for eid,gain in [('b',10.),('a',10.),('c',0.),('d',-20.)]:
  rows.append({'episode_id':eid,'window_start':2,'gain_percent':gain,
               'episode_endpoint_means':{'autoregressive':1.,'focal':1.-gain/100}})
 ranked=sorted(rows,key=lambda x:(-x['gain_percent'],x['episode_id']))
 return {'all_ranked_episodes':rows,'selected':[ranked[0],ranked[1],ranked[-1]]}


def test_deterministic_rank_ties_lower_median_and_negative():
 result=c.selected_cases(group(),'focal')
 assert [x['episode_id']for x in result]==['a','b','d']
 assert result[-1]['gain_percent']==-20


@pytest.mark.parametrize('damage',['duplicate','moved_window','wrong_gain','zero_denominator','nan'])
def test_invalid_selection_rejected(damage):
 q=group()
 if damage=='duplicate':q['all_ranked_episodes'][1]=deepcopy(q['all_ranked_episodes'][0])
 elif damage=='moved_window':q['selected']=deepcopy(q['selected']);q['selected'][0]['window_start']=3
 elif damage=='wrong_gain':q['all_ranked_episodes'][0]['gain_percent']=20
 elif damage=='zero_denominator':q['all_ranked_episodes'][0]['episode_endpoint_means']['autoregressive']=0
 elif damage=='nan':q['all_ranked_episodes'][0]['episode_endpoint_means']['focal']=float('nan')
 with pytest.raises(ValueError):c.selected_cases(q,'focal')


def tensors():
 rng=np.random.default_rng(32);T=rng.uniform(size=(3,16,16)).astype('float32');T/=T.sum(-1,keepdims=True)
 g=rng.uniform(size=(3,16,1)).astype('float32');delta=rng.uniform(-1,1,size=(3,16,384)).astype('float32');anchor=rng.normal(size=(16,384)).astype('float32')
 prediction=(1-g)*anchor+g*(T@anchor)+delta
 return T,g,delta,prediction,anchor


def test_actual_decomposition_effective_weights_and_bound():
 T,g,d,p,a=tensors();M,checks=c.measured_details(T,g,d,p,a,True)
 np.testing.assert_allclose(M.sum(-1),1,atol=2e-7)
 assert checks['max_abs_correction']<=1
 assert np.all(np.diagonal(M,axis1=-2,axis2=-1)>=np.diagonal(T,axis1=-2,axis2=-1))
 d=d*2;p=(1-g)*a+g*(T@a)+d
 c.measured_details(T,g,d,p,a,False)
 with pytest.raises(ValueError,match='unit bound'):c.measured_details(T,g,d,p,a,True)


@pytest.mark.parametrize('damage',['negative_weight','nonstochastic','gate','prediction','nan','dtype'])
def test_corrupt_mechanism_rejected(damage):
 T,g,d,p,a=tensors()
 if damage=='negative_weight':T[0,0,0]=-1
 elif damage=='nonstochastic':T[0,0]*=.5
 elif damage=='gate':g[0,0,0]=1.1
 elif damage=='prediction':p[0,0,0]+=.1
 elif damage=='nan':d[0,0,0]=np.nan
 elif damage=='dtype':T=T.astype('float64')
 with pytest.raises((ValueError,AssertionError)):c.measured_details(T,g,d,p,a,True)


def test_patch_layout_matches_frozen_metric_and_channel_order():
 import torch
 evaluator=r.module('scripts/real_video_iws/evaluate.py','closest_synthetic_frozen_metric')
 rng=np.random.default_rng(1);p=rng.normal(size=(3,6144)).astype('float32');target=rng.normal(size=p.shape).astype('float32');std=np.repeat(rng.uniform(.3,2,size=384).astype('float32'),16)
 patches=c.patch_error(p,target,std)
 expected=evaluator.feature_errors(torch.from_numpy(p)[None],torch.from_numpy(target)[None],torch.from_numpy(std))['standardized_mse'][0].numpy()
 np.testing.assert_allclose(patches.mean((-1,-2)),expected,rtol=2e-6,atol=2e-7)
 single=np.zeros((1,6144),dtype='float32');single[0,5]=2
 actual=c.patch_error(single,np.zeros_like(single),np.ones(6144,dtype='float32'))
 assert actual[0,1,1]==pytest.approx(4/384)and np.count_nonzero(actual)==1


def test_mean_error_is_not_error_of_mean_prediction_and_mean_M_is_direct():
 rows=[]
 for seed,value in enumerate([-1,0,1]):
  p=np.full((1,6144),value,dtype='float32');patch=c.patch_error(p,np.zeros_like(p),np.ones(6144,dtype='float32'))
  row=c.numeric_record(patch,[1]);row['seed']=seed;rows.append(row)
 assert c.average_records(rows)['mse_curve'][0]==pytest.approx(2/3)
 # Correlated gates and mixing matrices make recomposition from means wrong.
 T=np.stack([np.eye(16),np.roll(np.eye(16),1,axis=-1),np.eye(16)]).astype('float32')
 gates=np.array([0.,1.,.5],dtype='float32')[:,None,None];M=(1-gates)*np.eye(16)+gates*T
 assert not np.allclose(M.mean(0),(1-gates.mean(0))*np.eye(16)+gates.mean(0)*T.mean(0))


def test_missing_seed_rejected():
 with pytest.raises(ValueError,match='three seeds'):c.average_records([{'seed':0},{'seed':1}])


def test_missing_source_review_precedes_source_or_payload_hashing(tmp_path,monkeypatch):
 monkeypatch.setattr(r,'sha',lambda *a:pytest.fail('Unreviewed gate read bytes'))
 with pytest.raises(ValueError,match='source review'):r.gate(tmp_path)
 assert list(tmp_path.iterdir())==[]


def test_stale_source_review_rejected_before_payload_access(tmp_path):
 (tmp_path/r.REVIEW).parent.mkdir(parents=True);(tmp_path/r.REG).parent.mkdir(parents=True)
 (tmp_path/r.REG).write_text('{}');(tmp_path/r.REVIEW).write_text(json.dumps({'status':'passed','registration_sha256':'stale'}))
 with pytest.raises(ValueError,match='Mismatched'):r.gate(tmp_path)


def test_tampered_bound_source_rejected(tmp_path):
 source=tmp_path/'input.py';source.write_text('immutable')
 reg={'policy':r.POLICY,'dependencies':{'input.py':r.sha(source)},'selected_payloads':{}}
 (tmp_path/r.REG).parent.mkdir(parents=True);(tmp_path/r.REG).write_text(json.dumps(reg))
 (tmp_path/r.REVIEW).parent.mkdir(parents=True);(tmp_path/r.REVIEW).write_text(json.dumps({'status':'passed','registration_sha256':r.sha(tmp_path/r.REG)}))
 assert r.gate(tmp_path)==reg
 source.write_text('changed')
 with pytest.raises(ValueError,match='Changed registered'):r.gate(tmp_path)


def test_public_export_requires_independent_matching_complete_result(tmp_path):
 (tmp_path/r.RESULT_REVIEW).parent.mkdir(parents=True);(tmp_path/r.RESULT_REVIEW).write_text(json.dumps({'status':'pending'}))
 (tmp_path/r.OUTPUT).mkdir(parents=True);(tmp_path/r.OUTPUT/'manifest.json').write_text('{}')
 with pytest.raises(ValueError,match='completed replay review'):r.prepare_public(tmp_path)
 assert not (tmp_path/r.PUBLIC).exists()


def test_unsafe_path_rejected(tmp_path):
 for p in ('../outside','/tmp/outside'):
  with pytest.raises(ValueError,match='Unsafe'):r.local(p,tmp_path)
