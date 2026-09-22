"""Synthetic-only tests: censoring, pairing, complete grids and seed dependence."""
import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('physical_extract',Path(__file__).with_name('physical_extract.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def core():
    rows=[]
    for s in range(64):
        for o in range(3):
            for d in range(3):
                rows.append(dict(seed=s,observation_id=o,dynamics_id=d,trajectory_id=str(s),goal_index=7,
                    success=0,final_success=0,success_during_context=0,policy_eligible=1,
                    native_steps=50,steps_to_success_or_budget=50,block_translation_error_px=3.,
                    agent_position_error_px=30.,block_angle_error_rad=.1,initial_distance_after_context=40.))
    src={k:'hash' for k in ['checkpoint_sha256','evaluator_sha256','data_manifest_sha256']}
    d=dict(status='complete',environment='pusht',model_mode='factorized',training_seed=0,**src,
        planning=dict(status='complete',split='test',policy='world_model',records=rows,
                      planner=dict(native_budget=50,history_steps_charged=10,action_block=5)))
    return d,src

def test_complete_grid_and_physical_units():
    d,s=core();r=m.validate_core(d,'pusht','factorized',0,'test',s)
    assert len(r)==576 and r[0]['errors']['position_error_px']==pytest.approx((9+900)**.5)

@pytest.mark.parametrize('defect',['missing','duplicate','budget','criterion'])
def test_core_fail_closed(defect):
    d,s=core();r=d['planning']['records']
    if defect=='missing':r.pop()
    if defect=='duplicate':r[-1]=copy.deepcopy(r[0])
    if defect=='budget':r[0]['steps_to_success_or_budget']=12
    if defect=='criterion':r[0]['final_success']=1;r[0]['success']=1
    with pytest.raises(ValueError):m.validate_core(d,'pusht','factorized',0,'test',s)

def extension():
    rows=[]
    for s in range(8):
        metric=dict(success=False,valid_action=True,stable_deformation=True,distance_m=.01)
        rows.append(dict(seed=s,trajectory_id=str(s),observation_id=1,dynamics_id=1,
            native_budget=200,support_budget=10,native_calls=20,metrics_per_native_step=[copy.deepcopy(metric) for _ in range(20)],
            final_metrics=metric,final_distance_m=.01,initial_distance_m=.02,success=False,
            first_success_native_step=None,success_during_support=False,stop_reason='invalid_action'))
    return dict(status='completed',tasks=8,successes=0,records=rows)

def test_early_failure_never_masquerades_as_fast_success():
    d=extension();r=m.validate_extension(d,'surgery')[0]
    assert r['native_calls']==20 and r['first_success'] is None and r['budget_penalized_commands']==200
    assert r['failure_flags']['early_termination'] and not r['failure_flags']['budget_exhausted']

def test_extension_first_hit_must_match_ledger():
    d=extension();d['records'][0]['metrics_per_native_step'][7]['success']=True
    with pytest.raises(ValueError):m.validate_extension(d,'surgery')

def pair(seed_effects):
    aa=[];bb=[]
    for s,e in enumerate(seed_effects):
        def run(v):
            return dict(training_seed=s,mode='framewise',records=[dict(task_seed=i,observation_id=0,dynamics_id=0,
                support_success=False,support_identity={'start':2.},success=False,errors={'error':v}) for i in range(8)])
        aa.append(run(10-e));bb.append(run(10))
    return aa,bb

def test_signed_physical_gain_and_fixed_seed_uncertainty():
    a,b=pair([-4,0,4]);r=m.paired_effect(a,b,'error')
    assert r['mean']==0 and r['ci95']==[0,0] and r['training_seed_sd']==4
    c=m.paired_effect(a,b,'error',crossed=True)
    assert c['ci95'][0]<0<c['ci95'][1]

def test_paired_initial_condition_and_three_seed_gates():
    a,b=pair([1,1,1]);b[0]['records'][0]['support_identity']['start']=3
    with pytest.raises(ValueError):m.paired_effect(a,b,'error')
    with pytest.raises(ValueError):m.paired_effect(a[:2],b[:2],'error')

def test_source_hash_and_unchanged_guard(tmp_path):
    p=tmp_path/'a.json';p.write_text('{}');s=m.Sources(tmp_path)
    with pytest.raises(ValueError):s.read('a.json','wrong')
    s.read('a.json');p.write_text('{"changed":true}')
    with pytest.raises(ValueError):s.unchanged()
