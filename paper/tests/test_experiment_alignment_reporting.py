"""Fail-closed reporting gates, using synthetic receipts only in temporary paths."""
from pathlib import Path
import copy
import importlib.util
import json
import hashlib

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / 'scripts/refresh_experiment_alignment.py'
spec = importlib.util.spec_from_file_location('alignment_reporter', MODULE)
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(reporter,'ROOT',tmp_path)
    final_path=tmp_path/'reports/real_video_iws/development_finalization.json'
    monkeypatch.setattr(reporter,'FINAL',final_path)
    dependencies={}
    def save(name,data):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data));dependencies[name]=sha(path);return path
    config={'tasks':{task:{} for task in reporter.TASKS},'modes':list(reporter.MODES),'seeds':[0,1,2],
            'training':{'epochs':30,'horizon':60,'validation_stride':5,
                        'selector':'internal_dev_equal_trajectory_endpoint_H60_standardized_mse'},
            'evaluation':{'primary_method':'bounded_spatial_mix','primary_comparator':'anchored_additive',
                          'primary_metric':'standardized_feature_mse','bootstrap':{'draws':10000,'seed':173},
                          'official_validation_allowed_during_training':False}}
    save('configs/real_video_iws/training_v1.json',config)
    registration=save('registration.json',{'fixture':'synthetic-only, not scientific evidence'})
    per_run=[];caches=[];scales=1+np.arange(59)/100
    for task in reporter.TASKS:
        caches.append({'task':task,'development_population':{'a':1,'b':3}})
        for mode in reporter.MODES:
            for seed in range(3):
                name=f'{task}_{mode}_{seed}'
                checkpoint=save('models/'+name+'.json',{'synthetic_model_identity':name})
                factor={'autoregressive':1.2,'anchored_additive':1.,'bounded_spatial_mix':.9}[mode]
                episodes=[{'episode_id':episode,'windows':windows,
                           'standardized_mse_by_offset':(value*factor*scales).tolist(),
                           'persistence_standardized_mse_by_offset':(value*1.4*scales).tolist()}
                          for episode,windows,value in [('a',1,1.),('b',3,9.)]]
                evaluation={'schema':'shiftwm_iws_development_evaluation_v1','status':'passed',
                            'scope':'internal_development','horizons':[15,30,45,60],
                            'total_windows':4,'eligible_trajectories':2,'selected_checkpoint_sha256':sha(checkpoint),
                            'episodes':episodes}
                receipt=save('evaluations/'+name+'.json',evaluation)
                per_run.append({'task':task,'mode':mode,'seed':seed,'evaluation_path':str(receipt.relative_to(tmp_path)),
                                'evaluation_sha256':sha(receipt),'completed_epochs':30,'selected_checkpoint_sha256':sha(checkpoint)})
    final={'schema':reporter.SCHEMA,'status':'passed','scope':'internal_development',
           'expected_runs':27,'completed_runs':27,'registration_sha256':sha(registration),
           'source_dependencies':dependencies,'per_run':per_run}
    def write_final():
        final_path.parent.mkdir(parents=True,exist_ok=True);final_path.write_text(json.dumps(final))
    def change_evaluation(transform):
        run=final['per_run'][0];path=tmp_path/run['evaluation_path'];value=json.loads(path.read_text())
        transform(value);path.write_text(json.dumps(value));run['evaluation_sha256']=sha(path)
        dependencies[run['evaluation_path']]=sha(path);write_final()
    write_final()
    return final,caches,write_final,change_evaluation


def test_absence_and_incomplete_never_emit_scores(campaign):
    final,caches,write,_=campaign
    reporter.FINAL.unlink()
    assert reporter.completed_development({},caches)['numerical_results'] is None
    final['status']='running';write()
    assert reporter.completed_development({},caches)['numerical_results'] is None


def test_complete_episode_weighting_and_macro(campaign):
    _,caches,_,_=campaign
    result=reporter.completed_development({},caches)
    assert result['status']=='complete_validated_development'
    for task,summary in result['numerical_results'].items():
        # One and three windows receive equal TRAJECTORY weight: mean(1,9)=5,
        # rather than the wrong window-weighted value7.
        assert summary['means_all59_offsets']['anchored_additive'][58]==pytest.approx(5*1.58)
        assert summary['h60_comparisons']['anchored_additive']['relative_mse_reduction_percent']==pytest.approx(10)
    assert result['macro_primary_relative_gain_percent']==pytest.approx(10)
    assert result['bootstrap']['macro_gain_interval']==pytest.approx([10,10])
    assert result['bootstrap']['draws']==10000


@pytest.mark.parametrize('mutation',[lambda f:f.update(completed_runs=26),
    lambda f:f['per_run'].pop(),lambda f:f['per_run'].__setitem__(0,copy.deepcopy(f['per_run'][1])),
    lambda f:f.update(scope='official_validation'),lambda f:f['per_run'][0].update(completed_epochs=29),
    lambda f:f['per_run'][0].update(selected_checkpoint_sha256='0'*64),
    lambda f:f.update(registration_sha256='0'*64)])
def test_completion_scope_identity_coverage_rejections(campaign,mutation):
    final,caches,write,_=campaign;mutation(final);write()
    with pytest.raises(ValueError):reporter.completed_development({},caches)


@pytest.mark.parametrize('mutation',[lambda e:e['episodes'].pop(),
    lambda e:e['episodes'][0].update(episode_id='official-reserved'),
    lambda e:e['episodes'][0].update(windows=2),lambda e:e.update(total_windows=5),
    lambda e:e['episodes'][0].update(standardized_mse_by_offset=[0.]*58),
    lambda e:e['episodes'][0]['standardized_mse_by_offset'].__setitem__(0,float('nan')),
    lambda e:e['episodes'][0]['persistence_standardized_mse_by_offset'].__setitem__(0,123.),
    lambda e:e.update(scope='official_validation')])
def test_invalid_or_mismatched_evaluation_rejected(campaign,mutation):
    _,caches,_,change=campaign;change(mutation)
    with pytest.raises(ValueError):reporter.completed_development({},caches)


def test_source_tampering_is_not_a_new_result(campaign):
    final,caches,_,_=campaign
    path=reporter.ROOT/final['per_run'][0]['evaluation_path'];path.write_text('{}')
    with pytest.raises(ValueError,match='evidence has changed'):reporter.completed_development({},caches)


def test_official_report_alone_is_never_read(campaign):
    _,caches,_,_=campaign;reporter.FINAL.unlink()
    (reporter.FINAL.parent/'official_finalization.json').write_text('invalid data deliberately not parsed')
    assert reporter.completed_development({},caches)['status']=='pending'


def test_zero_baseline_ratio_stays_undefined():
    arrays={task:{mode:np.zeros((3,2)) for mode in (*reporter.MODES,'persistence')} for task in reporter.TASKS}
    result=reporter.crossed_intervals(arrays,draws=20)
    assert result['macro_gain_interval'] is None
    assert result['task_intervals']['pusht']['anchored_additive']['gain'] is None


def test_bootstrap_joint_task_seed_semantics():
    # One trajectory per task means all variability comes from the SAME seed
    # draw. Antisymmetric task effects cancel exactly in every macro draw.
    task_names=list(reporter.TASKS);arrays={}
    for task,delta in zip(task_names,[-.2,.2,0.]):
        base=np.ones((3,1));ours=base+delta*np.arange(3)[:,None]
        arrays[task]={'bounded_spatial_mix':ours,'anchored_additive':base,'autoregressive':base,'persistence':base}
    result=reporter.crossed_intervals(arrays,draws=200)
    assert result['macro_gain_interval']==pytest.approx([0,0],abs=1e-12)
