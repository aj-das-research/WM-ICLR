"""Finalizer integrity tests; all numerical examples are temporary test fixtures."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('iws_finalizer_test',ROOT/'scripts/real_video_iws/finalize.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


@pytest.fixture
def study(tmp_path,monkeypatch):
    report=tmp_path/'reports/real_video_iws';final=report/'development_finalization.json'
    monkeypatch.setattr(f,'ROOT',tmp_path);monkeypatch.setattr(f,'REPORT',report);monkeypatch.setattr(f,'FINAL',final)
    registration=tmp_path/'configs/real_video_iws/training_registration_v1.json'
    monkeypatch.setattr(f.campaign,'REGISTRATION',registration)
    def save(path,value):
        path=tmp_path/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
    config_path=save('configs/real_video_iws/training_v1.json',{'tasks':{t:{'action_dim':a} for t,a in f.campaign.TASKS.items()},
        'training':{'epochs':30},'model':{'test_fixture':True}})
    config=f.campaign.read(config_path);config_sha=f.campaign.sha(config_path)
    for name in ('scripts/real_video_iws/finalize.py','scripts/real_video_iws/evaluate.py','tests/test_iws_finalization.py',
                 'tests/test_iws_development_evaluation.py','paper/scripts/refresh_experiment_alignment.py',
                 'paper/tests/test_experiment_alignment_reporting.py'):
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    registry={'dependencies':{'configs/real_video_iws/training_v1.json':config_sha},'runs':[]}
    caches={};states={};summaries={};validated=[]
    for task,width in f.campaign.TASKS.items():
        index={str(i):{'episode_id':str(i),'split':'internal_development','task':task,'command_width':width,
                      'frames':n,'command_rows':n,'feature_dim':6144} for i,n in enumerate((61,66,60))}
        def authorize(eid,split,index=index,width=width):
            assert split=='internal_development'
            return {'shapes':{'target_qpos':[index[eid]['frames'],width]}},split
        cache=SimpleNamespace(index=index,inventory=SimpleNamespace(selected=lambda split:['0','1','2'],authorize=authorize))
        caches[task]=cache;audit=f.metadata_audit(cache)
        for mode in f.campaign.MODES:
            for seed in (0,1,2):
                name=f'{task}_{mode}_s{seed}';output=f'runs/{name}'
                row={'name':name,'task':task,'mode':mode,'seed':seed,'output':output};registry['runs'].append(row)
                factor={'autoregressive':1.2,'anchored_additive':1.,'bounded_spatial_mix':.9}[mode]
                arrays={'episode_index':np.array([0,1,1],dtype=np.int64),'window_start':np.array([0,0,5],dtype=np.int64)}
                for metric in f.evaluation.METRICS:
                    arrays[metric]=np.repeat(np.array([[9.],[1.],[3.]])*factor,59,axis=1)
                    arrays['persistence_'+metric]=np.repeat(np.array([[10.],[3.],[5.]]),59,axis=1)
                episodes=f.evaluation.aggregate_windows(arrays,audit)
                endpoint=float(np.mean([r['standardized_mse_by_offset'][-1] for r in episodes]))
                summary={'best_epoch':7,'best_validation_mse':endpoint}
                summaries[str(tmp_path/output)]=summary
                recipe={'task':task,'mode':mode,'seed':seed,'training':config['training'],'model':config['model'],
                        'task_config':config['tasks'][task],'study_config_sha256':config_sha}
                state={'config':{'metadata':{'identity':{'scientific_config':recipe,'populations':{'val':audit}}}}}
                for package in ('best','last'):
                    folder=tmp_path/output/package
                    for member in ('model.pt','config.json','training_state.pt'):
                        save(str(folder.relative_to(tmp_path)/member),{'fixture':name,'package':package,'member':member})
                    save(str(folder.relative_to(tmp_path)/'package_manifest.json'),{'files':{m:f.campaign.sha(folder/m)
                        for m in ('model.pt','config.json','training_state.pt')}})
                    states[str(folder)]=state
                for member in ('training_config.json','training_summary.json','metrics.jsonl'):
                    save(output+'/'+member,summary)
                path=report/'evaluations'/(name+'.json');ledger=path.with_suffix('.npz')
                f.evaluation.atomic_npz(ledger,arrays)
                value={'schema':'shiftwm_iws_development_evaluation_v1','status':'passed','scope':'internal_development',
                       'task':task,'mode':mode,'seed':seed,'completed_epochs':30,'selected_epoch':7,
                       'selected_checkpoint_sha256':f.campaign.sha(tmp_path/output/'best/model.pt'),'config_sha256':config_sha,
                       'episodes':episodes,'horizons':[15,30,45,60],'offsets':list(range(1,60)),
                       'metrics':list(f.evaluation.METRICS),'primary_aggregation':'equal_trajectory',
                       'population_audit':audit,'total_windows':3,'eligible_trajectories':2,'official_validation_payloads_read':0,
                       'h60_standardized_mse':endpoint,'prefix_maximum_absolute_difference':0.,
                       'window_ledger_path':str(ledger.relative_to(tmp_path)),'window_ledger_sha256':f.campaign.sha(ledger),
                       'evaluator_sha256':f.campaign.sha(f.evaluation.__file__)}
                save(str(path.relative_to(tmp_path)),value)
    save(str(registration.relative_to(tmp_path)),registry)
    for row in registry['runs']:
        path=report/'evaluations'/(row['name']+'.json');value=f.campaign.read(path)
        value['registration_sha256']=f.campaign.sha(registration);save(str(path.relative_to(tmp_path)),value)
    def validate(directory):
        validated.append(str(directory));return summaries[str(directory)]
    trainer=SimpleNamespace(validate_completed=validate,read_package=lambda path,*a:(path,states[str(path)]),
                            open_cache=lambda cfg,task:caches[task])
    monkeypatch.setattr(f.campaign,'trainer',lambda:trainer)
    monkeypatch.setattr(f.campaign,'check_registration',lambda path:copy.deepcopy(registry))
    refresh=[];monkeypatch.setattr(f,'refresh_paper',lambda:refresh.append(True))
    path=report/'evaluations'/(registry['runs'][0]['name']+'.json')
    def mutate(transform,ledger_transform=None):
        value=f.campaign.read(path)
        if ledger_transform:
            lp=tmp_path/value['window_ledger_path']
            with np.load(lp,allow_pickle=False) as loaded:arrays={k:loaded[k] for k in loaded.files}
            ledger_transform(arrays);f.evaluation.atomic_npz(lp,arrays);value['window_ledger_sha256']=f.campaign.sha(lp)
        transform(value);save(str(path.relative_to(tmp_path)),value)
    return SimpleNamespace(config=config_path,registry=registry,validated=validated,refresh=refresh,path=path,
                           mutate=mutate,caches=caches,states=states)


def test_full_campaign_raw_ledger_and_immutable_idempotence(study):
    result=f.finalize(study.config,True)
    assert result['status']=='passed' and len(study.validated)==27 and study.refresh==[True]
    original=f.FINAL.read_bytes();value=f.campaign.read(f.FINAL)
    assert len(value['per_run'])==27 and value['official_validation_payloads_read']==0
    assert len([p for p in value['source_dependencies'] if p.endswith('.npz')])==27
    assert len([p for p in value['source_dependencies'] if p.endswith('package_manifest.json')])==54
    assert f.finalize(study.config,True)['status']=='already_complete_verified'
    assert f.FINAL.read_bytes()==original and len(study.validated)==27


def test_missing_partial_never_updates_paper(study):
    study.path.unlink()
    result=f.finalize(study.config,True)
    assert result['status']=='pending' and result['completed_evaluation_receipts']==26
    assert not f.FINAL.exists() and study.refresh==[] and study.validated==[]
    with pytest.raises(ValueError):f.finalize(study.config,False)


@pytest.mark.parametrize('mutate',[
    lambda v:v.update(scope='official_validation'),lambda v:v.update(task='pusht-other'),
    lambda v:v.update(seed=99),lambda v:v.update(completed_epochs=29),
    lambda v:v.update(selected_checkpoint_sha256='bad'),lambda v:v.update(registration_sha256='bad'),
    lambda v:v.update(evaluator_sha256='bad'),lambda v:v.update(official_validation_payloads_read=1),
    lambda v:v.update(horizons=[60]),lambda v:v.update(total_windows=2),
    lambda v:v['episodes'][0]['standardized_mse_by_offset'].__setitem__(58,999.),
    lambda v:v.update(h60_standardized_mse=999.),lambda v:v.update(prefix_maximum_absolute_difference=float('nan'))])
def test_changed_identity_or_report_arithmetic_rejected(study,mutate):
    study.mutate(mutate)
    with pytest.raises(ValueError):f.finalize(study.config,True)
    assert not f.FINAL.exists() and not study.refresh


@pytest.mark.parametrize('mutate',[
    lambda a:a['window_start'].__setitem__(2,0),lambda a:a['episode_index'].__setitem__(2,0),
    lambda a:a['standardized_mse'].__setitem__((0,58),np.nan),
    lambda a:a.update(standardized_mse=a['standardized_mse'][:,:58]),
    lambda a:a.update(episode_index=a['episode_index'].astype(float)),
    lambda a:a.update(extra=np.zeros(3))])
def test_primitive_ledger_changes_fail_even_after_hash_rebind(study,mutate):
    study.mutate(lambda v:None,mutate)
    with pytest.raises(ValueError):f.finalize(study.config,True)
    assert not f.FINAL.exists()


def test_population_changes_and_cache_reserved_identity_rejected(study):
    study.caches['pusht'].index['0']['split']='official_validation'
    with pytest.raises(ValueError,match='metadata'):f.finalize(study.config,True)
    assert not f.FINAL.exists()


def test_completed_source_tamper_cannot_overwrite_final(study):
    f.finalize(study.config,True);original=f.FINAL.read_bytes()
    study.path.write_text('{}')
    with pytest.raises(ValueError,match='changed'):f.finalize(study.config,True)
    assert f.FINAL.read_bytes()==original


def test_official_report_is_never_opened(study):
    (f.REPORT/'official_finalization.json').write_text('deliberately invalid; must remain unread')
    assert f.finalize(study.config,True)['status']=='passed'


def test_path_escape_rejected(study):
    study.mutate(lambda v:v.update(window_ledger_path='/etc/passwd'))
    with pytest.raises(ValueError,match='escapes'):f.finalize(study.config,True)
