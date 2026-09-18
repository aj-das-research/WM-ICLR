#!/usr/bin/env python3
"""Freeze, execute, and finalize the complete 15-model spatial development study."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from shiftwm.real_video.data import sha256
from shiftwm.real_video_spatial.data import SpatialDataset
from shiftwm.real_video_spatial.model import SpatialWorldModel
CONFIG=ROOT/'configs/real_video_spatial/v1'; REPORT=ROOT/'reports/real_video_spatial'; REG=CONFIG/'registration.json'
PROTOCOL=ROOT/'reports/real_video_development/spatial_protocol.md'


def module(name):
    spec=importlib.util.spec_from_file_location('spatial_campaign_'+name,Path(__file__).with_name(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def now(): return datetime.now(timezone.utc).isoformat()


def register():
    if REG.exists(): raise ValueError('Spatial campaign already registered; immutable')
    train=module('train'); CONFIG.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    model={'feature_dim':6144,'channels':384,'grid_size':4,'action_dim':35,'hidden_dim':96,'history_length':3,'depth':4,
           'context_dim':32,'context_hidden':128,'innovation_bound':1.,'identity_bias':4.,'initial_gate_logit':-3.}
    rows=[]
    for seed in (0,1,2):
        for mode in SpatialWorldModel.MODES:
            name=f'{mode}_s{seed}'; path=CONFIG/(name+'.json')
            config={'mode':mode,'seed':seed,'model_config':model,'epochs':30,'train_horizon':10,'validation_horizon':10,
                    'lr':1e-4,'min_lr':1e-6,'weight_decay':.01,'batch_size':128,'grad_clip':1.,'bf16':True,
                    'stride':2,'validation_stride':5,'cpu_threads':8,'num_workers':0,'device':'cuda',
                    'cache_root':'data/features/droid_spatial_v1','original_cache':'data/features/droid_selected_v1',
                    'metadata_audit':'data/real_video/droid_selected/processed/data_audit.json',
                    'protocol_path':str(PROTOCOL.relative_to(ROOT)),'output_dir':'runs/real_video_spatial/v1/'+name}
            train.atomic_json(config,path);rows.append({'name':name,'mode':mode,'seed':seed,'config':str(path.relative_to(ROOT)),'sha256':sha256(path)})
    paths={Path(p) for p in train.source_files()}
    paths.update(Path(__file__).parent.glob('*.slurm'))
    paths.update([PROTOCOL,ROOT/'data/features/droid_selected_v1/manifest.json',ROOT/'data/features/droid_selected_v1/training_statistics.json',
                  ROOT/'data/real_video/droid_selected/processed/manifest.json',ROOT/'data/real_video/droid_selected/processed/data_audit.json',
                  ROOT/'data/pretrained/dinov2-small/provenance.json',ROOT/'tests/test_real_video_spatial.py',ROOT/'tests/test_spatial_ledger_validation.py'])
    result={'created_utc':now(),'status':'registered','scope':'Original train/validation development only; no test access',
            'expected_runs':15,'expected_epochs_per_run':30,'modes':list(SpatialWorldModel.MODES),'runs':rows,
            'dependencies':{str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths)},
            'resource_policy':{'max_concurrent_gpus':3,'batch_size':128,'max_projected_hours_per_model':20,'max_projected_total_gpu_hours':80,
                               'allocation_hours':7+50/60,'epoch_boundary_requeue_after_seconds':6*3600}}
    train.atomic_json(result,REG);print(json.dumps({'registration':str(REG),'sha256':sha256(REG)}))


def verify():
    reg=json.loads(REG.read_text())
    expected={(mode,seed) for mode in SpatialWorldModel.MODES for seed in (0,1,2)}
    if reg['expected_runs']!=15 or len(reg['runs'])!=15: raise ValueError('Registration incomplete')
    actual={(row.get('mode'),row.get('seed')) for row in reg['runs']}
    if actual!=expected or len({row.get('name') for row in reg['runs']})!=15 or len({row.get('config') for row in reg['runs']})!=15:
        raise ValueError('Registration must contain each mode/seed exactly once')
    if any(row['name']!=f"{row['mode']}_s{row['seed']}" for row in reg['runs']):
        raise ValueError('Registered name and mode/seed disagree')
    for name,expected in reg['dependencies'].items():
        if sha256(ROOT/name)!=expected: raise ValueError('Frozen spatial dependency changed: '+name)
    for row in reg['runs']:
        if sha256(ROOT/row['config'])!=row['sha256']: raise ValueError('Frozen spatial configuration changed')
        config=json.loads((ROOT/row['config']).read_text())
        if config.get('mode')!=row['mode'] or config.get('seed')!=row['seed']:
            raise ValueError('Configuration and registered mode/seed disagree')
    return reg


def cache_and_measure():
    reg=verify();from shiftwm.real_video_spatial.features import extract
    extract(ROOT/'data/real_video/droid_selected/processed',ROOT/'data/features/droid_spatial_v1',ROOT/'data/pretrained/dinov2-small',
            batch_size=32,device='cuda',original_cache=ROOT/'data/features/droid_selected_v1')
    verify(); config=json.loads((ROOT/reg['runs'][0]['config']).read_text())
    result=module('benchmark').benchmark(config,REPORT/'throughput.json')
    if result['estimated_15_run_total_gpu_hours_without_io']>80: raise ValueError('Campaign exceeds registered80-GPU-hour projected training budget')
    verify();module('train').atomic_json({'status':'passed','completed_utc':now(),'registration_sha256':sha256(REG),
         'cache_manifest_sha256':sha256(ROOT/config['cache_root']/'manifest.json'),
         'statistics_sha256':sha256(ROOT/config['cache_root']/'training_statistics.json'),
         'throughput_sha256':sha256(REPORT/'throughput.json')},REPORT/'cache_and_budget_gate.json')


def run(name):
    reg=verify();train=module('train');row=next(r for r in reg['runs'] if r['name']==name)
    gate=json.loads((REPORT/'cache_and_budget_gate.json').read_text());config=json.loads((ROOT/row['config']).read_text())
    if gate['status']!='passed' or gate['registration_sha256']!=sha256(REG): raise ValueError('Cache/resource gate absent or stale')
    for key,path in [('cache_manifest_sha256',ROOT/config['cache_root']/'manifest.json'),('statistics_sha256',ROOT/config['cache_root']/'training_statistics.json'),('throughput_sha256',REPORT/'throughput.json')]:
        if sha256(path)!=gate[key]: raise ValueError('Cache/resource identity changed')
    config['resume_if_present']=True;config['max_runtime_seconds']=6*3600
    result=train.train(config)
    if result['status']!='completed':
        job=os.environ.get('SLURM_JOB_ID')
        if not job or not job.isdigit(): raise RuntimeError('Incomplete full training requires a Slurm job for safe requeue')
        if int(os.environ.get('SLURM_RESTART_COUNT','0'))>=5: raise RuntimeError('Repeated allocation limit reached; preserve checkpoint and report failure')
        train.atomic_json({'status':'epoch_checkpointed_requeue_requested','job_id':job,'completed_epochs':result['completed_epochs'],'utc':now()},REPORT/(name+'_continuation.json'))
        subprocess.run(['scontrol','requeue',job],check=True)
        return
    verify();module('evaluate').evaluate(config,REPORT/(name+'_validation.json'))
    train.atomic_json({'status':'completed','utc':now(),'name':name,'epochs':30},REPORT/(name+'_completed.json'))


def offline_parity(config,destination):
    train=module('train'); source=ROOT/config['output_dir']/'best'; path,state=train.read_package(source)
    destination.mkdir(parents=True,exist_ok=False)
    for name in ('model.pt','config.json'): shutil.copy2(path/name,destination/name)
    train.atomic_json({'format_version':1,'package_kind':train.base.PACKAGE_KIND,'files':{n:sha256(destination/n) for n in ('model.pt','config.json')}},destination/'package_manifest.json')
    dataset=SpatialDataset(config['cache_root'],'val',horizon=10,stride=5)
    item=dataset[0]; support=item['features'][:3][None];past=item['actions'][:2][None];future=item['actions'][2:][None]
    original,_=train.load_package(source,'cpu');torch.set_num_threads(2)
    with torch.inference_mode(): expected=original.predict(support,past,future).numpy()
    with tempfile.TemporaryDirectory(prefix='spatial-offline-parity-') as tmp:
        tmp=Path(tmp); package=tmp/'weights';shutil.copytree(destination,package)
        src=tmp/'src'; files=['shiftwm/__init__.py','shiftwm/model.py','shiftwm/upstream.py','shiftwm/real_video_spatial/__init__.py',
                            'shiftwm/real_video_spatial/model.py','shiftwm/vendor/lewm/module.py','shiftwm/vendor/lewm/NOTICE.json']
        for name in files:
            target=src/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'src'/name,target)
        np.savez(tmp/'inputs.npz',support=support.numpy(),past=past.numpy(),future=future.numpy())
        code="""import json,sys,numpy as np,torch
from pathlib import Path
sys.path.insert(0,str(Path('src').resolve()))
from shiftwm.real_video_spatial.model import from_config
torch.set_num_threads(2)
c=json.loads(Path('weights/config.json').read_text());m=from_config(c)
s=torch.load('weights/model.pt',map_location='cpu',weights_only=True);m.load_state_dict(s['state_dict'],strict=True);m.eval()
x=np.load('inputs.npz',allow_pickle=False)
with torch.inference_mode(): y=m.predict(*(torch.from_numpy(x[k]) for k in ('support','past','future')))
np.save('output.npy',y.numpy())
"""
        (tmp/'check.py').write_text(code)
        env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')};env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        subprocess.run([sys.executable,'-I','check.py'],cwd=tmp,env=env,check=True,capture_output=True)
        actual=np.load(tmp/'output.npy',allow_pickle=False);np.testing.assert_array_equal(actual,expected)
    return {'status':'passed','max_abs_error':float(np.max(np.abs(actual-expected))),'device':'cpu',
            'relocated_isolated_process':True,'input_split':'original_validation','package_manifest_sha256':sha256(destination/'package_manifest.json')}


def paired_effects(rows, draws=10000, seed=173):
    """Use the independent tested session/seed bootstrap on validated ledgers."""
    helper=module('validate_ledger'); results=[row[3] for row in rows]; effects=[]
    for comparator in ('autoregressive','anchored_additive','context_off','action_free'):
        comparison=helper.paired_intervals(results,first_mode='transport',second_mode=comparator,draws=draws,bootstrap_seed=seed)
        for metric in ('native_mse','original_2x2_mse'):
            for horizon in (5,10):
                row=comparison['metrics'][metric][horizon-1];mean=row['first_mean'];baseline=row['second_mean'];interval=row['ci95']
                effects.append({'method':'transport','comparator':comparator,'metric':metric,'horizon':horizon,
                    'method_mean':mean,'comparator_mean':baseline,'method_minus_comparator':row['mean_difference'],
                    'relative_error_reduction_percent':100*(baseline-mean)/baseline if baseline>0 else None,
                    'paired_95_percent_interval':interval,'interval_includes_zero':interval[0]<=0<=interval[1],
                    'bootstrap_draws':draws,'bootstrap_seed':seed,'bootstrap_units':'recording sessions and training seeds, paired',
                    'scope':'exploratory original validation; no multiple-comparison adjustment'})
    return effects


def finalize():
    reg=verify();train=module('train'); rows=[]; parity=[]; release=ROOT/'artifacts/releases/spatial_v1'
    if release.exists(): raise ValueError('Spatial inference export exists; review before repeating finalizer')
    # Completion gate before any export.
    for r in reg['runs']:
        config=json.loads((ROOT/r['config']).read_text()); summary=train.validate_completed(ROOT/config['output_dir'])
        p=REPORT/(r['name']+'_validation.json');result=json.loads(p.read_text())
        _,selected_state=train.read_package(ROOT/config['output_dir']/'best')
        result['summary']=module('validate_ledger').validate_ledger(result,config,ROOT,selected_state)
        rows.append((r,config,summary,result))
    release.mkdir(parents=True)
    for r,config,summary,result in rows:
        proof=offline_parity(config,release/'models'/r['name']);parity.append({'name':r['name'],**proof})
    aggregates={}
    for mode in SpatialWorldModel.MODES:
        selected=[x[3] for x in rows if x[0]['mode']==mode]
        aggregates[mode]={key:np.mean([r['summary'][key] for r in selected],axis=0).tolist() for key in selected[0]['summary']}
    final={'status':'passed','scope':'original_validation_development_only','completed_utc':now(),'registration_sha256':sha256(REG),
           'completed_models':15,'epochs_per_model':30,'offline_cpu_parity':parity,'aggregate':aggregates,'paired_effects':paired_effects(rows),
           'runs':[{'name':r['name'],'selected_epoch':s['best_epoch'],'parameter_counts':s['parameter_counts'],
                    'validation':str((REPORT/(r['name']+'_validation.json')).relative_to(ROOT))} for r,c,s,e in rows],
           'limitations':'No fresh-test/physical-control/RGB-generation or novelty claim. All registered positive and negative arms included. Native4x4 and original2x2 errors are distinct metrics.'}
    train.atomic_json(final,REPORT/'finalization.json')
    lines=['# Spatial architecture development results','','Original validation only. All15models completed30epochs; all15relocated CPU inference packages passed exact prediction parity. No fresh test was used.','',
           '| Mode | Native h5 | Native h10 | Original2x2 h5 | Original2x2 h10 | Native h10 reduction vs autoregression |','|---|---:|---:|---:|---:|---:|']
    baseline=aggregates['autoregressive']['native_mse'][9]
    for mode,m in aggregates.items(): lines.append(f"| {mode} | {m['native_mse'][4]:.6f} | {m['native_mse'][9]:.6f} | {m['original_2x2_mse'][4]:.6f} | {m['original_2x2_mse'][9]:.6f} | {100*(baseline-m['native_mse'][9])/baseline:+.3f}% |")
    lines+=['','Window-then-episode averages, then equal training seeds. Cross-resolution metrics cannot be compared directly. Parameter counts, selected epochs, all per-episode horizon ledgers, registration, and parity proofs accompany this report. Transport and innovation bounds are not separately isolated by these five arms.']
    lines+=['','## Paired validation comparisons','',
            '| Metric | Horizon | Transport comparator | Relative reduction | Difference [paired95% interval] |',
            '|---|---:|---|---:|---|']
    for effect in final['paired_effects']:
        low,high=effect['paired_95_percent_interval'];reduction=effect['relative_error_reduction_percent']
        display=f'{reduction:+.3f}%' if reduction is not None else 'undefined (zero baseline)'
        lines.append(f"| {effect['metric']} | {effect['horizon']} | {effect['comparator']} | {display} | {effect['method_minus_comparator']:+.6f} [{low:+.6f}, {high:+.6f}] |")
    lines+=['','Intervals resample recording sessions and training seeds together in matched comparisons (10,000 draws). These are exploratory validation comparisons, unadjusted for multiple comparisons.']
    (REPORT/'results.md').write_text('\n'.join(lines)+'\n');verify();print(json.dumps({'status':'passed','models':15,'finalization':str(REPORT/'finalization.json')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('register','verify','cache','run','finalize'));p.add_argument('--name');a=p.parse_args()
    if a.command=='register':register()
    elif a.command=='verify': print(json.dumps({'status':'passed','runs':len(verify()['runs'])}))
    elif a.command=='cache':cache_and_measure()
    elif a.command=='run':run(a.name)
    else:finalize()
