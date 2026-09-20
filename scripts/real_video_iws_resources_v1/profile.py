#!/usr/bin/env python3
"""Registered, predictor-only resource comparison from immutable inference bundles."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import resource
import socket
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
CONFIG = ROOT / 'configs/real_video_iws_resources_v1/protocol.json'
REGISTRATION = ROOT / 'configs/real_video_iws_resources_v1/registration.json'
REPORT = ROOT / 'reports/real_video_iws_resources_v1'
TASK_WIDTHS = {'pusht': 4, 'bimanual_box': 14, 'bimanual_rope': 8}
MODES = ('autoregressive', 'anchored_additive', 'bounded_spatial_mix', 'unbounded_spatial_mix')


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def relative(p): return str(Path(p).relative_to(ROOT))


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() == text: return
        raise ValueError('Refusing to replace measurement: ' + str(path))
    temp = path.with_name(path.name + '.tmp')
    with temp.open('x') as f: f.write(text)
    temp.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def check_protocol(c):
    expected = {'schema':'iws_predictor_resource_protocol_v1', 'batch_size':1,
                'command_rows':60, 'predicted_offsets':59, 'feature_width':6144,
                'dtype':'float32', 'autocast':False, 'tf32':False,
                'torch_intraop_threads':2, 'torch_interop_threads':1,
                'warmup_calls_per_device':10, 'timed_calls_per_device':30,
                'fixture_episode':'000011', 'fixture_initial_frame':0,
                'required_complete_rows_per_device':39}
    if any(c.get(k) != v for k,v in expected.items()): raise ValueError('Unexpected profiling protocol')
    if (set(c['tasks']) != set(TASK_WIDTHS) or c['learned_modes'] != list(MODES)
            or c['seeds'] != [0,1,2] or c['devices'] != ['cpu','cuda']):
        raise ValueError('Incomplete matched profiling grid')


def collect_cases(c):
    import numpy as np
    cases = []; inputs = {}; bundle_hashes = {}
    for name in c['bundles']:
        bundle = (ROOT/name).resolve()
        if not bundle.is_relative_to((ROOT/'artifacts/releases').resolve()): raise ValueError('Not a local release')
        module = load_module(bundle/'runtime.py', '_resource_registration_'+bundle.name)
        manifest = module.verify_bundle(bundle)
        bundle_hashes[name+'/manifest.json'] = sha(bundle/'manifest.json')
        for path,h in manifest['files'].items(): bundle_hashes[name+'/'+path] = h
        for task,rec in manifest['training_inputs'].items():
            if (rec['split'] != 'internal_train' or rec['episode_id'] != c['fixture_episode']
                    or rec['initial_frame'] != 0 or rec['targets_included'] is not False
                    or rec['command_rows'] != [0,59]): raise ValueError('Fixture access scope differs')
            with np.load(bundle/'fixtures'/f'{task}.npz', allow_pickle=False) as fixture:
                if set(fixture.files) != {'initial_features','commands'}: raise ValueError('Non-input fixture')
                z,a=fixture['initial_features'],fixture['commands']
                validate_fixture(z,a,task)
                key=(hashlib.sha256(z.tobytes()).hexdigest(),hashlib.sha256(a.tobytes()).hexdigest())
                if key != (rec['initial_feature_values_sha256'],rec['command_values_sha256']): raise ValueError('Fixture identity differs')
                if task in inputs and inputs[task] != key: raise ValueError('Unmatched cross-bundle input')
                inputs[task]=key
        for row in manifest['models']:
            if row['completed_epochs'] != 30: raise ValueError('Unfinished selected predictor')
            cases.append({k:row[k] for k in ('name','task','mode','seed','directory','fixture')} | {'bundle':name})
    check_case_grid(cases)
    for task in c['tasks']:
        cases.append({'name':task+'_persistence','task':task,'mode':'persistence','seed':None,
                      'bundle':c['bundles'][0], 'fixture':'fixtures/'+task+'.npz'})
    random.Random(c['case_order_seed']).shuffle(cases)
    return cases, bundle_hashes, inputs


def check_case_grid(cases):
    actual=[(r['task'],r['mode'],r['seed']) for r in cases]
    expected={(t,m,s) for t in TASK_WIDTHS for m in MODES for s in range(3)}
    if len(actual)!=36 or set(actual)!=expected: raise ValueError('Expected all36 distinct learned predictors')


def validate_fixture(initial,commands,task):
    import numpy as np
    if (initial.shape!=(1,6144) or commands.shape!=(1,60,TASK_WIDTHS[task])
            or initial.dtype!=np.float32 or commands.dtype!=np.float32
            or not np.isfinite(initial).all() or not np.isfinite(commands).all()):
        raise ValueError('Expected finite FP32 batch1/H60 input fixture')


def persistence_forecast(initial):
    return initial[:,None,:].expand(-1,59,-1).clone()


def timed_call(predict,synchronize,clock=time.perf_counter_ns):
    synchronize();start=clock();value=predict();synchronize()
    return value,(clock()-start)/1e6


def validate_loaded_identity(case,identity,bindings):
    path=case['bundle']+'/'+case['directory']+'/model.pt'
    if identity.get('name')!=case['name'] or identity.get('checkpoint_sha256')!=bindings.get(path):
        raise ValueError('Case label or selected checkpoint differs from registration')


def register():
    if REGISTRATION.exists(): raise ValueError('Resource protocol is already registered')
    c=read(CONFIG);check_protocol(c);cases,bundles,inputs=collect_cases(c)
    paths=[CONFIG,SOURCE,SOURCE.with_name('test_profile.py'),SOURCE.with_name('run.slurm')]
    doc={'schema':'iws_resource_registration_v1','registered_utc':now(),'protocol':c,
         'source_sha256':{relative(p):sha(p) for p in paths},'bundle_sha256':bundles,
         'cases':cases,'fixture_array_sha256':inputs,'scientific_model_changes':False,
         'reserved_payload_access_authorized':False,'measurements_started':False}
    write_new(REGISTRATION,doc)
    print(json.dumps({'status':'registered','cases':len(cases),'registration_sha256':sha(REGISTRATION)}))


def verify_registration(full_bundles=True):
    r=read(REGISTRATION);check_protocol(r['protocol'])
    if r['schema']!='iws_resource_registration_v1' or r['reserved_payload_access_authorized'] is not False:
        raise ValueError('Wrong resource registration')
    required={relative(p) for p in (CONFIG,SOURCE,SOURCE.with_name('test_profile.py'),SOURCE.with_name('run.slurm'))}
    if set(r['source_sha256'])!=required or not r['bundle_sha256']:
        raise ValueError('Missing registered source or bundle bindings')
    check_case_grid([x for x in r['cases'] if x['mode']!='persistence'])
    persistence=[x for x in r['cases'] if x['mode']=='persistence']
    if len(persistence)!=3 or {x['task'] for x in persistence}!=set(TASK_WIDTHS):
        raise ValueError('Missing persistence task or duplicate case')
    for path,h in r['source_sha256'].items():
        if sha(ROOT/path)!=h: raise ValueError('Registered profiling source changed: '+path)
    if full_bundles:
        for path,h in r['bundle_sha256'].items():
            if sha(ROOT/path)!=h: raise ValueError('Immutable bundle changed: '+path)
    return r


def summarize(samples):
    import numpy as np
    x=np.asarray(samples,dtype=np.float64)
    if x.ndim!=1 or len(x)!=30 or not np.isfinite(x).all() or (x<=0).any():
        raise ValueError('Exactly30 positive finite latency samples are required')
    return {'samples_ms':x.tolist(),'mean_ms':float(x.mean()),'median_ms':float(np.median(x)),
            'p05_ms':float(np.quantile(x,.05)),'p95_ms':float(np.quantile(x,.95)),
            'sample_sd_ms':float(x.std(ddof=1)),'minimum_ms':float(x.min()),'maximum_ms':float(x.max())}


def guard_workspace(bundle):
    attempts=[]
    allowed=[bundle.resolve(),Path(sys.prefix).resolve(),SOURCE.parent.resolve(),CONFIG.parent.resolve(),REPORT.resolve()]
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.is_relative_to(ROOT) and not any(p.is_relative_to(a) for a in allowed):
                attempts.append(relative(p));raise RuntimeError('Non-bundle workspace access blocked')
        if event in ('socket.connect','socket.getaddrinfo'):
            attempts.append('network');raise RuntimeError('Network blocked')
    sys.addaudithook(audit)
    return attempts


def rss_bytes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    return None


def worker(case_name,output):
    import numpy as np
    import torch
    r=verify_registration(False);c=r['protocol']
    case=next(x for x in r['cases'] if x['name']==case_name)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(20260920)
    bundle=ROOT/case['bundle'];attempts=guard_workspace(bundle)
    loader=load_module(bundle/'runtime.py','_resource_runtime')
    loader.verify_bundle(bundle)
    model=None
    if case['mode']!='persistence':
        model,identity=loader.load_model(case['directory'],bundle)
        validate_loaded_identity(case,identity,r['bundle_sha256'])
    else:identity={'name':case['name'],'checkpoint_sha256':None}
    params={'total':0,'trainable':0} if model is None else model.parameter_counts
    with np.load(bundle/case['fixture'],allow_pickle=False) as f:
        if set(f.files)!={'initial_features','commands'}:raise ValueError('Targets in fixture')
        initial=torch.from_numpy(f['initial_features'].copy());commands=torch.from_numpy(f['commands'].copy())
    validate_fixture(initial.numpy(),commands.numpy(),case['task'])
    results={}
    for device in c['devices']:
        if device=='cuda' and not torch.cuda.is_available():
            results[device]={'status':'unsupported','reason':'CUDA unavailable in allocated process'};continue
        if device=='cuda':
            if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count()!=1:
                raise ValueError('GPU measurement requires one Slurm-allocated visible device')
            props=torch.cuda.get_device_properties(0)
            hardware={'name':props.name,'total_memory_bytes':props.total_memory,'cuda_build':torch.version.cuda}
        else:
            cpu_name=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                           if line.startswith('model name')),platform.processor())
            hardware={'cpu':cpu_name,'cpu_affinity':sorted(os.sched_getaffinity(0))}
        z=initial.to(device);a=commands.to(device)
        if model is not None:model=model.to(device).eval()
        def predict():
            return persistence_forecast(z) if model is None else model.predict(z,a)
        def synchronize():
            if device=='cuda':torch.cuda.synchronize()
        with torch.inference_mode():
            for _ in range(c['warmup_calls_per_device']):
                value=predict();synchronize();del value
            gc.collect();synchronize()
            baseline_rss=rss_bytes()
            if device=='cuda':
                allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
                torch.cuda.reset_peak_memory_stats()
            durations=[]
            for _ in range(c['timed_calls_per_device']):
                value,duration=timed_call(predict,synchronize);durations.append(duration)
                if value.shape!=(1,59,6144):raise ValueError('Invalid full forecast')
                del value
            synchronize()
            if device=='cuda':
                memory={'resident_allocated_bytes':allocated,'resident_reserved_bytes':reserved,
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                        'incremental_peak_allocated_bytes':torch.cuda.max_memory_allocated()-allocated,
                        'scope':c['cuda_memory']}
            else:
                memory={'process_rss_before_timing_bytes':baseline_rss,
                        'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                        'isolated_tensor_allocator_peak_bytes':None,'scope':c['cpu_memory']}
            # Sanity validation follows peak capture, so its temporary tensors
            # cannot inflate the measured predictor allocation peak.
            value=predict();synchronize()
            if not torch.isfinite(value).all():raise ValueError('Nonfinite forecast')
            del value
        results[device]={'status':'passed','latency':summarize(durations),'memory':memory,'hardware':hardware}
        del z,a
    if attempts:raise ValueError('Blocked data/network access was attempted')
    # Recheck only this bundle inside the access-restricted worker.
    loader.verify_bundle(bundle)
    doc={'schema':'iws_predictor_resource_case_v1','status':'passed' if all(x['status']=='passed' for x in results.values()) else 'partial',
         'registration_sha256':sha(REGISTRATION),'case':case,'parameters':params,'selected_identity':identity,
         'devices':results,'input_scope':'fixed exported internal_train000011/frame0 and commands0:60 only',
         'forecast_shape':[1,59,6144],'dtype':'float32','batch_size':1,'torch':torch.__version__,
         'numpy':np.__version__,'node':platform.node(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
         'pid':os.getpid(),'workspace_or_network_access_attempts':attempts,'official_validation_payloads_read':0}
    write_new(output,doc)
    print(json.dumps({'case':case_name,'status':doc['status'],'median_ms':{d:v.get('latency',{}).get('median_ms') for d,v in results.items()}}),flush=True)


def aggregate(rows):
    groups={}
    for row in rows:
        for device,result in row['devices'].items():
            key=(row['case']['task'],row['case']['mode'],device)
            groups.setdefault(key,[]).append((row,result))
    result=[]
    for (task,mode,device),items in sorted(groups.items()):
        ok=[(row,v) for row,v in items if v['status']=='passed']
        if len(ok)!=len(items):
            result.append({'task':task,'mode':mode,'device':device,'status':'unsupported'});continue
        lat=[v['latency']['median_ms'] for _,v in ok]
        result.append({'task':task,'mode':mode,'device':device,'status':'passed','models':len(ok),
                       'parameter_counts':[row['parameters'] for row,_ in ok],
                       'mean_of_model_median_ms':statistics.mean(lat),'per_model_median_ms':lat,
                       'minimum_model_median_ms':min(lat),'maximum_model_median_ms':max(lat),
                       'per_model_memory':[v['memory'] for _,v in ok],
                       'latency_scope':'Observed repeated-call timing on fixed training fixture; not statistical uncertainty over tasks.'})
    return result


def run():
    r=verify_registration();regsha=sha(REGISTRATION)
    job=os.environ.get('SLURM_JOB_ID')
    if not job:raise ValueError('Run full comparison through registered Slurm allocation')
    directory=REPORT/('job_'+job)
    if directory.exists():raise ValueError('Use a new job directory; measurements are immutable')
    directory.mkdir(parents=True)
    started=now();rows=[]
    for case in r['cases']:
        output=directory/(case['name']+'.json')
        subprocess.run([sys.executable,'-I',str(SOURCE),'worker','--case',case['name'],'--output',str(output)],
                       cwd='/tmp',check=True,timeout=300)
        row=read(output)
        if row['registration_sha256']!=regsha:raise ValueError('Mixed registration')
        rows.append(row)
    verify_registration()
    if sha(REGISTRATION)!=regsha:raise ValueError('Registration changed while measuring')
    status='passed' if len(rows)==39 and all(row['status']=='passed' for row in rows) else 'partial'
    doc={'schema':'iws_predictor_resources_complete_v1','status':status,'started_utc':started,'completed_utc':now(),
         'registration_sha256':regsha,'cases':len(rows),'learned_predictors':36,'persistence_tasks':3,
         'source_sha256':r['source_sha256'],'bundle_hashes_unchanged':True,'protocol':r['protocol'],
         'case_receipts':{relative(directory/(row['case']['name']+'.json')):sha(directory/(row['case']['name']+'.json')) for row in rows},
         'summaries':aggregate(rows),'official_validation_payloads_read':0,'dataset_payloads_opened':0,
         'model_or_registration_changed':False,'accuracy_metrics_computed':False}
    write_new(directory/'summary.json',doc)
    print(json.dumps({'status':status,'summary':relative(directory/'summary.json'),'cases':len(rows)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=['register','run','worker'])
    p.add_argument('--case');p.add_argument('--output',type=Path);args=p.parse_args()
    if args.command=='register':register()
    elif args.command=='run':run()
    else:worker(args.case,args.output)
