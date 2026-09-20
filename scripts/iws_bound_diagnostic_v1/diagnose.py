#!/usr/bin/env python3
"""Development-only analytical envelope audit; no model inference or refitting."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
REPORT=Path('reports/iws_bound_diagnostic_v1')
REG=REPORT/'registration.json'
TASKS=('pusht','bimanual_box','bimanual_rope')
MODES=('bounded_spatial_mix','unbounded_spatial_mix')
FINAL='reports/real_video_iws_unbounded/development_finalization.json'
SPLIT='internal_development'
CORE_SOURCES=[str(Path(__file__).relative_to(ROOT)),
 'scripts/iws_bound_diagnostic_v1/test_diagnose.py','scripts/iws_bound_diagnostic_v1/README.md',
 'scripts/iws_bound_diagnostic_v1/run.slurm']
PROTOCOL={
 'scope':'post-development exploratory analytical mechanism diagnostic; no causal attribution',
 'tasks':list(TASKS),'split':SPLIT,'horizon_command_rows':60,'target_offsets':list(range(1,60)),
 'stride':5,'eligibility':'range(0,N-60,5), strict s+60<N; no reselection or exclusions',
 'normalization':'frozen training statistics cast to float32; anchor subtraction/division exactly as model FP32',
 'layout':'channel-major 384 channels x 16 spatial patches; no coordinate mask',
 'bound':1.0,'envelope':'per-channel [min_patch(Z0)-1, max_patch(Z0)+1], broadcast to every target patch',
 'target_coordinates':'(raw_target-float32_mean)/float32_std in float64 to express real-valued raw-MSE/std^2',
 'floating_point':'real-arithmetic relaxation anchored to the executed FP32-normalized initial features; not a certified machine-rounding bound',
 'roundoff_screen':'secondary sensitivity expands each endpoint by 32*eps32*(1+max(abs(Z0))+abs(mean/std)); not a rigorous roundoff theorem',
 'primary_aggregation':'equal windows within trajectory, then equal eligible trajectories; each analytical envelope counted once, not once per seed',
 'secondary_aggregation':'equal windows, explicitly separate',
 'comparisons':'all three matched seed receipts, bounded and no-tanh; descriptive aligned mean curves only, no explained-gain ratio or new CI',
 'persistence_check':'all windows/all59 offsets reproduce frozen CPU persistence metric with rtol=2e-6,atol=2e-7',
 'resources':{'cpu_threads':2,'memory_gib':8,'gpus':0},
 'forbidden':['reserved/test payloads','raw RGB/HDF5','training/refitting','checkpoint reselection','model inference'],
}

def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def local(relative,root=ROOT):
 p=Path(relative);require(not p.is_absolute() and '..' not in p.parts,'Unsafe relative path')
 path=(root/p).resolve();require(path.is_relative_to(root.resolve()),'Path escapes workspace');return path
def atomic(value,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as f:
  json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');name=f.name
 os.replace(name,path)
def module(name,relative):
 spec=importlib.util.spec_from_file_location(name,ROOT/relative);m=importlib.util.module_from_spec(spec)
 sys.modules[name]=m;spec.loader.exec_module(m);return m

def check_payload_path(path,allowed,root=ROOT):
 """Deny raw inputs, model tensors and every unregistered feature/error NPZ."""
 path=Path(path).resolve()
 if not path.is_relative_to(root.resolve()):return
 relative=str(path.relative_to(root.resolve()))
 if path.suffix.lower() in ('.h5','.hdf5','.mp4','.avi','.pt'):
  raise ValueError('Diagnostic forbids raw/model payload: '+relative)
 if path.suffix.lower()=='.npz':require(relative in allowed,'Unregistered or reserved payload denied: '+relative)

def install_payload_guard(allowed):
 # Audit hooks cannot be removed. CLI processes are deliberately short lived.
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   check_payload_path(os.fsdecode(args[0]),allowed)
  if event in ('socket.connect','socket.connect_ex'):
   raise ValueError('Diagnostic does not use network access')
 sys.addaudithook(audit)

def stats_arrays(stats):
 require(stats['fit_split']=='internal_train','Statistics must remain training-only')
 mean=np.asarray(stats['feature_mean'],dtype=np.float32);std=np.asarray(stats['feature_std'],dtype=np.float32)
 require(mean.shape==std.shape==(6144,) and np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all(),'Invalid normalization')
 for a in (mean,std):require(np.array_equal(a.reshape(384,16),np.repeat(a.reshape(384,16)[:,:1],16,axis=1)),'Nonshared channel normalization')
 return mean,std

def envelope_diagnostics(initial,targets,mean,std):
 """One window, all59 targets. Pure function, also used by synthetic tests."""
 initial=np.asarray(initial);targets=np.asarray(targets)
 require(initial.shape==(6144,) and targets.shape==(59,6144),'Expected one initial and59 full-coordinate targets')
 require(initial.dtype==targets.dtype==np.float32,'Cache values must be original float32')
 require(np.isfinite(initial).all() and np.isfinite(targets).all(),'Nonfinite features')
 mean,std=stats_arrays({'fit_split':'internal_train','feature_mean':mean,'feature_std':std})
 # Exact operation order/dtype of frozen SingleObservationWorldModel.normalize_features.
 z=((torch.from_numpy(initial)-torch.from_numpy(mean))/torch.from_numpy(std)).numpy().reshape(384,16).astype(np.float64)
 target=(targets.astype(np.float64).reshape(59,384,16)-mean.reshape(384,16).astype(np.float64))/std.reshape(384,16).astype(np.float64)
 lo=z.min(1)[:,None]-1.;hi=z.max(1)[:,None]+1.
 outside=np.maximum(np.maximum(lo-target,target-hi),0.)
 guard=32*np.finfo(np.float32).eps*(1+np.abs(z).max(1)[:,None]+np.abs(mean.reshape(384,16).astype(np.float64)/std.reshape(384,16)))
 screened=np.maximum(outside-guard,0.)
 # Equivalent raw-coordinate envelope; same frozen positive std at every patch.
 raw_lo=lo*std.reshape(384,16)+mean.reshape(384,16)
 raw_hi=hi*std.reshape(384,16)+mean.reshape(384,16)
 raw_distance=np.maximum(np.maximum(raw_lo-targets.reshape(59,384,16),targets.reshape(59,384,16)-raw_hi),0.)/std.reshape(384,16)
 # The alternate raw reconstruction adds/subtracts the channel mean before
 # dividing by std. Its float64 cancellation scales with abs(mean/std), even
 # when the normalized distance itself is near zero. This check's tolerance
 # changes neither the primary distance nor the independent FP32 sensitivity.
 scale=1+np.abs(target)+np.maximum(np.abs(lo),np.abs(hi))+np.abs(mean.reshape(384,16).astype(np.float64)/std.reshape(384,16))
 identity_tolerance=32*np.finfo(np.float64).eps*scale
 require(np.all(np.abs(raw_distance-outside)<=identity_tolerance),'Raw/normalized envelope identity failed')
 return {
  'coordinate_violation_fraction':(outside>0).mean(axis=(1,2)),
  'lower_coordinate_violation_fraction':(target<lo).mean(axis=(1,2)),
  'upper_coordinate_violation_fraction':(target>hi).mean(axis=(1,2)),
  'any_coordinate_violation':(outside>0).any(axis=(1,2)).astype(np.float64),
  'relaxed_mse_lower_bound':np.square(outside).mean(axis=(1,2)),
  'roundoff_screened_coordinate_violation_fraction':(screened>0).mean(axis=(1,2)),
  'roundoff_screened_relaxed_mse_lower_bound':np.square(screened).mean(axis=(1,2)),
 }

def metadata_snapshot():
 """Metadata/source reads only. Never opens any cached arrays or model tensor."""
 train=module('_bound_frozen_train','scripts/real_video_iws/train.py')
 finalize=module('_bound_frozen_finalize','scripts/real_video_iws/finalize.py')
 config=read(ROOT/'configs/real_video_iws/training_v1.json');final=read(ROOT/FINAL)
 require(final['status']=='passed' and final['completed_new_runs']==9 and final['completed_v1_comparator_runs']==27,'Full36 completed development required')
 require(final['official_validation_payloads_read']==0,'Wrong evaluation scope')
 dependencies={};payloads={};tasks={};selected=[]
 def bind(relative,expected=None):
  p=local(relative);relative=str(p.relative_to(ROOT));digest=sha(p)
  old=final['source_dependencies'].get(relative)
  require(old is None or digest==old,'Changed completed-study dependency '+relative)
  require(expected is None or digest==expected,'Changed metadata/source '+relative)
  dependencies[relative]=digest;return read(p) if p.suffix=='.json' else None
 for p in CORE_SOURCES+[FINAL,'configs/real_video_iws/training_v1.json','src/shiftwm/real_video_iws_unbounded/model.py',
  'scripts/real_video_iws/evaluate.py','scripts/real_video_iws/finalize.py']+list(train.source_files()):bind(p)
 split=bind('configs/real_video_iws/split_v1.json')
 registries=[bind('configs/real_video_iws/training_registration_v1.json'),bind('configs/real_video_iws_unbounded/registration_v1.json')]
 runs={r['name']:r for reg in registries for r in reg['runs']}
 for task in TASKS:
  cache=train.open_cache(config,task)  # Static registration checks only; no .episode call.
  audit=finalize.metadata_audit(cache);require(audit==final['populations'][task],'Development population changed')
  stats_arrays(cache.statistics)
  cache_root=config['tasks'][task]['cache_root'];cache_reg=bind(config['tasks'][task]['cache_registration'])
  for p,s in cache_reg['dependencies'].items():bind(p,s)
  for name in ('manifest.json','identity.json','episode_index.json','training_statistics.json'):bind(cache_root+'/'+name)
  records=[]
  for row in audit['records']:
   eid=row['episode_id'];require(eid in split['partitions'][task][SPLIT],'Non-development identity')
   receipt=bind(cache_root+'/episodes/'+eid+'/receipt.json')
   require(receipt==cache.index[eid] and receipt['split']==SPLIT,'Changed development receipt')
   relative=cache_root+'/episodes/'+eid+'/arrays.npz';payloads[relative]=receipt['payload_sha256']
   records.append({**row,'payload_path':relative,'payload_sha256':receipt['payload_sha256']})
  tasks[task]={'cache_root':cache_root,'audit':audit,'records':records,'statistics_path':cache_root+'/training_statistics.json'}
 for row in final['per_run']:
  if row['mode'] not in MODES and not(row['mode']=='autoregressive' and row['seed']==0):continue
  receipt=bind(row['evaluation_path'],row['evaluation_sha256'])
  require(receipt['population_audit']==tasks[row['task']]['audit'] and receipt['scope']==SPLIT,'Comparator population mismatch')
  if row['mode']=='autoregressive':
   tasks[row['task']]['persistence_ledger_path']=row['window_ledger_path'];payloads[row['window_ledger_path']]=row['window_ledger_sha256'];continue
  best=(ROOT/runs[row['name']]['output']/'best').resolve();rel=str(best.relative_to(ROOT))
  package=bind(rel+'/package_manifest.json');pc=bind(rel+'/config.json',package['files']['config.json'])
  require(package['files']['model.pt']==row['selected_checkpoint_sha256'],'Selected weight identity differs')
  require(pc['model_config']['mode']==row['mode'] and pc['model_config']['innovation_bound']==1.0,'Different correction bound/mode')
  require(pc['coordinate_layout']=='channel_major_384x4x4_shared_channel_normalization','Different feature layout')
  identity=pc['metadata']['identity']
  require(identity['populations']['val']==tasks[row['task']]['audit'],'Selected model used another development population')
  require(all(identity['scientific_config'][k]==row[k] for k in ('task','mode','seed')),'Selected model has another task/mode/seed')
  stats=read(ROOT/tasks[row['task']]['statistics_path'])
  for key in ('feature_mean','feature_std','command_mean','command_std'):
   require(np.array_equal(np.asarray(pc[key],np.float32),np.asarray(stats[key],np.float32)),'Selected normalization differs')
  selected.append({**row,'selected_config_path':rel+'/config.json','selected_config_sha256':dependencies[rel+'/config.json']})
 require({(r['task'],r['mode'],r['seed']) for r in selected}=={(t,m,s) for t in TASKS for m in MODES for s in range(3)},'Incomplete18-model comparison')
 return {'schema':'iws_bound_diagnostic_registration_v1','status':'registered_before_development_target_audit',
  'protocol':PROTOCOL,'dependencies':dict(sorted(dependencies.items())),'payload_dependencies':dict(sorted(payloads.items())),
  'tasks':tasks,'selected_models':selected,'payloads_read_during_registration':0,
  'authorization':'Root delegated development-only mechanism diagnostic; execution requires detached source review.'}

def register():
 install_payload_guard(set())
 value=metadata_snapshot();p=ROOT/REG
 if p.exists():require(read(p)==value,'Refusing to alter existing diagnostic registration')
 else:atomic(value,p)
 return {'status':value['status'],'registration_sha256':sha(p),'metadata_sources':len(value['dependencies']),
         'deferred_payloads':len(value['payload_dependencies']),'trajectories':sum(t['audit']['eligible_episodes'] for t in value['tasks'].values()),
         'windows':sum(t['audit']['windows'] for t in value['tasks'].values())}

def checked_registration(root=ROOT):
 path=root/REG;review_path=root/REPORT/'source_review.json'
 require(review_path.is_file(),'Independent source review required before payload access')
 review=read(review_path);require(review.get('status')=='passed' and review.get('registration_sha256')==sha(path),'Missing or mismatched source approval')
 value=read(path);require(value.get('schema')=='iws_bound_diagnostic_registration_v1' and value.get('protocol')==PROTOCOL,'Changed diagnostic protocol')
 for p,s in value['dependencies'].items():require(sha(local(p,root))==s,'Changed registered source/metadata '+p)
 return value

def compare_saved(reg,task,episode_rows):
 selected=[r for r in reg['selected_models'] if r['task']==task]
 by_id={r['episode_id']:r for r in episode_rows};curves={m:[] for m in MODES}
 for r in selected:
  receipt=read(ROOT/r['evaluation_path']);require(sha(ROOT/r['evaluation_path'])==r['evaluation_sha256'],'Changed completed comparator')
  require([e['episode_id'] for e in receipt['episodes']]==list(by_id),'Comparator episode order differs')
  per_episode=[]
  for e in receipt['episodes']:
   out=by_id[e['episode_id']];require(out['windows']==e['windows'],'Comparator window count differs')
   values=np.asarray(e['standardized_mse_by_offset'],dtype=np.float64)
   require(values.shape==(59,) and np.isfinite(values).all() and (values>=0).all(),'Invalid saved comparator metric')
   out.setdefault('saved_model_standardized_mse',{}).setdefault(r['mode'],{})[str(r['seed'])]=values.tolist();per_episode.append(values)
  curves[r['mode']].append(np.mean(per_episode,axis=0))
 return {m:{'per_seed_curves':[v.tolist() for v in curves[m]],'equal_seed_equal_trajectory_curve':np.mean(curves[m],axis=0).tolist()} for m in MODES}

def execute():
 reg=checked_registration();start=time.monotonic();torch.set_num_threads(2)
 install_payload_guard(set(reg['payload_dependencies']))
 require(not (ROOT/REPORT/'results.json').exists(),'Refusing to overwrite completed diagnostic')
 train=module('_bound_run_train','scripts/real_video_iws/train.py')
 evaluator=module('_bound_run_metric','scripts/real_video_iws/evaluate.py')
 config=read(ROOT/'configs/real_video_iws/training_v1.json')
 from shiftwm.real_video_iws.windows import eligible_starts
 tasks={}
 for task in TASKS:
  meta=reg['tasks'][task];cache=train.open_cache(config,task);mean,std=stats_arrays(cache.statistics)
  ledger_path=meta['persistence_ledger_path'];require(sha(ROOT/ledger_path)==reg['payload_dependencies'][ledger_path],'Changed saved development errors')
  with np.load(ROOT/ledger_path,allow_pickle=False) as f:
   ledger={k:f[k] for k in ('episode_index','window_start','persistence_standardized_mse')}
  expected=[(r['episode_index'],s) for r in meta['records'] for s in eligible_starts(r['frames'])]
  require(list(zip(ledger['episode_index'].tolist(),ledger['window_start'].tolist()))==expected,'Saved window identity differs')
  require(ledger['persistence_standardized_mse'].shape==(len(expected),59),'Incomplete saved persistence')
  index=0;episode_rows=[];pooled={};max_gap=0.
  for record in meta['records']:
   p=record['payload_path'];require(sha(ROOT/p)==reg['payload_dependencies'][p],'Changed development feature payload')
   receipt,arrays=cache.episode(record['episode_id'],SPLIT)
   require(receipt['payload_sha256']==record['payload_sha256'],'Different cache identity')
   values={};features=arrays['features']
   for s in eligible_starts(record['frames']):
    initial=features[s];targets=features[s+1:s+60];d=envelope_diagnostics(initial,targets,mean,std)
    for k,v in d.items():values.setdefault(k,[]).append(v)
    pred=torch.from_numpy(initial.copy())[None,None].expand(1,59,-1)
    actual=evaluator.feature_errors(pred,torch.from_numpy(targets.copy())[None],torch.from_numpy(std))['standardized_mse'][0].double().numpy()
    saved=ledger['persistence_standardized_mse'][index];gap=float(np.max(np.abs(actual-saved)));max_gap=max(max_gap,gap)
    require(np.allclose(actual,saved,rtol=2e-6,atol=2e-7),'Persistence discrepancy: target/layout/scales/window mismatch')
    index+=1
   if not values:continue
   row={'episode_id':record['episode_id'],'windows':record['windows'],**{k:np.mean(v,axis=0).tolist() for k,v in values.items()}}
   for k,v in values.items():pooled.setdefault(k,[]).extend(v)
   episode_rows.append(row)
  require(index==meta['audit']['windows'] and len(episode_rows)==meta['audit']['eligible_episodes'],'Incomplete diagnostic population')
  keys=list(pooled);means={k:np.mean([r[k] for r in episode_rows],axis=0).tolist() for k in keys}
  saved=compare_saved(reg,task,episode_rows)
  tasks[task]={'population':meta['audit'],'episodes':episode_rows,'equal_trajectory':means,
   'equal_window':{k:np.mean(v,axis=0).tolist() for k,v in pooled.items()},'saved_model_errors':saved,
   'persistence_maximum_absolute_difference':max_gap,'normalization_sha256':reg['dependencies'][meta['statistics_path']]}
  print(json.dumps({'task':task,'status':'all_development_windows_audited','windows':index}),flush=True)
 checked_registration()
 result={'schema':'iws_bound_diagnostic_results_v1','status':'passed','registration_sha256':sha(ROOT/REG),
  'protocol':PROTOCOL,'tasks':tasks,'runtime_seconds':time.monotonic()-start,'new_model_inference':False,
  'training_payloads_read':0,'reserved_or_test_payloads_read':0,'raw_rgb_or_hdf5_read':0,
  'interpretation':'Analytical coordinate-box relaxation, not tight decoder expressivity and not a causal explanation of observed gains. A near-zero floor does not rule out channel-coupling or optimization effects.'}
 atomic(result,ROOT/REPORT/'results.json')
 atomic({'status':'passed','registration_sha256':sha(ROOT/REG),'results_sha256':sha(ROOT/REPORT/'results.json')},ROOT/REPORT/'completion.json')
 return {'status':'passed','results_sha256':sha(ROOT/REPORT/'results.json')}

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=['register','check','run']);a=p.parse_args()
 if a.command=='register':result=register()
 elif a.command=='check':
  checked_registration();result={'status':'passed','registration_sha256':sha(ROOT/REG)}
 else:result=execute()
 print(json.dumps(result,sort_keys=True))
