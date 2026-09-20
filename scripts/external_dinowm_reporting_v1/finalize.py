#!/usr/bin/env python3
"""Complete-only, read-only scientific aggregation of six adapted DINO-WM runs."""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
TRAIN_DIR=Path('scripts/external_dinowm_train_v1')
TRAIN_REPORT=Path('reports/external_dinowm_train_v1')
REPORT=Path('reports/external_dinowm_reporting_v1')
REG=REPORT/'registration.json'
BASE_REG=Path('configs/real_video_spatial/v1/registration.json')
OBJECTIVES=('official_one_step_shifted','matched_recursive_h10')
INTERNAL=('autoregressive','anchored_additive','transport','context_off','action_free')
MODES=(*INTERNAL,*OBJECTIVES)
METRICS=('native_mse','native_persistence_mse','original_2x2_mse','original_2x2_persistence_mse')
LABELS={'autoregressive':'Autoregressive','anchored_additive':'Additive anchor','transport':'ShiftWM (ours)',
 'context_off':'Context-off (ours, ablation)','action_free':'Action-free (ours, ablation)',
 'official_one_step_shifted':'Adapted DINO-WM: official shifted one-step objective',
 'matched_recursive_h10':'Adapted DINO-WM: matched recursive H10 objective'}
POLICY={'schema':'external_dinowm_reporting_policy_v1','external_runs':6,'internal_runs':15,'epochs':30,
 'objectives':list(OBJECTIVES),'internal_modes':list(INTERNAL),'seeds':[0,1,2],
 'metrics':list(METRICS),'horizons':list(range(1,11)),
 'aggregation':'mean windows within episode; equal episodes; equal matched training seeds',
 'bootstrap':{'draws':10000,'seed':20260919,'units':'recording sessions and matched training seeds','interval':'unadjusted exploratory percentile95'},
 'primary_comparisons':'ShiftWM transport minus each external objective; negative difference favors ShiftWM',
 'all_comparisons':'each of five internal modes minus each of two external objectives, all four metrics/all ten horizons',
 'scope':'posthoc development comparison with an adapted official DINO-WM baseline; not an official benchmark/SOTA reproduction',
 'no_partial_aggregates':True,'no_training_or_inference':True}


def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def local(path,root=ROOT):
 p=Path(path);require(not p.is_absolute() and '..' not in p.parts,'Unsafe source path')
 out=(root/p).resolve();require(out.is_relative_to(root.resolve()),'Source escapes workspace');return out
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
 sys.modules[name]=m;spec.loader.exec_module(m);return m
def ledger_module(root=ROOT):
 m=module('_external_reporting_private_ledger',root/'scripts/real_video_spatial/validate_ledger.py')
 # Only this private module instance accepts the externally registered names.
 # Header provenance is checked separately before reusing unchanged validation.
 m.MODES=MODES;return m
def atomic(value,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
 if path.exists() and path.read_bytes()==data:return
 with tempfile.NamedTemporaryFile('wb',dir=path.parent,delete=False) as f:f.write(data);p=f.name
 os.replace(p,path)
def write_text(text,path):
 path=Path(path)
 if path.exists() and path.read_text()==text:return
 with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as f:f.write(text);p=f.name
 os.replace(p,path)

def roster(reg,external):
 modes=OBJECTIVES if external else INTERNAL
 expected={(m,s) for m in modes for s in range(3)};rows=reg.get('runs',[])
 require(len(rows)==len(expected) and {(r.get('mode'),r.get('seed')) for r in rows}==expected,'Incomplete/duplicate registered roster')
 require(len({r.get('name') for r in rows})==len(rows) and len({r.get('config') for r in rows})==len(rows),'Repeated run/config')
 for r in rows:
  require(r['name']==f"{r['mode']}_s{r['seed']}",'Run identity differs')
  require(type(r['seed']) is int and r['seed'] in (0,1,2),'Invalid seed')
 return sorted(rows,key=lambda r:(modes.index(r['mode']),r['seed']))

def freeze(root=ROOT):
 """Freeze implementation and source identities before reading new results."""
 external=read(root/TRAIN_REPORT/'registration.json');roster(external,True)
 baseline=read(root/BASE_REG);roster(baseline,False)
 deps={}
 paths=[str(TRAIN_REPORT/'registration.json'),str(BASE_REG),'reports/real_video_spatial/finalization.json',
  'scripts/real_video_spatial/validate_ledger.py','scripts/real_video_spatial/train.py',
  'scripts/real_video_spatial/evaluate.py','scripts/real_video_spatial/campaign.py']
 paths += [str(TRAIN_DIR/n) for n in ('registry.py','train.py','evaluate.py','model.py')]
 paths += [str(Path('scripts/external_dinowm_reporting_v1')/n) for n in ('finalize.py','test_finalize.py','README.md')]
 paths += [r['config'] for reg in (external,baseline) for r in reg['runs']]
 # Hash completed comparator evidence now; new external results are not opened.
 paths += [f"reports/real_video_spatial/{r['name']}_validation.json" for r in baseline['runs']]
 for rel in sorted(set(paths)):deps[rel]=sha(local(rel,root))
 for reg in (external,baseline):
  for rel,expected in reg['dependencies'].items():
   require(sha(local(rel,root))==expected,'Changed registered scientific dependency: '+rel);deps[rel]=expected
 value={'schema':'external_dinowm_reporting_registration_v1','status':'frozen_before_external_results_read',
  'policy':POLICY,'dependencies':deps,'external_results_read':0}
 p=root/REG
 if p.exists():require(read(p)==value,'Reporting registration is immutable')
 else:atomic(value,p)
 return {'status':value['status'],'registration_sha256':sha(p),'source_count':len(deps)}

def reporting_gate(root=ROOT):
 review=root/REPORT/'source_review.json'
 require(review.exists(),'Independent reporter source review required')
 r=read(review);require(r.get('status')=='passed' and r.get('registration_sha256')==sha(root/REG),'Mismatched reporter approval')
 reg=read(root/REG);require(reg.get('policy')==POLICY,'Reporting policy changed')
 for rel,digest in reg['dependencies'].items():require(sha(local(rel,root))==digest,'Changed reporting dependency: '+rel)
 return reg

def pending_inputs(reg,root=ROOT):
 missing=[]
 for r in roster(reg,True):
  config=read(local(r['config'],root))
  for rel in (str(TRAIN_REPORT/(r['name']+'_validation.json')),str(TRAIN_REPORT/(r['name']+'_completed.json')),
              config['output_dir']+'/training_summary.json'):
   if not local(rel,root).is_file():missing.append(rel)
 return missing

def validate_external(result,config,state,config_path,registration_path,root=ROOT,ledger=None):
 ledger=ledger or ledger_module(root)
 expected={'schema':'adapted_official_dinowm_droid_validation_v1','model_family':'adapted_official_dinowm_v1',
  'run_name':f"{config['mode']}_s{config['seed']}",'training_objective':config['mode'],
  'registration_sha256':sha(registration_path),'config_sha256':sha(config_path),
  'training_identity':state['config']['metadata']['training_identity'],
  'normalization_sha256':sha(local(config['cache_root']+'/training_statistics.json',root)),
  'selection_metric':state['config']['metadata']['selection'],
  'evaluation_precision':state['config']['metadata']['validation_precision'],
  'source_sha256':sha(root/TRAIN_DIR/'evaluate.py'),
  'upstream_evaluation_source_sha256':sha(root/'scripts/real_video_spatial/evaluate.py')}
 require(config['mode'] in OBJECTIVES,'Unknown external objective')
 for k,v in expected.items():require(result.get(k)==v,'External evaluator identity differs: '+k)
 normalized=deepcopy(result);normalized['source_sha256']=result['upstream_evaluation_source_sha256']
 return ledger.validate_ledger(normalized,config,root,state)

def common_population_and_persistence(rows,ledger):
 reference=rows[0];keys=('cache_manifest_sha256','original_cache_manifest_sha256','original_normalization_sha256')
 window_key=lambda w:(w['episode_id'],w['session_id'],w['window_start'])
 ref={window_key(w):w for w in reference['windows']}
 for row in rows:
  require(all(row[k]==reference[k] for k in keys),'Methods use different feature/original populations')
  current={window_key(w):w for w in row['windows']}
  require(len(current)==len(row['windows']) and set(current)==set(ref),'Cross-method window population differs')
  for key in ref:
   for metric in ('native_persistence_mse','original_2x2_persistence_mse'):
    ledger.close(ledger.vector(current[key][metric],metric),ledger.vector(ref[key][metric],metric),'Cross-method persistence: '+metric)


def aggregate(rows,ledger,draws=10000):
 """Pure reduction; production caller must validate all21 complete ledgers first."""
 require(len(rows)==21 and {(r['mode'],r['seed']) for r in rows}=={(m,s) for m in MODES for s in range(3)},'Full six plus15 unique runs required')
 common_population_and_persistence(rows,ledger)
 aggregate={};per_seed={}
 for mode in MODES:
  selected=sorted((r for r in rows if r['mode']==mode),key=lambda r:r['seed'])
  per_seed[mode]={str(r['seed']):r['summary'] for r in selected}
  aggregate[mode]={metric:np.mean([ledger.vector(r['summary'][metric],metric) for r in selected],axis=0).tolist() for metric in METRICS}
 comparisons={}
 for external in OBJECTIVES:
  for internal in INTERNAL:
   ci=ledger.paired_intervals(rows,first_mode=internal,second_mode=external,draws=draws,bootstrap_seed=POLICY['bootstrap']['seed'])
   for metric,points in ci['metrics'].items():
    for p in points:
     a,b=p['first_mean'],p['second_mean']
     p['relative_error_reduction_percent']=None if b==0 else 100*(b-a)/b
     p['interval_includes_zero']=p['ci95'][0]<=0<=p['ci95'][1]
   ci['primary']=internal=='transport';comparisons[internal+'_vs_'+external]=ci
 first=rows[0];return {'aggregate':aggregate,'per_seed':per_seed,'comparisons':comparisons,
  'population':{'windows_per_run':len(first['windows']),'eligible_episodes':len(first['episodes']),
                'sessions':len({r['session_id'] for r in first['episodes']})}}

def collect(root=ROOT):
 reportreg=reporting_gate(root)
 external_api=module('_external_report_registry',root/TRAIN_DIR/'registry.py')
 external=external_api.verify();roster(external,True)
 base_api=module('_external_report_base_campaign',root/'scripts/real_video_spatial/campaign.py')
 baseline=base_api.verify();roster(baseline,False)
 require(not pending_inputs(external,root),'All six completed training/evaluation markers are required')
 exttrain=module('_external_report_train',root/TRAIN_DIR/'train.py')
 basetrain=module('_external_report_base_train',root/'scripts/real_video_spatial/train.py')
 ledger=ledger_module(root);sources=dict(reportreg['dependencies']);prepared=[]
 def bind(rel):
  path=local(rel,root);sources[str(path.relative_to(root))]=sha(path);return path
 # FIRST gate all six complete 30-epoch packages, and all15 fixed comparators.
 # No new validation ledger values are read until every package gate passes.
 for is_external,registry,trainer,report_dir in ((True,external,exttrain,TRAIN_REPORT),(False,baseline,basetrain,Path('reports/real_video_spatial'))):
  for row in roster(registry,is_external):
   config_path=local(row['config'],root);require(sha(config_path)==row['sha256'],'Frozen run config changed')
   config=read(config_path)
   require(config['mode']==row['mode'] and config['seed']==row['seed'] and config['epochs']==30,'Wrong mode/seed/training budget')
   directory=local(config['output_dir'],root);summary=trainer.validate_completed(directory)
   package,state=trainer.read_package(directory/'best')
   # Retain only metadata, not model tensors, across the21-run validation pass.
   selected={'epoch':state['epoch'],'config':state['config']};del state
   require(summary['completed_epochs']==30 and summary['best_epoch']==selected['epoch'],'Incomplete/incorrect selected checkpoint')
   marker_rel=str(report_dir/(row['name']+'_completed.json'));marker=read(bind(marker_rel))
   require(marker.get('status')=='completed' and marker.get('epochs')==30 and marker.get('name')==row['name'],'Missing complete-run marker')
   if is_external:
    require(marker.get('registration_sha256')==sha(root/TRAIN_REPORT/'registration.json')
            and marker.get('training_summary_sha256')==sha(directory/'training_summary.json')
            and marker.get('checkpoint_sha256')==sha(directory/'best/model.pt')
            and marker.get('selected_epoch')==selected['epoch'],'External completion identity differs')
   for name in ('training_summary.json','training_config.json','metrics.jsonl'):bind(str((directory/name).relative_to(root)))
   for kind in ('best','last'):
    package_path=(directory/kind).resolve();manifest=read(bind(str((package_path/'package_manifest.json').relative_to(root))))
    for name,digest in manifest['files'].items():
     path=bind(str((package_path/name).relative_to(root)));require(sources[str(path.relative_to(root))]==digest,'Changed selected/completed package payload')
   prepared.append((is_external,row,config,config_path,summary,selected,marker,str(report_dir/(row['name']+'_validation.json'))))
 rows=[];run_proofs=[]
 # THEN validate every primitive window, episode mean and scope/source identity.
 for is_external,row,config,config_path,summary,selected,marker,relative in prepared:
  result=read(bind(relative))
  if is_external:require(marker.get('validation_sha256')==sha(local(relative,root)),'External completion/validation binding differs')
  computed=(validate_external(result,config,selected,config_path,root/TRAIN_REPORT/'registration.json',root,ledger)
            if is_external else ledger.validate_ledger(result,config,root,selected))
  result['summary']=computed;rows.append(result)
  run_proofs.append({'name':row['name'],'mode':row['mode'],'seed':row['seed'],'external':is_external,
   'completed_epochs':30,'selected_epoch':summary['best_epoch'],'selected_checkpoint_sha256':result['checkpoint_sha256'],
   'evaluation_path':relative,'evaluation_sha256':sources[str(local(relative,root).relative_to(root))],
   'parameter_counts':result['parameter_counts'],'windows':len(result['windows']),'episodes':len(result['episodes'])})
 # ONLY NOW are any cross-run means or intervals constructed.
 result=aggregate(rows,ledger)
 for rel,digest in sources.items():require(sha(local(rel,root))==digest,'Evidence changed during finalization: '+rel)
 return {'schema':'external_dinowm_complete_development_comparison_v1','status':'passed','policy':POLICY,
  'completed_external_runs':6,'completed_internal_runs':15,'epochs_per_model':30,'results':result,'per_run':run_proofs,
  'reporting_registration_sha256':sha(root/REG),'training_registration_sha256':sha(root/TRAIN_REPORT/'registration.json'),
  'source_dependencies':sources,'new_inference_performed':False,'reserved_or_test_payloads_read':0}

def markdown(value):
 lines=['# Adapted DINO-WM: complete DROID development comparison','',POLICY['scope']+'.',
  '','All six external and15 internal runs completed30 epochs. Two training objectives are reported separately; both selected checkpoints with the same recursively evaluated H10 development metric.',
  '','| Method | Native h5 | Native h10 | Original2x2 h5 | Original2x2 h10 |','|---|---:|---:|---:|---:|']
 for m in MODES:
  a=value['results']['aggregate'][m];lines.append('| '+LABELS[m]+' | '+' | '.join(f'{a[k][h-1]:.6f}' for k,h in [('native_mse',5),('native_mse',10),('original_2x2_mse',5),('original_2x2_mse',10)])+' |')
 a=value['results']['aggregate']['transport']
 lines.append('| Persistence (same recorded anchor) | '+' | '.join(f'{a[k][h-1]:.6f}' for k,h in [('native_persistence_mse',5),('native_persistence_mse',10),('original_2x2_persistence_mse',5),('original_2x2_persistence_mse',10)])+' |')
 lines+=['','Lower MSE is better. Native4x4 and original2x2 are distinct standardized coordinate systems; their magnitudes are not interchangeable.',
  '','## Primary descriptive comparisons','',
  '| Coordinate system | Comparator | Horizon | ShiftWM reduction | ShiftWM−external MSE [95% interval] |',
  '|---|---|---:|---:|---:|']
 for external in OBJECTIVES:
  ci=value['results']['comparisons']['transport_vs_'+external]
  for metric in ('native_mse','original_2x2_mse'):
   for h in (5,10):
    p=ci['metrics'][metric][h-1];gain=p['relative_error_reduction_percent'];display='undefined (zero comparator)' if gain is None else f'{gain:+.3f}%'
    lines.append(f"| {metric} | {LABELS[external]} | {h} | {display} | {p['mean_difference']:+.6f} [{p['ci95'][0]:+.6f}, {p['ci95'][1]:+.6f}] |")
 lines+=['','Negative ShiftWM−external differences favor ShiftWM. Point reductions and signed intervals retain adverse or inconclusive results. Intervals resample matched training seeds and recording sessions10,000 times, unadjusted.',
  '','All ten horizons, four metrics (including both persistence controls), seven method means, per-seed summaries and ten paired method comparisons remain in finalization.json. All21 primitive window/episode ledgers remain at their source-bound evaluation paths. No RGB quality or physical control outcome was measured; no causal or official-benchmark SOTA claim follows.','']
 return '\n'.join(lines)

def main(if_ready=False,root=ROOT):
 training_reg=root/TRAIN_REPORT/'registration.json'
 if not training_reg.exists():
  if if_ready:return {'status':'pending','reason':'external_registration_absent'}
  raise ValueError('External registration absent')
 ext=read(training_reg);missing=pending_inputs(ext,root)
 if missing:
  if if_ready:return {'status':'pending','missing':missing}
  raise ValueError('External campaign incomplete: '+str(len(missing))+' inputs missing')
 reporting_gate(root)
 out=root/REPORT;out.mkdir(parents=True,exist_ok=True)
 with (out/'.finalization.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  final=out/'finalization.json';completion=out/'completion.json'
  if final.exists():
   require(completion.is_file(),'Existing finalization lacks completion binding')
   receipt=read(completion)
   require(receipt.get('status')=='passed' and receipt.get('finalization_sha256')==sha(final)
           and (out/'results.md').is_file() and receipt.get('results_md_sha256')==sha(out/'results.md'),
           'Existing finalization lacks valid completion binding')
   value=read(final);require(value.get('status')=='passed' and value.get('policy')==POLICY,'Invalid previous finalization')
   require(value.get('completed_external_runs')==6 and value.get('completed_internal_runs')==15
           and value.get('reporting_registration_sha256')==sha(root/REG),'Previous finalization scope differs')
   for rel,digest in value['source_dependencies'].items():require(sha(local(rel,root))==digest,'Changed completed evidence')
  else:
   value=collect(root);atomic(value,final)
   write_text(markdown(value),out/'results.md')
   atomic({'status':'passed','finalization_sha256':sha(final),'results_md_sha256':sha(out/'results.md')},completion)
  return {'status':'passed','completed_external_runs':6,'completed_internal_runs':15,'finalization_sha256':sha(final)}

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--if-ready',action='store_true');a=p.parse_args()
 print(json.dumps(freeze() if a.freeze else main(a.if_ready),sort_keys=True))
