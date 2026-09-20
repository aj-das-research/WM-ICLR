#!/usr/bin/env python3
"""Reviewed, fixed-case qualitative replay; raw artifacts local, derived pack gated."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(HERE)]
from core import require,selected_cases,patch_error,measured_details,numeric_record,average_records
AUDIT=Path('reports/qualitative_closest_v1/comparator_audit.json')
REG=Path('scripts/qualitative_closest_v1/registration.json')
REVIEW=Path('reports/qualitative_closest_v1/source_review.json')
RESULT_REVIEW=Path('reports/qualitative_closest_v1/result_review.json')
OUTPUT=Path('artifacts/qualitative/closest_v1')
PUBLIC=Path('paper/figure_sources/qualitative_closest_v1')
DROID=Path('artifacts/qualitative/spatial_v1_candidate')
IWS_REPORT=Path('reports/real_video_iws_reserved_recovery_v2')
IWS_REG=Path('configs/real_video_iws_reserved_recovery_v2/registration.json')
TASKS=('pusht','bimanual_box','bimanual_rope')
IWS_MODES=('autoregressive','bounded_spatial_mix','unbounded_spatial_mix')
DROID_MODES=('autoregressive','transport','unbounded_transport')
POLICY={'schema':'closest_qualitative_protocol_v1','baseline':'autoregressive',
 'baseline_claim':'fixed matched AR baseline; not strongest DROID control',
 'droid_scope':'original_validation_development_only','iws_scope':'reserved_upstream_validation',
 'droid_focal':'transport','iws_focal':'unbounded_spatial_mix',
 'primary_identity':'bounded ShiftWM; no-tanh is an ablation, including where it scores better',
 'selection':'largest/lower-median/smallest trajectory gain; ascending ID ties; smallest registered handle; no visual reselection',
 'seeds':[0,1,2],'droid_display_offsets':[5,10],'iws_display_offsets':[14,29,44,59],
 'target_patch_zero_based':5,'device':'cpu','dtype':'float32','threads':8,'interop_threads':1,'batch_size':1,
 'iws_backend':'one_command_row_gru_cpu_fp32_v1',
 'ledger_tolerance':{'rtol':2e-5,'atol':2e-6},'algebra_tolerance':{'rtol':5e-6,'atol':2e-6},
 'correction_aggregation':'arithmetic mean of per-seed RMS over384 channels; not pooled RMS',
 'maps':'Actual source-mixing weights, not semantic attention/saliency/flow or physical correspondence',
 'publication':'derived numeric JSON only; no raw RGB/features/commands/prediction archive',
 'no_model_or_training_selection':True,'posthoc':True}


def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def local(rel,root=ROOT):
 p=Path(rel);require(not p.is_absolute() and '..' not in p.parts,'Unsafe relative path')
 q=(root/p).resolve();require(q.is_relative_to(root.resolve()),'Path escapes workspace');return q

def module(rel,name):
 spec=importlib.util.spec_from_file_location(name,ROOT/rel);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def write(value,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 text=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
 if path.exists():require(path.read_text()==text,'Immutable output exists: '+str(path));return
 with path.open('x')as f:f.write(text)

def selections(audit):
 require(audit['status']=='complete_read_only_audit','Independent comparator audit missing')
 droid=selected_cases(audit['droid']['qualitative_cases_against_fixed_autoregression'],'transport')
 iws={t:selected_cases(audit['iws_reserved']['qualitative_cases_against_fixed_autoregression'][t],'unbounded_spatial_mix')for t in TASKS}
 require(all(len(audit['iws_reserved']['qualitative_cases_against_fixed_autoregression'][t]['all_ranked_episodes'])==10 for t in TASKS),'Wrong reserved trajectory count')
 return {'droid':droid,**iws}


def register():
 audit=read(ROOT/AUDIT);chosen=selections(audit)
 old=read(ROOT/DROID/'replay.json');require(old['arrays_sha256']==sha(ROOT/DROID/'replay_arrays.npz'),'Archived replay changed')
 require([(r['episode_id'],r['first_window_start'])for r in old['cases']]==[(r['episode_id'],r['window_start'])for r in chosen['droid']],'DROID case reuse differs')
 final=read(ROOT/IWS_REPORT/'finalization.json');review=read(ROOT/IWS_REPORT/'independent_result_review.json')
 require(final['status']=='passed' and final['completed_runs']==36 and final['total_unique_handles']==600,'Incomplete reserved study')
 require(review['status']=='passed' and review['finalization_sha256']==sha(ROOT/IWS_REPORT/'finalization.json'),'Reserved numerical review differs')
 comp=read(ROOT/'reports/real_video_spatial_components/finalization.json');require(comp['status']=='passed','Component study incomplete')
 dependencies={}
 def merge(sources):
  for p,h in sources.items():
   local(p);require(isinstance(h,str)and len(h)==64,'Invalid source binding')
   require(p not in dependencies or dependencies[p]==h,'Conflicting source binding: '+p);dependencies[p]=h
 merge(audit['source_sha256']);merge(old['sources']);merge(final['source_dependencies']);merge(comp['evidence_sha256'])
 merge(read(ROOT/'configs/real_video_spatial_components/v1/registration.json')['dependencies'])
 for p in [AUDIT,DROID/'replay.json',DROID/'replay_arrays.npz',IWS_REPORT/'finalization.json',IWS_REPORT/'independent_result_review.json',IWS_REG,
   Path('reports/real_video_spatial_components/finalization.json'),Path('configs/real_video_spatial_components/v1/registration.json'),
   Path('scripts/real_video_spatial_components/train.py'),Path('scripts/real_video_spatial_qualitative/replay.py'),
   Path('scripts/real_video_iws_reserved_recovery_v2/evaluate.py'),Path('src/shiftwm/real_video_iws_reserved_recovery/prefix_backend.py'),
   Path('src/shiftwm/real_video_iws_reserved/cache.py'),Path('src/shiftwm/real_video_iws/data.py'),
   Path('reports/qualitative_closest_v1/protocol_design.md'),*[Path('scripts/qualitative_closest_v1')/n for n in ('core.py','replay.py','test_replay.py','run.slurm')]]:
  merge({str(p):sha(ROOT/p)})
 # Raw source hashes come from already reviewed extraction receipts. Do not
 # decode any RGB or load reserved features to choose/freeze examples.
 payloads={}
 for task in TASKS:
  inputs=read(ROOT/f'data/features/iws_reserved_v1/{task}/inputs.json')
  records={r['episode_id']:r for r in inputs['episodes']}
  for row in chosen[task]:
   eid=row['episode_id'];r=records[eid];base=Path(f'data/features/iws_reserved_v1/{task}/episodes/{eid}')
   receipt=read(ROOT/base/'receipt.json')
   payloads[r['video_path']]=r['video_sha256'];payloads[r['metadata_path']]=r['metadata_sha256']
   payloads[str(base/'arrays.npz')]=receipt['payload_sha256'];merge({str(base/'receipt.json'):sha(ROOT/base/'receipt.json')})
 for p,h in dependencies.items():
  if Path(p).suffix in ('.py','.json','.md','.slurm'):require(sha(local(p))==h,'Changed metadata/source before registration: '+p)
 value={'schema':'closest_qualitative_registration_v1','status':'frozen_before_new_replay','policy':POLICY,'selected':chosen,
        'dependencies':dependencies,'selected_payloads':payloads,'new_inference_performed':False}
 write(value,ROOT/REG)
 return {'status':value['status'],'registration_sha256':sha(ROOT/REG),'source_count':len(dependencies)}


def gate(root=ROOT):
 require((root/REVIEW).exists(),'Independent qualitative source review required')
 review=read(root/REVIEW);require(review.get('status')=='passed'and review.get('registration_sha256')==sha(root/REG),'Mismatched source review')
 reg=read(root/REG);require(reg.get('policy')==POLICY,'Replay policy changed')
 for p,h in reg['dependencies'].items():require(sha(local(p,root))==h,'Changed registered source: '+p)
 for p,h in reg['selected_payloads'].items():require(sha(local(p,root))==h,'Changed selected payload: '+p)
 return reg


def image_export(rgb,path,native,role):
 from PIL import Image
 require(rgb.dtype==np.uint8 and rgb.ndim==3 and rgb.shape[-1]==3,'Not original RGB')
 Image.fromarray(rgb).save(path);require(np.array_equal(np.asarray(Image.open(path).convert('RGB')),rgb),'RGB PNG changed pixels')
 return {'native_index':int(native),'role':role,'path':str((ROOT/OUTPUT/path.name).relative_to(ROOT)),
         'sha256':sha(path),'pixel_sha256':array_sha(rgb),'local_only':True,'crop':'none'}


def details(model,initial,commands,droid=False):
 import torch
 with torch.inference_mode(),torch.autocast('cpu',enabled=False):
  if droid:
   z=model.normalize_features(initial[:,:3]);normal,rows=model._predict_normalized(z,commands[:,:2],commands[:,2:],return_details=True)
   ordinary=model.predict(initial[:,:3],commands[:,:2],commands[:,2:]);anchor=model.tokens(z)[:,-1]
  else:
   z=model.normalize_features(initial);normal,rows=model._predict_normalized(z,commands,return_details=True)
   ordinary=model.predict(initial,commands);anchor=model.tokens(z)
  require(torch.equal(ordinary,normal*model.feature_std+model.feature_mean),'Detail/public prediction mismatch')
  T=np.stack([r['transport'][0].numpy()for r in rows]);g=np.stack([r['gate'][0].numpy()for r in rows]);delta=np.stack([r['innovation'][0].numpy()for r in rows])
  predicted=model.tokens(normal)[0].numpy();anchor=anchor[0].numpy()
 return ordinary[0].numpy(),T,g,delta,predicted,anchor


def seed_record(mode,seed,prediction,target,std,expected,offsets,arrays,prefix,checkpoint,T=None,g=None,delta=None,normal=None,anchor=None):
 patches=patch_error(prediction,target,std);mse=patches.mean((-1,-2))
 np.testing.assert_allclose(mse,expected,**POLICY['ledger_tolerance'])
 checks={'ledger_max_abs':float(abs(mse-np.asarray(expected)).max()),'ledger_tolerance':POLICY['ledger_tolerance']}
 M=None
 if T is not None:
  M,algebra=measured_details(T,g,delta,normal,anchor,bounded=mode in ('transport','bounded_spatial_mix'));checks.update(algebra)
  for k,a in [('T',T),('g',g),('M',M),('delta',delta),('prediction_normalized',normal),('anchor_normalized',anchor)]:arrays[prefix+'_'+k]=a
 arrays[prefix+'_prediction']=prediction;arrays[prefix+'_patch_errors']=patches
 return {'seed':seed,'checkpoint_sha256':checkpoint,**numeric_record(patches,offsets,T,M,g,delta),'checks':checks}


def droid_replay(reg,temp,arrays):
 import torch
 old=read(ROOT/DROID/'replay.json');old_arrays=np.load(ROOT/DROID/'replay_arrays.npz',allow_pickle=False)
 trainer=module('scripts/real_video_spatial_components/train.py','qualitative_component_trainer')
 registry=read(ROOT/'configs/real_video_spatial_components/v1/registration.json')
 study={'id':'droid','scope':POLICY['droid_scope'],'baseline':'autoregressive','focal_method':'transport',
        'methods':list(DROID_MODES),'seeds':[0,1,2],'offsets':list(range(1,11)),'display_offsets':[5,10],'cases':[]}
 for index,selection in enumerate(reg['selected']['droid']):
  prefix=f'droid_case{index}';op=old['cases'][index]['prefix'];features=old_arrays[op+'_features'];actions=old_arrays[op+'_actions'];frames=old_arrays[op+'_frame_indices'];images=old_arrays[op+'_images']
  arrays[prefix+'_features']=features;arrays[prefix+'_commands']=actions
  case={'id':prefix,'episode_id':selection['episode_id'],'window_start':selection['window_start'],'selection':selection,'images':[],'methods':{}}
  for t in (0,1,2,7,12):case['images'].append(image_export(images[t],temp/f'{prefix}_frame{int(frames[t])}.png',frames[t],'observed'if t<3 else'target'))
  for mode in DROID_MODES:
   records=[]
   for seed in (0,1,2):
    label=f'{mode}_s{seed}';component=mode=='unbounded_transport'
    config=read(ROOT/(f'configs/real_video_spatial_components/v1/{label}.json'if component else f'configs/real_video_spatial/v1/{label}.json'))
    state_config=read((ROOT/config['output_dir']/'best/config.json').resolve());std=np.asarray(state_config['feature_std'],dtype=np.float32)
    source=ROOT/f'reports/{"real_video_spatial_components"if component else"real_video_spatial"}/{label}_validation.json';ledger=read(source)
    match=[w for w in ledger['windows']if w['episode_id']==selection['episode_id']and w['window_start']==selection['window_start']];require(len(match)==1,'Missing DROID source window')
    expected=match[0]['native_mse'];checkpoint=sha(ROOT/config['output_dir']/'best/model.pt');kw={}
    if component:
     trainer.validate_completed(ROOT/config['output_dir']);model,_=trainer.load_package(ROOT/config['output_dir']/'best','cpu');model.eval()
     require(not model.training,'DROID inference model must be in eval mode')
     prediction,T,g,delta,normal,anchor=details(model,torch.from_numpy(features)[None],torch.from_numpy(actions)[None],True)
     kw=dict(T=T,g=g,delta=delta,normal=normal,anchor=anchor);del model
    else:
     prediction=old_arrays[f'{op}_{label}_predictions']
     if mode=='transport':
      T=old_arrays[f'{op}_{label}_matrix'];g=old_arrays[f'{op}_{label}_gate'];delta=old_arrays[f'{op}_{label}_innovation']
      mean=np.asarray(state_config['feature_mean'],dtype=np.float32)
      anchor=((features[2]-mean)/std).reshape(384,16).T.copy();normal=((prediction-mean)/std).reshape(10,384,16).transpose(0,2,1).copy()
      kw=dict(T=T,g=g,delta=delta,normal=normal,anchor=anchor)
    record=seed_record(mode,seed,prediction,features[3:],std,expected,[5,10],arrays,f'{prefix}_{label}',checkpoint,**kw)
    record['origin']='new_component_replay'if component else'unchanged_archived_replay'
    if mode=='transport':np.testing.assert_array_equal(arrays[f'{prefix}_{label}_M'],old_arrays[f'{op}_{label}_effective_mixture'])
    records.append(record)
   case['methods'][mode]={'per_seed':records,'mean':average_records(records)}
  study['cases'].append(case);print(json.dumps({'study':'droid','case':prefix,'status':'numeric_replay_checked'}),flush=True)
 old_arrays.close();return study


def iws_replay(reg,temp,arrays):
 import torch
 from shiftwm.real_video_iws_reserved.cache import ReservedFeatureCache
 from shiftwm.real_video_iws.data import decode_native_rgb,load_commands
 evaluator=module('scripts/real_video_iws_reserved_recovery_v2/evaluate.py','qualitative_frozen_reserved_evaluator')
 registry=evaluator.checked_registration(ROOT);evaluator.check_grid(registry);studies=[]
 for task in TASKS:
  cache=ReservedFeatureCache(ROOT,task);records=[]
  study={'id':task,'scope':POLICY['iws_scope'],'baseline':'autoregressive','focal_method':'unbounded_spatial_mix',
         'methods':list(IWS_MODES),'seeds':[0,1,2],'offsets':list(range(1,60)),'display_offsets':[14,29,44,59],'cases':records}
  for i,selection in enumerate(reg['selected'][task]):
   eid,start=selection['episode_id'],selection['window_start'];prefix=f'{task}_case{i}'
   require(start==min(h['start']for h in cache.handles if h['episode_id']==eid),'Not smallest registered handle')
   payload=cache.episode(eid);receipt=cache.index[eid]
   rgb=decode_native_rgb(cache.inventory,eid,receipt['source_video_sha256'])
   require(array_sha(rgb)==receipt['decoded_rgb_sha256'],'Native RGB differs from feature-cache source')
   np.testing.assert_array_equal(load_commands(cache.inventory,eid),payload['commands'])
   features=payload['features'][start:start+60].copy();commands=payload['commands'][start:start+60].copy()
   arrays[prefix+'_features']=features;arrays[prefix+'_commands']=commands
   case={'id':prefix,'episode_id':eid,'window_start':start,'selection':selection,'images':[],'methods':{}}
   for offset in (0,14,29,44,59):case['images'].append(image_export(rgb[start+offset],temp/f'{prefix}_frame{start+offset}.png',start+offset,'observed'if offset==0 else'target'))
   del rgb,payload
   records.append(case)
  for mode in IWS_MODES:
   for seed in (0,1,2):
    run=next(r for r in registry['runs']if (r['task'],r['mode'],r['seed'])==(task,mode,seed))
    model,_=evaluator.load_selected(ROOT,run);model.eval();std=model.feature_std.numpy()
    for key in ('feature_mean','feature_std','command_mean','command_std'):np.testing.assert_array_equal(getattr(model,key).numpy(),np.asarray(cache.statistics[key],dtype=np.float32))
    source=read(ROOT/IWS_REPORT/'evaluations'/(run['name']+'.json'));npz=local(source['window_ledger_path'])
    require(sha(npz)==source['window_ledger_sha256'],'Reserved window ledger changed')
    with np.load(npz,allow_pickle=False)as ledger:
     for case in records:
      prefix=case['id'];features=arrays[prefix+'_features'];commands=arrays[prefix+'_commands']
      initial=torch.from_numpy(features[:1]);actions=torch.from_numpy(commands)[None];kw={}
      if mode=='autoregressive':
       with torch.inference_mode(),torch.autocast('cpu',enabled=False):prediction=model.predict(initial,actions)[0].numpy()
      else:
       prediction,T,g,delta,normal,anchor=details(model,initial,actions);kw=dict(T=T,g=g,delta=delta,normal=normal,anchor=anchor)
      matches=np.flatnonzero((ledger['episode_index']==cache.episode_ids.index(case['episode_id']))&(ledger['window_start']==case['window_start']));require(len(matches)==1,'Missing exact reserved handle')
      expected=ledger['standardized_mse'][matches[0]]
      record=seed_record(mode,seed,prediction,features[1:],std,expected,[14,29,44,59],arrays,f'{prefix}_{mode}_s{seed}',run['checkpoint_sha256'],**kw)
      case['methods'].setdefault(mode,{'per_seed':[]})['per_seed'].append(record)
    del model
  for case in records:
   for mode in IWS_MODES:case['methods'][mode]['mean']=average_records(case['methods'][mode]['per_seed'])
  studies.append(study);print(json.dumps({'study':task,'cases':3,'models':9,'status':'numeric_replay_checked'}),flush=True)
 return studies


def replay():
 reg=gate();import torch
 require(bool(os.environ.get('SLURM_JOB_ID')),'Allocated CPU job required')
 torch.set_num_threads(8);torch.set_num_interop_threads(1)
 require(torch.get_num_threads()==8 and torch.get_num_interop_threads()==1,'CPU thread contract differs')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 out=ROOT/OUTPUT;require(not out.exists(),'Preserve prior replay; destination already exists')
 out.parent.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(prefix='.closest-pending-',dir=out.parent));arrays={}
 studies=[droid_replay(reg,temp,arrays),*iws_replay(reg,temp,arrays)]
 for study in studies:
  study['error_scale']=[0.,max(float(np.max(v['mean']['by_offset'][str(h)]['patch_mse4x4']))for c in study['cases']for v in c['methods'].values()for h in study['display_offsets'])]
  study['weight_scale']=[0.,1.]
  study['correction_scale']=[0.,max(float(np.max(v['mean']['by_offset'][str(h)]['correction_rms4x4']))for c in study['cases']for v in c['methods'].values()for h in study['display_offsets']if'correction_rms4x4'in v['mean']['by_offset'][str(h)])]
  for case in study['cases']:
   means={m:v['mean']['mse_curve'][-1]for m,v in case['methods'].items()};base=means['autoregressive'];focal=study['focal_method']
   gain=100*(base-means[focal])/base
   np.testing.assert_allclose(gain,case['selection']['first_window_gain_percent'],rtol=3e-4,atol=.01)
   case['replayed_window_endpoint_means']=means;case['replayed_window_gain_percent']=gain
 gate()
 with (temp/'replay_arrays.npz').open('xb')as f:np.savez_compressed(f,**arrays)
 derived={'schema':'qualitative_closest_derived_v1','status':'passed','registration_sha256':sha(ROOT/REG),'policy':POLICY,
          'source_sha256':reg['dependencies'],'studies':studies,'rights':'Derived diagnostics only. Local IWS image references do not license or publish raw RGB; dataset license unspecified.',
          'images_are_predictions':False,'new_training_or_model_selection':False}
 validate_derived(derived);write(derived,temp/'derived.json')
 manifest={'schema':'qualitative_closest_local_complete_v1','status':'complete','registration_sha256':sha(ROOT/REG),'source_review_sha256':sha(ROOT/REVIEW),
           'files':{p.name:sha(p)for p in temp.iterdir()if p.is_file()},'local_only':True}
 write(manifest,temp/'manifest.json');os.rename(temp,out)
 return {'status':'complete_local_review_required','output':str(OUTPUT),'manifest_sha256':sha(out/'manifest.json')}


def validate_derived(value):
 require(value.get('schema')=='qualitative_closest_derived_v1' and value.get('status')=='passed','Wrong derived schema')
 require(value.get('policy')==POLICY and value.get('images_are_predictions') is False,'Wrong derived interpretation')
 require([s['id'] for s in value['studies']]==['droid',*TASKS],'Incomplete study grid')
 for study in value['studies']:
  require(len(study['cases'])==3 and study['seeds']==[0,1,2],'Incomplete case/seed grid')
  modes=DROID_MODES if study['id']=='droid' else IWS_MODES
  require(study['methods']==list(modes),'Wrong method identities')
  for case in study['cases']:
   require(set(case['methods'])==set(modes),'Missing displayed comparator')
   for image in case['images']:
    require(image['local_only'] is True and image['crop']=='none' and image['role'] in ('observed','target'),'Wrong image scope')
   for method in case['methods'].values():
    require([r['seed']for r in method['per_seed']]==[0,1,2],'Incomplete seeds')
    require(method['mean']==average_records(method['per_seed']),'Per-seed map averaging differs')
    for record in [*method['per_seed'],method['mean']]:
     require(not set(record)&{'features','commands','prediction','predictions','images','raw_rgb','anchor_normalized'},'Raw payload forbidden in derived pack')
     require(np.asarray(record['mse_curve']).shape==(len(study['offsets']),),'Wrong curve length')
     require(set(record['by_offset'])=={str(h)for h in study['display_offsets']},'Wrong display horizons')
     for maps in record['by_offset'].values():
      require(set(maps) in ({'patch_mse4x4'},{'patch_mse4x4','T16x16','M16x16','gate4x4','correction_rms4x4'}),'Unexpected/raw map fields')
      for key,x in maps.items():
       a=np.asarray(x);shape=(16,16)if key in ('T16x16','M16x16')else(4,4)
       require(a.shape==shape and np.isfinite(a).all()and(a>=0).all(),'Invalid derived map')
       if key in ('T16x16','M16x16','gate4x4'):require((a<=1).all(),'Invalid weight scale')
 return True


def prepare_public(root=ROOT):
 review=read(root/RESULT_REVIEW);manifest=read(root/OUTPUT/'manifest.json')
 require(review.get('status')=='passed'and review.get('manifest_sha256')==sha(root/OUTPUT/'manifest.json')
         and review.get('registration_sha256')==sha(root/REG),'Independent completed replay review required')
 require(manifest['status']=='complete'and manifest['registration_sha256']==sha(root/REG),'Incomplete local replay')
 source=root/OUTPUT/'derived.json';require(sha(source)==manifest['files']['derived.json'],'Derived numeric source changed')
 value=read(source);require(value['status']=='passed'and value['registration_sha256']==sha(root/REG),'Derived scope changed');validate_derived(value)
 write(value,root/PUBLIC/'derived.json');write(review,root/PUBLIC/'result_review.json')
 write({'schema':'qualitative_closest_public_derived_pack_v1','status':'complete_reviewed','registration_sha256':sha(root/REG),
        'files':{n:sha(root/PUBLIC/n)for n in ('derived.json','result_review.json')},'raw_files_included':False},root/PUBLIC/'manifest.json')
 return {'status':'complete_reviewed','derived_json':str(PUBLIC/'derived.json')}

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=('register','replay','prepare-public'));a=p.parse_args()
 print(json.dumps({'register':register,'replay':replay,'prepare-public':prepare_public}[a.command](),sort_keys=True))
