#!/usr/bin/env python3
"""Independent source-ledger, observed-pixel and saved-prediction audit; no model execution."""
from pathlib import Path
import json,hashlib,datetime,math
import numpy as np
import torch
from PIL import Image
torch.set_num_threads(2);root=Path(__file__).resolve().parents[2];p=root/'artifacts/qualitative/spatial_v1_candidate'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
r=read(p/'replay.json');x=dict(np.load(p/'replay_arrays.npz',allow_pickle=False));assert sha(p/'replay_arrays.npz')==r['arrays_sha256']
for path,digest in r['sources'].items():assert sha(root/path)==digest,path
final=read(root/'reports/real_video_spatial/finalization.json');assert final['status']=='passed'and final['completed_models']==15 and final['epochs_per_model']==30 and len(final['offline_cpu_parity'])==15
reg=read(root/'configs/real_video_spatial/v1/registration.json');assert sha(root/'reports/spatial_qualitative_selection_registration.json')==r['selection_registration_sha256']
ledgers={};stats={};seeds=(0,1,2);modes=('autoregressive','transport')
for mode in modes:
 for seed in seeds:
  d=read(root/f'reports/real_video_spatial/{mode}_s{seed}_validation.json');ledgers[mode,seed]=d
  cfg=read(root/f'runs/real_video_spatial/v1/{mode}_s{seed}/best/config.json');stats[mode,seed]={k:torch.tensor(cfg[k],dtype=torch.float32)for k in ('feature_mean','feature_std')}
  assert d['checkpoint_sha256']==sha(root/f'runs/real_video_spatial/v1/{mode}_s{seed}/best/model.pt')
episodes=sorted(e['episode_id']for e in ledgers[modes[0],0]['episodes']);inventory=[]
for eid in episodes:
 errs={m:math.fsum(next(e['native_mse'][9]for e in ledgers[m,s]['episodes']if e['episode_id']==eid)for s in seeds)/3 for m in modes};gain=100*(errs[modes[0]]-errs[modes[1]])/errs[modes[0]]
 inventory.append({'episode_id':eid,'gain':gain,'baseline':errs[modes[0]],'ours':errs[modes[1]]})
ranked=sorted(inventory,key=lambda z:(-z['gain'],z['episode_id']));assert [c['episode_id']for c in r['cases']]==[ranked[i]['episode_id']for i in (0,70,140)]
assert len(inventory)==141 and sum(v['gain']>0 for v in inventory)==136 and sum(v['gain']<0 for v in inventory)==5
pooled=100*(np.mean([a['baseline']for a in inventory])-np.mean([a['ours']for a in inventory]))/np.mean([a['baseline']for a in inventory]);assert abs(pooled-r['population']['pooled_relative_reduction_percent'])<1e-10
raw_manifest=read(root/'data/real_video/droid_selected/processed/manifest.json');val={e['episode_id']:e for e in raw_manifest['episodes']if e['split']=='val'};checks=[]
for case in r['cases']:
 prefix=case['prefix'];eid=case['episode_id'];start=case['first_window_start'];assert eid in val and start==0
 rawpath=root/'data/real_video/droid_selected/processed'/val[eid]['cameras']['exterior_image_1_left']['file'];assert sha(rawpath)==val[eid]['cameras']['exterior_image_1_left']['sha256']
 raw=np.load(rawpath,allow_pickle=False);np.testing.assert_array_equal(x[prefix+'_images'],raw['images'][:13]);np.testing.assert_array_equal(x[prefix+'_actions'],raw['actions'][:12]);np.testing.assert_array_equal(x[prefix+'_frame_indices'],raw['frame_indices'][:13])
 for entry,fi in zip(case['frame_exports'],(0,1,2,12)):
  rgb=np.asarray(Image.open(p/entry['path']).convert('RGB'));assert sha(p/entry['path'])==entry['file_sha256']and hashlib.sha256(rgb.tobytes()).hexdigest()==entry['decoded_pixel_sha256'];np.testing.assert_array_equal(rgb,x[prefix+'_images'][fi])
 for mode in modes:
  for seed in seeds:
   base=f'{prefix}_{mode}_s{seed}';pred=torch.from_numpy(x[base+'_predictions']);target=torch.from_numpy(x[prefix+'_features'][3:]);std=stats[mode,seed]['feature_std'];normalized_error=((pred-target)/std).square();patches=normalized_error.reshape(10,384,4,4).mean(1).numpy();curve=normalized_error.mean(-1).numpy()
   np.testing.assert_allclose(patches,x[base+'_patch_errors'],rtol=1e-6,atol=1e-7);np.testing.assert_array_equal(curve,x[base+'_native_mse'])
   source=next(w for w in ledgers[mode,seed]['windows']if w['episode_id']==eid and w['window_start']==start);np.testing.assert_allclose(curve,source['native_mse'],rtol=2e-5,atol=2e-6)
   proof=next(v for v in case['replay_checks']if v['mode']==mode and v['seed']==seed);np.testing.assert_array_equal(curve,proof['replayed_window_mse']);np.testing.assert_array_equal(source['native_mse'],proof['source_window_mse'])
   if mode=='transport':
    mean=stats[mode,seed]['feature_mean'];anchor=((torch.from_numpy(x[prefix+'_features'][2])-mean)/std).reshape(384,16).T[None];T=torch.from_numpy(x[base+'_matrix']);g=torch.from_numpy(x[base+'_gate']);innovation=torch.from_numpy(x[base+'_innovation']);assert torch.all(T>=0)and torch.all(g>=0)and torch.all(g<=1)and torch.all(innovation.abs()<=1)
    assert torch.allclose(T.sum(-1),torch.ones_like(T.sum(-1)),atol=1e-6,rtol=1e-6)
    np.testing.assert_array_equal(((1-g)*torch.eye(16)[None]+g*T).numpy(),x[base+'_effective_mixture'])
    for t in range(10):
     z=(1-g[t:t+1])*anchor+g[t:t+1]*(T[t:t+1]@anchor)+innovation[t:t+1];actual=z.transpose(-1,-2).reshape(1,6144)*std+mean;assert torch.equal(actual[0],pred[t])
   checks.append({'case':prefix,'mode':mode,'seed':seed,'source_window':start,'max_cpu_gpu_mse_discrepancy':float(np.max(np.abs(curve-np.array(source['native_mse']))))})
 b=np.mean([x[f'{prefix}_autoregressive_s{s}_native_mse'][-1]for s in seeds]);o=np.mean([x[f'{prefix}_transport_s{s}_native_mse'][-1]for s in seeds]);assert abs(100*(float(b)-float(o))/float(b)-case['first_window_gain_percent'])<1e-10
report={'status':'independent_numeric_review_passed','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'replay_sha256':sha(p/'replay.json'),'arrays_sha256':sha(p/'replay_arrays.npz'),'source_hashes_checked':len(r['sources']),'finalized_models':15,'completed_epochs_each':30,'ranked_episodes':141,'positive_episodes':136,'negative_episodes':5,'population_aggregated_native_h10_gain_percent':float(pooled),'selected_ranks':[0,70,140],'selected_cases':[{'episode_id':c['episode_id'],'label':c['label'],'episode_gain_percent':c['gain_percent'],'shown_window_gain_percent':c['first_window_gain_percent']}for c in r['cases']],'checks':checks,'raw_image_checks':'All13 saved frames/case, actions and frame indices match validation-only source NPZ; all12 displayed PNGs match decoded pixel SHA and bytes.','reconstruction':'All9 transport case/seed traces atall10steps reconstructed bit-exact saved predictions using actual T,g,innovation and frozen stats; no learned models rerun in this audit.','scope':'Gain-conditioned original-validation gallery, not independent test or physical-flow evidence.'}
(p/'independent_numeric_review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k]for k in ['status','source_hashes_checked','population_aggregated_native_h10_gain_percent']}));print('Max discrepancy',max(v['max_cpu_gpu_mse_discrepancy']for v in checks))
