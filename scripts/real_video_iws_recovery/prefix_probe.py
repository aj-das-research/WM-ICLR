#!/usr/bin/env python3
"""Bounded operational diagnostic of frozen selected IWS checkpoints, no metric gate changes."""
from pathlib import Path
import argparse,importlib.util,json,sys,time,hashlib,os
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from shiftwm.real_video_iws.windows import IWSWindowDataset
sp=importlib.util.spec_from_file_location('frozen_iws_evaluation',ROOT/'scripts/real_video_iws/evaluate.py');ev=importlib.util.module_from_spec(sp);sp.loader.exec_module(ev)
def difference(a,b):
 d=(a-b).abs();tol=2e-5+1e-5*b.abs()
 return {'max_absolute':float(d.max()),'mean_absolute':float(d.mean()),'failed_coordinates':int((d>tol).sum()),'total_coordinates':d.numel(),'frozen_allclose_pass':bool(torch.allclose(a,b,rtol=1e-5,atol=2e-5))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--tasks',nargs='+',default=['pusht','bimanual_box','bimanual_rope']);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--batch-size',type=int,default=64);args=p.parse_args()
 if args.output.exists():raise ValueError('Refusing to overwrite a prior diagnostic')
 config_path=ROOT/'configs/real_video_iws/training_v1.json';reg=ev.campaign.check_registration(config_path);config=ev.campaign.read(config_path);tr=ev.campaign.trainer()
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if args.device=='cuda':tr.allocated_device()
 result={'schema':'shiftwm_iws_prefix_operational_probe_v1','scope':'First internal-development batch only; selected completed checkpoint; no official/reserved access, no performance comparison or gate relaxation','device':args.device,'batch_size':args.batch_size,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'started_utc':ev.campaign.now(),'source_sha256':{p:ev.campaign.sha(ROOT/p) for p in ['scripts/real_video_iws/evaluate.py','src/shiftwm/real_video_iws/model.py','scripts/real_video_iws_recovery/prefix_probe.py']},'registration_sha256':ev.campaign.sha(ev.campaign.REGISTRATION),'runs':[]}
 for task in args.tasks:
  name=task+'_autoregressive_s0';row=next(r for r in reg['runs'] if r['name']==name);directory=ROOT/row['output'];summary=tr.validate_completed(directory);model,_=tr.load_package(directory/'best',args.device);cache=tr.open_cache(config,task);dataset=IWSWindowDataset(cache,'internal_development',horizon=60,stride=5);batch=next(iter(DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=0)))
  initial,commands=(batch[k].to(args.device) for k in ('initial_features','commands'));record={'name':name,'selected_epoch':summary['best_epoch'],'training_completed_epochs':summary['completed_epochs'],'first_episode_ids':batch['episode_id'],'window_start':batch['window_start'].tolist(),'prefixes':[]};t=time.perf_counter()
  with torch.inference_mode(),torch.autocast(device_type=args.device,enabled=False):
   prediction=model.predict(initial,commands);states=model.action_prefix(model.normalize_commands(commands))[0]
   for horizon in (15,30,45):
    prefix=model.predict(initial,commands[:,:horizon]);prefix_states=model.action_prefix(model.normalize_commands(commands[:,:horizon]))[0]
    entry={'horizon':horizon,'prediction_endpoint':difference(prefix[:,-1:],prediction[:,horizon-2:horizon-1]),'prediction_all_prefix_offsets':difference(prefix,prediction[:,:horizon-1]),'action_gru_prefix':difference(prefix_states,states[:,:horizon])}
    # Same-length suffix perturbation tests causal access separately from a
    # different length's numerical kernel selection. It does not waive either gate.
    if not entry['prediction_endpoint']['frozen_allclose_pass']:
     changed=commands.clone();changed[:,horizon:]+=1.;counter=model.predict(initial,changed);entry['same_length_changed_future']=difference(counter[:,:horizon-1],prediction[:,:horizon-1])
    record['prefixes'].append(entry)
  record['seconds']=time.perf_counter()-t;result['runs'].append(record);print(json.dumps(record),flush=True);del model,cache,dataset,batch,prediction,states
 result['completed_utc']=ev.campaign.now();result['status']='diagnostic_completed';ev.campaign.check_registration(config_path);ev.campaign.atomic_json(result,args.output)
if __name__=='__main__':main()
