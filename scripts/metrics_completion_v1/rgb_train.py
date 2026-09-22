"""Resumable 30-epoch shared RGB decoder fit and train-only resource profile."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import socket
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from rgb_core import ROOT, REPORT, REG, TASKS, RECIPE, RGBDecoder, configure_cpu, checked_registration, internal_cache, require, read, sha, atomic_json
from rgb_data import prepare, RGBPairs, verify_complete_targets


def atomic_torch(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent); os.close(fd)
    try:
        with open(temporary,'wb') as f:
            torch.save(value, f); f.flush(); os.fsync(f.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def validate(model, dataset, batch_size, device):
    model.eval(); sums = np.zeros(len(dataset.episode_ids)); counts = np.zeros(len(sums), dtype=np.int64)
    clipped_sums = np.zeros_like(sums)
    with torch.inference_mode():
        for feature, target, episode in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            target=target.to(device); prediction=model(feature.to(device))
            mse=(prediction-target).square().mean((1,2,3)).double().cpu().numpy()
            clipped=(prediction.clamp(0,1)-target).square().mean((1,2,3)).double().cpu().numpy()
            require(np.isfinite(mse).all() and np.isfinite(clipped).all(), 'Nonfinite decoder validation')
            np.add.at(sums,episode.numpy(),mse); np.add.at(counts,episode.numpy(),1)
            np.add.at(clipped_sums,episode.numpy(),clipped)
    require((counts>0).all() and counts.sum()==len(dataset), 'Incomplete reconstruction validation')
    return {'equal_trajectory_unclipped_mse': float((sums/counts).mean()),
            'equal_trajectory_clipped_mse': float((clipped_sums/counts).mean()),
            'episodes': [{'episode_id': eid,'frames':int(counts[i]),'unclipped_mse':float(sums[i]/counts[i]),
                          'clipped_mse':float(clipped_sums[i]/counts[i])} for i,eid in enumerate(dataset.episode_ids)]}


def fit(model, training, development, output, identity, device, *, epochs=30, batch_size=32, lr=1e-4, max_seconds=25200, verify=lambda:None, verify_final=lambda:None):
    """One atomic epoch state binds model, optimizer, RNG, history and selected state."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0)
    history=[];best=None;best_score=float('inf');best_epoch=0
    if (output/'last.pt').exists():
        state=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        require(state['identity']==identity,'Cannot resume changed decoder identity')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        if device=='cuda': torch.cuda.set_rng_state_all(state['cuda_rng'])
        history=state['history'];best=state['best_model'];best_score=state['best_score'];best_epoch=state['best_epoch']
        require([r['epoch'] for r in history]==list(range(1,len(history)+1)) and len(history)<=epochs,'Invalid epoch history')
    model.to(device)
    for state in optimizer.state.values():
        for k,v in state.items():
            if torch.is_tensor(v): state[k]=v.to(device)
    for epoch in range(len(history)+1,epochs+1):
        model.train();loss_sum=0.;count=0;epoch_start=time.monotonic()
        generator=torch.Generator().manual_seed(identity['seed']*1000003+epoch)
        loader=DataLoader(training,batch_size=batch_size,shuffle=True,generator=generator,num_workers=0,drop_last=False)
        for feature,target,_ in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction=model(feature.to(device));target=target.to(device)
            loss=(prediction-target).square().mean();require(torch.isfinite(loss).item(),'Nonfinite decoder training')
            loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(feature);count+=len(feature)
        require(count==len(training),'Training epoch omitted frames')
        result=validate(model,development,batch_size,device);score=result['equal_trajectory_unclipped_mse']
        if score<best_score:
            best_score=score;best_epoch=epoch;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append({'epoch':epoch,'train_frames':count,'development_frames':len(development),
                        'train_mse':loss_sum/count,'validation':result,'seconds':time.monotonic()-epoch_start})
        verify()
        state={'identity':identity,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
               'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else [],
               'history':history,'best_model':best,'best_score':best_score,'best_epoch':best_epoch}
        atomic_torch(state,output/'last.pt')
        atomic_json({'status':'training' if epoch<epochs else 'training_complete','epoch':epoch,'epochs':epochs,
                     'best_epoch':best_epoch,'best_unclipped_mse':best_score},output/'progress.json')
        print(json.dumps({'stage':'rgb_epoch','task':identity.get('task'),'epoch':epoch,'selector':score,'seconds':history[-1]['seconds']}),flush=True)
        if epoch<epochs and time.monotonic()-started>=max_seconds: return {'status':'requeue','epoch':epoch}
    require(len(history)==epochs and best is not None,'Decoder training incomplete')
    # Earliest strict minimum, independent of selected.pt from an earlier attempt.
    require(best_epoch==min(range(1,epochs+1),key=lambda e:history[e-1]['validation']['equal_trajectory_unclipped_mse']), 'Selected epoch differs')
    model.load_state_dict(best,strict=True)
    selected_validation=validate(model,development,batch_size,device)
    require(selected_validation==history[best_epoch-1]['validation'],'Selected decoder validation did not reproduce exactly')
    verify();verify_final()
    package={'kind':'iws_shared_rgb_decoder_v1','identity':identity,'selected_epoch':best_epoch,
             'model':best,'selection':RECIPE['selection'],'validation':selected_validation,'epochs':epochs}
    atomic_torch(package,output/'selected.pt')
    summary={'schema':'iws_shared_rgb_training_summary_v1','status':'completed','identity':identity,'epochs':epochs,
             'selected_epoch':best_epoch,'selection':RECIPE['selection'],'history':history,
             'selected_sha256':sha(output/'selected.pt'),'last_sha256':sha(output/'last.pt'),
             'reconstruction_quality':selected_validation}
    atomic_json(summary,output/'training_summary.json')
    marker={'status':'completed','epochs':epochs,'identity':identity,'selected_epoch':best_epoch,
            'selected_sha256':sha(output/'selected.pt'),'summary_sha256':sha(output/'training_summary.json')}
    atomic_json(marker,output/'completion.json')
    return marker


def allocated():
    require(bool(os.environ.get('SLURM_JOB_ID')) and torch.cuda.is_available(),'RGB GPU work requires Slurm allocation; login GPU forbidden')
    free,total=torch.cuda.mem_get_info();require(free>=8*1024**3,'Need8GiB actually free allocated GPU')
    return {'slurm_job_id':os.environ['SLURM_JOB_ID'],'hostname':socket.gethostname(),
            'device':torch.cuda.get_device_name(),'free_bytes':free,'total_bytes':total,
            'torch':str(torch.__version__),'numpy':np.__version__,'cuda_build':torch.version.cuda}


def profile(task, reg, resource):
    data=RGBPairs(task,'internal_train',reg,first_train_frames=RECIPE['resource_profile_train_frames'])
    seed_all(173);model=RGBDecoder(internal_cache(task).statistics).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=RECIPE['lr'])
    loader=iter(DataLoader(data,batch_size=32,shuffle=False,num_workers=0));times=[]
    torch.cuda.reset_peak_memory_stats()
    for i in range(2+RECIPE['resource_profile_updates']):
        try:feature,target,_=next(loader)
        except StopIteration:loader=iter(DataLoader(data,batch_size=32));feature,target,_=next(loader)
        feature,target=feature.cuda(),target.cuda();torch.cuda.synchronize();start=time.monotonic()
        optimizer.zero_grad(set_to_none=True);loss=(model(feature)-target).square().mean();require(torch.isfinite(loss).item(),'Nonfinite training-only resource profile')
        loss.backward();optimizer.step();torch.cuda.synchronize()
        if i>=2:times.append(time.monotonic()-start)
    value={'status':'passed','registration_sha256':sha(REG),'task':task,'resource':resource,
           'updates':len(times),'update_seconds':times,'mean_update_seconds':float(np.mean(times)),
           'maximum_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
           'maximum_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
           'model_parameters':sum(p.numel() for p in model.parameters()),
           'training_frames_only':len(data),'development_or_reserved_frames':0,
           'estimate_train_update_seconds_all30epochs':float(np.mean(times)*30*((reg['tasks'][task]['counts']['internal_train']['frames']+31)//32)),
           'estimate_excludes':'target cache, file IO, all development validation and checkpoint writes; no end-to-end completion promise'}
    atomic_json(value,REPORT/'profiles'/f'{task}.json')
    return value


def run(task, stage):
    allocation_start=time.monotonic()
    configure_cpu();reg=checked_registration();require(task in TASKS,'Unknown task')
    if stage=='cache':return prepare(task)
    resource=allocated()
    if stage=='pipeline':prepare(task)
    if stage in ('profile','pipeline'):
        result=profile(task,reg,resource)
        if stage=='profile':return result
        import gc
        gc.collect();torch.cuda.empty_cache()
    output=REPORT/'runs'/f'{task}_s173';output.mkdir(parents=True,exist_ok=True)
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cache_manifest=REPORT/'targets'/task/'manifest.json'
        identity={'registration_sha256':sha(REG),'task':task,'seed':173,'recipe':RECIPE,
                  'target_manifest_sha256':sha(cache_manifest),'statistics_sha256':reg['tasks'][task]['statistics_sha256']}
        if (output/'completion.json').exists():
            marker=read(output/'completion.json');require(marker['identity']==identity and marker['epochs']==30,'Wrong completed decoder')
            require(marker['summary_sha256']==sha(output/'training_summary.json') and marker['selected_sha256']==sha(output/'selected.pt'),'Completed decoder changed')
            return marker
        train=RGBPairs(task,'internal_train',reg);dev=RGBPairs(task,'internal_development',reg)
        seed_all(173);model=RGBDecoder(internal_cache(task).statistics).cuda()
        atomic_json(resource,output/'latest_resource.json')
        remaining=RECIPE['epoch_boundary_seconds']-(time.monotonic()-allocation_start)
        if remaining<=0:return {'status':'requeue'}
        return fit(model,train,dev,output,identity,'cuda',max_seconds=remaining,verify=lambda:checked_registration(),
                   verify_final=lambda:verify_complete_targets(task,reg))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=TASKS,required=True)
    parser.add_argument('--stage',choices=('cache','profile','train','pipeline'),default='pipeline')
    args=parser.parse_args();result=run(args.task,args.stage)
    print(json.dumps({'status':result['status'],'task':args.task}),flush=True)
    raise SystemExit(75 if result['status']=='requeue' else 0)
