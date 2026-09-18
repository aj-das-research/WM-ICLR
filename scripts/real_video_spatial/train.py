#!/usr/bin/env python3
"""Spatial development trainer reusing audited atomic/resumable real-video training.

The original trainer is loaded into an isolated module, never edited or globally
monkey-patched. Package, model, dataset, validation aggregation, and statistics
interfaces are explicitly replaced for this new registered experiment family.
"""
import argparse
from contextlib import nullcontext
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from shiftwm.real_video.data import sha256
from shiftwm.real_video_spatial.data import SpatialDataset, validate_spatial_manifest, statistics_from_episodes
from shiftwm.real_video_spatial.model import SpatialWorldModel, from_config
spec=importlib.util.spec_from_file_location('shiftwm_spatial_reused_trainer',ROOT/'scripts/real_video/train.py')
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
base.RealVideoWorldModel=SpatialWorldModel; base.from_config=from_config
base.RealVideoDataset=SpatialDataset; base.validate_manifest=validate_spatial_manifest
base.PACKAGE_KIND='shiftwm_real_video_spatial_v1'
base.SELECTION='window_mean_all10_shared_channel_standardized_mse'


def source_files():
    names=['scripts/real_video/train.py','scripts/real_video_spatial/train.py','scripts/real_video_spatial/campaign.py',
           'scripts/real_video_spatial/benchmark.py','scripts/real_video_spatial/evaluate.py',
           'scripts/real_video_spatial/validate_ledger.py',
           'src/shiftwm/real_video_spatial/model.py','src/shiftwm/real_video_spatial/data.py',
           'src/shiftwm/real_video_spatial/features.py','src/shiftwm/real_video_spatial/__init__.py',
           'src/shiftwm/real_video/data.py','src/shiftwm/model.py','src/shiftwm/upstream.py',
           'src/shiftwm/checkpoint.py','src/shiftwm/vendor/lewm/module.py','src/shiftwm/vendor/lewm/NOTICE.json']
    return {str(ROOT/p):sha256(ROOT/p) for p in names}
base.source_files=source_files


def verify_training_statistics(dataset,stats):
    actual=statistics_from_episodes(dataset.episodes)
    for key,value in actual.items():
        if key=='counts':
            if value!=stats[key]: raise ValueError('Training sample counts differ')
        elif not np.allclose(value,stats[key],rtol=1e-10,atol=1e-10): raise ValueError('Shared channel statistics differ')
base.verify_training_statistics=verify_training_statistics


def epoch_pass(model,loader,device,optimizer=None,bf16=False,grad_clip=1.):
    training=optimizer is not None; model.train(training)
    total=0.; elements=0; batches=0; windows=0; by_episode={}
    with (nullcontext() if training else torch.inference_mode()):
        for batch in loader:
            batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            if batch['features'].shape[1:]!=(13,6144) or batch['actions'].shape[1:]!=(12,35):
                raise ValueError('Spatial protocol requires three observed + ten query frames')
            if training: optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=training and bf16 and device.type=='cuda',dtype=torch.bfloat16):
                out=model(batch); errors=(out['standardized_predictions'].float()-out['standardized_targets'].float()).square()
                mse=errors.mean(); loss=out['loss'] if training else mse
            if not torch.isfinite(loss) or not torch.isfinite(mse): raise ValueError('Nonfinite loss')
            if training:
                loss.backward(); norm=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),grad_clip)
                if not torch.isfinite(norm): raise ValueError('Nonfinite gradients')
                optimizer.step()
            else:
                for eid,value in zip(batch['episode_index'].tolist(),errors.mean((1,2)).tolist()):
                    cumulative,count=by_episode.get(eid,(0.,0)); by_episode[eid]=(cumulative+value,count+1)
            total+=float(mse.detach())*errors.numel(); elements+=errors.numel(); batches+=1; windows+=len(errors)
    if elements==0: raise ValueError('Empty epoch')
    return {'standardized_mse':total/elements,
            'elements':elements,'batches':batches,'windows':windows,'query_steps':10,
            'aggregation':'window_weighted' if training else base.SELECTION,**({} if training else {'episodes':len(by_episode),'equal_episode_diagnostic_mse':float(np.mean([s/n for s,n in by_episode.values()]))})}
base.epoch_pass=epoch_pass


def train(config):
    torch.set_num_threads(config.get('cpu_threads',8))
    if (config.get('train_horizon')!=10 or config.get('validation_horizon')!=10
            or config.get('epochs')!=30 or config.get('seed') not in (0,1,2)
            or config.get('mode') not in SpatialWorldModel.MODES or config.get('validation_stride')!=5):
        raise ValueError('Configuration differs from spatial campaign')
    manifest,stats,identity=base.audit_inputs(config)
    if stats.get('normalization')!='shared_per_channel_over_train_frames_and_patches': raise ValueError('Wrong spatial statistics')
    train_data=SpatialDataset(config['cache_root'],'train',horizon=10,stride=config['stride'])
    val_data=SpatialDataset(config['cache_root'],'val',horizon=10,stride=5)
    verify_training_statistics(train_data,stats)
    base.seed_everything(config['seed'])
    model=SpatialWorldModel({**config['model_config'],'mode':config['mode']},**{k:stats[k] for k in ('feature_mean','feature_std','action_mean','action_std')})
    output=Path(config['output_dir']); output.mkdir(parents=True,exist_ok=True)
    with (output/'.training.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return base.fit(model,config,train_data,val_data,identity)


load_package=base.load_package
read_package=base.read_package
save_package=base.save_package
validate_completed=base.validate_completed
atomic_json=base.atomic_json

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',required=True)
    parser.add_argument('--resume-if-present',action='store_true'); args=parser.parse_args()
    config=json.loads(Path(args.config).read_text()); config['resume_if_present']=args.resume_if_present
    result=train(config); print(json.dumps(result),flush=True)
    raise SystemExit(0 if result['status']=='completed' else 75)
