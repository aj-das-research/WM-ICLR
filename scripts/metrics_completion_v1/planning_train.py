"""Train the unchanged current spatial architectures on new simulator DINO caches."""
from __future__ import annotations
from contextlib import nullcontext
import importlib.util
import json
import numpy as np
import torch
from planning_common import CACHE, PACKAGE_KIND, REG, ROOT, checked_registration, read, require, sha
from planning_data import PlanningDataset, load_cache, statistics
from shiftwm.real_video_spatial.model import SpatialWorldModel
from shiftwm.real_video_spatial_components.model import ComponentWorldModel

spec=importlib.util.spec_from_file_location('planning_private_atomic_training',ROOT/'scripts/real_video/train.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.PACKAGE_KIND=PACKAGE_KIND
base.SELECTION='equal_development_episode_mean_all5_training_channel_standardized_feature_mse'


def make_model(config, stats):
    cls=ComponentWorldModel if config['mode'] in ('bounded_additive','unbounded_transport') else SpatialWorldModel
    return cls(config,**{k:stats[k] for k in ('feature_mean','feature_std','action_mean','action_std')})


def from_config(config):
    require(config.get('package_kind')==PACKAGE_KIND and config.get('format_version')==1,'Wrong simulator package kind')
    require(config.get('coordinate_layout')=='channel_major_384x4x4_shared_channel_normalization','Wrong feature coordinates')
    return make_model(config['model_config'],config)


base.from_config=from_config


def epoch_pass(model, loader, device, optimizer=None, bf16=False, grad_clip=1.):
    training=optimizer is not None;model.train(training)
    total=0.;elements=0;batches=0;by_episode={}
    with (nullcontext() if training else torch.inference_mode()):
        for batch in loader:
            batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            require(batch['features'].shape[1:]==(8,6144) and batch['actions'].shape[1:]==(7,10),'Expected 3 observed +5 forecast frames')
            if training:optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=training and bf16 and device.type=='cuda'):
                out=model(batch)
                errors=(out['standardized_predictions'].float()-out['standardized_targets'].float()).square()
                loss=errors.mean()
            require(torch.isfinite(loss),'Nonfinite loss')
            if training:
                loss.backward();norm=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),grad_clip)
                require(torch.isfinite(norm),'Nonfinite gradient');optimizer.step()
            else:
                for eid,value in zip(batch['episode_index'].tolist(),errors.mean((1,2)).tolist()):
                    s,n=by_episode.get(eid,(0.,0));by_episode[eid]=(s+value,n+1)
            total+=float(loss.detach())*errors.numel();elements+=errors.numel();batches+=1
    require(elements>0,'Empty epoch')
    return {'standardized_mse':total/elements if training else float(np.mean([s/n for s,n in by_episode.values()])),
            'window_weighted_mse':total/elements,'elements':elements,'batches':batches,
            'episodes':len(by_episode) if not training else len(loader.dataset.episodes)}


base.epoch_pass=epoch_pass


def build_config(row, reg):
    return {**reg['training'],'seed':row['seed'],'mode':row['mode'],'task':row['task'],
            'model_config':{**reg['model_config'],'mode':row['mode']},
            'output_dir':str(ROOT/row['output']),'device':'cuda','cpu_threads':8,'num_workers':0,
            'resume_if_present':True,'max_runtime_seconds':5400}


def train(row):
    reg=checked_registration();config=build_config(row,reg)
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'Allocated BF16-capable CUDA required; no fallback')
    torch.set_num_threads(8)
    manifest,episodes=load_cache(row['task'],'train',sha(REG));_,development=load_cache(row['task'],'development',sha(REG))
    stats=read(CACHE/row['task']/'training_statistics.json')
    actual=statistics(episodes)
    require(all(actual[k]==stats[k] for k in actual),'Training normalization differs from training-only arrays')
    dependencies={str(ROOT/k):v for k,v in reg['dependencies'].items()}
    for path in (REG,CACHE/row['task']/'manifest.json',CACHE/row['task']/'training_statistics.json'):
        dependencies[str(path)]=sha(path)
    identity={'scientific_config':base.scientific_config(config),'dependencies':dependencies,
              'encoder_and_cache_identity':{k:manifest[k] for k in ('identity_sha256','preprocessing','registration_sha256')},
              'normalization':stats,'selection':base.SELECTION,'validation_precision':base.PRECISION,
              'runtime':{'torch':str(torch.__version__),'numpy':np.__version__,'cuda_build':torch.version.cuda}}
    base.seed_everything(row['seed'])
    model=make_model(config['model_config'],stats)
    return base.fit(model,config,PlanningDataset(episodes,1),PlanningDataset(development,1),identity)


load_package=base.load_package
