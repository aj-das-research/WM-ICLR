#!/usr/bin/env python3
"""Allocated-GPU full-batch throughput audit; never a substitute for full training."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from shiftwm.real_video_spatial.data import SpatialDataset
from shiftwm.real_video_spatial.model import SpatialWorldModel
spec=importlib.util.spec_from_file_location('spatial_benchmark_trainer',Path(__file__).with_name('train.py'))
train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)


def benchmark(config,output):
    if not torch.cuda.is_available(): raise RuntimeError('Requires an allocated GPU')
    torch.set_num_threads(config.get('cpu_threads',8)); train.base.seed_everything(173)
    dataset=SpatialDataset(config['cache_root'],'train',horizon=10,stride=2)
    stats=json.loads((Path(config['cache_root'])/'training_statistics.json').read_text())
    batch=next(iter(DataLoader(dataset,batch_size=config['batch_size'],shuffle=False)))
    batch={k:v.cuda() for k,v in batch.items()}; rows=[]
    for mode in SpatialWorldModel.MODES:
        model=SpatialWorldModel({**config['model_config'],'mode':mode},**{k:stats[k] for k in ('feature_mean','feature_std','action_mean','action_std')}).cuda().train()
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=config['lr'])
        torch.cuda.reset_peak_memory_stats(); durations=[]
        for iteration in range(7):
            torch.cuda.synchronize(); began=time.monotonic();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): loss=model(batch)['loss']
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();torch.cuda.synchronize()
            if iteration>=2: durations.append(time.monotonic()-began)
        model.eval();torch.cuda.synchronize();began=time.monotonic()
        with torch.inference_mode():
            for _ in range(3): model(batch)
        torch.cuda.synchronize();validation_seconds=(time.monotonic()-began)/3
        train_batches=math.ceil(len(dataset)/config['batch_size'])
        # Validation eligibility is metadata/length based; no validation payload
        # is decoded for this resource measurement.
        manifest=dataset.manifest
        val_windows=sum(max(0,(row['cameras']['exterior_image_1_left']['frames']-13)//5+1) for row in manifest['episodes'] if row['split']=='val')
        seconds=sum(durations)/len(durations)
        estimated=30*(train_batches*seconds+math.ceil(val_windows/config['batch_size'])*validation_seconds)
        rows.append({'mode':mode,'batch_size':len(batch['features']),'train_windows':len(dataset),'validation_windows':val_windows,
                     'seconds_per_train_batch':seconds,'seconds_per_float32_validation_batch':validation_seconds,
                     'estimated_seconds_per_30_epoch_run_without_io':estimated,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
                     'parameter_counts':{'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}})
        del model,optimizer;torch.cuda.empty_cache()
    result={'status':'measured','device':torch.cuda.get_device_name(),'batch_size':config['batch_size'],'rows':rows,
            'estimated_15_run_wall_hours_three_gpus_without_io':sum(r['estimated_seconds_per_30_epoch_run_without_io'] for r in rows)/3600,
            'estimated_15_run_total_gpu_hours_without_io':3*sum(r['estimated_seconds_per_30_epoch_run_without_io'] for r in rows)/3600,
            'limitations':'Timing uses training examples only; no metric selection. Excludes cache extraction, checkpoint IO, contention and scheduler delay.'}
    train.atomic_json(result,output);print(json.dumps(result),flush=True)
    # A fail-closed scheduling budget, not permission to change registered batch
    # sizes/architectures after seeing metrics.
    if result['estimated_15_run_total_gpu_hours_without_io']>80:
        raise RuntimeError('Full15-model projected training exceeds80GPU-hours; do not launch full training')
    if max(r['estimated_seconds_per_30_epoch_run_without_io'] for r in rows)>20*3600:
        raise RuntimeError('Per-run forecast exceeds registered 20-hour resource budget; do not launch full training')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    benchmark(json.loads(Path(a.config).read_text()),a.output)
