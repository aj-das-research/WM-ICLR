#!/usr/bin/env python3
"""Matched original-validation spatial forecasts and original-coordinate diagnostics."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from shiftwm.real_video.data import sha256, RealVideoDataset
from shiftwm.real_video_spatial.data import SpatialDataset
from shiftwm.real_video_spatial.model import pool_to_original_2x2
spec=importlib.util.spec_from_file_location('spatial_evaluation_trainer',Path(__file__).with_name('train.py'))
train=importlib.util.module_from_spec(spec); spec.loader.exec_module(train)


def evaluate(config,output):
    torch.set_num_threads(config.get('cpu_threads',8)); train.validate_completed(config['output_dir'])
    device=config.get('device','cuda'); model,state=train.load_package(Path(config['output_dir'])/'best',device)
    dataset=SpatialDataset(config['cache_root'],'val',horizon=10,stride=5)
    original=Path(config['original_cache']); original_stats=json.loads((original/'training_statistics.json').read_text())
    old_std=torch.tensor(original_stats['feature_std'],device=device)
    if original_stats['cache_manifest_sha256']!=sha256(original/'manifest.json') or original_stats['fit_split']!='train':
        raise ValueError('Original normalization identity changed')
    old_data=RealVideoDataset(original,'val',horizon=10,stride=5,verify=True)
    old_episodes={row['episode_id']:row for row in old_data.episodes}
    old_windows={(old_data.episodes[e]['episode_id'],start) for e,start in old_data.windows}
    new_windows={(dataset.episodes[e]['episode_id'],start) for e,start in dataset.windows}
    if old_windows!=new_windows: raise ValueError('Original/spatial evaluation populations differ')
    sums={}; counts={}; window_rows=[]
    with torch.inference_mode():
        for batch in DataLoader(dataset,batch_size=config['batch_size'],shuffle=False):
            x=batch['features'].to(device); a=batch['actions'].to(device)
            pred=model.predict(x[:,:3],a[:,:2],a[:,2:]); target=x[:,3:]
            native=((pred-target)/model.feature_std).square().mean(-1)
            persistence=((x[:,2:3]-target)/model.feature_std).square().mean(-1)
            pp=pool_to_original_2x2(pred)
            exact_targets=[]; exact_anchors=[]
            for eid,start in zip(batch['episode_index'].tolist(),batch['window_start'].tolist()):
                current=dataset.episodes[eid]; original_episode=old_episodes[current['episode_id']]
                if original_episode['session_id']!=current['session_id']:
                    raise ValueError('Original session identity differs')
                exact_targets.append(original_episode['features'][start+3:start+13])
                exact_anchors.append(original_episode['features'][start+2:start+3])
            tt=torch.from_numpy(np.stack(exact_targets)).to(device)
            old_anchor=torch.from_numpy(np.stack(exact_anchors)).to(device)
            coarse=((pp-tt)/old_std).square().mean(-1)
            coarse_persistence=((old_anchor-tt)/old_std).square().mean(-1)
            if not all(torch.isfinite(v).all() for v in (native,persistence,coarse,coarse_persistence)): raise ValueError('Nonfinite evaluation')
            values=torch.stack((native,persistence,coarse,coarse_persistence),1).cpu().numpy().astype(np.float64)
            for eid,start,row in zip(batch['episode_index'].tolist(),batch['window_start'].tolist(),values):
                sums[eid]=sums.get(eid,0)+row; counts[eid]=counts.get(eid,0)+1
                episode=dataset.episodes[eid]
                window_rows.append({'episode_id':episode['episode_id'],'session_id':episode['session_id'],'window_start':start,
                    **{key:value.tolist() for key,value in zip(('native_mse','native_persistence_mse','original_2x2_mse','original_2x2_persistence_mse'),row)}})
    rows=[]
    for eid,total in sorted(sums.items()):
        e=dataset.episodes[eid]; average=total/counts[eid]
        rows.append({'episode_id':e['episode_id'],'session_id':e['session_id'],'windows':counts[eid],
                     **{key:value.tolist() for key,value in zip(('native_mse','native_persistence_mse','original_2x2_mse','original_2x2_persistence_mse'),average)}})
    summary={k:np.mean([r[k] for r in rows],axis=0).tolist() for k in ('native_mse','native_persistence_mse','original_2x2_mse','original_2x2_persistence_mse')}
    result={'status':'complete','scope':'original_validation_development_only','mode':config['mode'],'seed':config['seed'],
            'selected_epoch':state['epoch'],'checkpoint_sha256':sha256(Path(config['output_dir'])/'best/model.pt'),
            'cache_manifest_sha256':sha256(Path(config['cache_root'])/'manifest.json'),
            'original_normalization_sha256':sha256(original/'training_statistics.json'),
            'original_cache_manifest_sha256':sha256(original/'manifest.json'),
            'original_target_coordinates':'exact original cached float32 features; not recomputed pooled targets',
            'aggregation':'mean windows within episode; equal episodes','summary':summary,'episodes':rows,'windows':window_rows,
            'parameter_counts':state['config']['metadata']['parameter_counts'],'source_sha256':sha256(__file__)}
    train.atomic_json(result,output); return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(); result=evaluate(json.loads(Path(a.config).read_text()),a.output);print(json.dumps(result['summary']))
