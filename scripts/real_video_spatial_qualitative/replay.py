#!/usr/bin/env python3
"""Replay prespecified spatial examples after complete verified training only."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from selection import RULE,select_episodes
REG=ROOT/'reports/spatial_qualitative_selection_registration.json'
SPATIAL_REG=ROOT/'configs/real_video_spatial/v1/registration.json'
FINAL=ROOT/'reports/real_video_spatial/finalization.json'
OUTPUT=ROOT/'artifacts/qualitative/spatial_v1_candidate'
MODES=('autoregressive','transport')
ALL_MODES=('autoregressive','anchored_additive','transport','context_off','action_free')
CAMERA='exterior_image_1_left'
METRICS=('native_mse','native_persistence_mse','original_2x2_mse','original_2x2_persistence_mse')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())

def atomic_json(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(path)


def module(name):
    path=ROOT/'scripts/real_video_spatial'/f'{name}.py'
    spec=importlib.util.spec_from_file_location('spatial_qualitative_'+name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def check_rule():
    registered=read(REG)
    if registered['status']!='registered_before_spatial_scores' or registered['rule']!=RULE or registered['spatial_validation_result_files_present']!=0:
        raise ValueError('Prespecified qualitative selection identity differs')
    for relative,expected in registered['dependencies'].items():
        if sha(ROOT/relative)!=expected:raise ValueError('Prespecified qualitative source changed')
    return registered


def check_scores(document,expected_population):
    """Recompute full episode summaries from exact source window ledgers."""
    if document.get('status')!='complete' or document.get('scope')!='original_validation_development_only':
        raise ValueError('Only completed original-validation evidence is allowed')
    windows={}
    for row in document['windows']:
        key=(row['episode_id'],row['window_start'])
        if key in windows:raise ValueError('Duplicate evaluation window')
        if row['episode_id'] not in expected_population:raise ValueError('Non-validation window')
        expected=expected_population[row['episode_id']]
        if row['session_id']!=expected['session_id'] or row['window_start'] not in expected['starts']:
            raise ValueError('Evaluation window/session mismatch')
        for metric in METRICS:
            values=np.asarray(row[metric],dtype=np.float64)
            if values.shape!=(10,) or not np.isfinite(values).all() or (values<0).any():
                raise ValueError('Invalid per-window ten-step metric ledger')
        windows[key]=row
    expected_keys={(eid,start) for eid,e in expected_population.items() for start in e['starts']}
    if set(windows)!=expected_keys:raise ValueError('Incomplete expected window population')
    episodes={}
    for row in document['episodes']:
        eid=row['episode_id']
        if eid in episodes or eid not in expected_population:raise ValueError('Duplicate or non-validation episode')
        reference=expected_population[eid]
        if row['session_id']!=reference['session_id'] or row['windows']!=len(reference['starts']):
            raise ValueError('Episode window count/session mismatch')
        for metric in METRICS:
            recomputed=np.mean([windows[eid,start][metric] for start in reference['starts']],axis=0,dtype=np.float64)
            if not np.allclose(recomputed,row[metric],rtol=1e-11,atol=1e-13):raise ValueError('Window-derived episode metric mismatch')
        episodes[eid]=row
    if set(episodes)!=set(expected_population):raise ValueError('Incomplete episode population')
    for metric in METRICS:
        recomputed=np.mean([episodes[eid][metric] for eid in sorted(episodes)],axis=0,dtype=np.float64)
        if not np.allclose(recomputed,document['summary'][metric],rtol=1e-11,atol=1e-13):raise ValueError('Episode-derived aggregate mismatch')
    return episodes,windows


def checked_campaign():
    selection=check_rule();registered=read(SPATIAL_REG);final=read(FINAL)
    if (final.get('status')!='passed' or final.get('completed_models')!=15 or final.get('epochs_per_model')!=30
            or final.get('registration_sha256')!=sha(SPATIAL_REG)
            or len(final.get('offline_cpu_parity',[]))!=15
            or any(p.get('status')!='passed' or p.get('max_abs_error')!=0 or not p.get('relocated_isolated_process') for p in final['offline_cpu_parity'])):
        raise ValueError('Awaiting fully verified fifteen-model spatial campaign')
    expected={(m,s) for m in ALL_MODES for s in (0,1,2)}
    if len(registered['runs'])!=15 or {(r['mode'],r['seed']) for r in registered['runs']}!=expected:
        raise ValueError('Complete five-mode three-seed registration required')
    if len(final.get('runs',[]))!=15 or {r['name'] for r in final['runs']}!={r['name'] for r in registered['runs']}:
        raise ValueError('Finalized run population mismatch')
    sources={**registered['dependencies'],**selection['dependencies'],
             str(REG.relative_to(ROOT)):sha(REG),str(SPATIAL_REG.relative_to(ROOT)):sha(SPATIAL_REG),
             str(FINAL.relative_to(ROOT)):sha(FINAL)}
    for path in (Path(__file__),):
        sources[str(path.relative_to(ROOT))]=sha(path)
    for relative,expected_sha in sources.items():
        if sha(ROOT/relative)!=expected_sha:raise ValueError('Frozen spatial/qualitative source changed: '+relative)
    cache_root=ROOT/'data/features/droid_spatial_v1';manifest=read(cache_root/'manifest.json')
    gate_path=ROOT/'reports/real_video_spatial/cache_and_budget_gate.json';gate=read(gate_path)
    if (gate.get('status')!='passed' or gate['registration_sha256']!=sha(SPATIAL_REG)
            or gate['cache_manifest_sha256']!=sha(cache_root/'manifest.json')
            or gate['statistics_sha256']!=sha(cache_root/'training_statistics.json')):
        raise ValueError('Spatial cache/resource gate mismatch')
    sources[str(gate_path.relative_to(ROOT))]=sha(gate_path)
    for name in ('manifest.json','training_statistics.json'):
        sources[str((cache_root/name).relative_to(ROOT))]=sha(cache_root/name)
    population={}
    for episode in manifest['episodes']:
        if episode['split']!='val':continue
        count=episode['cameras'][CAMERA]['frames'];starts=list(range(0,count-13+1,5))
        if starts:population[episode['episode_id']]={'session_id':episode['session_id'],'starts':starts,'feature_record':episode['cameras'][CAMERA]}
    trainer=module('train');scores={};configs={}
    for row in registered['runs']:
        path=ROOT/row['config']
        if sha(path)!=row['sha256']:raise ValueError('Registered spatial config changed')
        sources[row['config']]=sha(path);config=read(path);configs[row['name']]=config
        directory=ROOT/config['output_dir'];summary=trainer.validate_completed(directory)
        if summary['completed_epochs']!=30:raise ValueError('Partial model forbidden')
        final_run=next(r for r in final['runs'] if r['name']==row['name'])
        evaluation_path=ROOT/final_run['validation'];document=read(evaluation_path)
        if (document['mode'],document['seed'],document['selected_epoch'])!=(row['mode'],row['seed'],summary['best_epoch']):
            raise ValueError('Evaluation model/seed/epoch mismatch')
        if document['checkpoint_sha256']!=sha(directory/'best/model.pt') or document['cache_manifest_sha256']!=sha(cache_root/'manifest.json'):
            raise ValueError('Evaluation model/data hash mismatch')
        if document['source_sha256']!=sha(ROOT/'scripts/real_video_spatial/evaluate.py'):
            raise ValueError('Evaluation implementation hash mismatch')
        episodes,windows=check_scores(document,population)
        for relative in ('training_summary.json','best/model.pt','best/config.json','best/package_manifest.json'):
            sources[str((directory/relative).relative_to(ROOT))]=sha(directory/relative)
        sources[str(evaluation_path.relative_to(ROOT))]=sha(evaluation_path)
        scores[row['mode'],row['seed']]={'episodes':episodes,'windows':windows,'summary':document['summary']}
    for mode in ALL_MODES:
        for metric in METRICS:
            mean=np.mean([scores[mode,seed]['summary'][metric] for seed in (0,1,2)],axis=0)
            if not np.allclose(mean,final['aggregate'][mode][metric],rtol=1e-11,atol=1e-13):raise ValueError('Final aggregate mismatch')
    inventory=[]
    for eid in sorted(population):
        errors={mode:float(np.mean([scores[mode,seed]['episodes'][eid]['native_mse'][9] for seed in (0,1,2)])) for mode in MODES}
        if errors['autoregressive']<=0:raise ValueError('Nonpositive percentage-gain denominator')
        inventory.append({'episode_id':eid,'session_id':population[eid]['session_id'],
                          'window_count':len(population[eid]['starts']),'first_window_start':min(population[eid]['starts']),
                          'baseline_mse':errors['autoregressive'],'transport_mse':errors['transport'],
                          'gain_percent':100*(errors['autoregressive']-errors['transport'])/errors['autoregressive']})
    return trainer,configs,scores,population,inventory,sources


def patch_errors(prediction,target,feature_std):
    if prediction.shape!=target.shape or prediction.shape[-1]!=6144:raise ValueError('Expected native 384x4x4 coordinates')
    error=((prediction-target)/feature_std).square()
    return error.reshape(*error.shape[:-1],384,4,4).mean(-3)


def replay(output=OUTPUT):
    output=Path(output)
    if output.resolve().is_relative_to((ROOT/'paper').resolve()):raise ValueError('Unreviewed candidates must remain outside paper')
    if (output/'replay.json').exists():raise ValueError('Candidate replay already exists; preserve measured identity')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    trainer,configs,scores,population,inventory,sources=checked_campaign()
    selected=select_episodes(inventory)
    raw_root=ROOT/'data/real_video/droid_selected/processed';raw_manifest_path=raw_root/'manifest.json'
    raw_manifest=read(raw_manifest_path);raw_by_id={e['episode_id']:e for e in raw_manifest['episodes'] if e['split']=='val'}
    sources[str(raw_manifest_path.relative_to(ROOT))]=sha(raw_manifest_path)
    cases=[];arrays={};frame_exports=[]
    for index,row in enumerate(selected):
        eid=row['episode_id'];start=row['first_window_start'];end=start+13
        if eid not in raw_by_id:raise ValueError('Selected RGB case is not original validation')
        feature_record=population[eid]['feature_record'];feature_path=ROOT/'data/features/droid_spatial_v1'/feature_record['file']
        raw_record=raw_by_id[eid]['cameras'][CAMERA];raw_path=raw_root/raw_record['file']
        for path,expected in ((feature_path,feature_record['sha256']),(raw_path,raw_record['sha256'])):
            if sha(path)!=expected:raise ValueError('Selected data payload changed')
            sources[str(path.relative_to(ROOT))]=sha(path)
        if feature_record['source_sha256']!=raw_record['sha256']:raise ValueError('Features and RGB originate from different recordings')
        with np.load(feature_path,allow_pickle=False) as data:
            features=data['features'][start:end].copy();actions=data['actions'][start:end-1].copy();frame_indices=data['frame_indices'][start:end].copy()
        with np.load(raw_path,allow_pickle=False) as data:
            images=data['images'][start:end].copy()
            if not np.array_equal(actions,data['actions'][start:end-1]) or not np.array_equal(frame_indices,data['frame_indices'][start:end]):
                raise ValueError('Recorded RGB and feature action/frame coordinates differ')
        if features.shape!=(13,6144) or actions.shape!=(12,35) or images.shape[0]!=13 or images.dtype!=np.uint8 or images.shape[-1]!=3:
            raise ValueError('Invalid complete matched source window')
        prefix=f'case{index}'
        arrays[prefix+'_images']=images;arrays[prefix+'_features']=features;arrays[prefix+'_actions']=actions;arrays[prefix+'_frame_indices']=frame_indices
        cases.append({**row,'prefix':prefix,'frame_indices':frame_indices.tolist(),'replay_checks':[],'frame_exports':[]})
    for mode_index,mode in enumerate(MODES):
        for seed in (0,1,2):
            config=configs[f'{mode}_s{seed}'];model,_=trainer.load_package(ROOT/config['output_dir']/'best','cpu')
            for case in cases:
                prefix=case['prefix'];features=torch.from_numpy(arrays[prefix+'_features'])[None];actions=torch.from_numpy(arrays[prefix+'_actions'])[None]
                with torch.inference_mode():
                    prediction=model.predict(features[:,:3],actions[:,:2],actions[:,2:])
                    patches=patch_errors(prediction,features[:,3:],model.feature_std)[0]
                    native=((prediction-features[:,3:])/model.feature_std).square().mean(-1)[0]
                    expected=np.asarray(scores[mode,seed]['windows'][case['episode_id'],case['first_window_start']]['native_mse'])
                    tolerance=RULE['replay_tolerance']
                    if not np.allclose(native.numpy(),expected,**tolerance):raise ValueError('CPU replay differs from registered window metrics')
                    if not np.allclose(patches.mean((-1,-2)).numpy(),native.numpy(),rtol=1e-6,atol=1e-7):raise ValueError('Patch and whole-feature scoring differ')
                    arrays[f'{prefix}_{mode}_s{seed}_predictions']=prediction[0].numpy()
                    arrays[f'{prefix}_{mode}_s{seed}_patch_errors']=patches.numpy()
                    arrays[f'{prefix}_{mode}_s{seed}_native_mse']=native.numpy()
                    case['replay_checks'].append({'mode':mode,'seed':seed,'source_window_mse':expected.tolist(),
                        'replayed_window_mse':native.tolist(),'max_abs_error':float(np.max(np.abs(native.numpy()-expected))),
                        'rtol':tolerance['rtol'],'atol':tolerance['atol']})
                    if mode=='transport':
                        normalized,details=model._predict_normalized(model.normalize_features(features[:,:3]),actions[:,:2],actions[:,2:],return_details=True)
                        if not torch.equal(normalized*model.feature_std+model.feature_mean,prediction):raise ValueError('Instrumented prediction differs from ordinary predictor')
                        anchor=model.tokens(model.normalize_features(features[:,:3]))[:,-1]
                        for t,detail in enumerate(details):
                            matrix,gate,innovation=(detail[k] for k in ('transport','gate','innovation'))
                            if not torch.allclose(matrix.sum(-1),torch.ones_like(matrix.sum(-1)),atol=1e-6,rtol=1e-6) or (matrix<0).any() or (gate<0).any() or (gate>1).any():raise ValueError('Invalid transport/gate probabilities')
                            reconstructed=(1-gate)*anchor+gate*(matrix@anchor)+innovation
                            if not torch.equal(model.flatten(reconstructed),normalized[:,t]):raise ValueError('Transport decomposition does not reconstruct actual features')
                        arrays[f'{prefix}_transport_s{seed}_matrix']=torch.stack([d['transport'][0] for d in details]).numpy()
                        arrays[f'{prefix}_transport_s{seed}_gate']=torch.stack([d['gate'][0] for d in details]).numpy()
                        arrays[f'{prefix}_transport_s{seed}_innovation']=torch.stack([d['innovation'][0] for d in details]).numpy()
                        arrays[f'{prefix}_transport_s{seed}_effective_mixture']=torch.stack([((1-d['gate'])*torch.eye(16)[None]+d['gate']*d['transport'])[0] for d in details]).numpy()
            del model
    output.mkdir(parents=True,exist_ok=True)
    for case in cases:
        prefix=case['prefix'];images=arrays[prefix+'_images']
        curves={mode:np.stack([arrays[f'{prefix}_{mode}_s{s}_native_mse'] for s in (0,1,2)]) for mode in MODES}
        case['first_window_endpoint_mean']={mode:float(curves[mode][:,9].mean()) for mode in MODES}
        b,p=(case['first_window_endpoint_mean'][m] for m in MODES)
        case['first_window_gain_percent']=100*(b-p)/b
        for frame in (0,1,2,12):
            image_path=output/f'{prefix}_recorded_frame_{int(arrays[prefix+"_frame_indices"][frame])}.png'
            Image.fromarray(images[frame]).save(image_path)
            decoded=np.asarray(Image.open(image_path).convert('RGB'))
            if not np.array_equal(decoded,images[frame]):raise ValueError('Export altered actual RGB pixels')
            case['frame_exports'].append({'path':image_path.name,'native_frame_index':int(arrays[prefix+'_frame_indices'][frame]),
                'file_sha256':sha(image_path),'decoded_pixel_sha256':hashlib.sha256(decoded.tobytes()).hexdigest()})
    temporary=output/'replay_arrays.npz.tmp'
    with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
    temporary.replace(output/'replay_arrays.npz')
    for relative,expected in sources.items():
        if sha(ROOT/relative)!=expected:raise ValueError('Source changed during qualitative replay')
    gains=np.asarray([r['gain_percent'] for r in inventory]);b=np.mean([r['baseline_mse'] for r in inventory]);p=np.mean([r['transport_mse'] for r in inventory])
    measured={'status':'numeric_replay_passed_visual_review_pending','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'selection_registration_sha256':sha(REG),'spatial_registration_sha256':sha(SPATIAL_REG),'sources':sources,
        'arrays_sha256':sha(output/'replay_arrays.npz'),'cases':cases,'population':{'episodes':len(inventory),
            'sessions':len({r['session_id'] for r in inventory}),'positive_episodes':int((gains>0).sum()),
            'negative_episodes':int((gains<0).sum()),'tied_episodes':int((gains==0).sum()),
            'median_episode_gain_percent':float(np.median(gains)),'pooled_relative_reduction_percent':float(100*(b-p)/b)},
        'inventory':inventory,'scope':'original validation development; gain-conditioned examples; no RGB forecasts or physical flow',
        'semantics':'Transport matrices mix observed anchor features; their visualization does not establish physical correspondence or causal benefit.',
        'publication':'candidate only; actual pixel review required before manuscript integration'}
    atomic_json(measured,output/'replay.json');print(json.dumps({'status':measured['status'],'candidate':str(output),'cases':3,'replays':18}))
    return measured


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args();replay(args.output)
