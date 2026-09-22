"""Post-hoc RGB endpoint experiment: all reserved handles, fixed shared readouts."""
from pathlib import Path
import sys, os, json, hashlib, argparse, socket, time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/metrics_completion_v1'))
from rgb_core import RGBDecoder, target_pixels, configure_cpu, atomic_json, sha, read, require, module
from rgb_eval_metrics import METRICS, CONVENTIONS, load_lpips, frame_metrics
OUT=ROOT/'reports/iclr_review_2026-09-22/rgb_endpoint'
REG=OUT/'registration.json'
ASSETS=ROOT/'reports/metrics_completion_v1/rgb_evaluation'
TRAIN=ROOT/'reports/metrics_completion_v1/rgb'
TASKS=('pusht','bimanual_box','bimanual_rope')
RECOVERY=ROOT/'configs/real_video_iws_reserved_recovery_v2/registration.json'

def evaluator():return module(ROOT/'scripts/real_video_iws_reserved_recovery_v2/evaluate.py','_endpoint_recovery')

def register():
    training=read(TRAIN/'decoder_finalization.json');require(training['status']=='three_decoders_complete','Decoders incomplete')
    reg=read(RECOVERY);evaluator().check_grid(reg)
    decoders={t:{'path':str((TRAIN/'runs'/f'{t}_s173/selected.pt').relative_to(ROOT)),
                 'sha256':sha(TRAIN/'runs'/f'{t}_s173/selected.pt')}for t in TASKS}
    assets=read(ASSETS/'assets/manifest.json')
    sources={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
             **{str(Path('scripts/metrics_completion_v1')/n):sha(ROOT/'scripts/metrics_completion_v1'/n)for n in ['rgb_core.py','rgb_eval_metrics.py']},
             str(RECOVERY.relative_to(ROOT)):sha(RECOVERY),str((TRAIN/'decoder_finalization.json').relative_to(ROOT)):sha(TRAIN/'decoder_finalization.json'),
             **{str((ASSETS/p).relative_to(ROOT)):v for p,v in assets['files'].items()}}
    for r in reg['runs']:
        p=ROOT/'reports/real_video_iws_reserved_recovery_v2/evaluations'/(r['name']+'.npz');sources[str(p.relative_to(ROOT))]=sha(p)
    document={'schema':'posthoc_iws_rgb_endpoint_v1','status':'frozen_before_rgb_endpoint_scoring',
      'scope':'secondary RGB endpoint analysis after feature outcomes; not confirmatory; no model or decoder reselection',
      'tasks':list(TASKS),'predictor_seeds':[0,1,2],'decoder_seed':173,'handles_per_task':200,'trajectories_per_task':10,
      'target_offset':59,'command_rows':60,'forecast_batch':64,'decode_batch':16,'decoder_bindings':decoders,'sources':sources,
      'references':['rgb_persistence','feature_persistence','gt_feature_reconstruction'],
      'metrics':list(METRICS),'conventions':CONVENTIONS,'aggregation':'equal original handles within each trajectory; equal10 trajectories and three predictor seeds',
      'forecast_runtime':'existing CPU FP32 row-wise GRU,8 threads,interop1; require bitwise reproduction of original endpoint MSE',
      'rgb_runtime':'GPU FP32; TF32 off; same task readout for all methods; clamp[0,1]',
      'qualification':'single selected decoder per task; uncertainty conditional on it; ground-truth reconstruction is a reference, not a guaranteed lower bound',
      'not_measured':['all-offset video quality','FID','FVD'],
      'environment':{'torch':str(torch.__version__),'numpy':np.__version__}}
    OUT.mkdir(parents=True,exist_ok=True)
    if REG.exists():require(read(REG)==document,'Existing endpoint protocol differs')
    else:atomic_json(document,REG)
    return document

def check():
    d=read(REG)
    for p,digest in d['sources'].items():require(sha(ROOT/p)==digest,'Changed endpoint source: '+p)
    for r in d['decoder_bindings'].values():require(sha(ROOT/r['path'])==r['sha256'],'Decoder changed')
    require(d['environment']=={'torch':str(torch.__version__),'numpy':np.__version__},'Numerical environment differs')
    return d

@torch.inference_mode()
def run(task):
    doc=check();configure_cpu();require(os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available(),'Allocated GPU required')
    ev=evaluator();reg=ev.checked_registration(ROOT,RECOVERY)
    from shiftwm.real_video_iws_reserved.cache import ReservedFeatureCache
    from shiftwm.real_video_iws.data import decode_native_rgb
    cache=ReservedFeatureCache(ROOT,task,registration_path=reg['cache_registration_path'])
    stats=read(ROOT/f'data/features/iws_{task}_spatial_v1/training_statistics.json')
    decoder=RGBDecoder(stats);package=torch.load(ROOT/doc['decoder_bindings'][task]['path'],map_location='cpu',weights_only=False)
    require(package['kind']=='iws_shared_rgb_decoder_v1' and package['epochs']==30 and package['identity']['task']==task,'Wrong decoder package')
    decoder.load_state_dict(package['model'],strict=True);decoder=decoder.cuda().eval().requires_grad_(False)
    perceptual=load_lpips(ASSETS,'cuda')
    # Decode each original trajectory once; verify the complete native RGB hash.
    rgb={}
    for eid in cache.episode_ids:
        value=decode_native_rgb(cache.inventory,eid,cache.records[eid]['video_sha256'])
        require(hashlib.sha256(value.tobytes()).hexdigest()==cache.index[eid]['decoded_rgb_sha256'],'RGB decoding differs from frozen extraction')
        needed=sorted({h['start']+delta for h in cache.handles if h['episode_id']==eid for delta in (0,59)})
        rgb[eid]=dict(zip(needed,target_pixels(value[needed])));del value
    target=np.stack([rgb[h['episode_id']][h['start']+59]for h in cache.handles])
    initial_rgb=np.stack([rgb[h['episode_id']][h['start']]for h in cache.handles]);del rgb
    store={};batches=[ev.make_batch(cache,cache.handles[i:i+64],task,cache.episode_ids,store)for i in range(0,200,64)]
    episode_index=np.concatenate([b['episode_index']for b in batches]);starts=np.concatenate([b['window_start']for b in batches])
    output=OUT/task;output.mkdir(exist_ok=True);began=time.monotonic();rows=[]
    def measure(name,features=None,pixels=None):
        destination=output/(name+'.npz');receipt=output/(name+'.json')
        if receipt.exists():
            r=read(receipt);require(r['registration_sha256']==sha(REG)and r['npz_sha256']==sha(destination),'Stale RGB result');rows.append(r);return
        chunks=[]
        for i in range(0,200,16):
            predicted=decoder(features[i:i+16].cuda()).clamp(0,1)if features is not None else torch.from_numpy(pixels[i:i+16].copy()).permute(0,3,1,2).cuda().float()/255
            # Official decoder returns Nx3x224x224.
            truth=torch.from_numpy(target[i:i+16].copy()).permute(0,3,1,2).cuda().float()/255
            chunks.append(frame_metrics(predicted,truth,perceptual).cpu().numpy())
        values=np.concatenate(chunks);require(values.shape==(200,5) and np.isfinite(values).all(),'Nonfinite/incomplete RGB metrics')
        ev.atomic_npz(destination,{'metrics':values,'episode_index':episode_index,'window_start':starts})
        trajectory=np.stack([values[episode_index==j].mean(0)for j in range(10)])
        r={'name':name,'status':'complete','registration_sha256':sha(REG),'npz_sha256':sha(destination),
           'metrics':list(METRICS),'equal_trajectory_mean':trajectory.mean(0).tolist(),'trajectory_metrics':trajectory.tolist()}
        atomic_json(r,receipt);rows.append(r);print(json.dumps({'task':task,'completed':name}),flush=True)
    measure('rgb_persistence',pixels=initial_rgb)
    measure('feature_persistence',features=torch.cat([b['initial_features']for b in batches]))
    measure('gt_feature_reconstruction',features=torch.cat([b['targets'][:,-1]for b in batches]))
    for row in [r for r in reg['runs']if r['task']==task]:
        model,_=ev.load_selected(ROOT,row);predictions=[];errors=[]
        for batch in batches:
            p=model.predict(batch['initial_features'],batch['commands']);predictions.append(p[:,-1])
            errors.append(ev.feature_errors(p,batch['targets'],model.feature_std)['standardized_mse'][:,-1].double().numpy())
        with np.load(ROOT/'reports/real_video_iws_reserved_recovery_v2/evaluations'/(row['name']+'.npz'),allow_pickle=False)as original:
            require(np.array_equal(np.concatenate(errors),original['standardized_mse'][:,-1]),'Original endpoint MSE does not reproduce bitwise')
        measure(row['name'],features=torch.cat(predictions));del model
    require(len(rows)==15,'Incomplete RGB task comparison');check()
    atomic_json({'status':'complete','task':task,'registration_sha256':sha(REG),'rows':rows,'metric_names':list(METRICS),
        'gpu':torch.cuda.get_device_name(),'job_id':os.environ['SLURM_JOB_ID'],'hostname':socket.gethostname(),
        'elapsed_seconds':time.monotonic()-begun,'original_feature_mse_reproduced_exactly':True},OUT/(task+'.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');p.add_argument('--task',choices=TASKS);a=p.parse_args()
    if a.register:register();print('registered')
    else:require(a.task is not None,'Task required');run(a.task)
