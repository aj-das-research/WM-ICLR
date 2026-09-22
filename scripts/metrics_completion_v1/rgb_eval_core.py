"""Gates for post-hoc common-decoder RGB evaluation. Training recipe untouched."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from rgb_core import ROOT, REPORT as TRAIN_REPORT, REG as TRAIN_REG, RGBDecoder, sha, read, atomic_json, module, require
from rgb_eval_metrics import METRICS, DIRECTION, CONVENTIONS

REPORT=ROOT/'reports/metrics_completion_v1/rgb_evaluation'
REG=REPORT/'registration.json'
ACTIVATION=REPORT/'decoder_bindings.json'
TASKS=('pusht','bimanual_box','bimanual_rope')
LEARNED=('autoregressive','anchored_additive','bounded_spatial_mix','unbounded_spatial_mix')
REFERENCES=('feature_persistence','gt_feature_reconstruction','rgb_persistence')
RECOVERY_REG=ROOT/'configs/real_video_iws_reserved_recovery_v2/registration.json'
RECOVERY_REPORT=ROOT/'reports/real_video_iws_reserved_recovery_v2'
POLICY={'schema':'posthoc_shared_decoder_iws_rgb_v1','tasks':list(TASKS),'learned_methods':list(LEARNED),
 'references':list(REFERENCES),'predictor_seeds':[0,1,2],'decoder_seed':173,
 'handles_per_task':200,'trajectories_per_task':10,'offsets':list(range(1,60)),
 'forecast':{'device':'cpu','dtype':'float32','threads':8,'interop_threads':1,'batch_size':64,'command_gru_dispatch':'one_native_row_per_call','autocast':False},
 'rgb':{'device':'cuda','dtype':'float32','batch_size':16,'autocast':False,'tf32':False},
 'metrics':list(METRICS),'directions':DIRECTION,'conventions':CONVENTIONS,
 'endpoint':'stored target offset59 of60command rows; index58; not temporal mean',
 'aggregation':'all59 per-frame metrics; equal handles within trajectory, equal10trajectories, equal3matched predictor seeds; separate equal-handle view',
 'intervals':{'draws':10000,'seed':173,'method':'paired crossed predictor-seed and trajectory percentile bootstrap, shared seeds across tasks; conditional on one task decoder; posthoc unadjusted95%'},
 'scope':'posthoc auxiliary RGB measurement after feature-space reserved outcomes; primary endpoints and selected feature predictors unchanged',
 'qualitative':'fixed previous feature-selected largest/middle/smallest trajectory cases and earliest registered window; no RGB outcome reselection; all3seeds retained',
 'fid_fvd':'not measured; separate official weights/preprocessing/sample and clip protocol required; no proxies'}


def recovery():
    return module(ROOT/'scripts/real_video_iws_reserved_recovery_v2/evaluate.py','_rgb_eval_frozen_recovery')


def configure():
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def checked_registration(require_review=True):
    value=read(REG)
    require(value['schema']=='posthoc_rgb_evaluation_registration_v1' and value['policy']==POLICY,'RGB evaluation registration/recipe changed')
    if require_review:
        review=read(REPORT/'source_review.json')
        require(review.get('status')=='passed' and review.get('registration_sha256')==sha(REG),'RGB evaluation source review missing/stale')
    for path,expected in value['dependencies'].items():
        p=Path(path);require(not p.is_absolute() and '..' not in p.parts,'Unsafe RGB evaluation source path')
        require(sha(ROOT/p)==expected,'RGB evaluation dependency changed: '+path)
    require(value['environment']['torch']==str(torch.__version__) and value['environment']['numpy']==np.__version__,'Evaluation numerical environment changed')
    return value


def activate():
    reg=checked_registration()
    training=module(ROOT/'scripts/metrics_completion_v1/rgb_finalize.py','_rgb_eval_training_gate')
    result=training.finalize(if_ready=True)
    if result['status']=='pending':return result
    source=TRAIN_REPORT/'decoder_finalization.json';document=read(source)
    require(document['status']=='three_decoders_complete' and document['registration_sha256']==sha(TRAIN_REG),'Wrong decoder finalization')
    rows={}
    for task in TASKS:
        directory=TRAIN_REPORT/'runs'/f'{task}_s173'
        rows[task]={'package':str((directory/'selected.pt').relative_to(ROOT)),
                    'sha256':sha(directory/'selected.pt'),'summary_sha256':sha(directory/'training_summary.json')}
    binding={'schema':'shared_rgb_fixed_decoder_bindings_v1','status':'all3_decoders_fixed_before_rgb_forecast_scoring',
             'registration_sha256':sha(REG),'training_finalization_sha256':sha(source),'decoders':rows}
    if ACTIVATION.exists():require(read(ACTIVATION)==binding,'Decoder selection changed after RGB activation')
    else:atomic_json(binding,ACTIVATION)
    return binding


def gate():
    reg=checked_registration();binding=read(ACTIVATION)
    require(binding['registration_sha256']==sha(REG) and set(binding['decoders'])==set(TASKS),'Incomplete decoder activation')
    require(binding['training_finalization_sha256']==sha(TRAIN_REPORT/'decoder_finalization.json'),'Decoder finalization changed')
    for task,row in binding['decoders'].items():
        require(sha(ROOT/row['package'])==row['sha256'],'Selected shared decoder changed')
    return reg,binding


def decoder(task,binding,device):
    row=binding['decoders'][task];package=torch.load(ROOT/row['package'],map_location='cpu',weights_only=False)
    require(package['kind']=='iws_shared_rgb_decoder_v1' and package['epochs']==30 and package['identity']['task']==task and package['identity']['seed']==173,'Wrong shared decoder package')
    stats=read(ROOT/f'data/features/iws_{task}_spatial_v1/training_statistics.json')
    model=RGBDecoder(stats);model.load_state_dict(package['model'],strict=True)
    require(torch.equal(model.feature_mean,torch.tensor(stats['feature_mean'],dtype=torch.float32)) and torch.equal(model.feature_std,torch.tensor(stats['feature_std'],dtype=torch.float32)),'Decoder training normalization changed')
    return model.to(device).eval().requires_grad_(False)


def allocated_gpu():
    require(bool(os.environ.get('SLURM_JOB_ID')) and torch.cuda.is_available(),'RGB/LPIPS GPU scoring requires scheduler allocation')
    return {'job':os.environ['SLURM_JOB_ID'],'device':torch.cuda.get_device_name(),'torch':str(torch.__version__),'numpy':np.__version__}
