#!/usr/bin/env python3
"""Exact frozen scorer allocation with synthetic-zero targets, never accuracy.

The only real data are observed features and supplied commands streamed by the
reviewed v2 input loader. All metric arrays computed against zeros are discarded.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_input_only_v2',HERE/'probe_v2.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
ROOT=probe.ROOT


def score_synthetic(model, initial, commands, offset, rowwise):
    if torch.is_inference_mode_enabled() or any(p.is_inference() for p in model.parameters()):
        raise ValueError('Model must be loaded outside inference mode')
    context=probe.rowwise_gru_context(model) if rowwise else nullcontext(model)
    with context,torch.inference_mode(),torch.autocast(device_type='cpu',enabled=False):
        batch={'initial_features':torch.from_numpy(initial.numpy()),
               'commands':torch.from_numpy(commands.numpy()),
               'targets':torch.zeros((len(initial),59,6144),dtype=torch.float32)}
        try:
            arrays,invocations=probe.evaluation.score_batch(model,batch,offset)
            if set(arrays)!=set(probe.evaluation.metric_keys()):raise ValueError('Scorer returned wrong metric roster')
            if any(not np.isfinite(value).all() for value in arrays.values()):raise ValueError('Synthetic scorer became nonfinite')
            del arrays
            prefixes=[v for v in invocations if v['command_rows']!=60]
            return {'status':'passed','maximum_prefix_absolute_difference':max(v['maximum_absolute_difference'] for v in prefixes),
                    'maximum_prefix_tolerance_ratio':max(v['maximum_tolerance_ratio'] for v in prefixes),
                    'prefix_horizons':[v['command_rows'] for v in prefixes],
                    'full_prediction_sha256':invocations[0]['prediction_sha256'],
                    'all_synthetic_metric_values_discarded':True}
        except ValueError as error:
            return {'status':'failed','exception_type':type(error).__name__,'reason':str(error),
                    'all_synthetic_metric_values_discarded':True}


def run(output):
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('CPU Slurm allocation required')
    output=Path(output)
    if output.exists():raise ValueError('Refusing to overwrite diagnostic evidence')
    registry=probe.checked_registration(ROOT)
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dependencies={str(p.relative_to(ROOT)):probe.sha(p) for p in
                  (Path(__file__),HERE/'probe_v2.py',probe.HELPER,probe.EVALUATOR,ROOT/'scripts/real_video_iws/evaluate.py')}
    cache=probe.ReservedFeatureCache.metadata_only(ROOT,'bimanual_box')
    started=time.monotonic();rows=[];models=[]
    for seed in probe.SEEDS:
        identity=next(r for r in registry['runs'] if (r['task'],r['mode'],r['seed'])==('bimanual_box','autoregressive',seed))
        model,_=probe.evaluation.load_selected(ROOT,identity)
        before=probe.state_sha(model);keys=list(model.state_dict());counts=model.parameter_counts
        for offset in range(0,200,64):
            initial,commands,scope=probe.reserved_inputs(cache,cache.handles[offset:offset+64])
            row={'run':identity['name'],'first_handle':offset,'count':len(initial),
                 'initial_features_sha256':probe.values_sha(initial.numpy()),'commands_sha256':probe.values_sha(commands.numpy()),
                 'input_scope':scope,'targets_are_synthetic_zeros':True,
                 'native':score_synthetic(model,initial,commands,offset,False),
                 'rowwise':score_synthetic(model,initial,commands,offset,True)}
            rows.append(row)
            print(json.dumps({'run':identity['name'],'first_handle':offset,'native':row['native']['status'],'rowwise':row['rowwise']['status']}),flush=True)
        after=probe.state_sha(model)
        if before!=after or keys!=list(model.state_dict()) or counts!=model.parameter_counts:raise ValueError('Scorer probe changed model state')
        models.append({'run':identity['name'],'checkpoint_sha256':identity['checkpoint_sha256'],
                       'state_dict_before_sha256':before,'state_dict_after_sha256':after,'state_keys_unchanged':True,'parameter_counts':counts})
    if probe.checked_registration(ROOT)!=registry or any(probe.sha(ROOT/p)!=h for p,h in dependencies.items()):raise ValueError('Frozen source changed during diagnosis')
    report={'schema':'reserved_synthetic_scorer_diagnostic_20260920_v3','status':'completed',
            'completed_utc':datetime.now(timezone.utc).isoformat(),'slurm_job_id':os.environ['SLURM_JOB_ID'],'node':os.uname().nodename,
            'device':'cpu','dtype':'float32','threads':8,'batch_size':64,'affinity':sorted(os.sched_getaffinity(0)),
            'model_load_context':'outside_inference_mode','loader':'frozen_evaluator_load_selected',
            'scorer':'unchanged_frozen_score_batch','targets_are_synthetic_zeros':True,'future_targets_read':False,
            'synthetic_metric_values_saved_or_used_as_accuracy':False,'tolerance_changed':False,
            'registration_sha256':probe.sha(ROOT/'configs/real_video_iws_reserved_v1/registration.json'),
            'source_dependencies':dependencies,'models':models,'results':rows,'elapsed_seconds':time.monotonic()-started,
            'interpretation':'Allocation-path diagnostic only. Synthetic-zero losses are not forecast accuracy and are discarded. Original failure logs and both earlier probes remain unchanged.'}
    output.parent.mkdir(parents=True,exist_ok=True);temp=output.with_suffix('.tmp')
    temp.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n');temp.replace(output)
    print(json.dumps({'status':'completed','output':str(output),'elapsed_seconds':report['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
