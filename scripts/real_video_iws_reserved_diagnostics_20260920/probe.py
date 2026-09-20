#!/usr/bin/env python3
"""Input-only numerical prefix diagnosis; no targets, feature errors, or scoring.

Uses fixed Box training fixture first, then all exact reserved input handles for
AR seeds1/2. Reads only requested initial rows from compressed feature NPY streams;
future feature arrays are never materialized. Hashing cache bytes is provenance,
not target decoding. Existing source, weights, evaluation gates and logs stay fixed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from shiftwm.real_video_iws_reserved.protocol import checked_registration
from shiftwm.real_video_iws_reserved.cache import ReservedFeatureCache
from shiftwm.real_video_iws import training
from shiftwm.real_video_iws_reserved_recovery.prefix_backend import rowwise_gru_context, VERSION

PREFIXES = (15,30,45)
SEEDS = (1,2)
RTOL, ATOL = 1e-5, 2e-5
HELPER = ROOT/'src/shiftwm/real_video_iws_reserved_recovery/prefix_backend.py'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def values_sha(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def state_sha(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rows_npz(path,key,start,count,shape):
    """Read only selected rows; never construct the full reserved feature matrix."""
    with zipfile.ZipFile(path) as archive, archive.open(key+'.npy') as stream:
        version=np.lib.format.read_magic(stream)
        if version==(1,0):actual,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
        elif version==(2,0):actual,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
        else:raise ValueError('Unsupported NPY version')
        if tuple(actual)!=tuple(shape) or fortran or dtype!=np.dtype('float32') or start<0 or count<1 or start+count>shape[0]:
            raise ValueError('Unexpected input array layout/index')
        width=int(np.prod(shape[1:]));stream.seek(start*width*4,1)
        payload=stream.read(count*width*4)
        if len(payload)!=count*width*4:raise ValueError('Incomplete selected input rows')
        value=np.frombuffer(payload,dtype=dtype).copy().reshape(count,*shape[1:])
        if not np.isfinite(value).all():raise ValueError('Nonfinite diagnostic input')
        return value


def discrepancy(value,reference):
    if value.shape!=reference.shape or not torch.isfinite(value).all() or not torch.isfinite(reference).all():raise ValueError('Invalid prefix comparison')
    difference=(value-reference).abs(); tolerance=ATOL+RTOL*reference.abs(); ratio=difference/tolerance
    failed=ratio>1; coordinates=torch.nonzero(failed,as_tuple=False)
    flat=int(ratio.reshape(-1).argmax()); coord=list(np.unravel_index(flat,ratio.shape))
    return {'maximum_absolute_difference':float(difference.max()),'maximum_tolerance_ratio':float(ratio.max()),
            'maximum_reference_absolute_value':float(reference.abs().max()),'allclose':bool(torch.allclose(value,reference,rtol=RTOL,atol=ATOL)),
            'bitwise_equal':bool(torch.equal(value,reference)),'failed_coordinates':int(failed.sum()),
            'maximum_ratio_coordinate':[int(v) for v in coord],
            'first_failed_coordinates':coordinates[:8].tolist(),
            'rtol':RTOL,'atol':ATOL}


def gru_difference(value,reference):
    result=discrepancy(value,reference)
    different=(value!=reference).any(dim=0).any(dim=-1)
    positions=torch.nonzero(different,as_tuple=False)
    result['first_bitwise_different_command_row']=None if not len(positions) else int(positions[0,0])
    return result


def diagnose(model,initial,commands):
    result={'initial_sha256':values_sha(initial.numpy()),'commands_sha256':values_sha(commands.numpy()),
            'batch_size':len(initial),'feature_shape':list(initial.shape),'command_shape':list(commands.shape),'prefixes':{}}
    original_gru=model.action_prefix
    full_commands=model.normalize_commands(commands)
    full_states=original_gru(full_commands)[0]
    full_prediction=model.predict(initial,commands)
    for horizon in PREFIXES:
        raw_slice=commands[:,:horizon];normalized=model.normalize_commands(raw_slice)
        prefix=model.predict(initial,raw_slice)
        contiguous=model.predict(initial,raw_slice.contiguous())
        row={'normalized_commands_match_full_slice':bool(torch.equal(normalized,full_commands[:,:horizon])),
             'gru_full_vs_separately_normalized_prefix':gru_difference(original_gru(normalized)[0],full_states[:,:horizon]),
             'gru_full_vs_noncontiguous_slice':gru_difference(original_gru(full_commands[:,:horizon])[0],full_states[:,:horizon]),
             'gru_full_vs_contiguous_slice':gru_difference(original_gru(full_commands[:,:horizon].contiguous())[0],full_states[:,:horizon]),
             'native_endpoint':discrepancy(prefix[:,-1:],full_prediction[:,horizon-2:horizon-1]),
             'native_all_prefix_outputs':discrepancy(prefix,full_prediction[:,:horizon-1]),
             'contiguous_endpoint':discrepancy(contiguous[:,-1:],full_prediction[:,horizon-2:horizon-1]),
             'sliced_vs_contiguous_endpoint':discrepancy(prefix[:,-1:],contiguous[:,-1:])}
        result['prefixes'][str(horizon)]=row
    with rowwise_gru_context(model):
        stable_states=model.action_prefix(full_commands)[0]
        stable_prediction=model.predict(initial,commands)
        result['rowwise_vs_native_H60']=discrepancy(stable_prediction,full_prediction)
        for horizon in PREFIXES:
            stable_prefix=model.predict(initial,commands[:,:horizon])
            result['prefixes'][str(horizon)].update(
                rowwise_gru=gru_difference(model.action_prefix(model.normalize_commands(commands[:,:horizon]))[0],stable_states[:,:horizon]),
                rowwise_endpoint=discrepancy(stable_prefix[:,-1:],stable_prediction[:,horizon-2:horizon-1]),
                rowwise_all_prefix_outputs=discrepancy(stable_prefix,stable_prediction[:,:horizon-1]))
    return result


def training_inputs(batch_size):
    bundle=ROOT/'artifacts/releases/iws_single_observation_local_v1'
    manifest=json.loads((bundle/'manifest.json').read_text());path=bundle/'fixtures/bimanual_box.npz'
    if sha(path)!=manifest['files']['fixtures/bimanual_box.npz']:raise ValueError('Training-only fixture changed')
    metadata=manifest['training_inputs']['bimanual_box']
    if metadata['split']!='internal_train' or metadata['episode_id']!='000011' or metadata['targets_included'] is not False:raise ValueError('Unexpected fixture scope')
    with np.load(path,allow_pickle=False) as loaded:
        if set(loaded.files)!={'initial_features','commands'}:raise ValueError('Non-input fixture')
        z,a=loaded['initial_features'],loaded['commands']
    if z.shape!=(1,6144) or a.shape!=(1,60,14):raise ValueError('Wrong training fixture dimensions')
    return torch.from_numpy(np.repeat(z,batch_size,axis=0)),torch.from_numpy(np.repeat(a,batch_size,axis=0)),{
        'kind':'fixed_training_input_replicated_to_registered_batch_shape','episode_id':'000011','frame':0,
        'fixture_sha256':sha(path),'fixture_manifest_sha256':sha(bundle/'manifest.json'),
        'replications':batch_size,'replicas_are_not_independent_examples':True}


def reserved_inputs(cache,handles):
    z,a=[],[];bindings={}
    for handle in handles:
        eid,start=handle['episode_id'],handle['start']
        record=cache.index[eid];n=record['frames']
        path=cache.output/'episodes'/eid/'arrays.npz'
        if eid not in bindings:
            cache.inventory.authorize(eid)
            if sha(path)!=record['payload_sha256']:raise ValueError('Reserved input package changed')
            bindings[eid]=record['payload_sha256']
        if start+60>=n:raise ValueError('Ineligible original handle')
        z.append(rows_npz(path,'features',start,1,(n,6144))[0])
        a.append(rows_npz(path,'commands',start,60,(n,14)))
    for eid,expected in bindings.items():
        if sha(cache.output/'episodes'/eid/'arrays.npz')!=expected:raise ValueError('Reserved input package changed during reading')
    return torch.from_numpy(np.stack(z)),torch.from_numpy(np.stack(a)),{
        'kind':'exact_reserved_observed_feature_and_supplied_commands_only','handles':handles,'cache_payload_sha256':bindings,
        'feature_rows_materialized_per_handle':1,'command_rows_materialized_per_handle':60,'target_arrays_created':False}


def run(output,threads=8,batch_size=64):
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('Diagnostic inference requires a CPU Slurm allocation')
    if threads not in (2,8) or batch_size!=64:raise ValueError('Use declared CPU threads2/8 and registered batch64')
    output=Path(output)
    if output.exists():raise ValueError('Refusing to overwrite a diagnostic report')
    registry=checked_registration(ROOT)
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source_hashes={'probe':sha(__file__),'prefix_backend':sha(HELPER)}
    fixture_root=ROOT/'artifacts/releases/iws_single_observation_local_v1'
    fixture_bindings={name:sha(fixture_root/name) for name in ('manifest.json','fixtures/bimanual_box.npz')}
    cache=None;results=[];identities=[];started=time.monotonic()
    with torch.inference_mode(),torch.autocast(device_type='cpu',enabled=False):
        for seed in SEEDS:
            row=next(r for r in registry['runs'] if (r['task'],r['mode'],r['seed'])==('bimanual_box','autoregressive',seed))
            package=ROOT/row['package_path']
            if sha(package/'model.pt')!=row['checkpoint_sha256']:raise ValueError('Selected weight identity differs')
            model,state=training.load_package(package,'cpu')
            model.eval()
            recipe=state['config']['metadata']['identity']['scientific_config']
            if any(recipe[k]!=row[k] for k in ('task','mode','seed')) or state['epoch']!=row['selected_epoch']:raise ValueError('Selected recipe differs')
            before=state_sha(model);keys=list(model.state_dict());counts=model.parameter_counts
            z,a,scope=training_inputs(batch_size)
            results.append({'run':row['name'],'input_scope':scope,**diagnose(model,z,a)})
            print(json.dumps({'run':row['name'],'phase':'training_input_checked'}),flush=True)
            if cache is None:cache=ReservedFeatureCache.metadata_only(ROOT,'bimanual_box')
            for offset in range(0,200,batch_size):
                z,a,scope=reserved_inputs(cache,cache.handles[offset:offset+batch_size])
                result={'run':row['name'],'first_handle':offset,'input_scope':scope,**diagnose(model,z,a)}
                results.append(result)
                print(json.dumps({'run':row['name'],'phase':'reserved_inputs_checked','first_handle':offset,
                                  'native_pass':all(x['native_endpoint']['allclose'] for x in result['prefixes'].values()),
                                  'rowwise_pass':all(x['rowwise_endpoint']['allclose'] for x in result['prefixes'].values())}),flush=True)
            after=state_sha(model)
            if before!=after or keys!=list(model.state_dict()) or counts!=model.parameter_counts:raise ValueError('Probe changed model state/parameters')
            identities.append({'run':row['name'],'checkpoint_sha256':row['checkpoint_sha256'],'state_dict_before_sha256':before,
                               'state_dict_after_sha256':after,'state_keys_unchanged':True,'parameter_counts':counts})
    if checked_registration(ROOT)!=registry or source_hashes!={'probe':sha(__file__),'prefix_backend':sha(HELPER)}:raise ValueError('Sources changed during probe')
    if fixture_bindings!={name:sha(fixture_root/name) for name in fixture_bindings}:raise ValueError('Training fixture changed during probe')
    document={'schema':'reserved_prefix_input_only_diagnostic_20260920','status':'completed','completed_utc':datetime.now(timezone.utc).isoformat(),
              'slurm_job_id':os.environ['SLURM_JOB_ID'],'node':os.uname().nodename,'threads':threads,'batch_size':batch_size,
              'device':'cpu','dtype':'float32','torch':torch.__version__,'numpy':np.__version__,'source_sha256':source_hashes,
              'training_fixture_bindings_before_after':fixture_bindings,
              'registration_sha256':sha(ROOT/'configs/real_video_iws_reserved_v1/registration.json'),'backend_candidate':VERSION,
              'elapsed_seconds':time.monotonic()-started,'models':identities,'results':results,
              'future_targets_scored_or_materialized':False,'feature_errors_called':False,'tolerance_changed':False,
              'scientific_evaluation_or_checkpoint_files_modified':False,
              'interpretation':'Input-only numerical diagnosis. Differences are between executions, not forecast errors or reserved accuracy. A passing candidate is not an authorization to mix backends or retain selected result rows.'}
    output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_suffix('.tmp')
    temporary.write_text(json.dumps(document,sort_keys=True,indent=2,allow_nan=False)+'\n');temporary.replace(output)
    print(json.dumps({'status':'completed','output':str(output),'elapsed_seconds':document['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threads',type=int,choices=(2,8),default=8);parser.add_argument('--batch-size',type=int,default=64)
    args=parser.parse_args();run(args.output,args.threads,args.batch_size)
