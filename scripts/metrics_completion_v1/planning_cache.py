"""Extract actual frozen DINOv2 4x4 tokens from authorized training/development RGB."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import socket
import time
import numpy as np
from planning_common import CACHE, DATA, REG, REPORT, ROOT, array_sha, atomic_json, checked_registration, digest, read, require, sha
from planning_data import grouped_rows, load_cache, statistics


def prepare(task):
    reg = checked_registration(); require(task in reg['datasets'], 'Unknown task')
    alignment=read(REPORT/f'{task}_training_alignment.json')
    require(alignment.get('status')=='passed' and alignment.get('registration_sha256')==sha(REG),'Training-only simulator/action alignment audit missing')
    import h5py
    import hdf5plugin  # noqa: F401
    import torch
    from shiftwm.real_video_iws.features import DinoSpatialEncoder, PREPROCESSING
    torch.set_num_threads(8)
    require(torch.cuda.is_available(), 'Allocated CUDA GPU required; no CPU fallback')
    out = CACHE/task; out.mkdir(parents=True, exist_ok=True)
    with (out/'.cache.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reg_sha = sha(REG)
        identity = {'schema': 'current_spatial_simulator_cache_v1', 'task': task, 'registration_sha256': reg_sha,
                    'preprocessing': PREPROCESSING, 'encoder_provenance_sha256': sha(ROOT/'data/pretrained/dinov2-small/provenance.json'),
                    'precision': 'BF16 frozen encoder; FP32 4x4 pooling/storage',
                    'action_alignment': 'pixels at native rows0,5,...; action[i:i+5] produces next observed frame; flattened time-major',
                    'test_payloads_read': False}
        if (out/'identity.json').exists():
            require(read(out/'identity.json') == identity, 'Cache identity differs')
        else:
            atomic_json(identity, out/'identity.json')
        if (out/'manifest.json').exists():
            for split in ('train','development'):
                load_cache(task,split,reg_sha)
            require(sha(out/'training_statistics.json') == read(out/'manifest.json')['training_statistics_sha256'], 'Statistics changed')
            return read(out/'manifest.json')
        encoder = DinoSpatialEncoder(ROOT, ROOT/'data/pretrained/dinov2-small','cuda')
        began=time.monotonic(); records=[]
        with h5py.File(ROOT/DATA[task],'r') as source:
            for split in ('train','development'):
                for row in reg['datasets'][task]['partitions'][split]:
                    name=f'{split}_{row["episode_id"]:06d}'
                    destination=out/'episodes'/(name+'.npz'); receipt=destination.with_suffix('.json')
                    meta={'episode_id':row['episode_id'],'split':split,'source_offset':row['offset'],'source_length':row['length'],
                          'file':str(destination.relative_to(out)),'identity_sha256':sha(out/'identity.json')}
                    if destination.exists() or receipt.exists():
                        require(destination.is_file() and receipt.is_file(), 'Partial cache transaction; inspect before retry')
                        prior=read(receipt)
                        require(all(prior.get(k)==v for k,v in meta.items()) and sha(destination)==prior['sha256'],'Stale cached episode')
                        records.append(prior); continue
                    local, rows=grouped_rows(row)
                    pixels=np.asarray(source['pixels'][rows],dtype=np.uint8)
                    # Final source action may be an unused NaN terminal sentinel.
                    native=np.asarray(source['action'][row['offset']:int(rows[-1])],dtype=np.float32)
                    require(native.shape==(int(local[-1]),2) and np.isfinite(native).all(),'Incomplete/nonfinite executed native actions')
                    actions=native.reshape(-1,10)
                    features=encoder(pixels,batch_size=64)
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    temp=destination.with_suffix('.npz.tmp')
                    with temp.open('wb') as stream:
                        np.savez(stream,features=features,actions=actions,frame_indices=local)
                        stream.flush();os.fsync(stream.fileno())
                    os.replace(temp,destination)
                    row_meta={**meta,'frames':len(features),'sha256':sha(destination),'rgb_sha256':array_sha(pixels),
                              'native_actions_sha256':array_sha(native),'features_sha256':array_sha(features)}
                    atomic_json(row_meta,receipt); records.append(row_meta)
                    if len(records)%25==0:
                        print(json.dumps({'task':task,'encoded_episodes':len(records),'total':1100,'elapsed_seconds':time.monotonic()-began}),flush=True)
        # Compute normalization exclusively from the complete training partition.
        training=[]
        for r in records:
            if r['split']=='train':
                with np.load(out/r['file'],allow_pickle=False) as arrays:
                    training.append({'features':arrays['features'],'actions':arrays['actions']})
        stats=statistics(training); del training
        stats['training_episode_ids']=[r['episode_id'] for r in records if r['split']=='train']
        atomic_json(stats,out/'training_statistics.json')
        checked_registration()
        result={**identity,'status':'complete','identity_sha256':sha(out/'identity.json'),'episodes':records,
                'training_statistics_sha256':sha(out/'training_statistics.json'),
                'elapsed_seconds':time.monotonic()-began,'hostname':socket.gethostname(),'gpu':torch.cuda.get_device_name(),
                'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()}
        atomic_json(result,out/'manifest.json'); return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--task',choices=('pusht','reacher'),required=True)
    args=parser.parse_args(); result=prepare(args.task)
    print(json.dumps({'status':result['status'],'task':args.task,'episodes':len(result['episodes'])}),flush=True)
