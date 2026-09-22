"""Fixed stride-five native reconstruction pairs, internal partitions only."""
from collections import OrderedDict
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset

from rgb_core import ROOT, REPORT, REG, RECIPE, sha, read, require, atomic_json, internal_cache, checked_registration, target_pixels
from shiftwm.real_video_iws.data import decode_native_rgb


def verify_episode(path, record, reg_sha):
    path = Path(path); receipt = read(path / 'receipt.json')
    require(not path.is_symlink() and all(not (path/n).is_symlink() for n in ('receipt.json','features.npy','targets.npy','native_indices.npy')), 'RGB pair symlinks forbidden')
    require(receipt['registration_sha256'] == reg_sha and receipt['record'] == record, 'RGB pair identity changed')
    require(set(receipt['files']) == {'features.npy', 'targets.npy', 'native_indices.npy'}, 'Wrong RGB pair files')
    for name, expected in receipt['files'].items():
        require(sha(path / name) == expected, 'RGB pair payload changed: ' + name)
    indices = np.load(path / 'native_indices.npy', allow_pickle=False)
    require(indices.dtype == np.int64 and indices.tolist() == record['selected_indices'], 'RGB native indices changed')
    features = np.load(path / 'features.npy', mmap_mode='r', allow_pickle=False)
    targets = np.load(path / 'targets.npy', mmap_mode='r', allow_pickle=False)
    require(features.dtype == np.float32 and features.shape == (len(indices),6144) and np.isfinite(features).all(), 'Bad RGB pair features')
    require(targets.dtype == np.uint8 and targets.shape == (len(indices),224,224,3), 'Bad RGB pair targets')
    return receipt


def verify_complete_targets(task, reg):
    output=REPORT/'targets'/task;manifest=read(output/'manifest.json')
    require(manifest['status']=='complete' and manifest['registration_sha256']==sha(REG), 'RGB cache incomplete/stale')
    records=reg['tasks'][task]['records']
    require([r['episode_id'] for r in manifest['episodes']]==[r['episode_id'] for r in records], 'RGB cache population differs')
    for record,row in zip(records,manifest['episodes']):
        path=output/'episodes'/record['episode_id']
        receipt=verify_episode(path,record,sha(REG))
        require(sha(path/'receipt.json')==row['receipt_sha256'] and receipt['files']==row['files'],'RGB manifest changed')


def prepare(task):
    reg = checked_registration(); require(task in reg['tasks'], 'Unknown task')
    cache = internal_cache(task); reg_sha = sha(REG)
    output = REPORT / 'targets' / task; output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = reg['tasks'][task]['records']; rows = []
        for i, record in enumerate(records):
            eid = record['episode_id']; split = record['split']
            cache.inventory.authorize(eid, split)  # Deny before any payload open.
            final = output / 'episodes' / eid
            if not final.exists():
                receipt, arrays = cache.episode(eid, split)
                require(receipt['payload_sha256'] == record['feature_payload_sha256'], 'Feature source changed')
                rgb = decode_native_rgb(cache.inventory, eid, record['video_sha256'], split)
                require(hashlib.sha256(rgb.tobytes()).hexdigest() == record['decoded_rgb_sha256'], 'Decoded full RGB differs from frozen feature extraction')
                indices = np.asarray(record['selected_indices'], dtype=np.int64)
                targets = target_pixels(rgb[indices]); features = arrays['features'][indices]
                final.parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(tempfile.mkdtemp(prefix='.' + eid, dir=final.parent))
                try:
                    for name, value in [('features.npy', features), ('targets.npy', targets), ('native_indices.npy', indices)]:
                        with (temporary / name).open('xb') as stream:
                            np.save(stream, value, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
                    payload = {'schema': 'iws_rgb_native_pairs_v1', 'registration_sha256': reg_sha, 'task': task,
                               'record': record, 'files': {n: sha(temporary/n) for n in ('features.npy','targets.npy','native_indices.npy')}}
                    atomic_json(payload, temporary / 'receipt.json')
                    verify_episode(temporary, record, reg_sha)
                    os.rename(temporary, final)
                finally:
                    if temporary.exists(): shutil.rmtree(temporary)
            receipt = verify_episode(final, record, reg_sha)
            rows.append({'episode_id': eid, 'receipt_sha256': sha(final / 'receipt.json'), 'files': receipt['files']})
            if i % 25 == 0: print({'stage': 'rgb_pairs', 'task': task, 'episodes': i+1, 'total': len(records)}, flush=True)
        checked_registration()
        manifest = {'schema': 'iws_rgb_target_cache_v1', 'status': 'complete', 'task': task,
                    'registration_sha256': reg_sha, 'counts': reg['tasks'][task]['counts'], 'episodes': rows,
                    'reserved_payloads_opened': 0, 'target_recipe': RECIPE['target']}
        if (output / 'manifest.json').exists(): require(read(output / 'manifest.json') == manifest, 'Completed RGB cache changed')
        else: atomic_json(manifest, output / 'manifest.json')
        return manifest


class RGBPairs(Dataset):
    def __init__(self, task, split, reg, *, first_train_frames=None):
        require(split in ('internal_train', 'internal_development'), 'Reserved RGB decoder data forbidden')
        require(task in reg['tasks'], 'Unknown task')
        require(first_train_frames is None or split == 'internal_train', 'Resource profile cannot open development')
        self.output = REPORT / 'targets' / task
        manifest = read(self.output / 'manifest.json')
        require(manifest['status'] == 'complete' and manifest['registration_sha256'] == sha(REG), 'RGB cache incomplete/stale')
        expected = {r['episode_id']: r for r in manifest['episodes']}
        require(len(expected) == len(manifest['episodes']) == len(reg['tasks'][task]['records']), 'RGB cache population differs')
        self.records = []; features = []; self.lookup = []; self.maps = OrderedDict()
        self.episode_ids = []
        for record in reg['tasks'][task]['records']:
            if record['split'] != split: continue
            eid = record['episode_id']; path = self.output / 'episodes' / eid
            require(sha(path/'receipt.json') == expected[eid]['receipt_sha256'], 'RGB receipt changed')
            receipt = verify_episode(path, record, sha(REG))
            require(receipt['files'] == expected[eid]['files'], 'Manifest payload differs')
            values = np.load(path/'features.npy', allow_pickle=False)
            count = len(values)
            if first_train_frames is not None: count = min(count, first_train_frames - len(self.lookup))
            index = len(self.episode_ids); self.episode_ids.append(eid)
            features.append(values[:count]); self.records.append(record)
            self.lookup.extend((index, j) for j in range(count))
            if first_train_frames is not None and len(self.lookup) == first_train_frames: break
        require(self.lookup, 'Empty RGB dataset')
        self.features = np.concatenate(features)
        self.split = split

    def __len__(self): return len(self.lookup)

    def __getitem__(self, index):
        episode, local = self.lookup[index]; eid = self.episode_ids[episode]
        if eid not in self.maps:
            if len(self.maps) >= 16: self.maps.popitem(last=False)
            self.maps[eid] = np.load(self.output/'episodes'/eid/'targets.npy', mmap_mode='r', allow_pickle=False)
        self.maps.move_to_end(eid)
        pixels = torch.from_numpy(self.maps[eid][local].copy()).permute(2,0,1).float()/255
        return torch.from_numpy(self.features[index]), pixels, episode
