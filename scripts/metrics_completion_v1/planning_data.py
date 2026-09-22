"""Episode-disjoint metadata selection and native-row action alignment."""
from __future__ import annotations
import numpy as np
import torch
from planning_common import CACHE, DATA, ROOT, array_sha, read, require, sha


def metadata_inventory(task):
    import h5py
    import hdf5plugin  # noqa: F401
    path = ROOT / DATA[task]
    with h5py.File(path, 'r') as f:
        offsets = np.asarray(f['ep_offset'][:], dtype=np.int64)
        lengths = np.asarray(f['ep_len'][:], dtype=np.int64)
        require(offsets.shape == lengths.shape and offsets.ndim == 1, 'Invalid episode metadata')
        require(np.all(lengths >= 1) and np.array_equal(offsets, np.r_[0, np.cumsum(lengths)[:-1]]), 'Noncontiguous episode inventory')
        require(int(lengths.sum()) == f['pixels'].shape[0] and f['pixels'].shape[1:] == (224, 224, 3), 'RGB shape differs')
        key = 'episode_idx' if 'episode_idx' in f else 'ep_idx'
        # IDs and time indices are metadata only; RGB/actions/states are unopened.
        ids = np.asarray(f[key][offsets], dtype=np.int64)
        require(len(set(ids.tolist())) == len(ids), 'Repeated physical episode ID')
        require(np.all(f['step_idx'][offsets] == 0), 'Episode start metadata differs')
        columns = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in f.items()}
    eligible = [{'episode_id': int(e), 'offset': int(o), 'length': int(n)}
                for e, o, n in zip(ids, offsets, lengths) if n >= 36]
    require(len(eligible) >= 1200, 'Insufficient eligible whole episodes')
    order = np.random.default_rng(173).permutation(len(eligible))[:1200]
    selected = [eligible[i] for i in order]
    parts = {}
    for split, start, end in [('train', 0, 1000), ('development', 1000, 1100), ('test', 1100, 1200)]:
        parts[split] = sorted(selected[start:end], key=lambda r: r['episode_id'])
        for row in parts[split]:
            # One outcome-independent planning start per episode, with 10 calls
            # for support and a goal 25 calls after the observed anchor.
            row['planning_start'] = int(np.random.default_rng(173 + row['episode_id']).integers(0, row['length'] - 35))
    stat = path.stat()
    return {'file': DATA[task], 'file_stat': [stat.st_size, stat.st_mtime_ns],
            'columns': columns, 'physical_episodes': len(ids), 'eligible_episodes': len(eligible),
            'metadata_sha256': {'offsets': array_sha(offsets), 'lengths': array_sha(lengths), 'ids': array_sha(ids)},
            'selection': 'seed173 uniform eligible whole-episode permutation; no RGB/action/state/outcome reads',
            'partitions': parts}


def grouped_rows(row, block=5):
    local = np.arange(0, row['length'], block, dtype=np.int64)
    return local, row['offset'] + local


def load_cache(task, split, registration):
    require(split in ('train', 'development'), 'Final test cache access prohibited in training dataset')
    root = CACHE / task
    manifest = read(root / 'manifest.json')
    require(manifest['status'] == 'complete' and manifest['registration_sha256'] == registration, 'Incomplete/stale feature cache')
    episodes = []
    for r in manifest['episodes']:
        if r['split'] != split:
            continue
        path = root / r['file']
        require(sha(path) == r['sha256'], 'Cached episode changed')
        with np.load(path, allow_pickle=False) as data:
            x, a, frames = data['features'].copy(), data['actions'].copy(), data['frame_indices'].copy()
        require(x.shape == (len(frames), 6144) and a.shape == (len(frames)-1, 10), 'Feature/action dimensions differ')
        require(np.array_equal(frames, np.arange(len(x))*5), 'Native frame indexing differs')
        require(x.dtype == a.dtype == np.float32 and np.isfinite(x).all() and np.isfinite(a).all(), 'Nonfinite/wrong dtype cache')
        episodes.append({'episode_id': r['episode_id'], 'features': x, 'actions': a})
    require(len(episodes) == {'train': 1000, 'development': 100}[split], 'Incomplete episode population')
    return manifest, episodes


class PlanningDataset(torch.utils.data.Dataset):
    def __init__(self, episodes, stride=1, horizon=5):
        self.episodes = episodes
        self.horizon = horizon
        self.windows = [(i, s) for i, e in enumerate(episodes)
                        for s in range(0, len(e['features'])-horizon-2, stride)]
        require(self.windows, 'No complete chronological windows')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        i, s = self.windows[index]; e = self.episodes[i]; n = 3+self.horizon
        return {'features': torch.from_numpy(e['features'][s:s+n]),
                'actions': torch.from_numpy(e['actions'][s:s+n-1]),
                'episode_index': i, 'start': s}


def statistics(episodes):
    sums = {}; squares = {}; counts = {}
    for e in episodes:
        for key, v in [('feature', e['features'].reshape(-1,384,16).transpose(0,2,1).reshape(-1,384)), ('action', e['actions'])]:
            v = v.astype(np.float64)
            sums[key] = sums.get(key,0)+v.sum(0); squares[key] = squares.get(key,0)+np.square(v).sum(0)
            counts[key] = counts.get(key,0)+len(v)
    result = {'counts': counts, 'fit_split': 'train', 'normalization': 'shared_per_channel_over_training_frames_and_16_patches', 'ddof': 1, 'std_floor': 1e-5}
    for key in sums:
        n=counts[key]; mean=sums[key]/n
        std=np.maximum(np.sqrt(np.maximum((squares[key]-n*mean**2)/(n-1),0)),1e-5)
        result[key+'_mean']=(np.repeat(mean,16) if key=='feature' else mean).tolist()
        result[key+'_std']=(np.repeat(std,16) if key=='feature' else std).tolist()
    return result
