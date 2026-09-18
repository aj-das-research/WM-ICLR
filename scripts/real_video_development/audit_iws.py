#!/usr/bin/env python3
"""Audit official IWS split/handle metadata without opening video observations."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/real_video/iws_public_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    acquisition = json.loads((ROOT / 'reports/evidence/iws_acquisition.json').read_text())
    if acquisition['status'] != 'download_verified_and_extracted_no_evaluation':
        raise ValueError('Verified complete acquisition required')
    if acquisition['registry_sha256'] != sha(ROOT / 'configs/real_video_development/iws_acquisition_v1.json'):
        raise ValueError('Acquisition registry changed')
    root = DATA / 'extracted/iws_converted'
    summary = json.loads((root / 'summary.json').read_text())
    rows, handle_rows, split_ids = [], [], {}
    for name, splits in sorted(summary['subdatasets'].items()):
        ranges = json.loads((root / name / 'split_ranges.json').read_text())
        index = json.loads((root / name / 'index.json').read_text())['trajectories']
        ids = {key: set(range(value['start'], value['end'] + 1)) for key, value in ranges.items()}
        if set(ids) != {'train', 'val'} or ids['train'] & ids['val']:
            raise ValueError('Split identities invalid')
        if {int(key) for key in index} != ids['train'] | ids['val']:
            raise ValueError('Trajectory index and splits disagree')
        for split in ('train', 'val'):
            if splits[split]['failed'] or splits[split]['converted'] != len(ids[split]):
                raise ValueError('Upstream conversion incomplete')
        split_ids[name] = ids
        rows.append({'task': name, 'train_episodes': len(ids['train']),
                     'validation_episodes': len(ids['val']),
                     'action_dimensions': splits['train']['action_dims'],
                     'index_sha256': sha(root / name / 'index.json'),
                     'split_sha256': sha(root / name / 'split_ranges.json')})
    for path in sorted((DATA / 'download/eval_handles/iws').glob('*.json')):
        dataset = json.loads(path.read_text())
        if set(dataset['handles_by_horizon']) != {'60'}:
            raise ValueError('Unexpected official handle horizon')
        episodes = set()
        for item in dataset['handles_by_horizon']['60']:
            if (item['sampled_horizon'] != 60 or item['feasible_horizon'] < 60
                    or int(item['traj_id']) not in split_ids[item['group_name']]['val']):
                raise ValueError('Handle is outside official validation split')
            episodes.add((item['group_name'], item['traj_id']))
        handle_rows.append({'file': path.name, 'horizon': 60,
                            'windows': len(dataset['handles_by_horizon']['60']),
                            'validation_episodes': len(episodes), 'sha256': sha(path)})
    output = {'status': 'metadata_audit_passed_no_model_evaluation', 'tasks': rows,
              'audit_source_sha256': sha(Path(__file__)),
              'acquisition_sha256': sha(ROOT / 'reports/evidence/iws_acquisition.json'),
              'total_train': sum(row['train_episodes'] for row in rows),
              'total_validation': sum(row['validation_episodes'] for row in rows),
              'official_handles': handle_rows, 'images_decoded_for_audit': 0,
              'separation': 'Official trajectory split; session/person/scene independence is unestablished.'}
    (ROOT / 'reports/evidence/iws_metadata_audit.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({key: output[key] for key in ('status', 'total_train', 'total_validation')}))


if __name__ == '__main__':
    main()
