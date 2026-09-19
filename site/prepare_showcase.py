#!/usr/bin/env python3
"""Export unaltered recorded simulator frames and source-derived forecast summaries."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'site'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    ledger_path = ROOT / 'reports/evidence/qualitative_examples_candidates.json'
    ledger = json.loads(ledger_path.read_text())
    forecast_path = ROOT / 'paper/generated/forecast_comparison.json'
    forecast = json.loads(forecast_path.read_text())
    if forecast['status'] != 'source_validated':
        raise ValueError('Forecast source is not validated')
    assets = HERE / 'assets'
    assets.mkdir(exist_ok=True)
    result = {'schema_version': 1, 'source_sha256': {'qualitative': sha(ledger_path), 'forecast': sha(forecast_path)},
              'frame_policy': 'Unaltered RGB arrays exported as lossless PNG; each displayed frame has its recorded native call. No interpolated frames.',
              'selection': 'Existing first-lexicographic development examples in each of ours-only success, baseline-only success, and joint failure; selected illustrations are not success rates.',
              'benchmarks': {}, 'media': []}
    names = {'ours_only': 'Historical context model reaches the goal', 'baseline_only': 'Framewise reaches the goal', 'neither': 'Both miss the goal'}
    for environment in ('pusht', 'reacher'):
        examples = []
        selected = [r for r in ledger['recommended'] if r['environment'] == environment and r['category'] in names]
        for selection in selected:
            original = next(c for c in ledger['environments'][environment]['candidates'] if c['trajectory_id'] == selection['trajectory_id'] and c['observation_id'] == selection['observation_id'])
            key = f"{environment}-{selection['category']}"
            example = {'id': key, 'category': selection['category'], 'name': names[selection['category']], 'trajectory_id': original['trajectory_id'], 'methods': {}}
            common_goal, common_start = None, None
            for mode in ('factorized', 'framewise'):
                source = original['methods'][mode]
                path = ROOT / source['video_path']
                if sha(path) != source['video_sha256']:
                    raise ValueError(f'Video source changed: {path}')
                with np.load(path, allow_pickle=False) as data:
                    frames, goal = data['frames'], data['goal_image']
                if frames.dtype != np.uint8 or frames.shape[1:] != (224, 224, 3):
                    raise ValueError('Unexpected source frame format')
                pixel_hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]
                if pixel_hashes != source['frame_pixel_sha256']:
                    raise ValueError('Decoded frames differ from audited pixels')
                if common_goal is None:
                    common_goal, common_start = goal, frames[0]
                    filename = f'sim-{key}-goal.png'
                    Image.fromarray(goal).save(assets / filename)
                    example['goal'] = 'assets/' + filename
                    result['media'].append(example['goal'])
                elif not (np.array_equal(common_goal, goal) and np.array_equal(common_start, frames[0])):
                    raise ValueError('Methods do not share exactly the same goal and initial image')
                exported = []
                for index, (frame, step) in enumerate(zip(frames, source['frame_native_steps'], strict=True)):
                    filename = f'sim-{key}-{mode}-{index}.png'
                    Image.fromarray(frame).save(assets / filename)
                    exported.append({'src': 'assets/' + filename, 'native_call': step, 'pixel_sha256': pixel_hashes[index]})
                    result['media'].append('assets/' + filename)
                record = source['record']
                metrics = ({'Block error': {'value': record['block_translation_error_px'], 'unit': 'px'},
                            'Pusher error': {'value': record['agent_position_error_px'], 'unit': 'px'},
                            'Angle error': {'value': record['block_angle_error_rad'], 'unit': 'rad'}}
                           if environment == 'pusht' else {'Joint L2 error': {'value': record['final_distance'], 'unit': 'rad'}})
                example['methods'][mode] = {'frames': exported, 'success': bool(record['success']), 'stop_call': record['native_steps'],
                                            'endpoint_metrics': metrics, 'checkpoint_sha256': source['checkpoint_sha256'],
                                            'source': source['video_path'], 'source_sha256': source['video_sha256']}
            example['timeline'] = sorted(set(f['native_call'] for m in example['methods'].values() for f in m['frames']))
            examples.append(example)
        summary = [{**r, 'label': 'Historical context model' if r['mode'] == 'factorized' else r['label']}
                   for r in forecast['results'] if r['environment'] == environment]
        comparisons = [r for r in forecast['comparison_deltas'] if r['environment'] == environment]
        result['benchmarks'][environment] = {'name': 'PushT' if environment == 'pusht' else 'Reacher', 'examples': examples,
                                            'forecast_rows': summary, 'comparisons': comparisons,
                                            'metric': forecast['metric'], 'qualitative_split': 'development', 'qualitative_training_seed': 0}
    (HERE / 'showcase.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'showcase': 'site/showcase.json', 'examples': 6, 'unaltered_frame_files': len(result['media'])}))


if __name__ == '__main__':
    prepare()
