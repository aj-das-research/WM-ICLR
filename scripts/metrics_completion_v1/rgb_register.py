"""Metadata/source-only registration; does not decode videos or open feature arrays."""
from pathlib import Path
from rgb_core import ROOT, REPORT, REG, RECIPE, TASKS, UPSTREAM, sha, read, atomic_json, internal_cache, require, selected_indices, module


def freeze():
    require(not REG.exists(), 'RGB registration already exists; immutable')
    dependencies = {}
    runner=module(ROOT/'scripts/real_video_iws/train.py','_rgb_registration_iws_runner')
    dependencies.update(runner.source_files())
    own = sorted(Path(__file__).parent.glob('rgb_*'))
    for p in own:
        if p.is_file(): dependencies[str(p.relative_to(ROOT))] = sha(p)
    for relative in [UPSTREAM, 'external/official-dino-wm/conf/decoder/transposed_conv.yaml',
                     'external/official-dino-wm/models/visual_world_model.py',
                     'external/rla-wm/src/models/dino_to_image_unet_v1.py',
                     'configs/real_video_iws/training_v1.json', 'scripts/real_video_iws/train.py',
                     'src/shiftwm/real_video_iws/training.py', 'src/shiftwm/real_video_iws/windows.py',
                     'src/shiftwm/real_video_iws/model.py']:
        dependencies[relative] = sha(ROOT / relative)
    tasks = {}
    for task in TASKS:
        cache = internal_cache(task)
        original = read(ROOT / f'configs/real_video_iws/{task}_cache_registration_v1.json')
        for path, expected in original['dependencies'].items():
            require(sha(ROOT / path) == expected, 'Frozen cache source changed')
            dependencies[path] = expected
        registration = f'configs/real_video_iws/{task}_cache_registration_v1.json'
        dependencies[registration] = sha(ROOT / registration)
        for name in ('manifest.json', 'identity.json', 'episode_index.json', 'training_statistics.json'):
            p = cache.output / name; dependencies[str(p.relative_to(ROOT))] = sha(p)
        records = []
        for eid in cache.inventory.selected():
            record = cache.index[eid]
            cache.inventory.authorize(eid, record['split'])
            records.append({'episode_id': eid, 'split': record['split'], 'native_frames': record['frames'],
                            'selected_indices': selected_indices(record['frames']).tolist(),
                            'feature_payload_sha256': record['payload_sha256'],
                            'decoded_rgb_sha256': record['decoded_rgb_sha256'],
                            'video_sha256': record['source_video_sha256']})
        tasks[task] = {'cache_root': str(cache.output.relative_to(ROOT)), 'records': records,
                       'counts': {split: {'trajectories': sum(r['split'] == split for r in records),
                            'frames': sum(len(r['selected_indices']) for r in records if r['split'] == split)}
                            for split in ('internal_train', 'internal_development')},
                       'statistics_sha256': sha(cache.output / 'training_statistics.json')}
    import torch,numpy,cv2,einops
    document = {'schema': 'iws_shared_rgb_decoder_registration_v1', 'status': 'frozen_before_decoder_training',
                'recipe': RECIPE, 'tasks': tasks, 'dependencies': dependencies,
                'environment':{'torch':str(torch.__version__),'numpy':numpy.__version__,'opencv':cv2.__version__,'einops':einops.__version__,'cuda_build':torch.version.cuda},
                'runs': [{'task': task, 'seed': 173, 'epochs': 30,
                          'output': f'reports/metrics_completion_v1/rgb/runs/{task}_s173'} for task in TASKS],
                'reserved_payloads_opened': 0, 'source_review': 'detached source_review.json binding this file SHA required before any target decoding/training'}
    atomic_json(document, REG)
    return {'status': 'frozen', 'registration_sha256': sha(REG), 'tasks': {t: v['counts'] for t,v in tasks.items()}}


if __name__ == '__main__':
    import json
    print(json.dumps(freeze(), sort_keys=True))
