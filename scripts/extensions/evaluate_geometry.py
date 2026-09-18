#!/usr/bin/env python3
"""Evaluate the isolated wide-gain model with the frozen extension evaluator."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from shiftwm.extensions import geometry_revision
from shiftwm.extensions.checkpoint import atomic_json, file_sha256
from shiftwm.extensions.evaluate import (evaluation_identity, evaluate_forecasts,
                                         evaluate_task, tasks_for_split, SimulatorClient)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--kind', choices=['forecast', 'planning'], required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if Path(args.checkpoint).name != 'best':
        p.error('The registered comparison evaluates only validation-selected best packages')
    args.domain, args.split, args.policy = 'drone', 'development', 'world_model'
    args.data, args.feature_cache = 'data/extensions/drone_v1', 'data/features/drone_v1'
    args.episodes_per_gain, args.planner_seed = 8, 101
    torch.set_num_threads(4)
    model, state = geometry_revision.load_geometry_package(args.checkpoint, args.device)
    geometry_revision.validate_completed(Path(args.checkpoint).parent,
                                          state['config']['metadata']['training_identity'])
    identity = evaluation_identity(args, model)
    for path in (Path(__file__), Path(geometry_revision.__file__), Path('reports/geometry_revision_protocol.md')):
        identity['sources'][str(path)] = file_sha256(path)
    args.output.mkdir(parents=True, exist_ok=True)
    identity_path = args.output / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Geometry evaluation sources/settings changed; use a new output namespace')
    atomic_json(identity, identity_path)
    if args.kind == 'forecast':
        result = evaluate_forecasts(model, args.data, args.split, feature_cache=args.feature_cache,
                                    num_workers=0, stride=4)
        atomic_json({'status': 'completed', 'identity': identity, **result}, args.output / 'results.json')
        return
    tasks, selection = tasks_for_split(args.domain, args.data, args.split, args.episodes_per_gain)
    records = []
    with SimulatorClient('drone', args.output / 'simulator.log') as client:
        for task in tasks:
            key = task['trajectory_id'] + f"-o{task['observation_id']}"
            path, trace = args.output / (key + '.json'), args.output / (key + '.npz')
            if path.exists():
                row = json.loads(path.read_text())
                if file_sha256(trace) != row['trace_sha256']:
                    raise ValueError('Geometry trace payload changed')
            else:
                row, arrays = evaluate_task(model, client, task, 'drone',
                                            planner_seed=args.planner_seed)
                np.savez_compressed(trace, **arrays)
                row.update(trace_file=trace.name, trace_sha256=file_sha256(trace))
                atomic_json(row, path)
            records.append(row)
            print(json.dumps({'task': key, 'success': row['success'], 'native_calls': row['native_calls']}), flush=True)
    result = {'status': 'completed', 'identity': identity, 'selection': selection, 'records': records,
              'successes': sum(r['success'] for r in records), 'tasks': len(records),
              'support_successes': sum(r['success_during_support'] for r in records),
              'claim_scope': 'one-change development capacity ablation; final test not evaluated'}
    atomic_json(result, args.output / 'results.json')
    print(json.dumps({k: result[k] for k in ('status', 'successes', 'tasks')}), flush=True)


if __name__ == '__main__':
    main()
