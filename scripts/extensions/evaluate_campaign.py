#!/usr/bin/env python3
"""Evaluate completed registered extension runs on development or locked test."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from shiftwm.extensions.checkpoint import atomic_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--domain', required=True, choices=['drone', 'surgery'])
    parser.add_argument('--architecture', choices=['transformer', 'gru'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--split', choices=['development', 'test'], default='development')
    parser.add_argument('--kind', choices=['forecast', 'planning', 'both'], default='both')
    parser.add_argument('--episodes-per-gain', type=int, default=8)
    parser.add_argument('--output-root', default='results/extensions_v1')
    parser.add_argument('--include-controls', action='store_true')
    args = parser.parse_args()
    rows = [r for r in json.loads(Path('configs/extensions/campaign.json').read_text())['runs']
            if r['domain'] == args.domain and (args.architecture is None or r['architecture'] == args.architecture)
            and (args.seed is None or r['seed'] == args.seed)]
    if not rows:
        parser.error('No selected runs')
    tasks = []
    # Forecast diagnostics are collected for every model before closed-loop work.
    for kind in ('forecast', 'planning'):
        if args.kind not in (kind, 'both'):
            continue
        tasks += [(row, kind, 'world_model') for row in rows]
    if args.include_controls:
        # The checkpoint supplies only the unused feature encoder for controls;
        # no learned decisions are taken. Run once per domain, not per model.
        tasks += [(rows[0], 'planning', policy) for policy in ('replay_oracle', 'random')]
    results = []
    tag = f'{args.domain}_{args.architecture or "all"}_s{args.seed if args.seed is not None else "all"}_{args.split}_{args.kind}'
    report = Path(args.output_root) / 'campaigns' / (tag + '.json')
    for row, kind, policy in tasks:
        name = Path(row['output']).name if policy == 'world_model' else f'{args.domain}_{policy}'
        output = Path(args.output_root) / name / args.split / kind
        output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-m', 'shiftwm.extensions.evaluate', '--domain', args.domain,
                   '--checkpoint', row['output'] + '/best', '--data', f'data/extensions/{args.domain}_v1',
                   '--feature-cache', f'data/features/{args.domain}_v1', '--output', str(output),
                   '--split', args.split, '--kind', kind, '--policy', policy,
                   '--episodes-per-gain', str(args.episodes_per_gain)]
        print(json.dumps({'event': 'evaluate', 'run': name, 'kind': kind, 'policy': policy}), flush=True)
        with (output / 'evaluate.log').open('a') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        result = {'run': name, 'kind': kind, 'output': str(output), 'returncode': completed.returncode}
        results.append(result)
        atomic_json({'status': 'running', 'results': results}, report)
        if completed.returncode:
            return completed.returncode
        if kind == 'planning':
            data = json.loads((output / 'results.json').read_text())
            print(json.dumps({'event': 'complete', **result,
                              **{k: data[k] for k in ('successes', 'tasks', 'support_successes')}}), flush=True)
    atomic_json({'status': 'completed', 'results': results}, report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
