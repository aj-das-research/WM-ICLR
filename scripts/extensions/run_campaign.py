#!/usr/bin/env python3
"""Run full registered training runs in one shard; never subsample epochs/data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from shiftwm.extensions.checkpoint import atomic_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--domain', required=True, choices=['drone', 'surgery'])
    parser.add_argument('--architecture', choices=['transformer', 'gru'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--max-runtime-seconds', type=float, default=27000)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error('Invalid shard')
    manifest = json.loads(Path('configs/extensions/campaign.json').read_text())
    rows = [r for r in manifest['runs'] if r['domain'] == args.domain
            and (args.architecture is None or r['architecture'] == args.architecture)
            and (args.seed is None or r['seed'] == args.seed)][args.shard::args.shards]
    if not rows:
        parser.error('No registered runs selected')
    tag = f'{args.domain}_{args.architecture or "all"}_s{args.seed if args.seed is not None else "all"}_{args.shard}of{args.shards}'
    report = Path('runs/extensions/campaigns') / (tag + '.json')
    sources = [Path(r['config']) for r in rows] + [Path(manifest['protocol'])]
    identity = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    records = []
    started = time.monotonic()
    for row in rows:
        if identity != {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}:
            raise RuntimeError('Registered protocol or configuration changed during campaign')
        remaining = args.max_runtime_seconds - (time.monotonic() - started)
        if remaining < 120:
            return 75
        output = Path(row['output'])
        output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-m', 'shiftwm.extensions.train', '--config', row['config'],
                   '--resume-if-present', '--max-runtime-seconds', str(remaining - 60)]
        print(json.dumps({'event': 'start_run', **row}), flush=True)
        with (output / 'train.log').open('a') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        records.append({**row, 'returncode': result.returncode})
        atomic_json({'status': 'running', 'identity': identity, 'runs': records}, report)
        print(json.dumps({'event': 'finish_run', **records[-1]}), flush=True)
        if result.returncode:
            return result.returncode
    atomic_json({'status': 'completed', 'identity': identity, 'runs': records,
                 'elapsed_seconds': time.monotonic() - started}, report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
