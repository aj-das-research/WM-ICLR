#!/usr/bin/env python3
"""Refresh local extension status while training/evaluations run independently."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--interval', type=int, default=120)
    parser.add_argument('--max-hours', type=float, default=24)
    args = parser.parse_args()
    if args.interval < 30 or args.max_hours <= 0:
        parser.error('Positive duration and >=30s interval required')
    started = time.monotonic()
    while time.monotonic() - started < args.max_hours * 3600:
        result = subprocess.run([sys.executable, 'scripts/extensions/summarize.py'])
        if result.returncode:
            print(json.dumps({'event': 'summary_error', 'returncode': result.returncode}), flush=True)
        else:
            report = json.loads(Path('reports/domain_extension_progress.json').read_text())
            if report['completed_training_runs'] == 36 and report['completed_planning_evaluations'] == 36:
                return 0
        time.sleep(args.interval)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
