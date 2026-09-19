#!/usr/bin/env python3
"""Finalize all27 registered IWS development runs before paper ingestion.

No raw video, command or official-validation payload is opened. Completion is
immutable; a process lock serializes concurrent last-run finalizers. Primitive
window errors, complete training packages and every source identity are checked
again before the completion receipt becomes visible.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/real_video_iws'
FINAL = REPORT / 'development_finalization.json'
SCHEMA = 'shiftwm_iws_development_finalization_v1'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


evaluation = module('_iws_finalizer_evaluator', Path(__file__).with_name('evaluate.py'))
campaign = evaluation.campaign


def local(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Finalization source escapes the workspace')
    return path


def bind(path, sources, expected=None):
    path = local(path)
    digest = campaign.sha(path)
    if expected is not None and digest != expected:
        raise ValueError('Changed finalization evidence: ' + str(path))
    key = str(path.relative_to(ROOT))
    if key in sources and sources[key] != digest:
        raise ValueError('Evidence changed while finalization was running: ' + key)
    sources[key] = digest
    return digest


def metadata_audit(cache):
    """Reconstruct the exact registered window population without payload reads."""
    ids = cache.inventory.selected('internal_development')
    if ids != sorted(set(ids)) or not ids:
        raise ValueError('Invalid development trajectory inventory')
    records = []
    tasks = set()
    for index, eid in enumerate(ids):
        metadata, split = cache.inventory.authorize(eid, 'internal_development')
        row = cache.index[eid]
        task = row['task']; tasks.add(task)
        if (split != 'internal_development' or row['split'] != split or row['episode_id'] != eid
                or task not in campaign.TASKS or row['command_width'] != campaign.TASKS[task]
                or metadata['shapes']['target_qpos'] != [row['frames'], campaign.TASKS[task]]
                or row['command_rows'] != row['frames'] or row['feature_dim'] != 6144):
            raise ValueError('Development metadata differs from the native task contract')
        starts = list(range(0, row['frames']-60, 5))
        records.append({'episode_id': eid, 'episode_index': index, 'split': split,
                        'frames': row['frames'], 'windows': len(starts),
                        'first_start': starts[0] if starts else None, 'last_start': starts[-1] if starts else None,
                        'status': 'eligible' if starts else 'no_window_under_strict_unused_tail_rule'})
    if len(tasks) != 1:
        raise ValueError('Mixed native task identities')
    return {'task': next(iter(tasks)), 'split': 'internal_development', 'horizon_command_rows': 60,
            'predicted_offsets': list(range(1,60)), 'window_start_stride': 5,
            'eligibility': 'range(0,N-H,stride), requiring s+H<N', 'episodes': len(records),
            'eligible_episodes': sum(r['windows'] > 0 for r in records),
            'ineligible_episodes': sum(r['windows'] == 0 for r in records),
            'windows': sum(r['windows'] for r in records), 'records': records}


def validate_evaluation(path, row, summary, checkpoint_sha, audit, config_sha, registration_sha, sources):
    bind(path, sources)
    receipt = campaign.read(path)
    required = {'schema': 'shiftwm_iws_development_evaluation_v1', 'status': 'passed',
                'scope': 'internal_development', 'task': row['task'], 'mode': row['mode'], 'seed': row['seed'],
                'completed_epochs': 30, 'selected_epoch': summary['best_epoch'],
                'selected_checkpoint_sha256': checkpoint_sha, 'registration_sha256': registration_sha,
                'config_sha256': config_sha, 'horizons': [15,30,45,60], 'offsets': list(range(1,60)),
                'metrics': list(evaluation.METRICS), 'primary_aggregation': 'equal_trajectory',
                'official_validation_payloads_read': 0, 'population_audit': audit,
                'total_windows': audit['windows'], 'eligible_trajectories': audit['eligible_episodes'],
                'evaluator_sha256': campaign.sha(Path(evaluation.__file__))}
    if any(receipt.get(k) != v for k,v in required.items()):
        raise ValueError('Evaluation identity, completeness, population or access mismatch: ' + row['name'])
    gap = receipt.get('prefix_maximum_absolute_difference')
    if not isinstance(gap, (int,float)) or not math.isfinite(gap) or gap < 0:
        raise ValueError('Invalid causal-prefix check receipt')
    ledger = local(receipt['window_ledger_path'])
    if ledger != path.resolve().with_suffix('.npz'):
        raise ValueError('Evaluation ledger must be its sibling immutable NPZ')
    bind(ledger, sources, receipt['window_ledger_sha256'])
    metric_keys = set(evaluation.METRICS) | {'persistence_'+k for k in evaluation.METRICS}
    with np.load(ledger, allow_pickle=False) as loaded:
        if set(loaded.files) != metric_keys | {'episode_index','window_start'}:
            raise ValueError('Unexpected evaluation ledger fields')
        arrays = {key: loaded[key] for key in loaded.files}
    for key in ('episode_index','window_start'):
        if arrays[key].dtype != np.int64 or arrays[key].shape != (audit['windows'],):
            raise ValueError('Invalid integer native window indices')
    if any(arrays[k].dtype != np.float64 for k in metric_keys):
        raise ValueError('Primitive metrics must retain evaluator float64 storage')
    episodes = evaluation.aggregate_windows(arrays, audit)
    if episodes != receipt['episodes']:
        raise ValueError('Reported trajectory curves differ from raw-window reconstruction')
    h60 = float(np.mean([r['standardized_mse_by_offset'][-1] for r in episodes]))
    if h60 != receipt['h60_standardized_mse'] or not math.isclose(h60, summary['best_validation_mse'], rel_tol=2e-6, abs_tol=2e-7):
        raise ValueError('Selected-checkpoint endpoint does not reproduce the full ledger')
    return receipt


def collect(config_path, registry):
    config_path = local(config_path)
    config = campaign.read(config_path)
    sources = dict(registry['dependencies'])
    for name,digest in list(sources.items()):
        bind(local(name), sources, digest)
    config_sha = bind(config_path, sources)
    registration_sha = bind(campaign.REGISTRATION, sources)
    for name in ('scripts/real_video_iws/finalize.py','scripts/real_video_iws/evaluate.py',
                 'tests/test_iws_finalization.py','tests/test_iws_development_evaluation.py',
                 'paper/scripts/refresh_experiment_alignment.py','paper/tests/test_experiment_alignment_reporting.py'):
        bind(ROOT/name, sources)
    train = campaign.trainer()
    audits = {task: metadata_audit(train.open_cache(config,task)) for task in campaign.TASKS}
    runs = []
    persistence = {}
    for row in registry['runs']:
        directory = local(row['output'])
        summary = train.validate_completed(directory)
        package, state = train.read_package(directory/'best')
        last_package, _ = train.read_package(directory/'last', True)
        identity = state['config']['metadata']['identity']
        expected_recipe = {'task': row['task'], 'mode': row['mode'], 'seed': row['seed'],
                           'training': config['training'], 'model': config['model'],
                           'task_config': config['tasks'][row['task']], 'study_config_sha256': config_sha}
        if identity['scientific_config'] != expected_recipe or identity['populations']['val'] != audits[row['task']]:
            raise ValueError('Completed package recipe or development population changed')
        for filename in ('training_config.json','training_summary.json','metrics.jsonl'):
            bind(directory/filename, sources)
        for folder in (package,last_package):
            manifest_path = folder/'package_manifest.json'
            bind(manifest_path,sources)
            manifest = campaign.read(manifest_path)
            for name,digest in manifest['files'].items():
                bind(folder/name,sources,digest)
        checkpoint_sha = bind(package/'model.pt',sources)
        path = REPORT/'evaluations'/(row['name']+'.json')
        receipt = validate_evaluation(path,row,summary,checkpoint_sha,audits[row['task']],config_sha,registration_sha,sources)
        baseline = [{k:v for k,v in episode.items() if k.startswith('persistence_') or k in ('episode_id','windows')}
                    for episode in receipt['episodes']]
        if row['task'] in persistence and baseline != persistence[row['task']]:
            raise ValueError('Persistence reference changed between matched arms/seeds')
        persistence[row['task']] = baseline
        runs.append({'name': row['name'], 'task': row['task'], 'mode': row['mode'], 'seed': row['seed'],
                     'evaluation_path': str(path.relative_to(ROOT)), 'evaluation_sha256': campaign.sha(path),
                     'completed_epochs': 30, 'selected_epoch': summary['best_epoch'],
                     'selected_checkpoint_sha256': checkpoint_sha,
                     'window_ledger_path': receipt['window_ledger_path'], 'window_ledger_sha256': receipt['window_ledger_sha256']})
    if len(runs) != 27 or {(r['task'],r['mode'],r['seed']) for r in runs} != {
        (task,mode,seed) for task in campaign.TASKS for mode in campaign.MODES for seed in (0,1,2)}:
        raise ValueError('Finalization must cover exactly27 distinct registered runs')
    if campaign.check_registration(config_path) != registry:
        raise ValueError('Registration changed during finalization')
    for name,digest in sources.items():
        if campaign.sha(local(name)) != digest:
            raise ValueError('Evidence changed before completion commit: '+name)
    return {'schema': SCHEMA, 'status': 'passed', 'scope': 'internal_development',
            'expected_runs': 27, 'completed_runs': 27, 'completed_utc': campaign.now(),
            'registration_sha256': registration_sha, 'source_dependencies': sources, 'per_run': runs,
            'official_validation_payloads_read': 0,
            'verification': 'All30-epoch packages, native-window ledgers and equal-trajectory curves revalidated; all arms and signs retained.'}


def verify_existing(value, registry):
    if (value.get('schema') != SCHEMA or value.get('status') != 'passed'
            or value.get('scope') != 'internal_development' or value.get('expected_runs') != 27
            or value.get('completed_runs') != 27 or value.get('registration_sha256') != campaign.sha(campaign.REGISTRATION)
            or value.get('official_validation_payloads_read') != 0):
        raise ValueError('Existing completion receipt is not a valid immutable IWS finalizer')
    expected = {(r['task'],r['mode'],r['seed']) for r in registry['runs']}
    runs = value.get('per_run',[])
    if len(runs) != 27 or {(r['task'],r['mode'],r['seed']) for r in runs} != expected:
        raise ValueError('Existing finalizer no longer covers the complete grid')
    dependencies = value.get('source_dependencies',{})
    mandatory = {str(campaign.REGISTRATION.relative_to(ROOT)), 'scripts/real_video_iws/finalize.py',
                 'scripts/real_video_iws/evaluate.py','paper/scripts/refresh_experiment_alignment.py'}
    if not mandatory <= set(dependencies) or not set(registry['dependencies']) <= set(dependencies):
        raise ValueError('Existing finalizer lacks its complete source ledger')
    for name,digest in dependencies.items():
        if campaign.sha(local(name)) != digest:
            raise ValueError('Existing completed evidence changed: '+name)
    for run in runs:
        if (dependencies.get(run['evaluation_path']) != run['evaluation_sha256']
                or dependencies.get(run['window_ledger_path']) != run['window_ledger_sha256']
                or run['selected_checkpoint_sha256'] not in dependencies.values() or run['completed_epochs'] != 30):
            raise ValueError('Existing result identity is not source-bound')


def refresh_paper():
    reporter = module('_iws_finalized_paper', ROOT/'paper/scripts/refresh_experiment_alignment.py')
    reporter.render()


def finalize(config_path=campaign.CONFIG, if_ready=False):
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'.development_finalization.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        registry = campaign.check_registration(config_path)
        if FINAL.exists():
            value = campaign.read(FINAL)
            verify_existing(value,registry)
            refresh_paper()
            return {'status':'already_complete_verified','completed_runs':27,'finalization_sha256':campaign.sha(FINAL)}
        missing = [r['name'] for r in registry['runs'] if not (REPORT/'evaluations'/(r['name']+'.json')).is_file()]
        if missing:
            if not if_ready:
                raise ValueError(f'{len(missing)} registered evaluation receipts are missing')
            return {'status':'pending','completed_evaluation_receipts':27-len(missing),'expected_runs':27,
                    'missing_runs':missing,'paper_numerical_results_updated':False}
        value = collect(config_path,registry)
        campaign.atomic_json(value,FINAL)
        refresh_paper()
        return {'status':'passed','completed_runs':27,'finalization_sha256':campaign.sha(FINAL)}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,default=campaign.CONFIG)
    parser.add_argument('--if-ready',action='store_true')
    args=parser.parse_args()
    print(json.dumps(finalize(args.config,args.if_ready),sort_keys=True),flush=True)
