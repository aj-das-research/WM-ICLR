"""New spatial simulator campaign; immutable inputs and explicit access gates.

This namespace does not modify or relabel any historical simulator result.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPORT = ROOT / 'reports/metrics_completion_v1/planning'
REG = REPORT / 'registration.json'
sys.path.insert(0, str(ROOT / 'src'))
TASKS = ('pusht', 'reacher')
MODES = ('transport', 'autoregressive', 'bounded_additive', 'unbounded_transport')
DATA = {'pusht': 'data/upstream/pusht/pusht_expert_train.h5',
        'reacher': 'data/upstream/reacher/reacher.h5'}
CACHE = ROOT / 'data/features/current_spatial_planning_v1'
RUNS = ROOT / 'runs/current_spatial_planning_v1'
RUNTIME = ROOT / 'artifacts/upstream_planning_reproduction/stable-worldmodel'
SCHEMA = 'current_spatial_closed_loop_planning_v1'
PACKAGE_KIND = 'shiftwm_current_spatial_simulator_v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_sha(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('w') as out:
            json.dump(value, out, sort_keys=True, indent=2, allow_nan=False)
            out.write('\n'); out.flush(); os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def module(name):
    key = 'metrics_completion_' + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / (name + '.py'))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


def checked_registration(require_review=True):
    reg = read(REG)
    require(reg.get('schema') == SCHEMA and reg.get('status') == 'frozen_before_training_and_test_access', 'Unrecognized campaign registration')
    require(reg['methods'] == list(MODES) and reg['seeds'] == [0, 1, 2], 'Campaign grid differs')
    expected = {f'{t}_{m}_s{s}' for t in TASKS for m in MODES for s in range(3)}
    require({r['name'] for r in reg['runs']} == expected and len(reg['runs']) == 24, 'Incomplete run grid')
    if require_review:
        review = read(REPORT / 'source_review.json')
        require(review.get('status') == 'passed' and review.get('registration_sha256') == sha(REG), 'Missing or stale independent source review')
    for path, expected_sha in reg['dependencies'].items():
        require(sha(ROOT / path) == expected_sha, 'Registered dependency changed: ' + path)
    for task in TASKS:
        spec = reg['datasets'][task]
        stat = (ROOT / DATA[task]).stat()
        require([stat.st_size, stat.st_mtime_ns] == spec['file_stat'], 'Source HDF5 changed after metadata registration')
        parts = spec['partitions']
        ids = [r['episode_id'] for p in parts.values() for r in p]
        require(len(ids) == len(set(ids)), 'Episodes cross partitions')
        require({k: len(v) for k, v in parts.items()} == {'train': 1000, 'development': 100, 'test': 100}, 'Registered population differs')
    return reg


def select_run(reg, name=None, index=None):
    if index is not None:
        require(type(index) is int and 0 <= index < 24, 'Run index out of range')
        return reg['runs'][index]
    rows = [r for r in reg['runs'] if r['name'] == name]
    require(len(rows) == 1, 'Unknown registered run')
    return rows[0]


def install_simulator_runtime():
    """Use the unchanged API-compatible historical backend, with pinned sources."""
    require((RUNTIME / '.source_revision').read_text().strip() == 'abdced49809d5eae38e24b27dc7b635c502c4812', 'Wrong simulator revision')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    sys.path.insert(0, str(RUNTIME))
    import stable_worldmodel
    require(Path(stable_worldmodel.__file__).resolve().is_relative_to(RUNTIME), 'Unexpected simulator import')


def all_training_complete(reg):
    """Final test access requires all 24 selected packages, not partial results."""
    trainer = module('planning_train')
    selected = {}
    for row in reg['runs']:
        directory = ROOT / row['output']
        summary = trainer.base.validate_completed(directory)
        require(summary['status'] == 'completed', 'Full training gate not met')
        selected[row['name']] = {'selected_epoch': summary['best_epoch'],
                                'checkpoint_sha256': sha(directory / 'best/model.pt'),
                                'config_sha256': sha(directory / 'best/config.json'),
                                'summary_sha256': sha(directory / 'training_summary.json')}
    return selected
