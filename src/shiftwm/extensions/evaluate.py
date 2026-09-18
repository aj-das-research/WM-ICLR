"""Matched closed-loop extension evaluation with native interaction accounting.

Runs development by default. Simulator state belongs to this evaluator and is
never provided to the learned cost. Goal images are actual verified renders.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch import nn

from shiftwm.data import pixels_to_tensor, split_combinations, validate_manifest
from shiftwm.evaluate import evaluate_forecasts, LatentGoalCost
from .checkpoint import atomic_json, file_sha256, load_package
from .rpc import ROOT, SimulatorClient


class ExtensionGoalCost(LatentGoalCost):
    def __init__(self, model):
        nn.Module.__init__(self)
        self.model = model
        self.register_buffer('native_low', torch.full((10,), -1.))
        self.register_buffer('native_high', torch.full((10,), 1.))


def verified_npz(path, expected_hash):
    if file_sha256(path) != expected_hash:
        raise ValueError(f'Changed data payload {path}')
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def tasks_for_split(domain, root, split, episodes_per_gain):
    """Select entirely from pre-outcome eligibility, never from policy results."""
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    validate_manifest(manifest)
    combos = split_combinations(split)
    by_id = {e['trajectory_id']: e for e in manifest['episodes'] if e['split'] == split}
    tasks, exclusions = [], []
    if domain == 'drone':
        for ep in by_id.values():
            data = verified_npz(root / ep['audit_file'], ep['audit_sha256'])
            distance = float(np.linalg.norm(data['simulator_states'][0, :2] - data['goal_state'][:2]))
            if distance < .08:
                exclusions.append({'trajectory_id': ep['trajectory_id'], 'reason': 'initial_xy_distance_below_8cm'})
                continue
            tasks.append({**ep, 'goal_state': data['goal_state'].tolist(),
                          'goal_image': data['goal_image'], 'initial_image': data['images'][0],
                          'commands': data['actions'], 'initial_distance_m': distance})
    elif domain == 'surgery':
        registry = json.loads((root / 'planning_goals' / split / 'manifest.json').read_text())
        if registry['status'] != 'independent_replay_verified':
            raise ValueError('Surgery image goals have not passed independent replay')
        exclusions = registry['exclusions']
        for row in registry['episodes']:
            ep = by_id[row['trajectory_id']]
            if row['source_episode_sha256'] != ep['sha256'] or row['source_audit_sha256'] != ep['audit_sha256']:
                raise ValueError('Surgery goal belongs to a different data version')
            images = {}
            for name in ('goal_image', 'initial_image'):
                path = root / row[name + '_file']
                if file_sha256(path) != row[name + '_sha256']:
                    raise ValueError('Goal reference image changed')
                images[name] = np.asarray(Image.open(path).convert('RGB'))
            commands = verified_npz(root / row['command_file'], row['command_file_sha256'])['actions_commanded']
            tasks.append({**ep, **row, **images, 'commands': commands})
    else:
        raise ValueError(domain)
    selected = []
    counts = {}
    for gain in sorted({d for _, d in combos}):
        eligible = sorted([t for t in tasks if t['dynamics_id'] == gain], key=lambda t: (t['seed'], t['trajectory_id']))
        chosen = eligible[:episodes_per_gain]
        counts[str(gain)] = {'eligible': len(eligible), 'selected': len(chosen), 'requested': episodes_per_gain}
        for task in chosen:
            for appearance, dynamics in combos:
                if dynamics == gain:
                    selected.append({**task, 'observation_id': appearance})
    if not selected:
        raise ValueError('No eligible planning tasks')
    return selected, {'counts_by_gain': counts, 'exclusions': exclusions,
                      'selection': 'first eligible seeds before observing model outcomes'}


def distance(metrics, domain):
    value = metrics['goal_distance_m' if domain == 'drone' else 'distance_m']
    if value is None or not np.isfinite(value):
        raise ValueError('Invalid physical distance from simulator')
    return float(value)


def stopped(response):
    return bool(response['metrics']['success'] or response.get('terminated') or response.get('truncated'))


def valid_success(metrics):
    return bool(metrics['success'] and metrics.get('valid_action', True)
                and metrics.get('stable_deformation', True)
                and not metrics.get('crash', False) and not metrics.get('workspace_escape', False))


def accept_execution(response, requested):
    executed = np.asarray(response['executed_commands'], dtype=np.float32).reshape(-1, 2)
    if len(executed) > len(requested) or len(response['metrics_per_step']) != len(executed):
        raise ValueError('Simulator action accounting mismatch')
    if not np.array_equal(executed, np.asarray(requested, dtype=np.float32)[:len(executed)]):
        raise ValueError('Simulator changed commanded action history')
    if len(executed) != len(requested) and not stopped(response):
        raise ValueError('Simulator dropped actions without a terminal event')
    if not len(executed) and not stopped(response):
        raise ValueError('Nonterminal simulator made no progress')
    return executed


@torch.inference_mode()
def evaluate_task(model, client, task, domain, policy='world_model', native_budget=200, planner_seed=101):
    from gymnasium.spaces import Box
    from stable_worldmodel.planning.solver.cem import CEMSolver

    device = next(model.parameters()).device
    appearance = task['observation_id']
    response = client.request({'op': 'reset', 'seed': task['seed'], 'dynamics_id': task['dynamics_id'],
                               'goal_state': task['goal_state'], 'image_size': 128, 'max_steps': native_budget})
    if not np.array_equal(response['image'], task['initial_image']):
        raise ValueError('Initial simulator RGB differs from verified independent reference')
    initial = distance(response['metrics'], domain)
    if response['metrics']['success'] or initial < (.08 if domain == 'drone' else .004):
        raise ValueError('Prespecified task eligibility changed on live reset')
    encode = lambda image: model.encode_images(pixels_to_tensor(image, appearance)[None].to(device))
    features, past, frames, frame_steps = [encode(response['image'])], [], [response['image']], [0]
    native_commands, metrics, decisions = [], [], []
    initial_metrics = response['metrics']
    # Support and every subsequent native command consume the same total budget.
    for start in (0, 5):
        requested = task['commands'][start:start + 5]
        response = client.request({'op': 'step', 'actions': requested.tolist(), 'stop_on_success': True})
        executed = accept_execution(response, requested)
        native_commands.extend(executed.tolist())
        metrics.extend(response['metrics_per_step'])
        frames.append(response['image']); frame_steps.append(len(native_commands))
        if stopped(response):
            break
        past.append(torch.as_tensor(executed.reshape(1, 10), device=device))
        features.append(encode(response['image']))
    support_success = valid_success(response['metrics'])
    goal = pixels_to_tensor(task['goal_image'], appearance)[None].to(device)
    cost = ExtensionGoalCost(model).to(device)
    solver = CEMSolver(cost=cost, batch_size=1, num_samples=128, n_steps=5,
                       topk=16, device=device, seed=planner_seed + task['seed'])
    solver.configure(action_space=Box(-1, 1, shape=(1, 2), dtype=np.float32), n_envs=1,
                     config=SimpleNamespace(horizon=5, action_block=5))
    warm = None
    rng = np.random.default_rng(planner_seed + task['seed'])
    while len(native_commands) < native_budget and not stopped(response):
        remaining = min(5, native_budget - len(native_commands))
        decision = {'native_start': len(native_commands)}
        if policy == 'world_model':
            history = torch.stack(features[-3:], dim=1)
            actions = torch.stack(past[-2:], dim=1)
            obs, dyn = model.infer_context(history, actions)
            goal_features = model.goal_embedding(goal, obs)
            info = dict(history_features=history, past_actions=actions,
                        observation_context=obs, dynamics_context=dyn, goal_features=goal_features)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            before = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                solution = solver.solve(info, init_action=warm)
            normalized = solution['actions'].to(device)
            raw = cost.to_native(normalized)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            decision['solve_seconds'] = time.perf_counter() - before
            forecast = model.rollout_features(history, actions, raw, contexts=(obs, dyn))
            decision.update(predicted_terminal_goal_mse=float((forecast[:, -1] - goal_features).square().mean()),
                            predicted_next_goal_mse=float((forecast[:, 0] - goal_features).square().mean()),
                            dynamics_context_norm=float(dyn.norm()), observation_context_norm=float(obs.norm()))
            requested = raw[:, 0].reshape(5, 2).cpu().numpy()[:remaining]
            warm = normalized[:, 1:]
        elif policy == 'random':
            requested = rng.uniform(-1, 1, (remaining, 2)).astype(np.float32)
        elif policy == 'replay_oracle':
            requested = task['commands'][len(native_commands):len(native_commands) + remaining]
        else:
            raise ValueError(policy)
        before_distance = distance(response['metrics'], domain)
        response = client.request({'op': 'step', 'actions': requested.tolist(), 'stop_on_success': True})
        executed = accept_execution(response, requested)
        native_commands.extend(executed.tolist())
        metrics.extend(response['metrics_per_step'])
        frames.append(response['image']); frame_steps.append(len(native_commands))
        decision.update(native_end=len(native_commands), executed_commands=executed.tolist(),
                        physical_distance_before_m=before_distance,
                        physical_distance_after_m=distance(response['metrics'], domain))
        decisions.append(decision)
        if stopped(response):
            break
        if len(executed) != 5:
            raise ValueError('Nonterminal partial block cannot be added as full model history')
        past.append(torch.as_tensor(executed.reshape(1, 10), device=device))
        features.append(encode(response['image']))
    hits = [i + 1 for i, row in enumerate(metrics) if valid_success(row)]
    row = {'trajectory_id': task['trajectory_id'], 'seed': task['seed'],
           'observation_id': appearance, 'dynamics_id': task['dynamics_id'], 'policy': policy,
           'success': bool(hits), 'first_success_native_step': hits[0] if hits else None,
           'success_during_support': support_success, 'native_calls': len(native_commands),
           'initial_distance_m': initial, 'final_distance_m': distance(response['metrics'], domain),
           'initial_metrics': initial_metrics, 'final_metrics': response['metrics'],
           'stop_reason': response.get('stop_reason'), 'metrics_per_native_step': metrics,
           'decisions': decisions, 'native_budget': native_budget, 'support_budget': 10}
    arrays = {'images': np.stack(frames), 'image_native_steps': np.asarray(frame_steps),
              'commands': np.asarray(native_commands, np.float32).reshape(-1, 2),
              'goal_image': task['goal_image']}
    return row, arrays


def evaluation_identity(args, model):
    paths = [Path(__file__), Path(__file__).with_name('rpc.py'),
             Path('scripts/extensions') / ('serve_' + args.domain + '.py'),
             Path(__file__).with_name(args.domain + '.py'),
             Path(args.data) / 'manifest.json', Path(args.checkpoint) / 'model.pt',
             Path('reports/domain_extension_protocol.md')]
    if args.domain == 'surgery':
        paths.append(Path(args.data) / 'planning_goals' / args.split / 'manifest.json')
    import stable_worldmodel.planning.solver.cem as cem
    paths.append(Path(cem.__file__))
    module_root = Path(__file__).resolve().parent
    paths += [module_root / name for name in ('model.py', 'checkpoint.py')]
    paths += [module_root.parent / name for name in ('model.py', 'data.py', 'evaluate.py', 'upstream.py', 'checkpoint.py')]
    paths += [module_root.parent / 'vendor/lewm' / name for name in ('NOTICE.json', 'module.py', 'jepa.py')]
    paths += [Path(cem.__file__).parent / name for name in ('utils.py', 'solver.py')]
    paths += sorted(p for p in (ROOT / 'environments' / args.domain).iterdir()
                    if p.is_file() and (p.suffix in ('.sh', '.txt') or '.lock' in p.name))
    return {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'sources': {str(p): file_sha256(p) for p in paths},
            'learning_runtime': {'torch': str(torch.__version__), 'numpy': np.__version__,
                                 'cuda': torch.version.cuda},
            'model_config': model.package_config}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--domain', choices=['drone', 'surgery'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--kind', choices=['planning', 'forecast'], default='planning')
    parser.add_argument('--feature-cache')
    parser.add_argument('--split', choices=['development', 'test'], default='development')
    parser.add_argument('--episodes-per-gain', type=int, default=8)
    parser.add_argument('--policy', choices=['world_model', 'random', 'replay_oracle'], default='world_model')
    parser.add_argument('--planner-seed', type=int, default=101)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.episodes_per_gain < 1:
        parser.error('Positive episode count required')
    torch.set_num_threads(4)
    model, state = load_package(args.checkpoint, args.device)
    if state['epoch'] < 1 or state['config']['metadata']['config']['epochs'] != 30:
        raise ValueError('Use a checkpoint from the registered full training campaign')
    summary = json.loads((Path(args.checkpoint).parent / 'training_summary.json').read_text())
    if summary.get('status') != 'completed' or summary.get('completed_epochs') != 30:
        raise ValueError('Full training must finish before evaluation')
    args.output.mkdir(parents=True, exist_ok=True)
    identity = evaluation_identity(args, model)
    identity_path = args.output / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Evaluation sources or protocol changed; use a new output directory')
    atomic_json(identity, identity_path)
    if args.kind == 'forecast':
        result = evaluate_forecasts(model, args.data, args.split, feature_cache=args.feature_cache,
                                    num_workers=0, stride=4)
        atomic_json({'status': 'completed', 'identity': identity, **result}, args.output / 'results.json')
        return
    tasks, selection = tasks_for_split(args.domain, args.data, args.split, args.episodes_per_gain)
    rows = []
    with SimulatorClient(args.domain, args.output / 'simulator.log') as client:
        for task in tasks:
            key = task['trajectory_id'] + f"-o{task['observation_id']}"
            path = args.output / (key + '.json')
            arrays_path = args.output / (key + '.npz')
            if path.exists():
                row = json.loads(path.read_text())
                if file_sha256(arrays_path) != row['trace_sha256']:
                    raise ValueError('Saved planning trace changed')
            else:
                row, arrays = evaluate_task(model, client, task, args.domain, args.policy,
                                            planner_seed=args.planner_seed)
                np.savez_compressed(arrays_path, **arrays)
                row.update(trace_file=arrays_path.name, trace_sha256=file_sha256(arrays_path))
                atomic_json(row, path)
            rows.append(row)
            print(json.dumps({'task': key, 'success': row['success'], 'native_calls': row['native_calls'],
                              'final_distance_m': row['final_distance_m']}), flush=True)
    result = {'status': 'completed', 'identity': identity, 'selection': selection, 'records': rows,
              'successes': sum(r['success'] for r in rows), 'tasks': len(rows),
              'support_successes': sum(r['success_during_support'] for r in rows),
              'claim_scope': 'adapted simulation development tasks; not clinical evidence or original leaderboard'}
    atomic_json(result, args.output / 'results.json')
    print(json.dumps({k: result[k] for k in ('status', 'successes', 'tasks', 'support_successes')}), flush=True)


if __name__ == '__main__':
    main()
