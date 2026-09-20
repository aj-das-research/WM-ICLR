#!/usr/bin/env python3
"""Complete-only reserved IWS tables and a signed-comparison forest plot.

Live mode validates the frozen finalizer and raw error ledgers, then copies JSON
metric receipts into a portable pack. Public replay needs no checkpoint,
feature cache, raw dataset, model execution or network. Pending writes no results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / 'paper/figure_sources/iws_reserved_evidence'
OUT = ROOT / 'paper/generated/iws_reserved_evidence'
FINAL = ROOT / 'reports/real_video_iws_reserved_recovery_v2/finalization.json'
REGISTRATION = ROOT / 'configs/real_video_iws_reserved_recovery_v2/registration.json'
RECOVERY = {'command_gru_dispatch': 'one_native_row_per_call', 'same_weights': True,
            'same_normalization': True, 'same_tolerance': True, 'same_metrics': True,
            'same_examples': True, 'same_comparators': True, 'all36_rerun': True,
            'original_results_retained': True, 'future_targets_used_to_choose_backend': False,
            'accuracy_used_to_choose_backend': False, 'initial_reserved_access_precedes_this_revision': True}
TASKS = ('pusht', 'bimanual_box', 'bimanual_rope')
NAMES = dict(zip(TASKS, ('PushT', 'Box', 'Rope')))
MODES = ('autoregressive', 'anchored_additive', 'bounded_spatial_mix', 'unbounded_spatial_mix', 'persistence')
LABELS = dict(zip(MODES, ('Autoregressive', 'Additive anchor', 'ShiftWM (ours)', 'No tanh (ours, ablation)', 'Persistence')))
METRICS = ('standardized_mse', 'standardized_mae', 'raw_dinov2_l1', 'feature_cosine_distance')
METRIC_LABELS = dict(zip(METRICS, ('Standardized MSE', 'Standardized MAE', 'Raw feature L1', 'Cosine distance')))
PAIRS = {
    'bounded_vs_additive': ('bounded_spatial_mix', 'anchored_additive', 'ShiftWM / additive'),
    'bounded_vs_autoregressive': ('bounded_spatial_mix', 'autoregressive', 'ShiftWM / AR'),
    'bounded_vs_persistence': ('bounded_spatial_mix', 'persistence', 'ShiftWM / persistence'),
    'no_tanh_vs_bounded': ('unbounded_spatial_mix', 'bounded_spatial_mix', 'No tanh (ours) / ShiftWM'),
    'no_tanh_vs_autoregressive': ('unbounded_spatial_mix', 'autoregressive', 'No tanh (ours) / AR'),
}
PREFIX = 'reserved_comparisons'
RECOVERY_NOTE = ('All 36 predictors were rerun with the same row-wise command-GRU backend after an input-only '
                 'post-access numerical prefix audit; weights, examples, metrics and tolerances are unchanged.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def unique(pairs):
        result = dict(pairs)
        require(len(result) == len(pairs), 'Duplicate JSON keys')
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + v)))


def relative_path(root, name):
    name = Path(name)
    require(not name.is_absolute() and '..' not in name.parts and name.parts, 'Unsafe source path')
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), 'Source path escapes root')
    return path


def write_changed(path, payload):
    path = Path(path)
    if isinstance(payload, str):
        payload = payload.encode()
    if path.exists() and path.read_bytes() == payload:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_bytes(payload); temp.replace(path)
    return True


def live_pack(pack=PACK, if_ready=False):
    if not FINAL.exists():
        if if_ready:
            return None
        raise ValueError('Complete36-run reserved finalization is absent')
    registry = read(REGISTRATION)
    finalizer_path = ROOT / 'scripts/real_video_iws_reserved_recovery_v2/finalize.py'
    require(registry['dependencies'].get(str(finalizer_path.relative_to(ROOT))) == sha(finalizer_path), 'Frozen finalizer source differs')
    spec = importlib.util.spec_from_file_location('_reserved_presenter_frozen_finalizer', finalizer_path)
    finalizer = importlib.util.module_from_spec(spec); sys.modules[spec.name] = finalizer; spec.loader.exec_module(finalizer)
    checked = finalizer.evaluation.checked_registration(ROOT, REGISTRATION)
    finalizer.evaluation.check_grid(checked)
    value = read(FINAL)
    finalizer.verify_existing(ROOT, FINAL, value, checked, REGISTRATION)
    source_files = {'data.json': FINAL}
    runs = []
    for run in value['per_run']:
        receipt = relative_path(ROOT, run['evaluation_path'])
        ledger = relative_path(ROOT, run['window_ledger_path'])
        require(sha(receipt) == run['evaluation_sha256'] and sha(ledger) == run['window_ledger_sha256'], 'Finalized per-run source changed')
        evaluation_name = 'evaluations/' + run['name'] + '.json'
        source_files[evaluation_name] = receipt
        runs.append({'name': run['name'], 'evaluation_file': evaluation_name,
                     'local_primitive_ledger_sha256': run['window_ledger_sha256']})
    manifest = {'schema': 'iws_reserved_evidence_pack_recovery_v2', 'status': 'complete36_validated',
                'numerical_recovery': RECOVERY,
                'finalization_sha256': sha(FINAL), 'registration_sha256': sha(REGISTRATION),
                'files_sha256': {name: sha(path) for name, path in source_files.items()}, 'runs': runs,
                'per_trajectory_metric_receipts': 36, 'local_primitive_ledgers_validated': 36,
                'public_binary_dependencies': 0, 'unique_handles': 600, 'trajectories': 30,
                'source_finalizer_sha256': sha(finalizer_path), 'no_features_commands_targets_or_weights_in_pack': True}
    pack = Path(pack)
    if (pack / 'manifest.json').exists():
        require(read(pack / 'manifest.json') == manifest, 'Completed pack changed; use a reviewed new version')
        return load_pack(pack)
    pack.mkdir(parents=True, exist_ok=True)
    for name, source in source_files.items():
        destination = pack / name; destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            require(sha(destination) == sha(source), 'Existing result snapshot differs')
        else:
            shutil.copyfile(source, destination)
    write_changed(pack / 'manifest.json', json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    return load_pack(pack)


def recompute_intervals(arrays):
    """Independent presentation check of the frozen10k paired draw algorithm."""
    rng = np.random.default_rng(173); draws = 10000
    sampled = {t: {k: np.empty((draws, 5, 4)) for k in ('difference', 'gain_percent')} for t in TASKS}
    macro = np.empty((draws, 5, 4))
    stacked = {t: np.stack([arrays[t][m][:, :, -1, :] for m in MODES], axis=2) for t in TASKS}
    for i in range(draws):
        seeds = rng.integers(0, 3, size=3); gains = []
        for task in TASKS:
            trajectories = rng.integers(0, 10, size=10)
            means = stacked[task][seeds][:, trajectories].mean(axis=(0, 1))
            for ci, (method, comparator, _) in enumerate(PAIRS.values()):
                a, b = means[MODES.index(method)], means[MODES.index(comparator)]
                difference = a - b
                sampled[task]['difference'][i, ci] = difference
                sampled[task]['gain_percent'][i, ci] = np.divide(-100 * difference, b, out=np.full(4, np.nan), where=b != 0)
            gains.append(sampled[task]['gain_percent'][i])
        macro[i] = np.mean(gains, axis=0)
    def interval(x):
        return {'percentile95': np.quantile(x, [.025, .975]).tolist() if np.isfinite(x).all() else None,
                'undefined_draws': int((~np.isfinite(x)).sum())}
    return ({t: {metric: {name: {kind: interval(sampled[t][kind][:, ci, mi]) for kind in ('difference', 'gain_percent')}
                          for ci, name in enumerate(PAIRS)} for mi, metric in enumerate(METRICS)} for t in TASKS},
            {metric: {name: interval(macro[:, ci, mi]) for ci, name in enumerate(PAIRS)} for mi, metric in enumerate(METRICS)})


def validate_numerics(value, receipts):
    expected = {(task, mode, seed) for task in TASKS for mode in MODES[:-1] for seed in range(3)}
    rows = value['per_run']
    require(value.get('schema') == 'shiftwm_iws_reserved_recovery_finalization_v2' and value.get('status') == 'passed', 'Invalid finalization schema/status')
    require(value.get('numerical_recovery') == RECOVERY, 'Missing post-access numerical recovery disclosure')
    require(value.get('scope') == 'reserved_upstream_validation' and value.get('expected_runs') == value.get('completed_runs') == 36, 'Wrong finalization scope/count')
    require(len(rows) == 36 and {(r['task'], r['mode'], r['seed']) for r in rows} == expected, 'Incomplete36 grid')
    require(value.get('total_unique_handles') == 600 and value.get('total_unique_trajectories') == 30, 'Wrong reserved denominator')
    arrays = {t: {m: np.empty((3, 10, 59, 4)) for m in MODES} for t in TASKS}
    prefixes = {t: {m: np.empty((3, 10, 3, 4)) for m in MODES} for t in TASKS}
    handles = {t: {m: np.empty((3, 1, 59, 4)) for m in MODES} for t in TASKS}
    handle_prefixes = {t: {m: np.empty((3, 1, 3, 4)) for m in MODES} for t in TASKS}
    baselines = {}; normalization = {}; inputs = {}
    metric_keys = (*METRICS, *('persistence_' + m for m in METRICS), *('prefix_' + m for m in METRICS))
    for row in rows:
        task, mode, seed, name = (row[k] for k in ('task', 'mode', 'seed', 'name'))
        receipt, pop = receipts[name], value['populations'][task]
        require(receipt['schema'] == 'shiftwm_iws_reserved_recovery_evaluation_v2' and receipt['status'] == 'passed', 'Invalid evaluation receipt')
        require(receipt.get('numerical_recovery') == RECOVERY, 'Mixed or undisclosed numerical backend')
        for key in ('task', 'mode', 'seed', 'name', 'selected_epoch', 'package_kind'):
            require(receipt[key] == row[key], 'Evaluation identity differs')
        require(receipt['selected_checkpoint_sha256'] == row['checkpoint_sha256'] and receipt['registration_sha256'] == value['registration_sha256'], 'Selected checkpoint differs')
        require(receipt['population'] == pop and len(pop['episode_ids']) == 10 and len(pop['handles']) == 200, 'Population differs')
        ids = pop['episode_ids']; expected_pairs = [(ids.index(h['episode_id']), h['start']) for h in pop['handles']]
        require(len(set(expected_pairs)) == 200, 'Duplicate source handles')
        episodes = receipt['episodes']
        require([(e['episode_id'], e['handles']) for e in episodes] == [(r['episode_id'], r['handles']) for r in pop['records']], 'Trajectory denominator differs')
        weights = np.array([e['handles'] for e in episodes], dtype=np.float64)
        require(weights.sum() == 200 and (weights > 0).all(), 'Wrong handle counts')
        for key in metric_keys:
            width = 3 if key.startswith('prefix_') else 59
            x = np.array([e[key + '_by_offset'] for e in episodes], dtype=np.float64)
            require(x.shape == (10, width) and np.isfinite(x).all() and (x >= 0).all(), 'Invalid per-trajectory metric')
            require(x.mean(0).tolist() == receipt['equal_trajectory'][key], 'Equal-trajectory means differ')
            # Only the order of floating-point additions differs from the raw
            # handle ledger. Original authoritative means remain unchanged.
            require(np.allclose(np.average(x, axis=0, weights=weights), receipt['equal_handle'][key], rtol=2e-14, atol=2e-15), 'Weighted handle mean differs')
        baseline = np.stack([np.array([e['persistence_' + m + '_by_offset'] for e in episodes]) for m in METRICS], axis=-1)
        if task in baselines:
            require(np.array_equal(baselines[task], baseline), 'Persistence differs across methods')
            require(normalization[task] == receipt['normalization_sha256'], 'Normalization differs')
        baselines[task] = baseline; normalization[task] = receipt['normalization_sha256']
        input_rows = [{k: x[k] for k in ('first_handle', 'count', 'command_rows', 'initial_features_sha256', 'commands_sha256')} for x in receipt['invocations']]
        if task in inputs: require(inputs[task] == input_rows, 'Matched forecast inputs differ')
        inputs[task] = input_rows
        for mi, metric in enumerate(METRICS):
            arrays[task][mode][seed, :, :, mi] = [e[metric + '_by_offset'] for e in episodes]
            arrays[task]['persistence'][seed, :, :, mi] = [e['persistence_' + metric + '_by_offset'] for e in episodes]
            prefixes[task][mode][seed, :, :, mi] = [e['prefix_' + metric + '_by_offset'] for e in episodes]
            prefixes[task]['persistence'][seed, :, :, mi] = arrays[task]['persistence'][seed, :, :, mi][:, [13, 28, 43]]
            handles[task][mode][seed, 0, :, mi] = receipt['equal_handle'][metric]
            handles[task]['persistence'][seed, 0, :, mi] = receipt['equal_handle']['persistence_' + metric]
            handle_prefixes[task][mode][seed, 0, :, mi] = receipt['equal_handle']['prefix_' + metric]
            handle_prefixes[task]['persistence'][seed, 0, :, mi] = np.array(receipt['equal_handle']['persistence_' + metric])[[13, 28, 43]]
    intervals, macro_intervals = recompute_intervals(arrays)
    results = value['results']
    for task in TASKS:
        for mi, metric in enumerate(METRICS):
            actual = results['task_results'][task][metric]
            for label, data, short in (('equal_trajectory', arrays, prefixes), ('equal_handle', handles, handle_prefixes)):
                curves = {m: data[task][m][:, :, :, mi].mean(axis=(0, 1)) for m in MODES}
                for mode in MODES:
                    require(np.allclose(curves[mode], actual[label]['mean_all59_offsets'][mode], rtol=2e-14, atol=2e-15), 'All59 curves differ')
                for j, horizon in enumerate((15, 30, 45, 60)):
                    means = {m: float(curves[m][-1]) if horizon == 60 else float(short[task][m][:, :, j, mi].mean()) for m in MODES}
                    per_seed = {m: (data[task][m][:, :, -1, mi] if horizon == 60 else short[task][m][:, :, j, mi]).mean(axis=1).tolist() for m in MODES}
                    for mode in MODES:
                        require(math.isclose(means[mode], actual[label]['horizon_means'][str(horizon)][mode], rel_tol=2e-14, abs_tol=2e-15)
                                and np.allclose(per_seed[mode], actual[label]['per_seed_horizon_means'][str(horizon)][mode], rtol=2e-14, atol=2e-15), 'Horizon point means differ')
            for name, (method, comparator, _) in PAIRS.items():
                a = actual['equal_trajectory']['horizon_means']['60'][method]; b = actual['equal_trajectory']['horizon_means']['60'][comparator]
                effect = actual['h60_comparisons'][name]
                require(effect['method'] == method and effect['comparator'] == comparator and effect['method_mean'] == a and effect['comparator_mean'] == b, 'Contrast identity/means differ')
                require(effect['method_minus_comparator'] == a - b and effect['relative_error_reduction_percent'] == (None if b == 0 else 100 * (b - a) / b), 'Signed contrast differs')
                require(effect['paired95'] == intervals[task][metric][name], 'Paired task uncertainty differs')
    for metric in METRICS:
        for name in PAIRS:
            gains = [results['task_results'][t][metric]['h60_comparisons'][name]['relative_error_reduction_percent'] for t in TASKS]
            actual = results['macro_h60'][metric][name]
            require(actual['equal_task_relative_error_reduction_percent'] == (None if any(x is None for x in gains) else float(np.mean(gains))), 'Equal-task macro differs')
            require(actual['paired95'] == macro_intervals[metric][name], 'Shared-seed macro uncertainty differs')
    return {'all36_per_trajectory_receipts_reconstructed': True, 'local_raw_ledgers_verified_by_frozen_finalizer_before_snapshot': True,
            'public_binary_dependencies': 0, 'equal_handle_reassociation_tolerance': {'rtol': 2e-14, 'atol': 2e-15},
            'unique_handles': 600, 'trajectories': 30,
            'full_curve_means_checked': 7080, 'h60_score_cells': 60, 'signed_gain_intervals': 80,
            'absolute_difference_intervals': 60, 'bootstrap_draws': 10000, 'bootstrap_seed': 173}


def load_pack(pack=PACK):
    pack = Path(pack).resolve(); manifest = read(pack / 'manifest.json')
    require(manifest.get('schema') == 'iws_reserved_evidence_pack_recovery_v2' and manifest.get('status') == 'complete36_validated', 'Incomplete portable pack')
    require(manifest.get('numerical_recovery') == RECOVERY, 'Portable recovery scope differs')
    require(len(manifest.get('runs', [])) == 36 and manifest.get('per_trajectory_metric_receipts') == 36 and manifest.get('public_binary_dependencies') == 0, 'Incomplete portable roster')
    expected_files = {'data.json'} | {r['evaluation_file'] for r in manifest['runs']}
    require(len(expected_files) == 37 and set(manifest['files_sha256']) == expected_files, 'Missing/unexpected portable file')
    for name, expected in manifest['files_sha256'].items():
        require(sha(relative_path(pack, name)) == expected, 'Portable input hash differs: ' + name)
    require(sha(pack / 'data.json') == manifest['finalization_sha256'], 'Finalization snapshot differs')
    value = read(pack / 'data.json')
    require(value['registration_sha256'] == manifest['registration_sha256'], 'Portable registration differs')
    receipts = {}
    run_index = {r['name']: r for r in value['per_run']}
    require(len(run_index) == 36 and {r['name'] for r in manifest['runs']} == set(run_index), 'Portable run identities differ')
    for row in manifest['runs']:
        name = row['name']; source = run_index[name]
        require(manifest['files_sha256'][row['evaluation_file']] == source['evaluation_sha256'] and row['local_primitive_ledger_sha256'] == source['window_ledger_sha256'], 'Finalizer/source ledger binding differs')
        receipts[name] = read(relative_path(pack, row['evaluation_file']))
    checked = validate_numerics(value, receipts)
    return value, manifest, checked


def signed(value, decimals=2):
    return 'undefined' if value is None else f'{value:+.{decimals}f}'


def gain_cell(point, interval):
    if point is None:
        return r'\textemdash{} (undefined)'
    if interval is None:
        return '$' + signed(point) + r'\;[\mathrm{undefined}]$'
    return '$' + signed(point) + r'\;[' + rf'{interval[0]:+.2f},\,{interval[1]:+.2f}' + ']$'


def tables(value):
    results = value['results']; body = []
    body += [r'\begin{table}[t]\centering',
             r'\caption{Reserved upstream-validation feature forecasts at $H=60$ (stored offset 59). All 36 selected learned predictors and persistence use the same 200 original handles per task (ten trajectories); learned methods have three seeds. Means weight handles within trajectory, trajectories and seeds equally. Lower is better; black bold marks the actual lowest point mean only. MSE/MAE use fixed task-specific training scales; raw L1 and cosine distance use DINOv2 coordinates. The no-$\tanh$ component study was declared after development. Reserved observations did not select or tune weights. ' + RECOVERY_NOTE + '}',
             r'\label{tab:iws-reserved-scores}',r'\begingroup\fontsize{9}{10.8}\selectfont\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.10}',
             r'\begin{tabular}{@{}lrrrr@{}}\toprule Method & Std. MSE & Std. MAE & Raw L1 & Cosine \\ \midrule']
    for task in TASKS:
        body.append(r'\multicolumn{5}{@{}l}{\textbf{' + NAMES[task] + r'} --- 200 handles, 10 trajectories} \\')
        for mode in ('persistence', *MODES[:-1]):
            cells = []
            for metric in METRICS:
                means = results['task_results'][task][metric]['equal_trajectory']['horizon_means']['60']; x = means[mode]
                text = f'{x:.5f}'
                cells.append(r'\textbf{' + text + '}' if x == min(means.values()) else text)
            body.append(' & '.join([LABELS[mode], *cells]) + r' \\')
        if task != TASKS[-1]: body.append(r'\addlinespace[3pt]')
    body += [r'\bottomrule\end{tabular}\endgroup\end{table}', '']
    outputs = {'scores.tex': '\n'.join(body)}
    for metric in METRICS:
        caption = ('Reserved ' + METRIC_LABELS[metric] + r' relative error reductions at $H=60$, in percent. Each cell gives the signed point estimate and paired95\% interval (10,000 seed--trajectory draws, seed173). Positive favors the first method; all signs remain. Macro averages the three task-specific relative reductions with shared seed draws. The first MSE contrast is primary; other contrasts/metrics are unadjusted secondary analyses. AR means autoregressive; each slash denotes comparison, not division of metric values.')
        lines = [r'\begin{table}[t]\centering',r'\caption{' + caption + '}',r'\label{tab:iws-reserved-' + metric.replace('_', '-') + '-contrasts}',
                 r'\begingroup\fontsize{8}{9.6}\selectfont\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{1.15}',
                 r'\begin{tabular}{@{}lrrrr@{}}\toprule Comparison & PushT & Box & Rope & Macro \\ \midrule']
        for name, (_, _, label) in PAIRS.items():
            cells = []
            for task in TASKS:
                e = results['task_results'][task][metric]['h60_comparisons'][name]
                cells.append(gain_cell(e['relative_error_reduction_percent'], e['paired95']['gain_percent']['percentile95']))
            e = results['macro_h60'][metric][name]
            cells.append(gain_cell(e['equal_task_relative_error_reduction_percent'], e['paired95']['percentile95']))
            lines.append(' & '.join([label, *cells]) + r' \\')
        lines += [r'\bottomrule\end{tabular}\endgroup\end{table}', '']
        outputs['contrasts_' + metric + '.tex'] = '\n'.join(lines)
    return outputs


CAPTION = (r'\textbf{Reserved IWS feature forecasting: the declared comparisons.} '
    r'Points show $H=60$ relative standardized-MSE reductions, $100(b-m)/b$, for the row\textquotesingle s first method against its reference. '
    r'The first row (bounded ShiftWM, ours, versus additive anchoring) is primary; remaining rows are secondary, including our post-development no-$\tanh$ component study. '
    r'Bars are paired 95\% seed--trajectory bootstrap intervals (10,000 draws, seed 173). All three learned seeds and the same 200 original handles from ten trajectories per task are retained. '
    r'Means weight handles within trajectory, trajectories and seeds equally. Positive favors the first method; zero indicates equal point error. '
    r'Full method/metric means, equal-task macro effects and separate command-prefix endpoints accompany the source pack. ' + RECOVERY_NOTE + ' This evaluates feature forecasts, not robot control or RGB generation.')


def make_figure(value):
    plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8, 'axes.titlesize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'iws-reserved-evidence-v1', 'text.color': '#243447', 'axes.labelcolor': '#243447'})
    figure = plt.figure(figsize=(5.5, 2.6), facecolor='white'); axes = []
    all_values = [0.]
    for task in TASKS:
        for name in PAIRS:
            e = value['results']['task_results'][task]['standardized_mse']['h60_comparisons'][name]
            if e['relative_error_reduction_percent'] is not None: all_values.append(e['relative_error_reduction_percent'])
            ci = e['paired95']['gain_percent']['percentile95']
            if ci is not None: all_values.extend(ci)
    low, high = min(all_values), max(all_values); span = max(high - low, 1.)
    limits = (low - span * .12, high + span * .12)
    marks = []
    for ti, task in enumerate(TASKS):
        ax = figure.add_axes([.285 + ti * .235, .255, .205, .585]); axes.append(ax)
        ax.set(xlim=limits, ylim=(-.65, 4.65), yticks=range(5))
        ax.axhspan(3.55, 4.45, color='#EDF5EE', zorder=0)
        ax.axvline(0, color='#74808C', lw=.8, zorder=1)
        ax.set_yticklabels([]); ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', length=2.5, pad=3)
        ticks = MaxNLocator(nbins=3, min_n_ticks=2).tick_values(*limits)
        ax.set_xticks([tick for tick in ticks if limits[0] <= tick <= limits[1]])
        ax.spines[['top', 'left', 'right']].set_visible(False); ax.spines['bottom'].set_color('#7A8590')
        ax.set_title(f'{chr(97 + ti)}  {NAMES[task]}', loc='left', pad=7)
        for ci, (name, (method, _, label)) in enumerate(PAIRS.items()):
            y = 4 - ci; color = '#166534' if method == 'bounded_spatial_mix' else '#984A87'
            marker = 'o' if method == 'bounded_spatial_mix' else 'D'
            effect = value['results']['task_results'][task]['standardized_mse']['h60_comparisons'][name]
            point, interval = effect['relative_error_reduction_percent'], effect['paired95']['gain_percent']['percentile95']
            if point is not None:
                if interval is not None:
                    ax.hlines(y, interval[0], interval[1], color=color, lw=1.25)
                    ax.vlines(interval, y - .085, y + .085, color=color, lw=.8)
                ax.plot(point, y, marker=marker, ms=4, mew=.85, color=color,
                        markerfacecolor=color if ci == 0 else 'white', zorder=3)
            else:
                ax.text(.5, y, 'undefined', transform=ax.get_yaxis_transform(), ha='center', va='center', fontsize=8)
            marks.append({'task': task, 'contrast': name, 'gain_percent': point, 'interval95_percent': interval})
            if ti == 0:
                ax.text(-.13, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=8,
                        fontweight='bold' if ci == 0 else 'normal')
        if ti == 0:
            figure.text(.014, .93, 'First method / reference', fontsize=8, va='center')
    figure.text(.622, .07, 'Relative MSE reduction (%) · positive favors first method', ha='center', fontsize=8)
    figure.canvas.draw(); renderer = figure.canvas.get_renderer(); issues = []
    for text in figure.findobj(Text):
        if not text.get_visible() or not text.get_text(): continue
        box = text.get_window_extent(renderer)
        if text.get_fontsize() < 8: issues.append('Text below8pt: ' + text.get_text())
        if box.x0 < -.5 or box.x1 > figure.bbox.x1 + .5 or box.y0 < -.5 or box.y1 > figure.bbox.y1 + .5:
            issues.append('Text outside canvas: ' + text.get_text())
    tick_gaps = []
    for left, right in zip(axes, axes[1:]):
        a = [t.get_window_extent(renderer) for t in left.get_xticklabels() if t.get_visible() and limits[0] <= t.get_position()[0] <= limits[1]]
        b = [t.get_window_extent(renderer) for t in right.get_xticklabels() if t.get_visible() and limits[0] <= t.get_position()[0] <= limits[1]]
        if a and b:
            gap = (min(x.x0 for x in b) - max(x.x1 for x in a)) * 72 / figure.dpi; tick_gaps.append(gap)
            if gap < 3: issues.append('Adjacent tick labels closer than3pt')
    require(not issues, 'Reserved figure layout rejected: ' + json.dumps(issues))
    return figure, {'dimensions_inches': [5.5, 2.6], 'minimum_font_pt': 8, 'marks': marks,
                    'shared_gain_axis_limits': list(limits), 'adjacent_tick_gap_pt': tick_gaps, 'issues': []}


def render(if_ready=False, from_pack=False, pack=PACK, output=OUT):
    loaded = load_pack(pack) if from_pack else live_pack(pack, if_ready)
    if loaded is None:
        return {'status': 'pending', 'outputs_written': False, 'reason': 'Complete36 reserved finalization absent'}
    value, manifest, validation = loaded; output = Path(output)
    font = Path(font_manager.findfont(font_manager.FontProperties(family='Liberation Sans'), fallback_to_default=False))
    bound = {'renderer_sha256': sha(__file__), 'pack_manifest_sha256': sha(Path(pack) / 'manifest.json'),
             'finalization_sha256': manifest['finalization_sha256'], 'font_sha256': sha(font),
             'matplotlib': matplotlib.__version__, 'numpy': np.__version__}
    fingerprint = digest(bound); evidence_path = output / 'evidence.json'
    if evidence_path.exists():
        old = read(evidence_path)
        require(old.get('fingerprint') == fingerprint, 'Existing presentation differs; use a newly reviewed output version')
        for name, expected in old['outputs_sha256'].items(): require(sha(output / name) == expected, 'Presentation output changed')
        return {'status': 'unchanged_verified', 'outputs_written': False}
    figure, geometry = make_figure(value)
    output.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.reserved-evidence-', dir=output))
    try:
        outputs = tables(value)
        outputs[PREFIX + '_figure.tex'] = (r'\begin{figure}[t]\centering' + '\n' + r'\includegraphics[width=\linewidth]{generated/iws_reserved_evidence/' + PREFIX + '.pdf}\n' + r'\caption{' + CAPTION + '}\n' + r'\label{fig:iws-reserved-comparisons}\end{figure}' + '\n')
        names = []
        for ext in ('pdf', 'svg', 'png'):
            name = PREFIX + '.' + ext; names.append(name)
            figure.savefig(stage / name, dpi=300, metadata={'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {'Date': None} if ext == 'svg' else None)
        for name, text in outputs.items():
            (stage / name).write_text(text); names.append(name)
        plt.close(figure)
        evidence = {'schema': 'iws_reserved_presentation_v1', 'status': 'complete_data_rendered_visual_review_pending',
                    'fingerprint': fingerprint, 'source_bindings': bound, 'numerical_validation': validation, 'geometry': geometry,
                    'outputs_sha256': {name: sha(stage / name) for name in names}, 'generated_imagery': False,
                    'native_drawio': 'not applicable: quantitative vector plot',
                    'actual_pdf_color_grayscale_and_integrated_review': 'pending independent review of this exact snapshot'}
        for name in names:
            (stage / name).replace(output / name)
        write_changed(evidence_path, json.dumps(evidence, sort_keys=True, indent=2) + '\n')
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {'status': 'rendered', 'outputs_written': True, 'scores': 60, 'signed_gain_intervals': 80,
            'figure_task_contrasts': 15, 'visual_review': 'pending'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--if-ready', action='store_true')
    parser.add_argument('--from-pack', action='store_true')
    parser.add_argument('--pack-dir', type=Path, default=PACK)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(render(args.if_ready, args.from_pack, args.pack_dir, args.output_dir), sort_keys=True), flush=True)
