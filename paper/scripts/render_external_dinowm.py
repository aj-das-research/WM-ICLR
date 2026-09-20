#!/usr/bin/env python3
"""Complete-only presentation of the frozen adapted DINO-WM development study.

Live mode verifies the immutable reporter's completed 6+15 run evidence. It does
not finalize an incomplete study or train/evaluate models. --from-pack replays the
source-bound JSON beside these tables without checkpoints or feature caches.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path('reports/external_dinowm_reporting_v1')
OUT = Path('paper/generated/external_dinowm')
REPORTER = Path('scripts/external_dinowm_reporting_v1/finalize.py')
REPORTER_SHA = '0367ff732b2febd66c47bb90390294dd6b2c4a2f89e1604f4ad03ac704c9ac52'
REPORTING_SHA = '6d612346cb0ebadec37d2cbee8ff6f5e8bc54d8ab2d4dd61ef17c0ebd64c5e87'
TRAINING_SHA = '4594f4c69914fea835cbaa0eb20639a8312c91e3f4746eeb77e82f9e2a94f343'
INTERNAL = ('autoregressive', 'anchored_additive', 'transport', 'context_off', 'action_free')
EXTERNAL = ('official_one_step_shifted', 'matched_recursive_h10')
MODES = (*INTERNAL, *EXTERNAL)
METRICS = ('native_mse', 'native_persistence_mse', 'original_2x2_mse', 'original_2x2_persistence_mse')
LABELS = dict(zip(MODES, ('Autoregressive', 'Additive anchor', 'ShiftWM (ours)',
    'No context (ours, ablation)', 'No actions (ours, ablation)',
    'DINO-WM: shifted one-step', 'DINO-WM: recursive H10')))
GRID = (('native_mse', 5), ('native_mse', 10), ('original_2x2_mse', 5), ('original_2x2_mse', 10))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'


def atomic(path, text):
    path = Path(path)
    if path.exists() and path.read_text() == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
        stream.write(text)
        temp = stream.name
    os.replace(temp, path)


def vector(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (10,) and np.isfinite(values).all() and (values >= 0).all(),
            'Incomplete or invalid ten-horizon error curve')
    return values


def close(a, b):
    require(np.allclose(a, b, rtol=1e-11, atol=1e-13), 'Displayed aggregate differs from its source values')


def validate(value):
    require(value.get('schema') == 'external_dinowm_complete_development_comparison_v1'
            and value.get('status') == 'passed', 'Invalid completed comparison')
    require(value.get('completed_external_runs') == 6 and value.get('completed_internal_runs') == 15
            and value.get('epochs_per_model') == 30, 'Full six plus fifteen 30-epoch models required')
    require(value.get('reporting_registration_sha256') == REPORTING_SHA
            and value.get('training_registration_sha256') == TRAINING_SHA, 'Wrong frozen study')
    require(value.get('new_inference_performed') is False and value.get('reserved_or_test_payloads_read') == 0,
            'Wrong development scope')
    policy = value['policy']
    require(policy['horizons'] == list(range(1, 11)) and policy['metrics'] == list(METRICS)
            and policy['objectives'] == list(EXTERNAL) and policy['internal_modes'] == list(INTERNAL)
            and policy['seeds'] == [0, 1, 2], 'Changed reporting policy')
    require(policy['bootstrap']['draws'] == 10000 and policy['bootstrap']['seed'] == 20260919,
            'Changed paired bootstrap')
    runs = value['per_run']
    require(len(runs) == 21 and {(r['mode'], r['seed']) for r in runs} ==
            {(m, s) for m in MODES for s in range(3)}, 'Incomplete or duplicate model roster')
    for r in runs:
        require(r['name'] == f"{r['mode']}_s{r['seed']}" and r['completed_epochs'] == 30
                and 1 <= r['selected_epoch'] <= 30 and r['external'] == (r['mode'] in EXTERNAL)
                and r['windows'] == 1631 and r['episodes'] == 141, 'Changed selected model/population')
        require(value['source_dependencies'].get(r['evaluation_path']) == r['evaluation_sha256'],
                'Unbound primitive evaluation evidence')
    result = value['results']
    require(result['population'] == {'windows_per_run': 1631, 'eligible_episodes': 141, 'sessions': 59},
            'Changed DROID development population')
    aggregate, seeds = result['aggregate'], result['per_seed']
    require(set(aggregate) == set(seeds) == set(MODES), 'Missing method')
    for m in MODES:
        require(set(aggregate[m]) == set(METRICS) and set(seeds[m]) == {'0', '1', '2'}, 'Missing metric/seed')
        for metric in METRICS:
            curve = vector(aggregate[m][metric])
            close(curve, np.mean([vector(seeds[m][str(s)][metric]) for s in range(3)], axis=0))
            if 'persistence' in metric:
                close(curve, vector(aggregate['transport'][metric]))
    comparisons = result['comparisons']
    require(set(comparisons) == {m + '_vs_' + e for m in INTERNAL for e in EXTERNAL}, 'Missing contrast')
    for m in INTERNAL:
        for e in EXTERNAL:
            c = comparisons[m + '_vs_' + e]
            require(c['first_mode'] == m and c['second_mode'] == e and c['draws'] == 10000
                    and c['bootstrap_seed'] == 20260919 and c['episode_count'] == 141
                    and c['session_count'] == 59 and c['training_seed_count'] == 3
                    and c['primary'] == (m == 'transport') and set(c['metrics']) == set(METRICS),
                    'Changed contrast identity/population')
            for metric in METRICS:
                points = c['metrics'][metric]
                require(len(points) == 10, 'Missing contrast horizon')
                for index, p in enumerate(points):
                    a, b = aggregate[m][metric][index], aggregate[e][metric][index]
                    require(p['horizon'] == index + 1, 'Wrong horizon mapping')
                    close([p['first_mean'], p['second_mean'], p['mean_difference']], [a, b, a-b])
                    ci = np.asarray(p['ci95'], dtype=float)
                    require(ci.shape == (2,) and np.isfinite(ci).all() and ci[0] <= ci[1], 'Invalid signed interval')
                    require(p['interval_includes_zero'] == bool(ci[0] <= 0 <= ci[1]), 'Wrong interval label')
                    if b == 0:
                        require(p['relative_error_reduction_percent'] is None, 'Undefined ratio must remain undefined')
                    else:
                        close(p['relative_error_reduction_percent'], 100*(b-a)/b)
    return value


def live(root=ROOT):
    source = root / REPORT
    final, completion = source / 'finalization.json', source / 'completion.json'
    if not final.exists() and not completion.exists():
        return None
    require(final.is_file() and completion.is_file(), 'Partial finalization is not publishable')
    require(sha(root / REPORTER) == REPORTER_SHA, 'Frozen finalizer changed')
    require(sha(source / 'registration.json') == REPORTING_SHA, 'Reporter registration changed')
    spec = importlib.util.spec_from_file_location('_external_presentation_finalizer', root / REPORTER)
    reporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reporter)
    # Both completed files already exist: the unchanged reporter revalidates all
    # source/checkpoint/evaluation hashes, never starts a new aggregation here.
    status = reporter.main(if_ready=False, root=root)
    require(status['status'] == 'passed', 'Reporter did not confirm completion')
    value = validate(read(final))
    require(status['finalization_sha256'] == sha(final), 'Finalization changed while loading')
    files = [final, completion, source/'registration.json', source/'source_review.json', root/REPORTER]
    return {'schema': 'external_dinowm_presentation_pack_v1', 'finalization': value,
            'sources_sha256': {str(p.relative_to(root)): sha(p) for p in files}}


def from_pack(path):
    path = Path(path)
    manifest = read(path/'manifest.json')
    require(manifest.get('schema') == 'external_dinowm_presentation_manifest_v1'
            and manifest.get('status') == 'complete_6_external_15_internal'
            and manifest.get('data_sha256') == sha(path/'data.json'), 'Corrupt portable source pack')
    data = read(path/'data.json')
    require(data.get('schema') == 'external_dinowm_presentation_pack_v1', 'Wrong portable pack')
    validate(data['finalization'])
    require(hashlib.sha256(encoded(data['finalization']).encode()).hexdigest() ==
            data['sources_sha256'].get(str(REPORT/'finalization.json')),
            'Portable finalization no longer matches its frozen completion source')
    for rel, digest in ((str(REPORT/'registration.json'), REPORTING_SHA),
                        (str(REPORTER), REPORTER_SHA)):
        require(data['sources_sha256'].get(rel) == digest, 'Portable source identity changed')
    return data


def tables(value):
    result = value['results']
    aggregate = result['aggregate']
    rows = [(LABELS[m], [aggregate[m][k][h-1] for k, h in GRID]) for m in MODES]
    rows += [('Persistence', [aggregate['transport'][k.replace('_mse', '_persistence_mse')][h-1] for k, h in GRID])]
    minima = np.min([r[1] for r in rows], axis=0)
    scores = r'''% Complete source-bound development results; lower MSE is better.
\begin{table}[!htbp]\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrr@{}}\toprule
& \multicolumn{2}{c}{Native $4\!\times\!4$ MSE $\downarrow$}
& \multicolumn{2}{c}{Original $2\!\times\!2$ MSE $\downarrow$} \\
Method & $h=5$ & $h=10$ & $h=5$ & $h=10$ \\ \midrule
'''
    for label, numbers in rows:
        cells = [r'\textbf{' + f'{v:.6f}' + '}' if v == minima[i] else f'{v:.6f}' for i, v in enumerate(numbers)]
        scores += label + ' & ' + ' & '.join(cells) + r' \\' + '\n'
    scores += r'''\bottomrule\end{tabular}
\caption{Posthoc DROID development comparison with adapted DINO-WM.
Both external training objectives retain all three seeds and 30 epochs:
the official shifted one-step objective and matched recursive ten-step training.
All learned models use the same training/development split; means weight windows
within episode, 141 episodes, and three seeds equally (1,631 windows, 59 sessions).
Black bold marks the lowest point mean, not significance. Native and original
pooled coordinates have separate training scales; their errors are not interchangeable.
$h$ counts action blocks. This is not an official benchmark reproduction or a
control-success comparison.}
\label{tab:external-dinowm-scores}\end{table}
'''
    contrasts = r'''% Absolute paired intervals, not relative-reduction intervals.
\begin{table}[!htbp]\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llrrl@{}}\toprule
Coordinates & External objective & $h$ & Reduction (\%) & $\Delta$MSE [95\% CI] \\ \midrule
'''
    for metric, coordinate in (('native_mse', r'Native $4\!\times\!4$'), ('original_2x2_mse', r'Original $2\!\times\!2$')):
        for external, label in zip(EXTERNAL, ('Shifted one-step', 'Recursive H10')):
            for horizon in (5, 10):
                p = result['comparisons']['transport_vs_'+external]['metrics'][metric][horizon-1]
                gain = p['relative_error_reduction_percent']
                gain_text = 'undefined' if gain is None else f'{gain:+.2f}'
                if gain is not None and gain > 0:
                    gain_text = r'\textcolor[RGB]{0,102,51}{\textbf{' + gain_text + '}}'
                interval = f"{p['mean_difference']:+.6f} [{p['ci95'][0]:+.6f}, {p['ci95'][1]:+.6f}]"
                contrasts += f'{coordinate} & {label} & {horizon} & {gain_text} & {interval}' + r' \\' + '\n'
    contrasts += r'''\bottomrule\end{tabular}
\caption{ShiftWM (ours) versus each adapted DINO-WM objective on DROID development.
$\Delta$MSE is ShiftWM minus the named external comparator: negative favors ShiftWM.
Reduction is $100(\mathrm{external}-\mathrm{ShiftWM})/\mathrm{external}$;
green bold marks a favorable point reduction only. Intervals apply to absolute
MSE differences and resample recording sessions and matched training seeds
(10,000 draws, unadjusted 95\%). Adverse and zero-crossing results remain visible.
All ten horizons, both persistence controls and all five internal comparisons
are retained in the source pack.}
\label{tab:external-dinowm-contrasts}\end{table}
'''
    main_rows = '% Two complete external objectives; preserve existing internal rows.\n'
    main_gains = '% ShiftWM versus named external objective; signed point reductions.\n'
    for external, short in zip(EXTERNAL, ('shifted one-step', 'recursive H10')):
        values = [aggregate[external]['native_mse'][h-1] for h in (5, 10)]
        main_rows += f'Adapted DINO-WM ({short}) & {values[0]:.6f} & {values[1]:.6f}' + r' \\' + '\n'
        gains = []
        for h in (5, 10):
            gain = result['comparisons']['transport_vs_'+external]['metrics']['native_mse'][h-1]['relative_error_reduction_percent']
            text = 'undefined' if gain is None else f'{gain:+.2f}' + r'\%'
            if gain is not None and gain > 0:
                text = r'\positivegain{' + text + '}'
            gains.append(text)
        main_gains += f'ShiftWM gain vs DINO-WM ({short}) & ' + ' & '.join(gains) + r' \\' + '\n'
    return {'scorecard.tex': scores, 'contrasts.tex': contrasts,
            'external_main_rows.tex': main_rows, 'external_main_gains.tex': main_gains}


def main(if_ready=False, pack=None, output=None, root=ROOT):
    out = Path(output) if output is not None else root/OUT
    data = from_pack(pack) if pack is not None else live(root)
    if data is None:
        require(not (out/'evidence.json').exists(), 'Published display has lost completed source')
        if not if_ready:
            raise ValueError('Complete external-study finalization is not available')
        return {'status': 'pending', 'outputs_written': 0}
    rendered = tables(data['finalization'])
    # All validation precedes the first output mutation.
    atomic(out/'data.json', encoded(data))
    manifest = {'schema': 'external_dinowm_presentation_manifest_v1',
                'status': 'complete_6_external_15_internal', 'data_sha256': sha(out/'data.json'),
                'reporting_registration_sha256': REPORTING_SHA, 'training_registration_sha256': TRAINING_SHA}
    atomic(out/'manifest.json', encoded(manifest))
    for name, text in rendered.items():
        atomic(out/name, text)
    evidence = {'status': 'complete_verified', 'renderer_sha256': sha(Path(__file__)),
                'sources_sha256': data['sources_sha256'],
                'outputs_sha256': {name: sha(out/name) for name in ('data.json', 'manifest.json', *rendered)},
                'learned_methods': 7, 'learned_runs': 21, 'persistence_rows': 1,
                'displayed_score_cells': 32, 'displayed_paired_intervals': 8,
                'scope': 'posthoc development adapted official baseline; no official benchmark/SOTA claim'}
    atomic(out/'evidence.json', encoded(evidence))
    return {'status': 'rendered', 'output_directory': str(out), 'outputs': [*rendered, 'data.json', 'manifest.json', 'evidence.json']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--if-ready', action='store_true')
    parser.add_argument('--from-pack', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(json.dumps(main(args.if_ready, args.from_pack, args.output), sort_keys=True))
