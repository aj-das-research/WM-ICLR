#!/usr/bin/env python3
"""Render the complete closest-comparator audit; never opens model/data payloads.

Only copied existing paired intervals are shown. Rankings and signed relative
reductions are recomputed from full-precision completed finalizer means. The
bound audit additionally verifies the complete primitive score ledgers.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[2]
AUDIT = 'reports/qualitative_closest_v1/comparator_audit.json'
AUDIT_SHA256 = '1026351ab078080bd84e0fd4f907c147dfaaa4438019d4502badd7c29ed690cf'
OUT = ROOT / 'paper/generated/qualitative_closest_v1'
TASKS = ('pusht', 'bimanual_box', 'bimanual_rope')
METRICS = ('standardized_mse', 'standardized_mae', 'raw_dinov2_l1', 'feature_cosine_distance')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def close(first, second):
    return math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-14)


def gain(method, reference):
    require(reference > 0 and math.isfinite(reference) and math.isfinite(method),
            'Invalid relative-gain denominator')
    return 100 * (reference - method) / reference


def verify(root=ROOT):
    audit_path = root / AUDIT
    require(sha(audit_path) == AUDIT_SHA256, 'Closest-comparator audit changed')
    audit = load(audit_path)
    require(audit['status'] == 'complete_read_only_audit', 'Incomplete audit')
    for relative, expected in audit['source_sha256'].items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root.resolve()), 'Nonlocal audit source')
        require(sha(path) == expected, f'Changed audited source: {relative}')
    sources = {'audit': {'path': AUDIT, 'sha256': AUDIT_SHA256},
               'completed_sources': audit['source_sha256']}
    spatial = load(root / 'reports/real_video_spatial/finalization.json')
    components = load(root / 'reports/real_video_spatial_components/finalization.json')
    external = load(root / 'reports/external_dinowm_reporting_v1/finalization.json')
    reserved = load(root / 'reports/real_video_iws_reserved_recovery_v2/finalization.json')
    require(all(x['status'] == 'passed' for x in (spatial, components, external, reserved)),
            'One complete campaign no longer passes')
    require((spatial['completed_models'], components['completed_new_models'],
             external['completed_external_runs'], reserved['completed_runs']) == (15, 6, 6, 36),
            'Incomplete study roster')
    agg = {**spatial['aggregate'], **components['aggregate'], **external['results']['aggregate']}
    agg['persistence'] = {'native_mse': spatial['aggregate']['transport']['native_persistence_mse']}
    droid = []
    for h, audited in zip((5, 10), audit['droid']['ranking']):
        rank = sorted([{'method': m, 'mean': x['native_mse'][h - 1]} for m, x in agg.items()],
                      key=lambda x: (x['mean'], x['method']))
        require(rank == audited['ranked_all_reported_methods'], 'DROID audited ranking changed')
        closest = next(r for r in rank if r['method'] != 'transport')
        require(closest['method'] == 'unbounded_transport', 'DROID closest component changed')
        primary = agg['transport']['native_mse'][h - 1]
        comp = next(x for x in components['reported_effects']
                    if x.get('contrast') == 'bounding_with_mixing'
                    and x['horizon'] == h and x['metric'] == 'native_mse')
        ar = next(x for x in spatial['paired_effects']
                  if x['method'] == 'transport' and x['comparator'] == 'autoregressive'
                  and x['horizon'] == h and x['metric'] == 'native_mse')
        require(comp == audited['primary_vs_closest_saved_paired_effect'], 'Changed component interval')
        require(ar == audited['primary_vs_autoregressive_saved_paired_effect'], 'Changed AR interval')
        for effect in (comp, ar):
            require(close(effect['method_mean'], primary), 'DROID primary mean mismatch')
            require(close(effect['relative_error_reduction_percent'],
                          gain(effect['method_mean'], effect['comparator_mean'])), 'DROID gain mismatch')
            interval = effect['paired_95_percent_interval']
            require(len(interval) == 2 and all(math.isfinite(v) for v in interval)
                    and interval[0] <= interval[1], 'Invalid saved DROID interval')
        droid.append({'horizon': h, 'metric': 'native_mse', 'primary_mean': primary,
                      'ranked_methods': rank, 'closest_component': comp, 'fixed_autoregressive_baseline': ar})
    iws_rows = []
    row_specs = (('pusht', 'bounded_spatial_mix', 'bounded (ours)'),
                 ('pusht', 'autoregressive', 'AR'),
                 ('bimanual_box', 'autoregressive', 'AR'),
                 ('bimanual_rope', 'autoregressive', 'AR'))
    labels = {'pusht': 'PushT', 'bimanual_box': 'Box', 'bimanual_rope': 'Rope'}
    for task, reference, ref_label in row_specs:
        row = {'task': task, 'label': labels[task], 'method': 'unbounded_spatial_mix',
               'method_role': 'secondary_no_tanh_ablation', 'reference': reference,
               'reference_label': ref_label, 'metrics': {}}
        for metric in METRICS:
            data = reserved['results']['task_results'][task][metric]
            means = data['equal_trajectory']['horizon_means']['60']
            rank = sorted([{'method': m, 'mean': v} for m, v in means.items()],
                          key=lambda x: (x['mean'], x['method']))
            audited = next(x for x in audit['iws_reserved']['ranking']
                           if x['task'] == task and x['metric'] == metric)
            require(rank == audited['ranked_all_reported_methods'], 'IWS audited ranking changed')
            comp = next(x for x in data['h60_comparisons'].values()
                        if x['method'] == row['method'] and x['comparator'] == reference)
            require(close(comp['relative_error_reduction_percent'],
                          gain(means[row['method']], means[reference])), 'IWS signed gain mismatch')
            ci = comp['paired95']['gain_percent']['percentile95']
            require(len(ci) == 2 and all(math.isfinite(v) for v in ci) and ci[0] <= ci[1],
                    'Invalid saved IWS interval')
            require(comp['paired95']['gain_percent']['undefined_draws'] == 0,
                    'Undefined saved IWS bootstrap draw')
            row['metrics'][metric] = {'means': means, 'existing_paired_effect': comp,
                                      'reference_is_actual_runner_up': reference == rank[1]['method'],
                                      'reference_is_fixed_matched_autoregression': reference == 'autoregressive'}
        iws_rows.append(row)
    return {'schema': 'shiftwm_closest_comparator_display_v1', 'status': 'complete_verified',
            'droid': droid, 'iws_reserved': iws_rows, 'source_bindings': sources,
            'interpretation': {
                'droid': 'Primary bounded versus closest alternative and fixed AR; no-tanh wins h5, primary bounded wins h10. Component intervals cross zero; this does not establish equivalence.',
                'iws': 'No-tanh remains a secondary ablation. PushT runner-up is bounded primary; Box/Rope runner-up is AR. AR is the fixed matched baseline across all qualitative cases, without an ownership-based best-control claim.',
                'rank_and_uncertainty': 'Ranking is descriptive. All displayed paired intervals are copied from complete studies, secondary and unadjusted; rank selection does not make them confirmatory.',
                'highlight': 'Green bold marks strictly positive point gains only, not significance.',
                'scales': 'DROID intervals are absolute MSE differences scaled by 1000. IWS intervals are relative error reductions in percent. These are distinct units.'}}


def gain_tex(value, places=2):
    text = f'{value:+.{places}f}' + r'\%'
    return r'\positivegain{' + text + '}' if value > 0 else text


def interval_tex(values, scale=1, places=2):
    # Math mode gives an actual minus glyph and avoids hyphen ambiguity.
    return '$[' + ', '.join(f'{v * scale:.{places}f}' for v in values) + ']$'


def render(data):
    tex = r'''% Generated from complete, source-bound results. Do not edit numeric cells.
\begin{table}[!htbp]\centering
\begingroup\fontsize{9}{10.8}\selectfont\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.15}
\begin{tabular}{@{}lrrrr@{}}
\toprule
\multicolumn{5}{@{}l}{\textbf{A. DROID development: ShiftWM (ours), bounded primary}} \\
& \multicolumn{2}{c}{Versus no-$\tanh$ (ours, ablation)} & \multicolumn{2}{c}{Versus autoregression} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
Endpoint & Gain & $10^3\Delta$ MSE [95\% CI] & Gain & $10^3\Delta$ MSE [95\% CI] \\
\midrule
'''
    for row in data['droid']:
        cells = [f"h{row['horizon']}"]
        for key in ('closest_component', 'fixed_autoregressive_baseline'):
            c = row[key]
            cells += [gain_tex(c['relative_error_reduction_percent'], 3),
                      interval_tex(c['paired_95_percent_interval'], 1000, 3)]
        tex += ' & '.join(cells) + r' \\' + '\n'
    tex += r'''\bottomrule\end{tabular}
\par\vspace{4pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
\multicolumn{5}{@{}l}{\textbf{B. Reserved IWS: no-$\tanh$ (ours, ablation), relative error reductions}} \\
Task / reference & Std. MSE & Std. MAE & Raw L1 & Cosine \\
\midrule
'''
    for row in data['iws_reserved']:
        cells = [row['label'] + ' / ' + row['reference_label']]
        for metric in METRICS:
            c = row['metrics'][metric]['existing_paired_effect']
            cells.append(r'\shortstack{' + gain_tex(c['relative_error_reduction_percent']) + r'\\' +
                         r'\fontsize{8}{9.6}\selectfont ' + interval_tex(c['paired95']['gain_percent']['percentile95']) + '}')
        tex += ' & '.join(cells) + r' \\' + '\n'
    tex += r'''\bottomrule\end{tabular}\endgroup
\caption{\textbf{Gains against the closest alternatives.}
(A) At h5 no-$\tanh$ ranks first and bounded ShiftWM second; h10 reverses this
order, with both paired intervals spanning zero. $\Delta$ denotes bounded minus
reference MSE. (B) No-$\tanh$ is lowest in all twelve reserved $H=60$ cells.
The runner-up is bounded ShiftWM on PushT and AR on Box/Rope. AR is the fixed
matched autoregressive baseline throughout; this does not rank additive controls
below AR in DROID. Intervals are existing paired 95\% estimates:
absolute differences in A, relative gains (\%) in B. Rank comparisons are
secondary, unadjusted and descriptive. Green bold marks positive point gains.
No-$\tanh$ remains an ablation.}
\label{tab:closest-comparator-gains}
\end{table}
'''
    return tex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    data = verify()
    tex = render(data)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'closest_table.tex'
    target.write_text(tex)
    data['renderer_sha256'] = sha(Path(__file__))
    data['tex_sha256'] = sha(target)
    (args.output / 'closest_table.json').write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'complete_verified', 'output': str(args.output),
                      'droid_endpoint_rows': 2, 'iws_comparison_rows': 4,
                      'copied_paired_intervals': 20, 'new_bootstrap_computations': 0}))


if __name__ == '__main__':
    main()
