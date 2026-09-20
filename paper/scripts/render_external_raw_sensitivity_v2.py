#!/usr/bin/env python3
"""Display the completed raw-coordinate external sensitivity study, separately from v1.

Copies only existing finalized paired intervals. Does not use the audit's new
posthoc external-versus-persistence intervals or run inference/bootstrapping.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[2]
REPORT = 'reports/external_dinowm_raw_reporting_v2'
AUDIT = 'reports/qualitative_temporal_v1/new_training_completion_audit.json'
AUDIT_SHA = '2ef7ee742b047192e8a6ddba2496d209df8e2c0db9949e788e90b10dd0a21b37'
FINAL_SHA = 'f8fe4e30ff12fec8903b784ec271d8492e75f35da5321e8426613cfed9b624d8'
OUT = ROOT / 'paper/generated/external_dinowm_raw_v2'
MODES = ('official_raw_one_step', 'official_raw_recursive_h10')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def verify():
    paths = {name: ROOT / REPORT / (name + '.json') for name in
             ('finalization', 'completion', 'registration', 'source_review')}
    require(sha(ROOT / AUDIT) == AUDIT_SHA, 'Independent audit changed')
    require(sha(paths['finalization']) == FINAL_SHA, 'Completed raw-v2 report changed')
    records = {k: json.loads(p.read_text()) for k, p in paths.items()}
    final, done, reg, review = (records[k] for k in paths)
    audit = json.loads((ROOT / AUDIT).read_text())
    require(audit['status'] == final['status'] == done['status'] == review['status'] == 'passed',
            'Incomplete report or independent review')
    require(done['finalization_sha256'] == FINAL_SHA and
            done['results_md_sha256'] == sha(ROOT / REPORT / 'results.md'), 'Invalid completion receipt')
    require(final['reporting_registration_sha256'] == review['registration_sha256'] == sha(paths['registration']),
            'Reporting source-review binding differs')
    require(final['epochs_per_model'] == {'external': 100, 'internal': 30} and
            final['completed_external_runs'] == 6 and final['completed_internal_runs'] == 15,
            'Wrong complete study roster/budget')
    require(final['results']['population'] == {'eligible_episodes': 141, 'sessions': 59, 'windows_per_run': 1631},
            'Changed common population')
    for relative, expected in final['source_dependencies'].items():
        require(sha(ROOT / relative) == expected, 'Changed completed source: ' + relative)
    agg = final['results']['aggregate']
    rows = []
    for mode, label in [('transport', 'ShiftWM (ours), bounded'), ('persistence', 'Persistence'),
                        (MODES[0], 'DINO-WM, raw one-step'), (MODES[1], 'DINO-WM, raw recursive H10')]:
        curve = agg['transport']['native_persistence_mse'] if mode == 'persistence' else agg[mode]['native_mse']
        means = {f'h{h}': curve[h - 1] for h in (5, 10)}
        require(means == audit['native_endpoint_aggregate'][mode], 'Audited means differ')
        rows.append({'mode': mode, 'label': label, 'native_mse': means})
    contrasts = []
    for mode in MODES:
        saved = final['results']['comparisons']['transport_vs_' + mode]
        require(saved['draws'] == 10000 and saved['bootstrap_seed'] == 20260919,
                'Unexpected existing paired interval protocol')
        require(saved['first_mode'] == 'transport' and saved['second_mode'] == mode,
                'Reversed paired comparison')
        values = []
        for h in (5, 10):
            effect = saved['metrics']['native_mse'][h - 1]
            require(effect['horizon'] == h and len(effect['ci95']) == 2 and
                    effect['ci95'][0] <= effect['ci95'][1], 'Invalid existing paired interval')
            require(math.isclose(effect['first_mean'], agg['transport']['native_mse'][h - 1], rel_tol=1e-12)
                    and math.isclose(effect['second_mean'], agg[mode]['native_mse'][h - 1], rel_tol=1e-12),
                    'Paired contrast differs from displayed means')
            require(math.isclose(effect['mean_difference'], effect['first_mean'] - effect['second_mean'],
                                 rel_tol=1e-12, abs_tol=1e-14), 'Paired difference mismatch')
            values.append(effect)
        contrasts.append({'external_mode': mode, 'saved_effects': values})
    previous_rel = 'reports/external_dinowm_reporting_v1/finalization.json'
    require(sha(ROOT / previous_rel) == audit['source_sha256'][previous_rel], 'Prior v1 report changed')
    previous = json.loads((ROOT / previous_rel).read_text())
    prior_diagnostics = []
    for mode, old_mode in zip(MODES, ('official_one_step_shifted', 'matched_recursive_h10')):
        for h in (5, 10):
            current = agg[mode]['native_mse'][h - 1]
            old = previous['results']['aggregate'][old_mode]['native_mse'][h - 1]
            persistence = agg['transport']['native_persistence_mse'][h - 1]
            require(current > old and current > persistence, 'Appendix sensitivity interpretation must be revised')
            prior_diagnostics.append({'mode': mode, 'horizon': h, 'raw100_mean': current,
                                      'standardized30_mean': old, 'persistence_mean': persistence})
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in paths.values()}
    bindings[AUDIT] = AUDIT_SHA
    bindings[previous_rel] = sha(ROOT / previous_rel)
    bindings[REPORT + '/results.md'] = done['results_md_sha256']
    return {'schema': 'external_dinowm_raw_sensitivity_display_v2', 'status': 'complete_verified',
            'source_sha256': bindings, 'scientific_source_closure_count': len(final['source_dependencies']),
            'rows': rows, 'contrasts': contrasts, 'prior_recipe_checks': prior_diagnostics,
            'scope': final['policy']['scope'],
            'new_CIs_computed': False, 'audit_posthoc_persistence_CIs_used': False,
            'training_budgets': final['epochs_per_model'], 'population': final['results']['population']}


def render(data):
    text = r'''% Complete raw-coordinate follow-up; preserve the separate original v1 study.
\begin{table}[!htbp]\centering
\begingroup\fontsize{9}{10.8}\selectfont\setlength{\tabcolsep}{6pt}\renewcommand{\arraystretch}{1.10}
\begin{tabular}{@{}lrr@{}}\toprule
Method & Native h5 MSE $\downarrow$ & Native h10 MSE $\downarrow$ \\
\midrule
'''
    for row in data['rows']:
        cells = [row['label'], f"{row['native_mse']['h5']:.6f}", f"{row['native_mse']['h10']:.6f}"]
        if row['mode'] == 'transport':
            cells = [r'\textbf{' + cells[0] + '}',
                     *[r'\positivegain{' + c + '}' for c in cells[1:]]]
        text += ' & '.join(cells) + r' \\' + '\n'
    text += r'''\bottomrule\end{tabular}
\par\vspace{4pt}
\begin{tabular}{@{}lll@{}}\toprule
ShiftWM minus external & $10^3\Delta_5$ [95\% CI] & $10^3\Delta_{10}$ [95\% CI] \\
\midrule
'''
    for row, label in zip(data['contrasts'], ('Raw one-step', 'Raw recursive H10')):
        cells = [label]
        for effect in row['saved_effects']:
            lo, hi = effect['ci95']
            cells.append(f"${1000 * effect['mean_difference']:.3f}\;[{1000 * lo:.3f}, {1000 * hi:.3f}]$")
        text += ' & '.join(cells) + r' \\' + '\n'
    return text + r'''\bottomrule\end{tabular}\endgroup
\caption{\textbf{Raw-coordinate DINO-WM follow-up on DROID development.}
The adapted external models complete 100 epochs per seed; the fixed bounded
ShiftWM reference completes 30, and persistence has no learned predictor.
Both external training objectives use raw pooled DINO features, while all shown
errors use the same training-standardized $4\times4$ evaluation coordinates,
141 episodes and three seeds. Existing paired recording-session/seed intervals
(10,000 draws, unadjusted) apply to absolute ShiftWM-minus-external MSE differences;
negative favors ShiftWM. Green bold marks the lowest shown endpoint errors.
Both external objectives still underperform persistence.
This is an adaptation sensitivity study with unequal training budgets, not an
official benchmark reproduction or a state-of-the-art claim.}
\label{tab:external-dinowm-raw-sensitivity}
\end{table}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    data = verify()
    args.output.mkdir(parents=True, exist_ok=True)
    table = args.output / 'sensitivity.tex'
    table.write_text(render(data))
    data['renderer_sha256'] = sha(Path(__file__))
    data['table_sha256'] = sha(table)
    (args.output / 'evidence.json').write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'complete_verified', 'score_rows': 4, 'existing_paired_intervals': 4,
                      'scientific_source_closure_count': data['scientific_source_closure_count'],
                      'new_CIs_computed': False, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
