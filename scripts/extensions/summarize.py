#!/usr/bin/env python3
"""Report extension progress and all paired development results, including losses."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from shiftwm.extensions.checkpoint import atomic_json, file_sha256


def paired_comparison(ours, baseline):
    """Crossed resampling of matched training seeds and task seeds, no unpairing."""
    seeds = sorted(set(ours) & set(baseline))
    if seeds != [0, 1, 2]:
        return {'status': 'pending_all_three_training_seeds', 'available_paired_seeds': seeds}
    key = lambda row: (row['seed'], row['dynamics_id'], row['observation_id'])
    keys = [set(map(key, result['records'])) for result in list(ours.values()) + list(baseline.values())]
    if any(k != keys[0] for k in keys[1:]):
        raise ValueError('Paired model comparisons have different physical tasks')
    order = sorted(keys[0])
    physical = sorted({k[0] for k in order})
    values = []
    for seed in seeds:
        a, b = ({key(r): float(r['success']) for r in results[seed]['records']}
                for results in (ours, baseline))
        values.append([np.mean([a[k] - b[k] for k in order if k[0] == task]) for task in physical])
    values = np.asarray(values)
    rng = np.random.default_rng(417)
    samples = []
    for _ in range(5000):
        training = rng.integers(len(seeds), size=len(seeds))
        tasks = rng.integers(len(physical), size=len(physical))
        samples.append(values[np.ix_(training, tasks)].mean())
    return {'status': 'complete', 'success_difference_percentage_points': float(100 * values.mean()),
            'ci95_percentage_points': (100 * np.quantile(samples, [.025, .975])).tolist(),
            'per_training_seed_difference_percentage_points': (100 * values.mean(1)).tolist(),
            'training_seeds': len(seeds), 'physical_task_seeds': len(physical),
            'interval': 'paired crossed bootstrap over training and physical task seeds; 5000 draws; unadjusted exploratory 95% interval',
            'consistent_positive_in_all_training_seeds': bool((values.mean(1) > 0).all())}


def summarize(result_root):
    campaign = json.loads(Path('configs/extensions/campaign.json').read_text())
    runs, grouped, evidence = [], {}, {}
    for row in campaign['runs']:
        output = Path(row['output'])
        summary_path = output / 'training_summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        record = {**row, 'training_status': summary.get('status', 'pending_or_running'),
                  'completed_epochs': summary.get('completed_epochs', 0),
                  'best_validation_mse': summary.get('best_validation_prediction_loss'),
                  'checkpoint_path': str(output / 'best') if (output / 'best').exists() else None}
        for kind in ('forecast', 'planning'):
            result_path = Path(result_root) / output.name / 'development' / kind / 'results.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result.get('status') != 'completed':
                    raise ValueError('Incomplete results cannot be summarized as complete')
                evidence[str(result_path)] = file_sha256(result_path)
                record[kind + '_status'] = 'completed'
                if kind == 'planning':
                    record.update(successes=result['successes'], tasks=result['tasks'],
                                  support_successes=result['support_successes'])
                    grouped.setdefault((row['domain'], row['architecture'], row['mode']), {})[row['seed']] = result
                else:
                    record['forecast_mse_h5'] = result['summary']['o1_d1']['mse_h5']['mean']
            else:
                record[kind + '_status'] = 'pending_or_running'
        runs.append(record)
    comparisons = []
    for domain in ('drone', 'surgery'):
        for architecture in ('transformer', 'gru'):
            ours = grouped.get((domain, architecture, 'factorized'), {})
            for mode in ('framewise', 'constant_dynamics'):
                baseline = grouped.get((domain, architecture, mode), {})
                comparisons.append({'domain': domain, 'architecture': architecture, 'baseline': mode,
                                    **paired_comparison(ours, baseline)})
    return {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'registered extension development only; not final test or SOTA evidence',
            'planned_training_runs': len(runs),
            'completed_training_runs': sum(r['training_status'] == 'completed' and r['completed_epochs'] == 30 for r in runs),
            'completed_forecasts': sum(r['forecast_status'] == 'completed' for r in runs),
            'completed_planning_evaluations': sum(r['planning_status'] == 'completed' for r in runs),
            'runs': runs, 'comparisons': comparisons, 'result_sources_sha256': evidence}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--result-root', default='results/extensions_v1')
    parser.add_argument('--output', default='reports/domain_extension_progress')
    args = parser.parse_args()
    report = summarize(args.result_root)
    atomic_json(report, args.output + '.json')
    lines = ['# Domain extension development progress', '', report['generated_at_utc'], '',
             f"Full training: {report['completed_training_runs']}/{report['planned_training_runs']}; "
             f"forecast evaluations: {report['completed_forecasts']}; "
             f"closed-loop evaluations: {report['completed_planning_evaluations']}.", '',
             'These are development results. All success/failure outcomes are included. '
             'Full training does not by itself establish a control gain or state of the art.', '',
             '| Domain | Predictor | Method | Seed | Training | Forecast MSE@5 | Planning successes |',
             '|---|---|---|---:|---|---:|---:|']
    labels = {'framewise': 'Framewise', 'constant_dynamics': 'Constant dynamics', 'factorized': 'ShiftWM (ours)'}
    for row in report['runs']:
        forecast = f"{row['forecast_mse_h5']:.6f}" if 'forecast_mse_h5' in row else 'pending'
        planning = f"{row['successes']}/{row['tasks']}" if 'successes' in row else 'pending'
        lines.append(f"| {row['domain']} | {row['architecture']} | {labels[row['mode']]} | {row['seed']} | "
                     f"{row['completed_epochs']}/30 ({row['training_status']}) | {forecast} | {planning} |")
    lines += ['', 'Paired comparisons (all three training seeds required):', '',
              '| Domain | Predictor | Comparator | Success difference, pp | 95% interval, pp |',
              '|---|---|---|---:|---|']
    for row in report['comparisons']:
        if row['status'] == 'complete':
            delta = f"{row['success_difference_percentage_points']:+.2f}"
            ci = '[%+.2f, %+.2f]' % tuple(row['ci95_percentage_points'])
        else:
            delta, ci = 'pending', 'pending'
        lines.append(f"| {row['domain']} | {row['architecture']} | {labels[row['baseline']]} | {delta} | {ci} |")
    lines += ['', 'Intervals use paired crossed resampling of training and physical task seeds. '
              'They are exploratory, unadjusted intervals; eight development tasks provide limited precision. '
              'The source JSON records result-file hashes and all individual run statuses.', '']
    Path(args.output + '.md').write_text('\n'.join(lines))
    print(json.dumps({k: report[k] for k in ('completed_training_runs', 'completed_forecasts', 'completed_planning_evaluations')}))


if __name__ == '__main__':
    main()
