#!/usr/bin/env python3
"""Plot completed development diagnostics; no fabricated or partial records."""
from pathlib import Path
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_goal_calibration_campaign import validate_completed_diagnostic
from diagnose_goal_calibration import sha256

MODES = ('frozen', 'single', 'factorized', 'framewise')
LABELS = {'frozen': 'Frozen LeWM', 'single': 'Shared context',
          'factorized': 'ShiftWM (ours)', 'framewise': 'Framewise calibration'}
DISPLAY_MODES = ('frozen', 'framewise', 'single', 'factorized')
METRICS = ('goal_calibration_mse', 'recorded_prediction_to_canonical_goal_mse')


def load_evidence():
    rows = []
    paired_keys = {}
    for environment in ('pusht', 'reacher'):
        for mode in MODES:
            run_id = f'{environment}_{mode}_s0'
            path = ROOT / 'results/diagnostics' / run_id / 'goal_calibration.json'
            result = json.loads(path.read_text())
            planning_path = ROOT / 'results/development_official_budget' / run_id / 'planning_development.json'
            planning = json.loads(planning_path.read_text())
            checkpoint = ROOT / 'runs/world' / run_id / 'best'
            data_name = 'pusht_relative' if environment == 'pusht' else environment
            identity = {
                'checkpoint_sha256': sha256(checkpoint / 'model.pt'),
                'config_sha256': sha256(checkpoint / 'config.json'),
                'data_manifest_sha256': sha256(ROOT / 'data/world' / data_name / 'manifest.json'),
                'evaluator_sha256': sha256(ROOT / 'src/shiftwm/evaluate.py'),
                'model_source_sha256': sha256(ROOT / 'src/shiftwm/model.py'),
                'generation_source_sha256': sha256(ROOT / 'src/shiftwm/generate.py'),
                'diagnostic_source_sha256': sha256(ROOT / 'scripts/diagnose_goal_calibration.py'),
            }
            validate_completed_diagnostic(result, planning, identity, planning_path)
            measured = [row for row in result['records'] if row['diagnostic_status'] == 'measured']
            keys = [(row['trajectory_id'], row['observation_id'], row['goal_index']) for row in measured]
            if environment in paired_keys and keys != paired_keys[environment]:
                raise ValueError('Methods do not share the same measured development cases')
            paired_keys[environment] = keys
            if len({row['seed'] for row in measured}) != len(measured):
                raise ValueError('Plot expects exactly one observation per independent episode seed')
            metrics = {}
            for metric in METRICS:
                summary = result['summary']['all'][metric]
                recomputed = sum(row[metric] for row in measured) / len(measured)
                if not math.isclose(recomputed, summary['mean'], rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(f'Mean mismatch for {run_id}/{metric}')
                lo, hi = summary['ci95']
                if not (0 <= lo <= summary['mean'] <= hi and math.isfinite(hi)):
                    raise ValueError('Invalid confidence interval')
                metrics[metric] = {**summary, 'source_key': f'summary.all.{metric}'}
            rows.append({'run_id': run_id, 'environment': environment, 'mode': mode,
                         'source': str(path.relative_to(ROOT)), 'source_sha256': sha256(path),
                         'counts': result['counts'], 'metrics': metrics})
    return {'scope': 'posthoc development diagnostic; seed0 trained checkpoints; not primary test evidence',
            'metric_units': 'mean squared error per immutable pretrained latent dimension; lower is better',
            'uncertainty': '95% trajectory-seed percentile bootstrap intervals from source records',
            'no_claim': 'Neither causal attribution nor improved planning follows from these errors alone.',
            'runs': rows}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    evidence = load_evidence()
    output = ROOT / 'paper/generated'
    output.mkdir(exist_ok=True)
    (output / 'goal_calibration.json').write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.edgecolor': '#647080',
                         'axes.labelcolor': '#243447', 'text.color': '#243447',
                         'xtick.color': '#243447', 'ytick.color': '#243447',
                         'axes.linewidth': .7, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.15), sharey=True)
    colors = ('#0072B2', '#B75B16')
    names = ('Goal calibration', 'Recorded-action prediction')
    markers = ('o', 'D')
    for ax, environment, title in zip(axes, ('pusht', 'reacher'), ('a  PushT · 31 tasks', 'b  Reacher · 30 tasks')):
        lookup = {row['mode']: row for row in evidence['runs'] if row['environment'] == environment}
        rows = [lookup[mode] for mode in DISPLAY_MODES]
        for j, metric in enumerate(METRICS):
            means = np.array([row['metrics'][metric]['mean'] for row in rows])
            bounds = np.array([row['metrics'][metric]['ci95'] for row in rows])
            if np.any(bounds <= 0):
                raise ValueError('Log-scale diagnostic requires strictly positive observed interval bounds')
            y = np.arange(len(MODES)) + (j - .5) * .27
            ax.errorbar(means, y, xerr=np.stack((means - bounds[:, 0], bounds[:, 1] - means)),
                        fmt=markers[j], markersize=4.6, color=colors[j], label=names[j],
                        markeredgecolor='white', markeredgewidth=.5, elinewidth=1.1, capsize=2)
        ax.set_yticks(range(len(DISPLAY_MODES)), [LABELS[mode] for mode in DISPLAY_MODES])
        if ax.get_yticklabels():
            ax.get_yticklabels()[-1].set_fontweight('bold')
        ax.set_title(title, fontweight='bold', fontsize=9, loc='left', pad=10)
        ax.set_ylim(len(MODES) - .5, -.5)
        ax.set_xscale('log')
        ax.set_xlim(.004, 1.7)
        ax.set_xticks([.01, .1, 1], ['0.01', '0.1', '1'])
        ax.set_xlabel('Latent MSE ↓  (log)', labelpad=6)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, which='major', color='#E4E8ED', linewidth=.6)
        ax.tick_params(axis='y', length=0, pad=7)
        ax.tick_params(axis='x', which='minor', bottom=False)
        ax.axhspan(2.53, 3.47, color='#F5F7FA', zorder=-2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.63, .00), frameon=False,
               fontsize=8, labelspacing=.5, handletextpad=.6)
    fig.text(.015, .975, 'Development diagnostic · mean and 95% bootstrap interval',
             fontsize=8, va='top', color='#465365')
    fig.subplots_adjust(left=.285, right=.985, bottom=.31, top=.83, wspace=.20)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(output / f'goal_calibration.{extension}', dpi=240)
    plt.close(fig)
    print(json.dumps({'validated_runs': len(evidence['runs']), 'outputs': 'paper/generated/goal_calibration.*'}))


if __name__ == '__main__':
    main()
