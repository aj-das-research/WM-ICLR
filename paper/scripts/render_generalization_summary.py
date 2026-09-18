#!/usr/bin/env python3
"""Render fully finalized validation comparisons; require pixel review to publish."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/real_droid_generalization_results.json'
FINAL = ROOT / 'reports/real_droid_generalization_finalization.json'
REGISTRY = ROOT / 'configs/real_video_development/generalization_v1/registration.json'
PROTOCOL = ROOT / 'reports/generalization_summary_figure_protocol.md'
CANDIDATE = ROOT / 'artifacts/publishing/generalization_summary_review'
PUBLIC = ROOT / 'paper/generated/real_video'
STEM = 'generalization_summary'
ARMS = ('slow', 'decay', 'compact')
MODES = ('framewise', 'constant_dynamics', 'factorized', 'action_free')
NAMES = {'framewise': 'Framewise', 'constant_dynamics': 'Constant dynamics', 'factorized': 'ShiftWM (ours)', 'action_free': 'Action-free'}
COLORS = {'framewise': '#256C9B', 'constant_dynamics': '#77678C', 'factorized': '#CC651E', 'action_free': '#58666D'}
MARKERS = {'framewise': 's', 'constant_dynamics': '^', 'factorized': 'D', 'action_free': 'x'}
INK, MUTED, GREEN, RUST = '#172B3A', '#5B6977', '#13734E', '#A34C18'
WIDTH, HEIGHT = 5.5, 4.3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def equal(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-13):
        raise ValueError(f'Numerical mismatch for {label}: {actual} != {expected}')


def registered_bootstrap(matrix, sessions, seed):
    """Independent reproduction from raw paired episode errors, original RNG order."""
    groups = [np.flatnonzero(np.asarray(sessions) == s) for s in sorted(set(sessions))]
    rng = np.random.default_rng(seed)
    distribution = np.empty(10000)
    for index in range(len(distribution)):
        seeds = rng.integers(0, 3, size=3)
        selected = rng.integers(0, len(groups), size=len(groups))
        episodes = np.concatenate([groups[i] for i in selected])
        distribution[index] = matrix[np.ix_(seeds, episodes)].mean()
    return np.quantile(distribution, [.025, .975])


def checked_inputs():
    if not all(p.is_file() for p in (REPORT, FINAL, REGISTRY)):
        raise ValueError('Awaiting the completed 36-run / 72-evaluation finalization reports')
    report, final, registry = read(REPORT), read(FINAL), read(REGISTRY)
    if (report.get('status') != 'completed' or report.get('completed_runs') != 36 or report.get('missing')
            or final.get('status') != 'completed' or final.get('completed_training_runs') != 36
            or final.get('completed_validation_evaluations') != 72
            or final.get('registered_checkpoint_parity_passes') != 36
            or final.get('independent_cpu_reload_parity_passes') != 36
            or final.get('review', {}).get('status') != 'passed'
            or final.get('paper_build', {}).get('status') != 'passed'):
        raise ValueError('The full campaign, independent reload checks and finalizer build must complete first')
    registration_hash = sha(REGISTRY)
    if report['registration_sha256'] != registration_hash or final['registration_sha256'] != registration_hash:
        raise ValueError('Report registration identity mismatch')
    expected = {(a, m, s) for a in ARMS for m in MODES for s in range(3)}
    if (len(registry['runs']) != 36 or {(x['arm'], x['mode'], x['seed']) for x in registry['runs']} != expected
            or len(final['runs']) != 36 or {(x['arm'], x['mode'], x['seed']) for x in final['runs']} != expected):
        raise ValueError('A registered/finalized arm, method or seed is missing or duplicated')
    if len(report['aggregate']) != 6 or len(final['aggregate']) != 6:
        raise ValueError('All six arm/horizon comparisons are required')
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in (REPORT, FINAL, REGISTRY, PROTOCOL, Path(__file__))}
    raw, run_metadata = {}, []
    for entry in registry['runs']:
        key = (entry['arm'], entry['mode'], entry['seed'])
        config_path = ROOT / entry['config']
        if sha(config_path) != entry['config_sha256'] or final['sources'].get(entry['config']) != entry['config_sha256']:
            raise ValueError('Registered config hash mismatch')
        config = read(config_path)
        directory = (ROOT / config['output_dir']).resolve()
        directory.relative_to((ROOT / 'runs/real_video_development/generalization_v1').resolve())
        if (config['mode'], config['seed'], config['epochs']) != (entry['mode'], entry['seed'], 30):
            raise ValueError('Registered config does not match run identity')
        finalized = next(x for x in final['runs'] if (x['arm'], x['mode'], x['seed']) == key)
        summary_path = directory / 'training_summary.json'
        summary = read(summary_path)
        if summary.get('status') != 'completed' or summary.get('completed_epochs') != 30 or summary != finalized['training']:
            raise ValueError('Training did not complete all 30 epochs or differs from finalization')
        source_hashes[entry['config']] = sha(config_path)
        source_hashes[str(summary_path.relative_to(ROOT))] = sha(summary_path)
        raw[key] = {}
        for horizon in (5, 10):
            path = directory / f'validation_h{horizon}.json'
            digest, relative = sha(path), str(path.relative_to(ROOT))
            if final['sources'].get(relative) != digest:
                raise ValueError('Validation result changed after independent finalization')
            document = read(path)
            if (document['scope'] != 'validation development only' or document['arm'] != entry['arm']
                    or document['mode'] != entry['mode'] or document['seed'] != entry['seed']
                    or document['horizon'] != horizon or document['registration_sha256'] != registration_hash
                    or document['checkpoint_sha256'] != finalized['checkpoint_sha256']):
                raise ValueError('Validation result identity mismatch')
            rows = sorted(document['result']['episodes'], key=lambda x: x['episode_id'])
            metric = f'h{horizon}_standardized_mse'
            values = np.asarray([r['errors']['model'][metric] for r in rows], dtype=np.float64)
            if not len(rows) or len({r['episode_id'] for r in rows}) != len(rows) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('Invalid episode population or MSE')
            equal(float(values.mean()), document['result']['summary']['model'][metric], relative)
            raw[key][horizon] = {'values': values, 'ids': [r['episode_id'] for r in rows],
                                 'sessions': [r['session_id'] for r in rows], 'source': relative, 'source_sha256': digest}
            source_hashes[relative] = digest
        run_metadata.append({'arm': entry['arm'], 'mode': entry['mode'], 'seed': entry['seed'],
                             'best_epoch': summary['best_epoch'], 'parameter_counts': summary['parameter_counts'],
                             'checkpoint_sha256': finalized['checkpoint_sha256']})
    comparisons = []
    for arm in ARMS:
        for horizon in (5, 10):
            formal = next(x for x in final['aggregate'] if (x['arm'], x['horizon']) == (arm, horizon))
            registered = next(x for x in report['aggregate'] if (x['arm'], x['horizon']) == (arm, horizon))
            reference = raw[arm, 'framewise', 0][horizon]
            matrices, methods = {}, {}
            for mode in MODES:
                records = [raw[arm, mode, seed][horizon] for seed in range(3)]
                for r in records:
                    if r['ids'] != reference['ids'] or r['sessions'] != reference['sessions']:
                        raise ValueError('Unmatched episode/session population across methods or seeds')
                values = np.stack([r['values'] for r in records])
                means = values.mean(1)
                average, sd = float(means.mean()), float(means.std(ddof=1))
                equal(average, formal['methods'][mode]['mean'], f'{arm}/{horizon}/{mode} finalized mean')
                equal(average, registered['means'][mode], f'{arm}/{horizon}/{mode} registered mean')
                equal(sd, formal['methods'][mode]['seed_sd'], f'{arm}/{horizon}/{mode} seed SD')
                if not np.allclose(means, formal['methods'][mode]['per_seed'], rtol=1e-11, atol=1e-13):
                    raise ValueError('Per-seed mean mismatch')
                matrices[mode] = values
                methods[mode] = {'mean': average, 'seed_sd': sd, 'per_seed': means.tolist(),
                                 'sources': [r['source'] for r in records], 'source_hashes': [r['source_sha256'] for r in records]}
            difference = matrices['factorized'] - matrices['framewise']
            delta = float(difference.mean())
            ci = registered_bootstrap(difference, reference['sessions'], seed=5197000+horizon)
            pair = formal['methods']['factorized']['paired_difference']
            for reported in (pair, registered['paired_ours_minus_framewise']):
                equal(delta, reported['mean_difference'], f'{arm}/{horizon} difference')
                if (not np.allclose(ci, reported['ci95'], rtol=1e-11, atol=1e-13)
                        or reported['draws'] != 10000 or reported['training_seed_count'] != 3
                        or reported['bootstrap_seed'] != 5197000+horizon):
                    raise ValueError('Registered paired bootstrap cannot be reproduced')
            gain = 100 * (methods['framewise']['mean'] - methods['factorized']['mean']) / methods['framewise']['mean']
            equal(gain, formal['methods']['factorized']['gain_percent'], f'{arm}/{horizon} gain')
            equal(gain, registered['ours_reduction_vs_framewise_percent'], f'{arm}/{horizon} registered gain')
            comparisons.append({'arm': arm, 'horizon': horizon, 'episodes': len(reference['ids']),
                                'sessions': len(set(reference['sessions'])), 'methods': methods,
                                'ours_minus_framewise': delta, 'ci95': ci.tolist(), 'relative_gain_percent': gain,
                                'interval_excludes_zero': bool(ci[1] < 0 or ci[0] > 0)})
    return {'schema_version': 1, 'status': 'numerically_verified', 'scope': 'original validation development only',
            'metric': 'endpoint train-standardized feature MSE', 'seeds': [0, 1, 2],
            'sources': source_hashes, 'rows': comparisons, 'runs': run_metadata,
            'uncertainty': {'left': 'sample SD of three equal-episode seed means; ddof=1',
                            'right': '95% paired crossed session/seed bootstrap, 10000 draws, seed 5197000+horizon; exploratory/unadjusted'},
            'physical_size_inches': [WIDTH, HEIGHT], 'minimum_font_points': 8.0}


def geometry_checks(fig, axes, rows):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    texts, problems = [], []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        if text.axes is not None and not text.axes.get_visible():
            continue
        box = text.get_window_extent(renderer)
        if text.get_fontsize() < 8 - 1e-6:
            problems.append({'kind': 'small_font', 'text': text.get_text(), 'size': text.get_fontsize()})
        if box.width and box.height and (box.x0 < bounds.x0-.5 or box.y0 < bounds.y0-.5 or box.x1 > bounds.x1+.5 or box.y1 > bounds.y1+.5):
            problems.append({'kind': 'canvas_clipping', 'text': text.get_text()})
        texts.append((text.get_text(), box))
    for i, (name, box) in enumerate(texts):
        for other, box2 in texts[i+1:]:
            overlap_w = min(box.x1, box2.x1)-max(box.x0, box2.x0)
            overlap_h = min(box.y1, box2.y1)-max(box.y0, box2.y0)
            if overlap_w > 2 and overlap_h > 2:
                problems.append({'kind': 'text_collision', 'a': name, 'b': other})
    for axis, extents in [(axes[0], [(m['mean']-m['seed_sd'], m['mean']+m['seed_sd']) for r in rows for m in r['methods'].values()]),
                          (axes[1], [r['ci95'] for r in rows])]:
        lo, hi = axis.get_xlim()
        if any(a < lo or b > hi for a, b in extents):
            problems.append({'kind': 'clipped_uncertainty_interval'})
    if not axes[1].get_xlim()[0] <= 0 <= axes[1].get_xlim()[1]:
        problems.append({'kind': 'missing_zero_reference'})
    if problems:
        raise ValueError('Figure geometry checks failed: ' + json.dumps(problems))
    return {'status': 'passed', 'font_floor_pt': 8, 'text_objects': len(texts), 'text_collisions': 0,
            'canvas_clipping': 0, 'clipped_uncertainty_intervals': 0, 'zero_reference_visible': True,
            'visual_review': 'pending actual pixel inspection'}


def render(ledger, directory):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.3, 'axes.labelsize': 8.3,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8.3, 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': INK,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .6})
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=180, facecolor='white')
    a = fig.add_axes([.165, .155, .32, .635])
    b = fig.add_axes([.595, .155, .29, .635])
    rows = ledger['rows']
    offsets = [-.24, -.08, .08, .24]
    for axis in (a, b):
        axis.set_ylim(-.55, 5.55)
        for low, high in ((3.5, 5.5), (-.5, 1.5)):
            axis.axhspan(low, high, color='#F4F7F8', lw=0, zorder=0)
        axis.set_axisbelow(True)
        axis.xaxis.grid(True, color='#DEE5E9', lw=.5)
        axis.spines[['top', 'right', 'left']].set_visible(False)
        axis.spines['bottom'].set_color('#BECAD2')
        axis.tick_params(axis='y', length=0)
        axis.tick_params(axis='x', length=2.5, width=.6)
    a.set_yticks(range(5, -1, -1), [f"{r['arm'].capitalize()} / {r['horizon']}" for r in rows])
    b.set_yticks([])
    all_extents = []
    for index, record in enumerate(rows):
        y = 5-index
        for offset, mode in zip(offsets, MODES):
            m = record['methods'][mode]
            a.errorbar(m['mean'], y+offset, xerr=m['seed_sd'], marker=MARKERS[mode],
                       markersize=4.8 if mode == 'factorized' else 4.0, color=COLORS[mode],
                       markerfacecolor=COLORS[mode], markeredgewidth=.75, elinewidth=.85,
                       capsize=2.0, capthick=.7, linewidth=0, zorder=4 if mode == 'factorized' else 3)
            all_extents.extend([m['mean']-m['seed_sd'], m['mean']+m['seed_sd']])
        delta, (lo, hi) = record['ours_minus_framewise'], record['ci95']
        color = GREEN if delta < 0 else RUST
        marker = 'o' if record['horizon'] == 5 else 'D'
        # Draw the reported interval independently of its point: a bootstrap
        # interval need not mathematically contain the point estimate.
        b.hlines(y, lo, hi, color=color, linewidth=1.3, zorder=3)
        b.vlines([lo, hi], y-.055, y+.055, color=color, linewidth=.9, zorder=3)
        b.plot(delta, y, marker=marker, markersize=5, linestyle='none',
               markerfacecolor=color if record['interval_excludes_zero'] else 'white',
               markeredgecolor=color, markeredgewidth=1.15, zorder=4)
        fig.text(.993, .155 + .635*(y+.55)/6.1, f"{record['relative_gain_percent']:+.2f}",
                 ha='right', va='center', fontsize=8.3, color=color, fontweight='bold' if delta < 0 else 'normal')
    span = max(all_extents)-min(all_extents)
    pad = max(span*.12, .002)
    a.set_xlim(max(0, min(all_extents)-pad), max(all_extents)+pad)
    a.set_xticks([x for x in MaxNLocator(3).tick_values(*a.get_xlim()) if a.get_xlim()[0] <= x <= a.get_xlim()[1]])
    a.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    limit = max(abs(v) for r in rows for v in [*r['ci95'], r['ours_minus_framewise']]) * 1.22
    if limit <= 0: limit = .00001
    b.set_xlim(-limit, limit)
    b.set_xticks([x for x in MaxNLocator(3, symmetric=True).tick_values(-limit, limit) if -limit <= x <= limit])
    decimals = max(2, min(6, math.ceil(-math.log10(limit))+1))
    b.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '0' if abs(x) < 1e-14 else f'{x:+.{decimals}f}'))
    b.axvline(0, color=INK, lw=.85, ls=(0, (3, 2)), zorder=1)
    a.set_xlabel('Endpoint feature MSE ↓', labelpad=7)
    b.set_xlabel('ΔMSE (ours − Framewise)', labelpad=7)
    fig.text(.01, .965, 'Matched controls on recorded-video forecasting', fontsize=10.2, fontweight='bold', va='top')
    fig.text(.01, .914, 'Original validation only · 36 runs · three seeds per method', fontsize=8.1, color=MUTED, va='top')
    handles = [Line2D([], [], linestyle='none', marker=MARKERS[m], color=COLORS[m], markersize=4.8, label=NAMES[m]) for m in MODES]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.01, .872), frameon=False,
               ncol=4, fontsize=8, handletextpad=.35, columnspacing=.9, borderaxespad=0)
    fig.text(.165, .814, 'a  Absolute error', fontsize=9.3, fontweight='bold')
    fig.text(.595, .814, 'b  Paired difference', fontsize=9.3, fontweight='bold')
    fig.text(.993, .814, 'Gain (%)', ha='right', fontsize=8, color=MUTED)
    fig.text(.01, .07, 'Whiskers: seed SD (a); paired 95% interval (b).', fontsize=8, color=MUTED)
    fig.text(.01, .03, 'Open marks: interval includes zero. Green: favorable point estimate.', fontsize=8, color=MUTED)
    geometry = geometry_checks(fig, (a, b), rows)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(directory / f'{STEM}.{extension}', dpi=300, facecolor='white')
    plt.close(fig)
    return geometry


def caption(ledger):
    counts = {h: next(r for r in ledger['rows'] if r['horizon'] == h) for h in (5, 10)}
    return (r'\textbf{Matched optimization and capacity controls on original validation.} '
            r'All 36 registered runs complete 30 epochs; selected checkpoints use the common all-five-query validation objective. '
            r'Rows identify arm and forecast horizon in action blocks. Slow lowers learning rate, Decay increases weight decay, '
            r'and Compact jointly reduces predictor/context capacity. '
            r'(a) Equal-episode three-seed mean endpoint feature MSE with sample seed standard deviation; all four methods are shown. '
            r'(b) ShiftWM (ours) minus matched Framewise with 95\% intervals from 10,000 paired session/seed bootstrap draws. '
            r'Green positive reductions and open/filled interval markers distinguish point direction from interval exclusion of zero. '
            r'The adjacent gain column is the relative MSE reduction, not the interval unit. '
            f"Five and ten blocks include {counts[5]['episodes']} and {counts[10]['episodes']} eligible validation episodes, respectively. "
            r'These exploratory development intervals are unadjusted for multiple comparisons; no additional held-out-test claim is made.')


def compile_proof(directory, ledger):
    text = (r'\documentclass{article}' '\n'
            r'\usepackage[paperwidth=8.5in,paperheight=11in,textwidth=5.5in,textheight=9in]{geometry}' '\n'
            r'\usepackage{graphicx,caption}\captionsetup{font=small}' '\n'
            r'\begin{document}\begin{figure}[ht]\centering' '\n'
            rf'\includegraphics[width=\linewidth]{{{STEM}.pdf}}' '\n'
            r'\caption{' + caption(ledger) + '}' '\n'
            r'\end{figure}\end{document}' '\n')
    (directory / 'proof.tex').write_text(text)
    for _ in range(2):
        result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'proof.tex'], cwd=directory, capture_output=True, text=True)
        if result.returncode: raise ValueError('Standalone figure proof failed: ' + result.stdout[-2000:])
    log = (directory / 'proof.log').read_text()
    if any(word in log for word in ('Overfull', 'undefined', 'multiply defined')):
        raise ValueError('Standalone figure proof has overflow or unresolved references')
    subprocess.run(['pdftoppm', '-r', '160', '-png', '-singlefile', str(directory/'proof.pdf'), str(directory/'proof')], check=True)
    return {'status': 'passed', 'compile_passes': 2, 'warnings': 0, 'proof_pdf_sha256': sha(directory/'proof.pdf')}


def verify_sources(ledger):
    for name, expected in ledger['sources'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Source changed after candidate rendering: ' + name)


def prepare(directory):
    ledger = checked_inputs()
    geometry = render(ledger, directory)
    compile_review = compile_proof(directory, ledger)
    verify_sources(ledger)
    atomic_json(directory / f'{STEM}_ledger.json', ledger)
    review = {'status': 'awaiting_visual_review', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'pdf_sha256': sha(directory/f'{STEM}.pdf'), 'ledger_sha256': sha(directory/f'{STEM}_ledger.json'),
              'geometry': geometry, 'compile': compile_review,
              'required_review': ['paper_size_readability', 'enlarged_intervals_and_zero_line', 'legend_and_labels', 'grayscale', 'caption'],
              'manuscript_include_written': False}
    atomic_json(directory / f'{STEM}_review.json', review)
    print(json.dumps({'status': review['status'], 'candidate': str(directory), 'pdf_sha256': review['pdf_sha256']}), flush=True)


def publish_reviewed(directory, review_path):
    review = read(review_path)
    ledger = read(directory/f'{STEM}_ledger.json')
    candidate_review = read(directory/f'{STEM}_review.json')
    required = set(candidate_review['required_review'])
    if (review.get('status') != 'passed' or review.get('pdf_sha256') != sha(directory/f'{STEM}.pdf')
            or review.get('ledger_sha256') != sha(directory/f'{STEM}_ledger.json')
            or not required.issubset(review.get('visual_checks', [])) or not review.get('reviewer')
            or candidate_review['geometry']['status'] != 'passed' or candidate_review['compile']['status'] != 'passed'):
        raise ValueError('Publication requires exact-hash visual review of all required checks')
    verify_sources(ledger)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for filename in (f'{STEM}.pdf', f'{STEM}.svg', f'{STEM}.png', f'{STEM}_ledger.json'):
        shutil.copyfile(directory/filename, PUBLIC/filename)
    atomic_json(PUBLIC/f'{STEM}_review.json', {**review, 'candidate_automated_checks': candidate_review})
    include = ('\\begin{figure}[p]\n\\centering\n'
               + rf'\includegraphics[width=\linewidth]{{generated/real_video/{STEM}.pdf}}' + '\n'
               + '\\caption{' + caption(ledger) + '}\n'
               + '\\label{fig:generalization-summary}\n\\end{figure}\n')
    target = PUBLIC/f'{STEM}_figure.tex'
    temporary = target.with_suffix('.tex.tmp')
    temporary.write_text(include)
    temporary.replace(target)
    print(json.dumps({'status': 'published_reviewed_figure', 'include': str(target.relative_to(ROOT))}), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--candidate-dir', type=Path, default=CANDIDATE)
    parser.add_argument('--publish-reviewed', type=Path, help='Review JSON bound to the exact candidate PDF and ledger')
    args = parser.parse_args()
    if args.publish_reviewed: publish_reviewed(args.candidate_dir, args.publish_reviewed)
    else: prepare(args.candidate_dir)


if __name__ == '__main__':
    main()
