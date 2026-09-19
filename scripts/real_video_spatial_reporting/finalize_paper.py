#!/usr/bin/env python3
"""Create the spatial appendix only after complete source/ledger/arithmetic gates."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
REGISTRY = 'configs/real_video_spatial/v1/registration.json'
FINAL = 'reports/real_video_spatial/finalization.json'
EXPECTED = 'ae99340f7a3761ece066ee7376f4bd119b61a92503948c142196c40a1a76e337'
MODES = ('autoregressive', 'anchored_additive', 'transport', 'context_off', 'action_free')
METRICS = ('native_mse', 'native_persistence_mse', 'original_2x2_mse', 'original_2x2_persistence_mse')
NAMES = dict(zip(MODES, ('Autoregressive', 'Anchored additive', r'\textbf{Spatial transport (ours)}', 'Context-off', 'Action-free')))
OUT = ROOT/'paper/generated/real_video/spatial'
SECTION = ROOT/'paper/sections/spatial_development.tex'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def same(actual, expected, label):
    if not np.allclose(actual, expected, rtol=1e-11, atol=1e-12):
        raise ValueError('Independent paper arithmetic differs: '+label)


def independent_comparison(first, second, metric, horizon, draws=10000):
    """Recompute one endpoint independently from full episode rows, seed 173."""
    matrices, reference = [], None
    for records in (first, second):
        if len(records) != 3 or sorted(r['seed'] for r in records) != [0, 1, 2]:
            raise ValueError('Need three distinct paired training seeds')
        rows = []
        for record in sorted(records, key=lambda r: r['seed']):
            episodes = sorted(record['episodes'], key=lambda e: e['episode_id'])
            population = [(e['episode_id'], e['session_id'], e['windows']) for e in episodes]
            if reference is None:
                reference = population
            if population != reference:
                raise ValueError('Paired episode/session/window populations differ')
            rows.append([e[metric][horizon-1] for e in episodes])
        matrices.append(np.asarray(rows, dtype=np.float64))
    a, b = matrices
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (b < 0).any():
        raise ValueError('Invalid endpoint errors')
    sessions = [r[1] for r in reference]
    groups = [np.flatnonzero(np.asarray(sessions) == s) for s in sorted(set(sessions))]
    if len(groups) < 2:
        raise ValueError('Need at least two recording sessions')
    differences = a-b
    generator = np.random.default_rng(173)
    distribution = np.empty(draws)
    for i in range(draws):
        seeds = generator.integers(0, 3, size=3)
        group_ids = generator.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[j] for j in group_ids])
        distribution[i] = differences[np.ix_(seeds, indices)].mean()
    low, high = np.quantile(distribution, [.025, .975])
    base, proposed = float(b.mean()), float(a.mean())
    return dict(method_mean=proposed, comparator_mean=base,
                method_minus_comparator=float(differences.mean()),
                relative_error_reduction_percent=100*(base-proposed)/base if base else None,
                paired_95_percent_interval=[float(low), float(high)],
                interval_includes_zero=bool(low <= 0 <= high))


def summarize(ledgers, official):
    if len(ledgers) != 15 or {(r['mode'], r['seed']) for r in ledgers} != {(m,s) for m in MODES for s in range(3)}:
        raise ValueError('All fifteen mode/seed ledgers required')
    summary = {}
    for mode in MODES:
        selected = [r for r in ledgers if r['mode'] == mode]
        summary[mode] = {}
        for metric in METRICS:
            array = np.array([[np.mean([e[metric][h] for e in r['episodes']]) for h in range(10)] for r in selected])
            if not np.isfinite(array).all() or (array < 0).any():
                raise ValueError('Invalid aggregate')
            mean = array.mean(axis=0)
            same(mean, official['aggregate'][mode][metric], mode+'/'+metric)
            summary[mode][metric] = {'mean': mean.tolist(), 'seed_sd': array.std(axis=0, ddof=1).tolist()}
    for metric in ('native_persistence_mse', 'original_2x2_persistence_mse'):
        reference = summary[MODES[0]][metric]['mean']
        for mode in MODES:
            same(summary[mode][metric]['mean'], reference, 'persistence consistency')
            same(summary[mode][metric]['seed_sd'], np.zeros(10), 'persistence seed consistency')
    expected = {(m,k,h) for m in MODES if m != 'transport' for k in ('native_mse','original_2x2_mse') for h in (5,10)}
    effects = official['paired_effects']
    if len(effects) != 16 or {(r['comparator'],r['metric'],r['horizon']) for r in effects} != expected:
        raise ValueError('All sixteen paired effects required')
    ours = [r for r in ledgers if r['mode'] == 'transport']
    for row in effects:
        result = independent_comparison(ours, [r for r in ledgers if r['mode'] == row['comparator']], row['metric'], row['horizon'])
        for key, value in result.items():
            if value is None:
                if row[key] is not None: raise ValueError('Zero denominator must remain undefined')
            elif isinstance(value, bool):
                if row[key] is not value: raise ValueError('Interval inclusion label differs')
            else: same(value, row[key], 'paired/'+key)
    return summary


def bind_finalizer_parity(name, selected, parity):
    """Anchor the finalizer's old relocation proof to this exact selected model."""
    exported = ROOT/'artifacts/releases/spatial_v1/models'/name
    manifest_path = exported/'package_manifest.json'
    if exported.is_symlink() or any(path.is_symlink() for path in exported.rglob('*')):
        raise ValueError('Finalized inference export must contain independent regular files')
    if (parity.get('name') != name or parity.get('status') != 'passed'
            or parity.get('max_abs_error') != 0
            or parity.get('relocated_isolated_process') is not True
            or sha(manifest_path) != parity.get('package_manifest_sha256')):
        raise ValueError('Selected model is not bound to finalizer relocation evidence')
    manifest = read(manifest_path)
    if (manifest.get('format_version') != 1
            or manifest.get('package_kind') != 'shiftwm_real_video_spatial_v1'
            or set(manifest.get('files', {})) != {'model.pt', 'config.json'}):
        raise ValueError('Finalized inference package schema differs')
    for filename, expected in manifest['files'].items():
        if sha(exported/filename) != expected or sha(selected/filename) != expected:
            raise ValueError('Finalized inference export differs from selected checkpoint')
    return {str(path.relative_to(ROOT)): sha(path)
            for path in (manifest_path, exported/'model.pt', exported/'config.json')}


def collect():
    campaign = module('scripts/real_video_spatial/campaign.py', 'spatial_paper_campaign')
    release = module('scripts/publishing/spatial_release.py', 'spatial_paper_gate')
    if sha(ROOT/REGISTRY) != EXPECTED:
        raise ValueError('Unexpected registered campaign')
    registry = campaign.verify()
    official = read(ROOT/FINAL)
    release.validate_gate(official, registry, EXPECTED)
    trainer, validator = campaign.module('train'), campaign.module('validate_ledger')
    evidence = {**registry['dependencies'], REGISTRY: EXPECTED, FINAL: sha(ROOT/FINAL)}
    for name in ('scripts/publishing/spatial_release.py', 'scripts/real_video_spatial_reporting/finalize_paper.py',
                 'scripts/real_video_spatial_reporting/finalize_paper.slurm', 'tests/test_spatial_paper.py',
                 'reports/spatial_paper_protocol.md'):
        evidence[name] = sha(ROOT/name)
    evidence.update(release.validate_cache_gate(ROOT, read(ROOT/registry['runs'][0]['config'])))
    rows, ledgers = [], []
    finalized = {r['name']: r for r in official['runs']}
    parity = {r['name']: r for r in official['offline_cpu_parity']}
    for row in registry['runs']:
        config = read(ROOT/row['config'])
        run = ROOT/config['output_dir']
        summary = trainer.validate_completed(run)
        selected, state = trainer.read_package(run/'best')
        executed = read(run/'training_config.json')
        if trainer.base.scientific_config(executed) != trainer.base.scientific_config(config):
            raise ValueError('Executed configuration differs')
        final_row = finalized[row['name']]
        result = read(ROOT/final_row['validation'])
        validator.validate_ledger(result, config, ROOT, state)
        if summary['best_epoch'] != final_row['selected_epoch'] or summary['parameter_counts'] != final_row['parameter_counts']:
            raise ValueError('Model selection/parameters differ')
        evidence.update(bind_finalizer_parity(row['name'], selected, parity[row['name']]))
        completion_files = [run/'training_summary.json', run/'training_config.json', run/'metrics.jsonl']
        for package in ('best', 'last'):
            completion_files.extend(run/package/filename for filename in
                                    ('package_manifest.json', 'model.pt', 'config.json'))
        for path in (ROOT/row['config'], ROOT/final_row['validation'], *completion_files,
                     selected/'model.pt', selected/'config.json'):
            evidence[str(path.relative_to(ROOT))] = sha(path)
        rows.append({**final_row, 'mode': row['mode'], 'seed': row['seed']})
        ledgers.append(result)
    summary = summarize(ledgers, official)
    first = ledgers[0]
    population = {'episodes': len(first['episodes']), 'sessions': len({e['session_id'] for e in first['episodes']}), 'windows': len(first['windows'])}
    return official, rows, summary, evidence, population


def gain(value, inconclusive=False):
    if value is None: return r'---'
    label = f'{value:+.3f}'+r'\%'
    if value > 0: label = r'\positivegain{'+label+'}'
    if inconclusive: label += r'$^{\dagger}$'
    return label


def table(body, caption, label, columns):
    return (r'\begin{table}[p]\centering\footnotesize'+'\n'+r'\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.12}'+'\n'+
            r'\begin{tabular}{'+columns+r'}\toprule'+'\n'+body+r'\bottomrule\end{tabular}'+'\n'+r'\caption{'+caption+'}\n'+r'\label{'+label+r'}\end{table}'+'\n')


def tables(official, runs, summary):
    files = {}
    for metric, title, prefix in (('native_mse', 'Native $4\\times4$ features', 'native'), ('original_2x2_mse', 'Original $2\\times2$ coordinates', 'pooled')):
        persistence = 'native_persistence_mse' if prefix == 'native' else 'original_2x2_persistence_mse'
        body = r'Method & $h=5$ MSE & $h=10$ MSE & $h=10$ gain\\\midrule'+'\n'
        p = summary['autoregressive'][persistence]['mean']
        body += f'Persistence & {p[4]:.6f} & {p[9]:.6f} & '+r'---\\'+'\n'
        baseline = summary['autoregressive'][metric]['mean'][9]
        for mode in MODES:
            mean, std = summary[mode][metric]['mean'], summary[mode][metric]['seed_sd']
            if mode == 'transport': body += r'\rowcolor{orange!9}'+'\n'
            value = 100*(baseline-mean[9])/baseline if baseline else None
            body += NAMES[mode]+f' & ${mean[4]:.6f}\\pm{std[4]:.6f}$ & ${mean[9]:.6f}\\pm{std[9]:.6f}$ & '+(r'---' if mode == 'autoregressive' else gain(value))+r'\\'+'\n'
        caption = (r'\textbf{'+title+r'.} All five arms share the $4\times4$ representation and support/query horizons; Context-off and Action-free explicitly remove information paths. Values are equal-seed means $\pm$ sample seed SD after within-episode window averaging; persistence is deterministic. Signed gain compares each arm with Autoregressive in the same column; green bold marks positive point estimates only. Paired uncertainty is reported separately. ' +
                   ('Native scores use shared-channel training normalization.' if prefix == 'native' else 'Predicted raw grids are pooled, then scored against exact original cached targets with original normalization. These values cannot be compared numerically with the native table.') + ' All results use development validation.')
        files[prefix+'.tex'] = table(body, caption, 'tab:spatial-'+prefix, 'lrrr')
        body = r'Transport comparator & $h$ & Gain & Difference [paired 95\% interval]\\\midrule'+'\n'
        for row in official['paired_effects']:
            if row['metric'] != metric: continue
            lo, hi = row['paired_95_percent_interval']
            body += f"{NAMES[row['comparator']]} & {row['horizon']} & "+gain(row['relative_error_reduction_percent'],row['interval_includes_zero'])+f" & ${row['method_minus_comparator']:+.6f}\\;[{lo:+.6f}, {hi:+.6f}]$"+r'\\'+'\n'
        files[prefix+'_intervals.tex'] = table(body, r'\textbf{Transport comparisons: '+title+r'.} Differences are Spatial transport (ours) minus comparator; negative favors ours. All four comparators and both endpoints are included. Intervals resample recording sessions and three training seeds jointly (10,000 draws; seed 173). $\dagger$ marks intervals containing zero. Green bold denotes positive point gain, not significance. Exploratory development evidence, without multiple-comparison adjustment.', 'tab:spatial-'+prefix+'-intervals', 'lrrl')
    body = r'Method & Selected epochs & Total params & Active params\\\midrule'+'\n'
    for mode in MODES:
        rows = sorted([r for r in runs if r['mode'] == mode], key=lambda r:r['seed'])
        counts = rows[0]['parameter_counts']
        if any(r['parameter_counts'] != counts for r in rows): raise ValueError('Seed parameter counts differ')
        if mode == 'transport': body += r'\rowcolor{orange!9}'+'\n'
        body += NAMES[mode]+' & '+'/'.join(str(r['selected_epoch']) for r in rows)+f" & {counts['total']:,} & {counts['trainable']:,}"+r'\\'+'\n'
    files['checkpoints.tex'] = table(body, r'\textbf{All fifteen completed spatial models.} Every model completed 30 epochs; selected epochs follow seeds 0/1/2 and minimize window-weighted all-ten validation error. Active counts exclude frozen unused heads. The shared trunk does not imply identical active parameter counts; Context-off and Action-free remove inputs and parameters. All selected models passed relocated CPU forecast parity.', 'tab:spatial-checkpoints', 'lrrr')
    return files


def section(population):
    return r'''\FloatBarrier
\section{Observation-anchored spatial development study}
\label{app:spatial-development}
This registered architecture study addresses recursive prediction drift while
preserving the locations of visual features. It is a new spatial variant;
it does not preserve the original separate observation/dynamics factorization.
The frozen DINOv2-small encoder produces a $4\times4$ grid of 384-dimensional
features. Training-only channel statistics are shared across all sixteen
positions, so mixtures remain in common feature coordinates. Three observed
grids pass through a shared spatial encoder and the pinned LeWM temporal
predictor. A causal action-prefix GRU supplies commands only through the
requested horizon, and a support-only transition context conditions the trunk
through FiLM. Context remains fixed throughout the forecast.

For horizon $h$, let $Z_0$ be the last actual observed grid, $H_h$ the
horizon state, and $E_0$ the spatial encoding of $Z_0$ before context FiLM.
The transport decoder computes
\begin{align}
 T_h &= \operatorname{softmax}_{\mathrm{row}}\!\left(
   Q(H_h)K(E_0)^\top/\sqrt{96}+4I\right),\\
 \widehat Z_h &= (1-g_h)\odot Z_0+g_h\odot(T_hZ_0)
                  +\tanh(R(H_h)),\quad g_h=\sigma(W_gH_h+b_g).
\end{align}
Every forecast retains the actual anchor instead of recursively replacing it
with predictions. The row-stochastic mixture and unit-bounded innovation
imply the componentwise bound $|\widehat Z_{h,p,c}|\leq
\max_q|Z_{0,q,c}|+1$ in shared normalized coordinates. This is an output
invariant, not an accuracy or stability guarantee. Transport mixes semantic
features; it is not a validated physical image warp or RGB generator.

The five registered arms share the spatial inputs and trunk: Autoregressive
recursively updates the feature history; Anchored additive keeps the actual
observation anchor; Spatial transport (ours) adds mixing and bounded innovation;
Context-off and Action-free disable the corresponding information paths.
Transport begins near persistence (identity score bias 4 and gate bias $-3$),
whereas the zero-initialized additive heads begin at exact persistence.
Active parameter counts differ slightly. These arms test the package-level
decoder effect; they do not separately identify mixing versus bounded innovation.

All fifteen models complete 30 epochs at seeds 0/1/2 with identical AdamW
settings, batch size 128 and ten query steps. Window-weighted all-ten
validation MSE selects checkpoints; final metrics average windows within
episodes, then weight episodes and seeds equally. '''+f"The evaluation contains {population['episodes']} episodes from {population['sessions']} recording sessions and {population['windows']} windows. "+r'''Neither original nor fresh
held-out test observations enter this development study.

Tables~\ref{tab:spatial-native} and~\ref{tab:spatial-pooled} report every arm
in native and original pooled coordinates. Each coordinate system has its own
normalization; comparisons are made within columns. Tables~\ref{tab:spatial-native-intervals}
and~\ref{tab:spatial-pooled-intervals} preserve all sixteen registered comparisons,
including regressions and intervals containing zero. These intervals are
exploratory and unadjusted for multiple comparisons; stronger claims require
an independently frozen evaluation. Table~\ref{tab:spatial-checkpoints}
records every selected checkpoint and the nonidentical active parameter counts.
'''+''.join(r'\input{generated/real_video/spatial/'+name+'}\n' for name in ('native.tex','pooled.tex','native_intervals.tex','pooled_intervals.tex','checkpoints.tex'))


def proof(stage, files):
    for name, text in files.items(): (stage/name).write_text(text)
    preamble = r'''\documentclass{article}
\usepackage{iclr2027_conference,times,booktabs,amsmath,xcolor,colortbl,hyperref}
\definecolor{gainpositive}{HTML}{166534}
\newcommand{\positivegain}[1]{\textcolor{gainpositive}{\textbf{#1}}}
\begin{document}
'''
    (stage/'proof.tex').write_text(preamble+''.join(r'\input{'+name+'}\n' for name in files)+r'\end{document}')
    env = dict(os.environ, TEXINPUTS=str(ROOT/'paper/template/official/iclr2027')+'//:'+os.environ.get('TEXINPUTS',''))
    for i in range(2):
        done = subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','proof.tex'], cwd=stage, env=env, capture_output=True, text=True)
        (stage/f'pass{i+1}.stdout').write_text(done.stdout+done.stderr)
        if done.returncode: raise ValueError('ICLR table proof failed: '+done.stdout[-2000:])
    log = (stage/'proof.log').read_text()
    if any(v in log for v in ('Overfull','undefined','multiply defined','Float too large')):
        raise ValueError('Table layout/reference proof failed')
    return {'status':'passed','pdf_sha256':sha(stage/'proof.pdf'),'actual_visual_review':'pending; automated typesetting is not visual review'}


def main(build=False):
    registration = read(ROOT/'reports/spatial_paper_submission.json')
    if registration.get('scientific_registration_sha256') != EXPECTED:
        raise ValueError('Unexpected paper execution registration')
    for name, digest in registration['reporting_dependencies'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Registered reporting dependency changed: '+name)
    torch.set_num_threads(2)
    official, runs, summary, evidence, population = collect()
    evidence['reports/spatial_paper_submission.json'] = sha(ROOT/'reports/spatial_paper_submission.json')
    files = tables(official, runs, summary)
    with tempfile.TemporaryDirectory(prefix='spatial-paper-proof-') as tmp:
        stage = Path(tmp)
        review = proof(stage, files)
        for name, digest in evidence.items():
            if sha(ROOT/name) != digest: raise ValueError('Source changed before paper generation: '+name)
        OUT.mkdir(parents=True, exist_ok=True)
        for name in (*files, 'proof.pdf', 'proof.log', 'proof.tex', 'pass1.stdout', 'pass2.stdout'):
            temp = OUT/(name+'.tmp'); shutil.copyfile(stage/name, temp);temp.replace(OUT/name)
        doc = {'status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),
               'sources':evidence,'population':population,'summary':summary,
               'paired_comparisons':official['paired_effects'],'runs':runs,'proof':review,
               'all_models':15,'all_epochs':30,'all_comparisons':16}
        (OUT/'evidence.json').write_text(json.dumps(doc,indent=2)+'\n')
        temp = SECTION.with_suffix('.tex.tmp');temp.write_text(section(population));temp.replace(SECTION)
        report = ROOT/'reports/spatial_paper_finalization.json'
        report.write_text(json.dumps(doc,indent=2)+'\n')
    if build:
        subprocess.run(['bash','paper/build.sh'], cwd=ROOT, check=True)
    print(json.dumps({'status':'completed','models':15,'comparisons':16,'build':build}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--build-paper',action='store_true');args=parser.parse_args()
    lock=ROOT/'artifacts/development/spatial_paper.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main(args.build_paper)
