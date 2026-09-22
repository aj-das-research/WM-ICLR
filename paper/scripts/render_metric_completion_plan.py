#!/usr/bin/env python3
"""Render explicit measurement dependencies; never infer numerical results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper/generated/metric_completion'
SOURCES = ('reports/metrics_completion_v1/measurement_contract.md',
           'reports/metric_literature_audit_2026-09-21.md')


def write(path, value):
    if not path.exists() or path.read_text() != value:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(value)
        temp.replace(path)


def main():
    from render_current_spatial_planning_completion import live_pack, REPORT, PACK
    planning_data = live_pack(REPORT, PACK)
    planning_ready = planning_data is not None
    planning_sources = ['reports/metrics_completion_v1/planning/finalization.json'] if planning_ready else []
    rgb_path = ROOT / 'reports/metrics_completion_v1/rgb/decoder_finalization.json'
    rgb_ready = rgb_path.exists() and json.loads(rgb_path.read_text()).get('status') == 'three_decoders_complete'
    rgb_final = ROOT / 'reports/iclr_review_2026-09-22/rgb_endpoint_v2/finalization.json'
    rgb_scored = rgb_final.exists()
    rgb_sources = []
    if rgb_scored:
        record = json.loads(rgb_final.read_text())
        if record.get('status') != 'complete' or record.get('rows') != 45 or record.get('scored_endpoints') != 9000:
            raise ValueError('RGB endpoint campaign is incomplete')
        for path, expected in record['source_sha256'].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                raise ValueError('Changed RGB source: ' + path)
        rgb_sources = [str(rgb_final.relative_to(ROOT))]
    droid_gate = ROOT / 'reports/metrics_completion_v1/droid/recovery_v2/completion.json'
    droid_ready = droid_gate.exists()
    droid_sources = []
    if droid_ready:
        final = droid_gate.with_name('finalization.json')
        marker = json.loads(droid_gate.read_text())
        if marker.get('status') != 'passed' or marker.get('completed_rows') != 34 or marker['finalization_sha256'] != hashlib.sha256(final.read_bytes()).hexdigest():
            raise ValueError('Invalid DROID completion gate')
        droid_sources = [str(p.relative_to(ROOT)) for p in (droid_gate, final)]
    rows = [
        ('DROID', 'Feature MSE / MAE / raw L1 / cosine',
         'Complete: Table~\\ref{tab:droid-complementary-metrics}' if droid_ready else
         'MSE complete; complementary metrics in evaluation pipeline'),
        ('IWS PushT / Box / Rope', 'Same four feature errors; fixed horizons',
         'Complete; reserved scores in Table~\\ref{tab:iws-reserved-scores}'),
        ('Current spatial PushT / Reacher', 'Success; physical error; actions; full CEM time',
         'Complete: 24 models, 2,400 trials; Table~\\ref{tab:current-spatial-planning-success}' if planning_ready else 'New matched training/control campaign; results pending'),
        ('Historical PushT / Reacher / drone / tissue', 'Success; physical error; failure-censored effort',
         'Completed traces; supplementary physical aggregation'),
        ('Decoded recorded video', 'RGB MSE; PSNR; SSIM; LPIPS; UIQI',
         'Complete: 45 rows, 9,000 endpoints; Table~\\ref{tab:rgb-endpoint-comparison}' if rgb_scored else 'Three decoder fits complete; forecast RGB scoring pending' if rgb_ready else 'Pending shared decoder and reconstruction reference'),
        ('Decoded frame / clip distributions', 'FID / FVD',
         'Pending decoder, fixed extractors and sample-size audit'),
        ('Particle / physical rope tasks', 'Chamfer distance / clips threaded',
         'Requires particle-state / executed-routing environment'),
        ('Efficiency', 'Latency; CEM time; memory; FLOPs; GPU-hours',
         'IWS predictor and simulator online costs complete; FLOPs/GPU-hours pending' if planning_ready else 'IWS predictor costs complete; remaining scopes pending'),
        ('Simulator state probes', 'State MSE; Pearson correlation',
         'Pending common training-only probe'),
        ('DROID inverse planning / hardware', 'Offline action error / robot success',
         'Offline protocol pending / requires robot access'),
    ]
    text = r'''% Working measurement ledger; pending entries are not numerical observations.
\begin{table}[!htb]\centering\small
\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.15}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{.22\linewidth}>{\raggedright\arraybackslash}p{.31\linewidth}>{\raggedright\arraybackslash}p{.41\linewidth}@{}}
\toprule Task / measurement scope & Metrics & Completion / dependency \\ \midrule
'''
    text += '\n'.join(' & '.join(row) + r' \\' for row in rows)
    text += r'''
\bottomrule\end{tabular}
\caption{\textbf{Complete measurement ledger for the working draft.} Metrics are
attached to the outputs and environments that can support them. A pending
dependency is not a measured zero. Prediction, visual fidelity and physical
execution answer different questions; their scores are not interchangeable.}
\label{tab:metric-completion-coverage}\end{table}
'''
    # Keep placeholders visibly pending until a separate source-bound control
    # renderer exists. Do not populate from partial files or historical studies.
    planning = r'''% Planned comparisons, explicitly unmeasured; not an empirical result table.
\begin{table}[!htb]\centering
\begingroup\fontsize{8.5}{10.2}\selectfont
\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.1}
\begin{tabular}{@{}lrrrrr@{}}
\toprule Method & \shortstack{Success\\(\%) $\uparrow$} & \shortstack{Goal error\\$\downarrow$} & \shortstack{Calls to\\success $\downarrow$} & \shortstack{CEM time\\(s) $\downarrow$} & \shortstack{Peak GPU\\(MiB) $\downarrow$} \\ \midrule
'''
    for task, units in [('PushT', 'native position px'), ('Reacher', 'joint angle rad')]:
        planning += r'\multicolumn{6}{@{}l}{\textbf{' + task + '} --- ' + units + r'} \\' + '\n'
        for method in ('Autoregressive', 'Additive anchor', 'ShiftWM (ours)', r'No-$\tanh$ ablation'):
            planning += method + ' & ' + ' & '.join(['Pending'] * 5) + r' \\' + '\n'
    planning += r'''\bottomrule\end{tabular}\endgroup
\caption{\textbf{Current spatial-model planning: pending measurement slots.}
All entries await the separate matched training and closed-loop campaign; none
reuse the historical architecture's scores. Goal-error units differ by task and
are never pooled. Interaction cost includes support; failures are censored at
the native budget. Complete CEM time and memory require hardware measurements.
Results enter only after the registered comparison population and seeds finish.}
\label{tab:current-planning-pending}\end{table}
'''
    write(OUT / 'coverage.tex', text)
    write(OUT / 'planning_pending.tex', r'\input{generated/current_spatial_planning_completion/section.tex}' + '\n' if planning_ready else planning)
    evidence = {'schema': 'metric_completion_working_ledger_v1',
                'status': 'presentation_complete_measurements_pending',
                'numerical_results_inferred': False,
                'rows': [dict(zip(('scope', 'metrics', 'status'), row)) for row in rows],
                'sources': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (*SOURCES, *droid_sources, *planning_sources, *rgb_sources)}}
    evidence['sources'][str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(OUT / 'evidence.json', json.dumps(evidence, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'status': 'completed', 'kind': 'working_measurement_ledger', 'numerical_results_added': 0}))


if __name__ == '__main__':
    main()
