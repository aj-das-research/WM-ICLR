"""Validate complete capacity controls and report paired development effects."""
from pathlib import Path
import json
import numpy as np
import capacity_control as c

ROOT, OUT = c.ROOT, c.OUT
DISPLAY = ROOT / 'paper/generated/iclr_review_v1'


def main():
    reg = json.loads((OUT / 'registration.json').read_text())
    sources = {}
    def read(path):
        path = Path(path)
        sources[str(path.relative_to(ROOT))] = c.sha256(path)
        return json.loads(path.read_text())
    read(OUT / 'registration.json')
    for path, expected in reg['source_sha256'].items():
        assert c.sha256(path) == expected, path
    for path, expected in reg['comparators'].items():
        assert c.sha256(ROOT / path) == expected, path
    ledger = c.private(ROOT / 'scripts/real_video_spatial/validate_ledger.py', '_capacity_ledger')
    ledger.MODES += ('bounded_additive',)
    rows = []
    # Validate new packages with their own loader, then original comparator loaders.
    for seed in range(3):
        cfg = reg['configs'][seed]
        run = Path(cfg['output_dir'])
        done = read(OUT / f's{seed}_completion.json')
        path = OUT / f's{seed}_evaluation.json'
        assert done['status'] == 'complete' and done['seed'] == seed
        assert done['registration_sha256'] == c.sha256(OUT / 'registration.json')
        assert done['evaluation_sha256'] == c.sha256(path)
        c.legacy.validate_completed(run)
        executed = read(run / 'training_config.json')
        assert c.legacy.base.scientific_config(executed) == c.legacy.base.scientific_config(cfg)
        _, state = c.legacy.read_package(run / 'best')
        row = read(path)
        ledger.validate_ledger(row, cfg, ROOT, state)
        assert row['parameter_counts']['trainable'] == reg['parameter_counts']['control']
        row['mode'] = 'capacity_additive'
        rows.append(row)
        for name in ('training_summary.json', 'metrics.jsonl', 'best/model.pt', 'last/model.pt'):
            sources[str((run / name).relative_to(ROOT))] = c.sha256(run / name)
    for mode, directory in [('bounded_additive', 'real_video_spatial_components'), ('transport', 'real_video_spatial')]:
        trainer = c.private(ROOT / f'scripts/{directory}/train.py', '_capacity_compare_' + mode)
        for seed in range(3):
            cfg = read(ROOT / f'configs/{directory}/v1/{mode}_s{seed}.json')
            run = ROOT / cfg['output_dir']
            trainer.validate_completed(run)
            _, state = trainer.read_package(run / 'best')
            row = read(ROOT / f'reports/{directory}/{mode}_s{seed}_validation.json')
            ledger.validate_ledger(row, cfg, ROOT, state)
            rows.append(row)
    comparisons = {other: ledger.paired_intervals(rows, 'transport', other, 10000, 173)
                   for other in ('bounded_additive', 'capacity_additive')}
    means = {mode: {metric: np.mean([r['summary'][metric] for r in rows if r['mode'] == mode], axis=0).tolist()
                    for metric in ledger.METRICS}
             for mode in ('bounded_additive', 'capacity_additive', 'transport')}
    result = {'status': 'complete_validated', 'models': 9, 'new_models': 3, 'epochs': 30,
              'scope': 'post-hoc original development; unadjusted paired session x seed intervals',
              'parameter_counts': reg['parameter_counts'], 'means': means,
              'comparisons': comparisons, 'source_sha256': sources}
    result['source_sha256'][str(Path(__file__).relative_to(ROOT))] = c.sha256(__file__)
    (OUT / 'finalization.json').write_text(json.dumps(result, indent=2) + '\n')
    tex = r'''\begin{table}[tbp]\centering\small
\begin{tabular}{@{}lrrrrr@{}}\toprule
Method & Active params. & Native h5 & Native h10 & Pooled h5 & Pooled h10 \\ \midrule
'''
    for mode, label, count in [('bounded_additive', 'Bounded additive', reg['parameter_counts']['bounded_additive']),
                               ('capacity_additive', 'Capacity-matched additive', reg['parameter_counts']['control']),
                               ('transport', 'Bounded ShiftWM', reg['parameter_counts']['mixing'])]:
        vals = [means[mode][metric][h-1] for metric in ('native_mse', 'original_2x2_mse') for h in (5, 10)]
        tex += label + f' & {count:,} & ' + ' & '.join(f'{v:.6f}' for v in vals) + r' \\' + '\n'
    tex += r'''\bottomrule\end{tabular}
\caption{\textbf{Active-parameter control on DROID development.} Three seeds per arm;
all trained for 30 epochs with the original selection rule. All errors are MSE
(lower is better), averaged within episode and then across episodes and seeds.
The new patch-local gated MLP adds exactly 18,529 active parameters without
mixing patch positions in the added head. This matches parameter count, not
all expressivity or optimization properties. No model is reselected.}
\label{tab:capacity-control}\end{table}
'''
    (DISPLAY / 'capacity_table.tex').write_text(tex)
    lines = ['# Completed capacity control', '', 'Post-hoc DROID development only. Three new complete 30-epoch runs; six existing comparators revalidated.', '']
    for other, effect in comparisons.items():
        for metric in ('native_mse', 'original_2x2_mse'):
            e = effect['metrics'][metric][9]
            lines.append(f"- Mixing minus {other}, h10 {metric}: {e['mean_difference']:.6f}, 95% CI {e['ci95']}; means {e['first_mean']:.6f} vs {e['second_mean']:.6f}.")
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
