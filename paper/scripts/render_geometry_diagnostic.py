#!/usr/bin/env python3
"""Evidence-backed latent-goal contraction diagnostic; real image examples."""
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from shiftwm.data import pixels_to_tensor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/diagnostics/goal_geometry'
OUT.parent.mkdir(parents=True, exist_ok=True)
source = ROOT / 'reports/evidence/extensions_mechanism_probe.json'
evidence = json.loads(source.read_text())
dataset = ROOT / 'data/extensions/drone_v1'
manifest = json.loads((dataset / 'manifest.json').read_text())
goals = []
for ep in manifest['episodes']:
    if ep['split'] != 'development' or ep['dynamics_id'] != 1:
        continue
    path = dataset / ep['audit_file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != ep['audit_sha256']:
        raise ValueError('Goal source hash changed')
    with np.load(path, allow_pickle=False) as data:
        goals.append((float(data['goal_state'][0]), ep, data['goal_image'].copy()))
goals.sort(key=lambda item: (item[0], item[1]['seed']))
chosen = [goals[i] for i in (0, len(goals) // 2, len(goals) - 1)]
canonical = evidence['goal_geometry']['canonical']['pairwise_latent_mse_mean']
warm = evidence['goal_geometry']['warm_uncorrected']['pairwise_latent_mse_mean']
rows = {r['mode']: r for r in evidence['rows'] if r['architecture'] == 'transformer'}
values = np.array([canonical, warm,
    rows['framewise']['goal_geometry_mean_over_fixed_support_contexts']['pairwise_latent_mse_mean'],
    rows['factorized']['goal_geometry_mean_over_fixed_support_contexts']['pairwise_latent_mse_mean'],
    1.21 * warm]) / canonical * 100
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 8.5,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#CED4DA',
                     'text.color': '#243447', 'axes.labelcolor': '#243447'})
fig = plt.figure(figsize=(5.5, 3.3), facecolor='white')
fig.text(.045, .94, 'a  Same physical goals', fontsize=10, weight='bold')
fig.text(.535, .94, 'b  Goal separation', fontsize=10, weight='bold')
for row, (appearance, title) in enumerate(((0, 'Canonical'), (1, 'Warm shift'))):
    y = .555 if row == 0 else .235
    fig.text(.045, y + .255, title, fontsize=8.5, color='#386A87' if row == 0 else '#A05D27')
    for col, (_, ep, image) in enumerate(chosen):
        ax = fig.add_axes([.045 + .135 * col, y, .125, .23])
        pixels = pixels_to_tensor(image, appearance).permute(1, 2, 0).numpy()
        ax.imshow(pixels[24:104, 24:104], interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(.6); spine.set_color('#CED4DA')
        if row == 1:
            ax.set_xlabel(f'Goal {col + 1}', labelpad=3, fontsize=8)
fig.text(.045, .105, 'Recorded RGB · identical crops', fontsize=8, color='#52616B')
ax = fig.add_axes([.71, .255, .235, .57])
labels = ['Canonical', 'Warm', 'Framewise', 'ShiftWM (ours)', 'FiLM ceiling']
colors = ['#386A87', '#C17C3A', '#7B7F86', '#147D78', '#FFFFFF']
bars = ax.barh(np.arange(5), values, height=.56, color=colors,
               edgecolor=['none'] * 4 + ['#98632B'], linewidth=1)
bars[-1].set_hatch('////')
ax.set_yticks(np.arange(5), labels=labels, fontsize=8)
ax.invert_yaxis(); ax.set_xlim(0, 119)
ax.set_xticks([0, 50, 100]); ax.tick_params(axis='both', length=0)
ax.set_xlabel('Separation (% of canonical)', fontsize=8, labelpad=6)
ax.xaxis.grid(True, color='#E9EDF0', linewidth=.6); ax.set_axisbelow(True)
ax.spines['left'].set_visible(False)
for index, value in enumerate(values):
    ax.text(value + 2, index, f'{value:.1f}', va='center', fontsize=8)
fig.text(.535, .105, '16 goals · 120 pairs · development', fontsize=8, color='#52616B')
for extension in ('pdf', 'svg', 'png'):
    fig.savefig(OUT.with_suffix('.' + extension), dpi=240, facecolor='white')
    shutil.copyfile(OUT.with_suffix('.' + extension), ROOT / 'paper/figures' / ('goal_geometry.' + extension))
plt.close(fig)
caption = ('Lighting shifts can compress latent goal geometry. (a) Three recorded drone endpoint goals '
           'under canonical and registered warm photometric transforms; identical central80×80 crops. '
           'Examples are selected by minimum, median and maximum X position, not model outcome. '
           '(b) Mean pairwise latent MSE over all16development goals/120pairs, normalized by canonical '
           'separation. Corrected-goal measurements use a fixed support context across all goals and '
           'average over16contexts; transformer seed0 is shown. The hatched bar is the analytic1.21× '
           'upper expansion bound of the original observation FiLM applied to warm features, not an '
           'achieved result or uncertainty interval. The full six-model diagnostic is reported separately. '
           'This descriptive geometry limitation does not establish a cause of planning failure.')
OUT.with_suffix('.caption.txt').write_text(caption + '\n')
ledger = {'source': str(source.relative_to(ROOT)), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'width_inches': 5.5, 'height_inches': 3.3, 'labels': labels, 'separation_percent': values.tolist(),
          'images': [{'trajectory_id': ep['trajectory_id'], 'file': ep['audit_file'], 'sha256': ep['audit_sha256']}
                     for _, ep, _ in chosen], 'crop_xyxy': [24, 24, 104, 104],
          'output_hashes': {extension: hashlib.sha256(OUT.with_suffix('.' + extension).read_bytes()).hexdigest()
                            for extension in ('pdf', 'svg', 'png')}}
OUT.with_suffix('.json').write_text(json.dumps(ledger, indent=2) + '\n')
print(OUT.with_suffix('.pdf'))
