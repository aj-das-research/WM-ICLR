#!/usr/bin/env python3
"""Make a four-frame annotated companion from the reviewed qualitative replay.

This is an animation of measured diagnostics and recorded evaluation targets,
not generated video. No inference, new selection, RGB interpolation or training.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / 'paper/figure_sources/qualitative_closest_v1'
OUT = ROOT / 'paper/generated/qualitative_temporal_v1'
EXPECTED = {
    'derived.json': 'da07f473966c180061d1ee6c67e27c94ce6ad4aef33b21af39492889ea45c7df',
    'manifest.json': 'ad35126e1df545af44d2b3552ca8de61717a5955bac81c54ab5cb4de69a13b92',
    'result_review.json': '8ef21f746952755965265d31c7ff5f237c39829b41238fbdfaf618ed72ef64fd',
}
OFFSETS = (14, 29, 44, 59)
TASKS = ('pusht', 'bimanual_box', 'bimanual_rope')
LABELS = {'pusht': 'PushT', 'bimanual_box': 'Box', 'bimanual_rope': 'Rope'}
CASES = {'pusht': ('000007', 7), 'bimanual_box': ('000006', 18), 'bimanual_rope': ('000000', 2)}
MODES = ('autoregressive', 'bounded_spatial_mix', 'unbounded_spatial_mix')
WIDTH, HEIGHT, DPI = 900, 490, 100
INK, MUTED, RULE = '#203547', '#586A79', '#D9E1E6'
GREEN, ORANGE, BLUE = '#126D51', '#A35426', '#275A91'
DELTA = LinearSegmentedColormap.from_list('companion_gain', ['#AD582B', '#FFFFFF', '#168268'])
WEIGHT = LinearSegmentedColormap.from_list('companion_weight', ['#F5F8FB', '#82ADB6', '#164F61'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify_pack():
    for name, expected in EXPECTED.items():
        require(sha(PACK / name) == expected, 'Frozen pack changed: ' + name)
    manifest = json.loads((PACK / 'manifest.json').read_text())
    require(manifest['status'] == 'complete_reviewed' and manifest['raw_files_included'] is False,
            'Expected the complete reviewed derived-only source pack')
    for name, expected in manifest['files'].items():
        require(Path(name).name == name and sha(PACK / name) == expected, 'Pack member changed')
    data = json.loads((PACK / 'derived.json').read_text())
    review = json.loads((PACK / 'result_review.json').read_text())
    require(data['status'] == review['status'] == 'passed', 'Incomplete replay or review')
    require(review['derived_sha256'] == EXPECTED['derived.json'], 'Review does not bind replay')
    require(data['images_are_predictions'] is False and data['new_training_or_model_selection'] is False,
            'Unexpected replay semantics')
    selected, checks = {}, 0
    for task in TASKS:
        study = next(s for s in data['studies'] if s['id'] == task)
        require(study['scope'] == 'reserved_upstream_validation', 'Wrong IWS split')
        require(study['display_offsets'] == list(OFFSETS) and study['seeds'] == [0, 1, 2],
                'Temporal offsets or seeds differ')
        case = study['cases'][1]
        require((case['episode_id'], case['window_start']) == CASES[task], 'Case identity changed')
        require(case['selection']['rank_descending'] == 4, 'Not the fixed middle-ranked trajectory')
        for mode in MODES:
            model = case['methods'][mode]
            require(len(model['per_seed']) == 3, 'Missing seeds')
            for offset in OFFSETS:
                field = model['mean']['by_offset'][str(offset)]
                maps = np.asarray([s['by_offset'][str(offset)]['patch_mse4x4'] for s in model['per_seed']])
                require(maps.shape == (3, 4, 4) and np.isfinite(maps).all() and (maps >= 0).all(),
                        'Invalid recorded error maps')
                np.testing.assert_allclose(maps.mean(0), field['patch_mse4x4'], rtol=3e-6, atol=2e-7)
                np.testing.assert_allclose(maps.mean(), model['mean']['mse_curve'][offset - 1],
                                           rtol=3e-6, atol=2e-7)
                if mode != 'autoregressive':
                    per = np.asarray([s['by_offset'][str(offset)]['M16x16'] for s in model['per_seed']])
                    require(per.shape == (3, 16, 16) and np.isfinite(per).all() and (per >= 0).all(),
                            'Invalid effective source weights')
                    np.testing.assert_allclose(per.sum(-1), 1, rtol=3e-6, atol=2e-7)
                    np.testing.assert_allclose(per.mean(0), field['M16x16'], rtol=3e-6, atol=2e-7)
                checks += 1
        selected[task] = {'study': study, 'case': case}
    return selected, checks


def image_record(case, native_index, role):
    found = [r for r in case['images'] if r['native_index'] == native_index and r['role'] == role]
    require(len(found) == 1, 'Missing or duplicate exact frame identity')
    rec = found[0]
    path = ROOT / rec['path']
    require(path.resolve().is_relative_to(ROOT.resolve()), 'Unexpected external image path')
    require(sha(path) == rec['sha256'] and rec['crop'] == 'none', 'Recorded frame changed')
    with Image.open(path) as source:
        pixels = np.asarray(source.convert('RGB'))
        require(hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest() == rec['pixel_sha256'],
                'Decoded recorded image identity changed')
        reduced = Image.fromarray(pixels)
        reduced.thumbnail((256, 192), Image.Resampling.LANCZOS)
    return np.asarray(reduced), rec


def label(fig, x, y, text, size=9.2, color=INK, weight='normal', ha='left', va='center'):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, text, fontsize=size, color=color,
                    weight=weight, ha=ha, va=va, linespacing=1.15)


def axes(fig, x, y, w, h):
    return fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])


def photograph(fig, pixels, x, y, query=False):
    ax = axes(fig, x, y, 144, 83)
    ax.imshow(pixels, interpolation='none')
    if query:
        h, w = pixels.shape[:2]
        ax.add_patch(Rectangle((w / 4 - .5, h / 4 - .5), w / 4, h / 4,
                               fill=False, lw=1.8, edgecolor='white'))
        ax.add_patch(Rectangle((w / 4 - .5, h / 4 - .5), w / 4, h / 4,
                               fill=False, lw=.8, edgecolor=GREEN))
    ax.set_axis_off()


def grid(fig, array, x, y, norm, cmap, query=False):
    ax = axes(fig, x, y, 75, 75)
    ax.imshow(array, cmap=cmap, norm=norm, interpolation='none')
    ax.set_xticks(np.arange(.5, 3.5), minor=True)
    ax.set_yticks(np.arange(.5, 3.5), minor=True)
    ax.grid(which='minor', color='#FFFFFF', lw=.5)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    if query:
        ax.add_patch(Rectangle((.5, .5), 1, 1, fill=False, lw=1, edgecolor=INK))
    for sp in ax.spines.values():
        sp.set_linewidth(.45)
        sp.set_color(RULE)
    return ax


def scale_bar(fig, x, y, w, norm, cmap, text, ticks):
    ax = axes(fig, x, y, w, 7)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{v:.2g}' for v in ticks])
    cb.ax.tick_params(labelsize=8, length=2, pad=1)
    cb.outline.set_linewidth(.4)
    label(fig, x + w / 2, y - 10, text, size=8.3, ha='center')


def layout_check(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for txt in fig.findobj(Text):
        if not txt.get_visible() or not txt.get_text().strip():
            continue
        bb = txt.get_window_extent(renderer)
        require(bb.x0 >= -1 and bb.y0 >= -1 and bb.x1 <= WIDTH + 1 and bb.y1 <= HEIGHT + 1,
                'Clipped text: ' + txt.get_text())
        if txt in fig.texts:
            for other, b in boxes:
                require(not bb.overlaps(b), 'Overlapping labels: ' + txt.get_text() + ' / ' + other)
            boxes.append((txt.get_text(), bb))
    return {'text_clipping': [], 'figure_label_overlaps': [], 'canvas_pixels': [WIDTH, HEIGHT]}


def render_frame(selected, offset, step, difference_limit, out):
    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor='white')
    label(fig, 13, 17, 'Recorded scenes · matched feature forecasts', size=12, weight='bold')
    label(fig, 887, 17, f'Saved offset {offset} / 59   |   {step + 1} of 4', size=10.5, ha='right', weight='bold')
    label(fig, 13, 39, 'No-tanh is an ablation; bounded ShiftWM remains the primary method.', size=9.1, color=MUTED)
    headers = [(13, 'Fixed middle\ncase'), (156, 'Recorded input\nfixed across forecasts'),
               (323, 'Recorded target\nevaluation only'), (458, 'AR − no-tanh\nfeature error'),
               (553, 'Source W[q, : ]\nno-tanh (abl.)'), (625, 'Matched MSE ↓  ·  three seeds')]
    for x, text in headers:
        label(fig, x, 66, text, size=8.8, ha='center' if x in (156, 323, 458, 553) else 'left')
    delta_norm = TwoSlopeNorm(0, -difference_limit, difference_limit)
    records = []
    for row, task in enumerate(TASKS):
        case = selected[task]['case']
        y = 89 + row * 103
        start = case['window_start']
        observed, obs_rec = image_record(case, start, 'observed')
        target, tgt_rec = image_record(case, start + offset, 'target')
        label(fig, 13, y + 29, LABELS[task], size=11, weight='bold')
        label(fig, 13, y + 47, 'traj. ' + case['episode_id'][-2:], size=9, color=MUTED)
        photograph(fig, observed, 84, y, query=True)
        photograph(fig, target, 251, y)
        label(fig, 156, y + 90, f'input f{obs_rec["native_index"]}  ·  q marked', size=8.6, ha='center', color=MUTED)
        label(fig, 323, y + 90, f'target f{tgt_rec["native_index"]}', size=8.6, ha='center', color=MUTED)
        maps = {m: np.asarray(case['methods'][m]['mean']['by_offset'][str(offset)]['patch_mse4x4']) for m in MODES}
        mse = {m: float(a.mean()) for m, a in maps.items()}
        diff = maps[MODES[0]] - maps[MODES[2]]
        difference_axes = grid(fig, diff, 420, y + 4, delta_norm, DELTA)
        # A centered slash preserves the unfavorable sign in grayscale.
        for rr in range(4):
            for cc in range(4):
                if diff[rr, cc] < 0:
                    difference_axes.plot([cc - .26, cc + .26], [rr + .26, rr - .26],
                                         color=INK, lw=.65)
        weights = np.asarray(case['methods'][MODES[2]]['mean']['by_offset'][str(offset)]['M16x16'])[5].reshape(4, 4)
        grid(fig, weights, 516, y + 4, Normalize(0, 1), WEIGHT, query=True)
        label(fig, 458, y + 90, f'mean {diff.mean():+.4f}', size=8.5, ha='center')
        label(fig, 553, y + 90, f'self {weights[1, 1]:.2f}', size=8.5, ha='center')
        labels = ('AR', 'Bounded (ours)', 'No-tanh (ours, abl.)')
        for j, (m, name) in enumerate(zip(MODES, labels)):
            label(fig, 625, y + 14 + j * 19, name, size=9.1)
            label(fig, 878, y + 14 + j * 19, f'{mse[m]:.4f}', size=9.5,
                  ha='right', weight='bold' if mse[m] == min(mse.values()) else 'normal')
        g = 100 * (mse[MODES[0]] - mse[MODES[2]]) / mse[MODES[0]]
        label(fig, 625, y + 77, f'No-tanh vs AR: {g:+.2f}% lower MSE', size=9.3,
              color=GREEN if g > 0 else ORANGE, weight='bold')
        if row < 2:
            fig.add_artist(plt.Line2D([.014, .985], [1 - (y + 99) / HEIGHT] * 2,
                                     transform=fig.transFigure, color=RULE, lw=.5))
        records.append({'task': task, 'case': case['id'], 'episode_id': case['episode_id'],
                        'window_start': start, 'offset': offset, 'observed': obs_rec, 'target': tgt_rec,
                        'mean_mse_by_method': mse, 'relative_no_tanh_vs_AR_MSE_reduction_percent': g,
                        'signed_AR_minus_no_tanh_patch_mse4x4': diff.tolist(),
                        'no_tanh_effective_source_weight_q5': weights.tolist()})
    scale_bar(fig, 61, 426, 183, delta_norm, DELTA,
              'AR − no-tanh MSE · slash = higher error', [-difference_limit, 0, difference_limit])
    scale_bar(fig, 362, 426, 158, Normalize(0, 1), WEIGHT,
              'Effective source weight W[q, : ]', [0, 1])
    label(fig, 616, 418, 'Same cases, model parameters\nand scales at every offset.', size=9, color=MUTED)
    label(fig, 13, 456, 'Four sampled forecast offsets · 2 s per display (not physical time). Source weights are not semantic attention.',
          size=8.5, color=MUTED)
    label(fig, 13, 475, 'Recorded RLA-WM / IWS excerpts + measured diagnostics. Companion animation, not model-generated video.',
          size=8.5, color=MUTED)
    qa = layout_check(fig)
    file = out / f'temporal_companion_offset{offset:02d}.png'
    fig.savefig(file, dpi=DPI, facecolor='white')
    plt.close(fig)
    return file, records, qa


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OUT)
    args = p.parse_args()
    selected, checks = verify_pack()
    args.output.mkdir(parents=True, exist_ok=True)
    # A single symmetric scale spans every displayed task and saved offset.
    # These are standardized-feature errors; task labels retain their separate statistics.
    limit = max(float(np.abs(np.asarray(v['case']['methods'][MODES[0]]['mean']['by_offset'][str(h)]['patch_mse4x4']) -
                                np.asarray(v['case']['methods'][MODES[2]]['mean']['by_offset'][str(h)]['patch_mse4x4'])).max())
                for v in selected.values() for h in OFFSETS)
    require(limit > 0, 'Degenerate difference scale')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.unicode_minus': True})
    files, rows, reviews = [], [], []
    for step, offset in enumerate(OFFSETS):
        f, rec, qa = render_frame(selected, offset, step, limit, args.output)
        files.append(f); rows.append(rec); reviews.append(qa)
    rgb = [Image.open(p).convert('RGB') for p in files]
    # One global 256-color palette prevents changing color meaning across GIF frames.
    montage = Image.new('RGB', (WIDTH, HEIGHT * len(rgb)))
    for j, im in enumerate(rgb):
        montage.paste(im, (0, j * HEIGHT))
    palette = montage.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in rgb]
    gif = args.output / 'temporal_companion.gif'
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=2000,
                   loop=0, disposal=2, optimize=False)
    with Image.open(gif) as check:
        require(check.n_frames == 4 and check.size == (WIDTH, HEIGHT), 'GIF frame count/size changed')
        durations = []
        for j in range(4):
            check.seek(j); durations.append(check.info.get('duration'))
            require(np.array_equal(np.asarray(check.convert('RGB')), np.asarray(frames[j].convert('RGB'))),
                    'Encoded GIF frame differs from its fixed-palette source composite')
        require(durations == [2000] * 4, 'GIF timing changed')
    # Duplicate each saved composite for two seconds; there is no temporal RGB interpolation.
    mp4 = args.output / 'temporal_companion.mp4'
    subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-framerate', '1/2', '-pattern_type', 'glob',
                    '-i', str(args.output / 'temporal_companion_offset??.png'),
                    '-vf', 'fps=24', '-t', '8', '-c:v', 'libx264', '-crf', '18',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(mp4)], check=True)
    probe = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                      '-show_entries', 'stream=width,height,nb_frames,r_frame_rate,duration,pix_fmt',
                                      '-of', 'json', str(mp4)], capture_output=True, text=True, check=True).stdout)
    stream = probe['streams'][0]
    require((stream['width'], stream['height']) == (WIDTH, HEIGHT)
            and stream['nb_frames'] == '192' and stream['r_frame_rate'] == '24/1'
            and abs(float(stream['duration']) - 8.0) < 1e-6 and stream['pix_fmt'] == 'yuv420p',
            'Companion video size/timing differs')
    poster = args.output / 'temporal_companion_poster.png'
    shutil.copyfile(files[0], poster)
    evidence = {'schema': 'shiftwm_temporal_companion_v1', 'status': 'rendered_pending_independent_review',
                'source_sha256': {str((PACK / n).relative_to(ROOT)): h for n, h in EXPECTED.items()},
                'renderer_sha256': sha(Path(__file__)), 'seeds': [0, 1, 2],
                'selection': 'Existing fixed middle-ranked reserved IWS trajectories and minimum registered starts; no reselection',
                'scope': 'Annotated recorded evaluation targets and measured feature diagnostics; not model-generated video',
                'primary': 'bounded ShiftWM', 'no_tanh_role': 'secondary ablation',
                'offsets': list(OFFSETS), 'command_call_length': 60, 'duration_ms_per_frame': 2000,
                'GIF_decoded_frames_exactly_match_shared_palette_sources': True,
                'loop': 'infinite for GIF; user-controlled for MP4', 'playback_seconds': 8,
                'playback_is_physical_time': False, 'mp4_encoding': stream,
                'temporal_interpolation': 'none; repeated source composites only', 'query_patch_zero_based': [1, 1],
                'scales': {'AR_minus_no_tanh_patch_MSE': [-limit, limit], 'effective_source_weights': [0, 1],
                           'constant_across_frames': True, 'negative_cells': 'centered diagonal slash; higher no-tanh error',
                           'MSE_coordinates': 'Each task uses its own fixed training channel scales; shared color range is visual, not a pooled task metric.'},
                'frame_records': rows, 'layout_checks': reviews, 'numeric_map_checks': checks,
                'images': {'max_source_excerpt_pixels': [256, 192], 'crop': 'none', 'resize': 'aspect-preserving PIL Lanczos',
                           'raw_file_and_decoded_pixel_hashes_verified': True, 'raw_frames_exported': False,
                           'only_composite_annotated_frames_exported': True, 'GIF_palette': 'one shared 256-color palette; PNGs retain full color',
                           'source': 'RLA-WM / IWS dataset; license unspecified; research excerpts retain attribution'},
                'no_training_inference_interpolation_or_model_selection': True,
                'outputs_sha256': {p.name: sha(p) for p in [*files, gif, mp4, poster]},
                'caption': 'Forecast evolution on the fixed middle-ranked reserved PushT, Box and Rope cases. Four recorded targets (evaluation only) align with measured AR-minus-no-tanh feature-error maps and no-tanh effective observed-source weights at offsets 14, 29, 44 and 59. MSE values average three seeds; both bounded primary and no-tanh ablation remain labeled. Color scales stay fixed. The 8-second playback holds each sampled offset for 2 seconds and does not represent physical time. This is an annotated diagnostic animation, not model-generated video; source weights are not semantic attention or causal attribution.'}
    (args.output / 'companion_evidence.json').write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': evidence['status'], 'frames': 4, 'output': str(args.output),
                      'duration_ms_per_frame': 2000, 'numeric_map_checks': checks}))


if __name__ == '__main__':
    main()
