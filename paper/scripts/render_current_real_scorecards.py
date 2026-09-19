#!/usr/bin/env python3
"""Presentation-only DROID/IWS scorecards from finalized, saved feature errors.

No model, checkpoint, cache, or reserved payload is opened. Public copies render
the small source-bound pack with --from-pack. A live refresh rechecks completed
finalization identities and their saved JSON metric receipts; it never evaluates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACK = Path('paper/figure_sources/current_real_scorecards')
OUT = Path('paper/generated/benchmark_scorecards')
SPATIAL = 'paper/generated/real_video/spatial_versions_evidence.json'
SPATIAL_SHA = 'f64ef1ee15bc7e01c26caf6febd1e1c3a1c2352a9f0b283364363b05288774be'
DROID_FINAL = 'reports/real_video_spatial/finalization.json'
COMPONENT_FINAL = 'reports/real_video_spatial_components/finalization.json'
IWS_FINAL = 'reports/real_video_iws/development_finalization.json'
IWS_FINAL_SHA = '80463fc4ed452d370259823a66091af2de0706c5b664fa5e3c2d308a6c0d9a43'
UNBOUNDED_FINAL = 'reports/real_video_iws_unbounded/development_finalization.json'
UNBOUNDED_REG_SHA = '39256bc91067fee19ff448b9ef9ec304b93cd0b6e7e1296e4142815820229eeb'
TASKS = ('pusht', 'bimanual_box', 'bimanual_rope')
TASK_NAMES = dict(zip(TASKS, ('PushT', 'Box', 'Rope')))
BASE = ('autoregressive', 'anchored_additive', 'bounded_spatial_mix')
METRICS = ('standardized_mse', 'standardized_mae', 'raw_dinov2_l1', 'feature_cosine_distance')
DROID_MODES = ('autoregressive', 'persistence', 'transport', 'bounded_additive',
               'unbounded_transport', 'anchored_additive', 'context_off', 'action_free')
DROID_LABELS = dict(zip(DROID_MODES, ('Autoregressive', 'Persistence', 'ShiftWM (ours)',
    'No mixing (ours, ablation)', 'No bound (ours, ablation)', 'Additive anchor (ours, ablation)',
    'No context (ours, ablation)', 'No actions (ours, ablation)')))
IWS_LABELS = {'persistence':'Persistence', 'autoregressive':'Autoregressive',
    'anchored_additive':'Additive anchor', 'bounded_spatial_mix':'ShiftWM (ours)',
    'unbounded_spatial_mix':r'No $\tanh$ (ours, ablation)'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'


def local(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Source path escapes project')
    return path


def read_bound(name, sources, expected=None):
    path = local(name)
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError('Changed finalized score source: ' + name)
    sources[name] = actual
    return json.loads(path.read_text())


def same(a, b, message):
    if not np.allclose(a, b, rtol=0, atol=2e-14):
        raise ValueError(message)


def finite(values, shape):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != shape or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Incomplete or nonfinite metric array')
    return values


def droid_scores(sources):
    evidence = read_bound(SPATIAL, sources, SPATIAL_SHA)
    if (evidence['completed_runs'] != 21 or evidence['epochs_each'] != 30
            or evidence['pending_results'] or evidence['population'] != {'episodes':141,'sessions':59,'windows':1631}):
        raise ValueError('All 21 completed DROID models are required')
    ledgers = evidence['source_sha256']
    old = read_bound(DROID_FINAL, sources, ledgers[DROID_FINAL])
    new = read_bound(COMPONENT_FINAL, sources, ledgers[COMPONENT_FINAL])
    if (old['status'] != 'passed' or old['completed_models'] != 15 or old['epochs_per_model'] != 30
            or new['status'] != 'passed' or new['completed_new_models'] != 6 or new['epochs_per_model'] != 30):
        raise ValueError('Incomplete DROID finalization')
    methods, population = {}, None
    for mode in DROID_MODES:
        owner = 'autoregressive' if mode == 'persistence' else mode
        report = new if owner in ('bounded_additive','unbounded_transport') else old
        rows = sorted([r for r in report['runs'] if r['name'].rsplit('_s',1)[0] == owner], key=lambda r:r['name'])
        if [r['name'] for r in rows] != [f'{owner}_s{s}' for s in range(3)]:
            raise ValueError('Missing DROID method/seed')
        values = {m:[] for m in ('native_mse','original_2x2_mse')}
        for seed, row in enumerate(rows):
            receipt = read_bound(row['validation'], sources, ledgers[row['validation']])
            if receipt['status'] != 'complete' or receipt['mode'] != owner or receipt['seed'] != seed:
                raise ValueError('Wrong DROID receipt identity')
            episodes = receipt['episodes']
            pop = [(e['episode_id'],e['session_id'],e['windows']) for e in episodes]
            if (len(pop) != 141 or len({e[0] for e in pop}) != 141
                    or len({e[1] for e in pop}) != 59 or sum(e[2] for e in pop) != 1631
                    or population is not None and pop != population):
                raise ValueError('DROID matched population changed')
            population = pop
            for metric in values:
                key = metric.replace('_mse','_persistence_mse') if mode == 'persistence' else metric
                mean = finite([e[key] for e in episodes], (141,10)).mean(0)
                same(mean, receipt['summary'][key], 'DROID episode aggregation differs')
                values[metric].append(mean.tolist())
        means = {m:np.asarray(v).mean(0).tolist() for m,v in values.items()}
        for metric, curve in means.items():
            key = metric.replace('_mse','_persistence_mse') if mode == 'persistence' else metric
            same(curve, evidence['absolute_error'][owner][key], 'DROID completed aggregate differs')
        methods[mode] = {'per_seed_curves':values, 'mean_curves':means}
    return {'scope':'original_validation_development_only', 'population':evidence['population'],
            'seed_ids':[0,1,2], 'steps':list(range(1,11)), 'methods':methods,
            'metrics':{'native_mse':'training-standardized native 4x4 feature MSE',
                       'original_2x2_mse':'original-normalized pooled 2x2 feature MSE'},
            'not_measured':['feature MAE','feature cosine distance','RGB quality','physical control success']}


def validate_grid(rows, modes, count):
    keys = [(r.get('task'),r.get('mode'),r.get('seed')) for r in rows]
    expected = {(t,m,s) for t in TASKS for m in modes for s in range(3)}
    if len(keys) != count or set(keys) != expected:
        raise ValueError('Missing, duplicate or substituted task/method/seed')


def iws_scores(sources):
    final = read_bound(IWS_FINAL, sources, IWS_FINAL_SHA)
    if (final.get('status') != 'passed' or final.get('completed_runs') != 27
            or final.get('expected_runs') != 27 or final.get('scope') != 'internal_development'
            or final.get('official_validation_payloads_read') != 0):
        raise ValueError('Complete internal-development v1 finalization required')
    rows = final['per_run']; validate_grid(rows, BASE, 27)
    bindings = dict(final['source_dependencies'])
    extra = None
    if local(UNBOUNDED_FINAL).exists():
        extra = read_bound(UNBOUNDED_FINAL, sources)
        if (extra.get('schema') != 'iws_unbounded_complete_comparison_v1' or extra.get('status') != 'passed'
                or extra.get('completed_new_runs') != 9 or extra.get('completed_v1_comparator_runs') != 27
                or extra.get('official_validation_payloads_read') != 0
                or extra.get('scope') != 'exploratory_internal_development_after_v1'
                or extra.get('registration_sha256') != UNBOUNDED_REG_SHA
                or extra.get('baseline_finalization_sha256') != IWS_FINAL_SHA):
            raise ValueError('Unbounded inclusion requires complete validated nine-plus-27 finalization')
        validate_grid(extra['per_run'], (*BASE,'unbounded_spatial_mix'), 36)
        original = {(r['task'],r['mode'],r['seed']):r for r in rows}
        for r in extra['per_run']:
            key = (r['task'],r['mode'],r['seed'])
            if key in original and r != original[key]:
                raise ValueError('Frozen v1 comparator changed')
        if any(extra['source_dependencies'].get(p) != v for p,v in bindings.items()):
            raise ValueError('Unbounded finalizer dropped a v1 source binding')
        bindings = extra['source_dependencies']; rows = extra['per_run']
    result = {t:{'scope':'internal_development','seed_ids':[0,1,2], 'methods':{}} for t in TASKS}
    baseline_arrays, population = {}, {}
    for row in rows:
        if (row.get('completed_epochs') != 30 or row.get('selected_checkpoint_sha256') not in bindings.values()
                or bindings.get(row['evaluation_path']) != row['evaluation_sha256']):
            raise ValueError('Incomplete or unbound selected checkpoint/receipt')
        receipt = read_bound(row['evaluation_path'], sources, row['evaluation_sha256'])
        required = {'status':'passed','scope':'internal_development','completed_epochs':30,
                    'offsets':list(range(1,60)), 'metrics':list(METRICS), 'official_validation_payloads_read':0,
                    **{k:row[k] for k in ('task','mode','seed','selected_epoch','selected_checkpoint_sha256')}}
        if any(receipt.get(k) != v for k,v in required.items()):
            raise ValueError('IWS receipt does not match completed identity')
        task,mode,seed = row['task'],row['mode'],row['seed']
        episodes = receipt['episodes']; n = 120 if task == 'pusht' else 121
        pop = [(e['episode_id'],e['windows']) for e in episodes]
        audit = receipt['population_audit']
        if (audit['split'] != 'internal_development' or len(pop) != n or len({x[0] for x in pop}) != n
                or sum(x[1] for x in pop) != n*28 or receipt['eligible_trajectories'] != n
                or receipt['total_windows'] != n*28 or task in population and pop != population[task]):
            raise ValueError('IWS development population differs')
        population[task] = pop
        value = finite([[e[m+'_by_offset'] for m in METRICS] for e in episodes], (n,4,59))
        persistence = finite([[e['persistence_'+m+'_by_offset'] for m in METRICS] for e in episodes], (n,4,59))
        if task in baseline_arrays and not np.array_equal(persistence, baseline_arrays[task]):
            raise ValueError('Common persistence differs across arms/seeds')
        baseline_arrays[task] = persistence
        for name, data in [(mode,value), *(([('persistence',persistence)]) if mode == 'autoregressive' else [])]:
            entry = result[task]['methods'].setdefault(name, {'per_seed_curves':{}})
            entry['per_seed_curves'][str(seed)] = dict(zip(METRICS,data.mean(0).tolist()))
    for task, study in result.items():
        study['population'] = {'trajectories':len(population[task]),'windows':sum(x[1] for x in population[task])}
        for mode, values in study['methods'].items():
            if set(values['per_seed_curves']) != {'0','1','2'}:
                raise ValueError('Partial seeds cannot populate a scorecard')
            values['mean_curves'] = {m:np.mean([values['per_seed_curves'][str(s)][m] for s in range(3)],axis=0).tolist() for m in METRICS}
            if extra is not None:
                for metric in METRICS:
                    same(values['mean_curves'][metric],extra['results']['task_results'][task][metric]['means_all59_offsets'][mode],
                         'Complete unbounded report disagrees with episode metrics')
    return {'tasks':result, 'offsets':list(range(1,60)), 'metrics':list(METRICS),
            'completion':{'v1_runs':27,'epochs_each':30,'v1_finalization_sha256':IWS_FINAL_SHA,
                          'unbounded_runs':9 if extra else 0,
                          'unbounded_finalization_sha256':sources.get(UNBOUNDED_FINAL)},
            'unbounded_included':extra is not None, 'not_measured':['RGB quality','physical control success'],
            'unbounded_scope':'exploratory follow-up after v1; distinct registered ablation' if extra else None}


def validate_payload(data):
    if data.get('schema') != 'current_real_scorecards_v1' or data.get('status') != 'complete_validated_scores':
        raise ValueError('Unknown portable scorecard pack')
    if data.get('official_validation_payloads_read') != 0 or not data.get('source_sha256'):
        raise ValueError('Missing source or access boundary')
    droid = data['droid']; iws = data['iws']
    if set(droid['methods']) != set(DROID_MODES) or droid['seed_ids'] != [0,1,2] or droid['steps'] != list(range(1,11)):
        raise ValueError('Incomplete DROID scorecard')
    for item in droid['methods'].values():
        for m in ('native_mse','original_2x2_mse'):
            a = finite(item['per_seed_curves'][m], (3,10))
            same(a.mean(0),finite(item['mean_curves'][m],(10,)), 'DROID pack means differ')
    expected = {*BASE,'persistence'} | ({'unbounded_spatial_mix'} if iws['unbounded_included'] else set())
    if set(iws['tasks']) != set(TASKS) or iws['metrics'] != list(METRICS) or iws['offsets'] != list(range(1,60)):
        raise ValueError('Incomplete IWS scorecard')
    if iws['unbounded_included'] and UNBOUNDED_FINAL not in data['source_sha256']:
        raise ValueError('No full-nine source for unbounded rows')
    completion = iws['completion']
    if (completion != {'v1_runs':27,'epochs_each':30,'v1_finalization_sha256':IWS_FINAL_SHA,
                       'unbounded_runs':9 if iws['unbounded_included'] else 0,
                       'unbounded_finalization_sha256':data['source_sha256'].get(UNBOUNDED_FINAL)}):
        raise ValueError('Portable completion gate differs')
    for study in iws['tasks'].values():
        if set(study['methods']) != expected or study['seed_ids'] != [0,1,2]:
            raise ValueError('Missing or extra IWS method')
        for item in study['methods'].values():
            if set(item['per_seed_curves']) != {'0','1','2'}:
                raise ValueError('Missing IWS seed')
            for m in METRICS:
                a = finite([item['per_seed_curves'][str(s)][m] for s in range(3)],(3,59))
                same(a.mean(0), finite(item['mean_curves'][m],(59,)), 'IWS pack means differ')


def score(value, best):
    text = f'{value:.6f}'
    return r'\textbf{' + text + '}' if value == best else text


def droid_table(data):
    methods = data['methods']; columns = [(m,h) for m in ('native_mse','original_2x2_mse') for h in (5,10)]
    rows = []
    for mode in DROID_MODES:
        cells = [score(methods[mode]['mean_curves'][m][h-1], min(x['mean_curves'][m][h-1] for x in methods.values())) for m,h in columns]
        rows.append(' & '.join([DROID_LABELS[mode],*cells]) + r' \\')
    return (r'\begin{table}[!htb]\centering'+'\n'+
        r'\caption{Complete current DROID spatial-development scorecard. Lower feature MSE is better; black bold marks the lowest point mean in each column, not significance. Native and pooled coordinates use different fixed training normalizations and cannot be compared numerically across groups. All seven learned variants completed 30 epochs at three seeds; persistence is untrained. Means weight windows within episode, then 141 episodes and seeds equally (59 sessions; 1,631 windows). Each query step spans five native transitions. Feature MAE/cosine, RGB quality and physical-control success were not measured in this spatial study. Paired uncertainty remains in the spatial comparison tables; the source pack retains every seed curve.}'+'\n'+
        r'\label{tab:spatial-native}\label{tab:spatial-pooled}\label{tab:current-droid-scorecard}'+'\n'+
        r'\begingroup\fontsize{9}{10.8}\selectfont\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.13}'+'\n'+
        r'\begin{tabular}{@{}lrrrr@{}}\toprule'+'\n'+
        r'& \multicolumn{2}{c}{Native $4\times4$ MSE $\downarrow$} & \multicolumn{2}{c}{Pooled $2\times2$ MSE $\downarrow$} \\'+'\n'+
        r'\cmidrule(lr){2-3}\cmidrule(l){4-5}Method & h5 & h10 & h5 & h10 \\ \midrule'+'\n'+
        '\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}\endgroup\end{table}'+'\n')


def iws_table(data):
    rows = []; modes = ('persistence', *BASE, *(('unbounded_spatial_mix',) if data['unbounded_included'] else ()))
    for task in TASKS:
        study = data['tasks'][task]; p = study['population']
        rows.append(r'\multicolumn{5}{@{}l}{\textbf{' + TASK_NAMES[task] + '}' +
                    f" --- {p['trajectories']} trajectories, {p['windows']:,} windows" + r'} \\')
        for mode in modes:
            cells = [score(study['methods'][mode]['mean_curves'][m][58], min(x['mean_curves'][m][58] for x in study['methods'].values())) for m in METRICS]
            rows.append(' & '.join([IWS_LABELS[mode],*cells]) + r' \\')
        rows.append(r'\addlinespace[3pt]')
    extra = (' The no-tanh row is a separately registered exploratory follow-up after v1 development, included only after all nine new and 27 original runs passed complete-study validation.' if data['unbounded_included'] else '')
    return (r'\begin{table}[!htb]\centering'+'\n'+
        r'\caption{Complete IWS internal-development feature metrics at $H=60$ (stored offset 59). Lower is better in every column; black bold marks each task/metric\textquotesingle s lowest point mean, not significance. Learned methods use one image and supplied causal commands; persistence ignores commands. Means weight windows within trajectory, trajectories and three seeds equally. MSE/MAE use task-specific training scales; raw L1 uses DINOv2 coordinates; cosine distance uses flattened raw features. RGB quality and physical-control success were not measured. Reserved upstream-validation data are excluded. The main MSE comparison table retains paired gains and intervals.'+extra+'}\n'+
        r'\label{tab:iws-secondary}\label{tab:current-iws-scorecard}'+'\n'+
        r'\begingroup\fontsize{9}{10.8}\selectfont\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.10}'+'\n'+
        r'\begin{tabular}{@{}lrrrr@{}}\toprule Method & Std. MSE $\downarrow$ & Std. MAE $\downarrow$ & Raw L1 $\downarrow$ & Cosine $\downarrow$ \\ \midrule'+'\n'+
        '\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}\endgroup\end{table}'+'\n')


def write_atomic(path, text):
    if path.exists() and path.read_text() == text:
        return False
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.scorecard-',delete=False) as stream:
        stream.write(text); temporary = Path(stream.name)
    temporary.replace(path)
    return True


def render(output=None, from_pack=False, if_ready=False):
    output = local(OUT) if output is None else Path(output).resolve()
    pack = local(PACK)
    if from_pack:
        if not (pack/'manifest.json').exists() and if_ready:
            return {'status':'pending','outputs_written':False}
        manifest = json.loads((pack/'manifest.json').read_text())
        if manifest.get('schema') != 'current_real_scorecards_pack_v1' or sha(pack/'data.json') != manifest['data_sha256']:
            raise ValueError('Portable pack hash/schema mismatch')
        data = json.loads((pack/'data.json').read_text())
    else:
        if not all(local(p).exists() for p in (SPATIAL,DROID_FINAL,COMPONENT_FINAL,IWS_FINAL)):
            if if_ready:
                return {'status':'pending','outputs_written':False}
            raise ValueError('Completed source studies required; public copies use --from-pack')
        sources = {}
        data = {'schema':'current_real_scorecards_v1','status':'complete_validated_scores',
                'droid':droid_scores(sources),'iws':iws_scores(sources),
                'source_sha256':sources,'official_validation_payloads_read':0}
        if any(sha(local(p)) != v for p,v in sources.items()):
            raise ValueError('Evidence changed during scorecard refresh')
    validate_payload(data)
    tables = {'droid_spatial.tex':droid_table(data['droid']), 'iws_development.tex':iws_table(data['iws'])}
    if not from_pack:
        content = encoded(data)
        manifest = {'schema':'current_real_scorecards_pack_v1','data_sha256':hashlib.sha256(content.encode()).hexdigest(),
                    'provenance':'Only finalized JSON metric receipts; hashes name private backing evidence but are not public runtime dependencies.',
                    'source_sha256':data['source_sha256']}
        write_atomic(pack/'data.json',content); write_atomic(pack/'manifest.json',encoded(manifest))
    record = {'schema':'current_real_scorecards_render_v1','renderer_sha256':sha(__file__),
              'pack_sha256':sha(pack/'data.json'),'manifest_sha256':sha(pack/'manifest.json'),
              'unbounded_included':data['iws']['unbounded_included'],
              'font_pt':9,'best_mean_style':'bold black; no significance claim',
              'outputs_sha256':{n:hashlib.sha256(t.encode()).hexdigest() for n,t in tables.items()}}
    changed = [write_atomic(output/n,t) for n,t in tables.items()]
    changed.append(write_atomic(output/'scorecards.json',encoded(record)))
    return {'status':'complete_validated_scores','outputs_written':any(changed),
            'unbounded_included':data['iws']['unbounded_included'],'droid_rows':8,
            'iws_rows':15 if data['iws']['unbounded_included'] else 12}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--if-ready',action='store_true')
    parser.add_argument('--from-pack',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.output_dir,args.from_pack,args.if_ready),sort_keys=True))
