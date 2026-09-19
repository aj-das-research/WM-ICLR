#!/usr/bin/env python3
"""Refresh the study matrix; incomplete IWS work never supplies method scores.

This reporter reads cache metadata and completed, hash-bound result receipts.
It does not open reserved data, launch models, or ingest training-log scores.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper/generated/experiment_alignment'
FINAL = ROOT / 'reports/real_video_iws/development_finalization.json'
TASKS = {'pusht': ('PushT', 4), 'bimanual_box': ('Box', 14), 'bimanual_rope': ('Rope', 8)}
MODES = ('autoregressive', 'anchored_additive', 'bounded_spatial_mix')
SCHEMA = 'shiftwm_iws_development_finalization_v1'
SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def local_path(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Evidence path escapes the research workspace')
    return path


def checked_json(path, sources, expected=None):
    path = Path(path)
    actual = SHA(path)
    if expected is not None and actual != expected:
        raise ValueError(f'Source identity mismatch: {path}')
    sources[str(path.relative_to(ROOT))] = actual
    return json.loads(path.read_text())


def cache_metadata(task, sources):
    folder = ROOT / 'data/features' / f'iws_{task}_spatial_v1'
    manifest = checked_json(folder / 'manifest.json', sources)
    if manifest['status'] != 'complete' or manifest['official_validation_payloads_read'] != 0:
        raise ValueError('IWS cache completion/access gate failed')
    if manifest['model_training_or_evaluation'] is not False or manifest['task'] != task:
        raise ValueError('Cache metadata is not predictor-performance evidence')
    if manifest['command_width'] != TASKS[task][1]:
        raise ValueError('Native action width changed')
    identity = checked_json(folder / 'identity.json', sources, manifest['identity_sha256'])
    index = checked_json(folder / 'episode_index.json', sources, manifest['episode_index_sha256'])
    stats = checked_json(folder / 'training_statistics.json', sources, manifest['training_statistics_sha256'])
    if identity['official_validation_payloads_allowed'] or stats['fit_split'] != 'internal_train':
        raise ValueError('Reserved-data or normalization split violation')
    if len(index['episodes']) != manifest['episodes']:
        raise ValueError('Cache episode inventory is incomplete')
    result = {k:manifest[k] for k in ('task','episodes','native_frames','command_width','counts')}
    result['development_population'] = {
        item['episode_id']: len(range(0,item['frames']-60,5))
        for item in index['episodes']
        if item['split']=='internal_development' and item['frames']>60}
    return result


def crossed_intervals(task_arrays, draws=10000, seed=173):
    """Paired seed × trajectory bootstrap; shared seed identities across tasks."""
    rng=np.random.default_rng(seed); comparators=('anchored_additive','autoregressive','persistence')
    sampled={task:{mode:{'difference':[],'gain':[]} for mode in comparators} for task in task_arrays}
    macro=[]
    for _ in range(draws):
        seeds=rng.integers(0,3,size=3);primary_gains=[]
        for task,arrays in task_arrays.items():
            n=arrays['bounded_spatial_mix'].shape[1];episodes=rng.integers(0,n,size=n)
            ours=float(arrays['bounded_spatial_mix'][seeds][:,episodes].mean())
            for mode in comparators:
                baseline=float(arrays[mode][seeds][:,episodes].mean())
                gain=None if baseline==0 else 100*(baseline-ours)/baseline
                sampled[task][mode]['difference'].append(ours-baseline)
                sampled[task][mode]['gain'].append(gain)
                if mode=='anchored_additive':primary_gains.append(gain)
        macro.append(None if any(g is None for g in primary_gains) else float(np.mean(primary_gains)))
    def interval(values):
        return None if any(x is None for x in values) else np.quantile(values,[.025,.975]).tolist()
    return {'task_intervals':{task:{mode:{k:interval(v) for k,v in values.items()} for mode,values in rows.items()} for task,rows in sampled.items()},
            'macro_gain_interval':interval(macro),'draws':draws,'seed':seed,
            'units':'matched training seeds and trajectories; trajectory draws independent by task; same seed draw across tasks',
            'confidence':'unadjusted exploratory 95 percent percentile intervals'}


def completed_development(sources,caches):
    """No numeric fallback: the authoritative completion contract is mandatory."""
    pending = {'status':'pending', 'reason':'No completed IWS predictor/evaluation finalizer.',
               'numerical_results':None, 'official_validation':'reserved; not ingested'}
    if not FINAL.exists():
        return pending
    final = checked_json(FINAL, sources)
    if final.get('schema') != SCHEMA:
        raise ValueError('Unknown IWS finalization schema; paper results remain unaccepted')
    if final.get('status') != 'passed':
        pending['reason'] = 'IWS finalizer has not passed all completion checks.'
        return pending
    if final.get('expected_runs') != 27 or final.get('completed_runs') != 27:
        raise ValueError('A subset of the registered IWS campaign cannot populate results')
    dependencies = final.get('source_dependencies', {})
    if not dependencies or not isinstance(dependencies, dict):
        raise ValueError('Missing scientific source/registration dependency ledger')
    for name, digest in dependencies.items():
        path = local_path(name)
        if SHA(path) != digest:
            raise ValueError(f'Completed IWS evidence has changed: {name}')
        sources[str(path.relative_to(ROOT))] = digest
    registration_sha = final.get('registration_sha256')
    if not registration_sha or registration_sha not in dependencies.values():
        raise ValueError('Current IWS registration is not bound by the finalizer')
    if final.get('scope') != 'internal_development':
        raise ValueError('Development reporter cannot ingest reserved official evaluation')
    config_path='configs/real_video_iws/training_v1.json'
    if config_path not in dependencies:
        raise ValueError('Finalizer must bind the executed IWS training/evaluation configuration')
    config=checked_json(ROOT/config_path,sources,dependencies[config_path])
    if (set(config['tasks'])!=set(TASKS) or set(config['modes'])!=set(MODES)
        or config['seeds']!=[0,1,2] or config['training']['epochs']!=30
        or config['training']['horizon']!=60 or config['training']['validation_stride']!=5
        or config['training']['selector']!='internal_dev_equal_trajectory_endpoint_H60_standardized_mse'):
        raise ValueError('Completed campaign differs from the reviewed paper interface')
    evaluation_config=config['evaluation']
    if (evaluation_config['primary_method']!='bounded_spatial_mix'
        or evaluation_config['primary_comparator']!='anchored_additive'
        or evaluation_config['primary_metric']!='standardized_feature_mse'
        or evaluation_config['bootstrap']['draws']!=10000
        or evaluation_config['bootstrap']['seed']!=173
        or evaluation_config['official_validation_allowed_during_training'] is not False):
        raise ValueError('Primary metric, bootstrap or reserved-data contract changed')
    population={c['task']:c['development_population'] for c in caches}
    runs=final.get('per_run',[]);expected={(task,mode,seed) for task in TASKS for mode in MODES for seed in (0,1,2)}
    observed=[(r.get('task'),r.get('mode'),r.get('seed')) for r in runs]
    if len(observed)!=27 or set(observed)!=expected:
        raise ValueError('Duplicate, missing or unregistered task/method/seed receipt')
    arrays={task:{mode:np.empty((3,len(population[task]),59),dtype=np.float64) for mode in (*MODES,'persistence')} for task in TASKS}
    persistence_by_task={};receipt_sources=[]
    for run in runs:
        task,mode,seed=run['task'],run['mode'],run['seed'];checkpoint=run['selected_checkpoint_sha256']
        if run['completed_epochs']!=30 or checkpoint not in dependencies.values():
            raise ValueError('Unverified training completion or selected checkpoint identity')
        evaluation=checked_json(local_path(run['evaluation_path']),sources,run['evaluation_sha256'])
        if (evaluation.get('schema')!='shiftwm_iws_development_evaluation_v1'
            or evaluation.get('status')!='passed' or evaluation.get('scope')!='internal_development'
            or evaluation.get('horizons')!=[15,30,45,60]
            or evaluation.get('selected_checkpoint_sha256')!=checkpoint):
            raise ValueError('Invalid evaluation identity, horizon or split')
        if run['evaluation_path'] not in dependencies or dependencies[run['evaluation_path']]!=run['evaluation_sha256']:
            raise ValueError('Evaluation receipt is not bound by the authoritative source ledger')
        episodes=evaluation['episodes'];ids=[ep['episode_id'] for ep in episodes]
        expected_ids=sorted(population[task]);by_id={ep['episode_id']:ep for ep in episodes}
        if (len(ids)!=len(set(ids)) or set(ids)!=set(expected_ids)
            or evaluation['eligible_trajectories']!=len(expected_ids)
            or evaluation['total_windows']!=sum(population[task].values())):
            raise ValueError('Incomplete or substituted internal-development population')
        persistence=[]
        for i,episode_id in enumerate(expected_ids):
            ep=by_id[episode_id]
            if ep['windows']!=population[task][episode_id]:
                raise ValueError('Per-trajectory development window count changed')
            for field in ('standardized_mse_by_offset','persistence_standardized_mse_by_offset'):
                value=np.asarray(ep[field],dtype=np.float64)
                if value.shape!=(59,) or not np.isfinite(value).all() or (value<0).any():
                    raise ValueError('Invalid complete standardized-error vector')
                if field=='standardized_mse_by_offset':arrays[task][mode][seed,i]=value
                else:persistence.append(value)
        persistence=np.asarray(persistence)
        if task in persistence_by_task and not np.array_equal(persistence,persistence_by_task[task]):
            raise ValueError('Persistence changed across matched arms or seeds')
        persistence_by_task[task]=persistence;arrays[task]['persistence'][seed]=persistence
        receipt_sources.append({'task':task,'mode':mode,'seed':seed,'evaluation_path':run['evaluation_path'],
                                'evaluation_sha256':run['evaluation_sha256'],'selected_checkpoint_sha256':checkpoint})
    endpoints={task:{mode:values[:,:,58] for mode,values in methods.items()} for task,methods in arrays.items()}
    bootstrap=crossed_intervals(endpoints)
    summaries={};macro=[]
    for task,methods in arrays.items():
        means={mode:values.mean(axis=(0,1)).tolist() for mode,values in methods.items()}
        ours=means['bounded_spatial_mix'][58];comparisons={}
        for mode in ('anchored_additive','autoregressive','persistence'):
            baseline=means[mode][58];gain=None if baseline==0 else 100*(baseline-ours)/baseline
            comparisons[mode]={'method_minus_comparator':ours-baseline,'relative_mse_reduction_percent':gain,
                               'paired95':bootstrap['task_intervals'][task][mode]}
            if mode=='anchored_additive':macro.append(gain)
        summaries[task]={'eligible_trajectories':len(population[task]),'total_windows':sum(population[task].values()),
                         'means_all59_offsets':means,'h60_comparisons':comparisons}
    return {'status':'complete_validated_development','scope':'internal_development','finalization_sha256':SHA(FINAL),
            'registration_sha256':registration_sha,'runs':receipt_sources,'numerical_results':summaries,
            'macro_primary_relative_gain_percent':None if any(g is None for g in macro) else float(np.mean(macro)),
            'bootstrap':bootstrap,'official_validation':'not read or ingested by this reporter'}


def render():
    sources = {}
    base = checked_json(ROOT/'reports/real_video_spatial/finalization.json', sources)
    component = checked_json(ROOT/'reports/real_video_spatial_components/finalization.json', sources)
    if base['status'] != 'passed' or component['status'] != 'passed':
        raise ValueError('DROID completion finalizers have not passed')
    if base['completed_models'] + component['completed_new_models'] != 21:
        raise ValueError('DROID model-count discrepancy')
    caches = [cache_metadata(task,sources) for task in TASKS]
    if sum(c['episodes'] for c in caches) != 1804 or sum(c['native_frames'] for c in caches) != 360473:
        raise ValueError('The completed IWS data inventory changed')
    development = completed_development(sources,caches)
    rows = [r'DROID & Three images; two past and ten query action blocks; future $4\!\times\!4$ features. & Native $h=10$ standardized MSE; matched AR and additive-anchor controls. & 21 models complete; development only. \\']
    for cache in caches:
        title,width=TASKS[cache['task']]
        result_status=(r'results \textit{pending}' if development['status']=='pending' else 'matched development complete')
        rows.append(f'IWS {title} & One image; 60 native {width}D command rows; 59 future feature grids. & '
                    r'$H=60$ standardized MSE; additive anchor is the primary comparator. & '
                    f"{cache['episodes']} cached; "+result_status+r'. \\')
    scope_caption=('IWS rows describe the intended single-observation study and contain no predictor measurements.'
                   if development['status']=='pending' else 'IWS development results use the completed, verified single-observation protocol; official evaluation remains separate.')
    table=(r'\begin{table}[!htb]'+'\n'+r'\centering'+'\n'
           r'\caption{Study interfaces and evidence status. DROID results use development validation; '+scope_caption+' Historical context-model protocols remain separate.}'+'\n'
           r'\label{tab:experiment-alignment}'+'\n'+r'\begingroup\setlength{\tabcolsep}{3pt}'+'\n'
           r'\begin{tabular}{@{}p{0.14\linewidth}p{0.30\linewidth}p{0.28\linewidth}p{0.22\linewidth}@{}}'+'\n'
           r'\toprule Task & Inputs and outputs & Score and comparison & Status \\ \midrule'+'\n'
           +'\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}\endgroup'+'\n'+r'\end{table}'+'\n')
    status={'schema':'shiftwm_experiment_alignment_status_v1','spatial_models':21,
            'spatial_contrasts':{'method_comparisons':32,'interactions':4},'iws_caches':caches,
            'iws_development':development,'numerical_ingestion_enabled':True,
            'reason':'Only complete27-run, source-bound evaluation receipts accepted; absent/incomplete results never yield numbers.',
            'sources_sha256':sources,'reporter_sha256':SHA(__file__)}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    (OUT/'status.tex').write_text(table+development_table(development))
    if development['status']=='pending':
        transfer=('We have prepared complete feature caches for IWS PushT, Box and Rope: '
                  '1,804 trajectories and 360,473 native frames. A matched single-observation '
                  'implementation is in progress; no predictor results from these tasks are reported yet. '
                  'It tests transfer of the fixed-source decoder with native command interfaces, without '
                  'inventing an observed transition for context inference. ')
        ingestion='The current status table contains no IWS predictor measurements.\n'
    else:
        transfer=('The IWS single-observation development campaign has completed all 27 task, method and seed runs. '
                  'Source-bound evaluation receipts support the separately reported transfer results. '
                  'This changes the input interface; it does not establish the historical context model or an upstream SOTA reproduction. ')
        ingestion='IWS development numbers are generated only from all 27 completed, source-bound evaluation receipts; official evaluation is not ingested.\n'
    transfer+=r'The task matrix and completion gates are in Appendix~\ref{app:experiment-alignment}. Official validation remains separate and requires a locked protocol and checkpoint identities.'+'\n'
    (OUT/'transfer_status.tex').write_text(transfer)
    (OUT/'ingestion_status.tex').write_text(ingestion)
    print(json.dumps({'spatial_models':21,'iws_recordings':1804,'iws_frames':360473,
                      'iws_results':development['status'],'numerical_ingestion_enabled':True}))


def development_table(development):
    if development['status']=='pending':
        return ''
    def number(value):
        return 'undefined' if value is None else f'{value:.3f}'
    rows=[]
    names={'autoregressive':'Autoregressive','anchored_additive':'Additive anchor',
           'bounded_spatial_mix':'ShiftWM (ours)','persistence':'Persistence'}
    for task,result in development['numerical_results'].items():
        for mode in ('persistence','autoregressive','anchored_additive','bounded_spatial_mix'):
            mse=result['means_all59_offsets'][mode][58]
            if mode=='bounded_spatial_mix':
                effect=result['h60_comparisons']['anchored_additive'];gain=effect['relative_mse_reduction_percent']
                change='undefined' if gain is None else (r'\positivegain{'+number(gain)+r'\%}' if gain>0 else number(gain)+r'\%')
                ci=effect['paired95']['difference'];interval=f'[{ci[0]:+.5f}, {ci[1]:+.5f}]'
            else:change='---';interval='---'
            rows.append(f"{TASKS[task][0]} & {names[mode]} & {mse:.6f} & {change} & {interval} "+r'\\')
    caption=(r'\caption{Complete IWS internal-development endpoints at $H=60$. MSE averages windows within trajectory, then trajectories and matched seeds equally. Gains and paired MSE-difference intervals compare ShiftWM with the additive anchor. Positive gains are bold green; intervals are unadjusted and may include zero. All three tasks, three learned arms, three seeds and persistence are retained. Official evaluation is not included.}')
    macro=number(development['macro_primary_relative_gain_percent']);interval=development['bootstrap']['macro_gain_interval']
    macro_interval='undefined' if interval is None else f'[{interval[0]:+.3f}, {interval[1]:+.3f}]'
    return ('\n'+r'\begin{table}[!htb]\centering'+'\n'+caption+'\n'+r'\label{tab:iws-development-complete}'+'\n'
            r'\begin{tabular}{@{}llrrl@{}}\toprule Task & Method & MSE & Gain vs anchor & Paired 95\% CI \\ \midrule'+'\n'
            +'\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}\end{table}'+'\n'
            +'Equal-task macro relative reduction against the additive anchor: '+macro+r'\%, paired 95\% interval '+macro_interval+'.\n')


if __name__ == '__main__':
    render()
