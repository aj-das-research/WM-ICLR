#!/usr/bin/env python3
"""Independently audit the complete h10 control and publish source-backed tables.

This postprocessor never mutates registered training/evaluation sources. It
requires the complete registered finalizer report before writing any TeX include.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
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
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts/real_video_development')]
import horizon10_train as trainer
from shiftwm.real_video.data import RealVideoDataset

REGISTRY = ROOT/'configs/real_video_development/horizon10_v1/registration.json'
RESULTS = ROOT/'reports/real_droid_horizon10_results.json'
PROTOCOL = ROOT/'reports/horizon10_paper_tables_protocol.md'
GENERATED = ROOT/'paper/generated/real_video/horizon10'
SECTION = ROOT/'paper/sections/horizon10_development.tex'
REPORT = ROOT/'reports/real_droid_horizon10_paper_finalization.json'
MODES = ('framewise','constant_dynamics','factorized','action_free')
NAMES = {'framewise':'Framewise','constant_dynamics':'Constant dynamics',
         'factorized':r'\textbf{ShiftWM (ours)}','action_free':'Action-free'}
EVALUATIONS = {'validation_h5':(5,'h10_trained_standard_h5'),
               'validation_h10':(10,'h10_trained_standard_h10'),
               'matched_new_h5':(5,'h10_trained_matched_h5_prefix'),
               'matched_original_h5':(5,'h5_trained_matched_h5'),
               'matched_original_h10':(10,'h5_trained_matched_h10')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def close(first,second,label):
    if not math.isclose(first,second,rel_tol=1e-11,abs_tol=1e-13):
        raise ValueError(f'Independent numeric check failed: {label}: {first} != {second}')


def bootstrap(matrix,sessions):
    """Independent implementation of registered crossed seed/session resampling."""
    if matrix.shape != (3,len(sessions)) or not np.isfinite(matrix).all():
        raise ValueError('Invalid three-seed paired matrix')
    groups=[np.flatnonzero(np.asarray(sessions)==session) for session in sorted(set(sessions))]
    rng=np.random.default_rng(5198010)
    draws=np.empty(10000)
    for index in range(10000):
        seeds=rng.integers(0,3,size=3)
        sampled=rng.integers(0,len(groups),size=len(groups))
        episodes=np.concatenate([groups[g] for g in sampled])
        draws[index]=matrix[np.ix_(seeds,episodes)].mean()
    return {'mean_difference':float(matrix.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),
            'draws':10000,'bootstrap_seed':5198010,'training_seed_count':3,
            'session_count':len(groups),'episode_count':len(sessions)}


def check_evaluation(doc,row,key,checkpoint_hash,population,registry_hash):
    horizon,kind=EVALUATIONS[key]
    if (doc.get('scope')!='original validation development only' or doc.get('mode')!=row['mode']
            or doc.get('seed')!=row['seed'] or doc.get('horizon')!=horizon or doc.get('kind')!=kind
            or doc.get('registration_sha256')!=registry_hash or doc.get('checkpoint_sha256')!=checkpoint_hash):
        raise ValueError('Evaluation provenance mismatch: '+row['name']+'/'+key)
    result=doc['result'];episodes=sorted(result['episodes'],key=lambda e:e['episode_id'])
    if [e['episode_id'] for e in episodes]!=sorted(population):
        raise ValueError('Missing/duplicate or non-validation episodes')
    methods={'model','persistence','constant_velocity','reversed_future_actions'}
    metrics={'mean_standardized_mse','mean_raw_mse','mean_cosine_error'}
    for h in ((1,3,5) if horizon==5 else (10,)):
        metrics.update(f'h{h}_{metric}' for metric in ('standardized_mse','raw_mse','cosine_error'))
    for episode in episodes:
        reference=population[episode['episode_id']]
        if (episode['session_id']!=reference['session_id'] or episode['window_starts']!=reference['starts']
                or episode['windows']!=len(reference['starts']) or set(episode['errors'])!=methods):
            raise ValueError('Session or exact prediction-window population mismatch')
        for values in episode['errors'].values():
            if set(values)!=metrics or any(not math.isfinite(v) or v<0 for v in values.values()):
                raise ValueError('Invalid evaluation metric')
    if (result['episode_count']!=len(episodes) or result['session_count']!=len({e['session_id'] for e in episodes})
            or result['window_count']!=sum(e['windows'] for e in episodes) or set(result['summary'])!=methods):
        raise ValueError('Evaluation population count mismatch')
    for method in methods:
        if set(result['summary'][method])!=metrics:
            raise ValueError('Evaluation summary metric mismatch')
        for metric in metrics:
            close(math.fsum(e['errors'][method][metric] for e in episodes)/len(episodes),
                  result['summary'][method][metric],method+'/'+metric)
    return episodes


def collect():
    registry,official=read(REGISTRY),read(RESULTS)
    registry_hash=sha(REGISTRY)
    if (official.get('status')!='completed' or official.get('completed_runs')!=12 or official.get('missing')
            or official.get('completed_standard_evaluations')!=24
            or official.get('completed_matched_diagnostic_evaluations')!=36
            or official.get('registration_sha256')!=registry_hash):
        raise ValueError('Awaiting all 12 full runs and 60 evaluations from the registered finalizer')
    expected={(m,s) for m in MODES for s in (0,1,2)}
    for rows in (registry['runs'],official['runs']):
        if len(rows)!=12 or {(r['mode'],r['seed']) for r in rows}!=expected or len({r['name'] for r in rows})!=12:
            raise ValueError('Complete matched mode/seed registry required')
    sources={**registry['dependencies'],**registry['original_h5_evidence'],
             str(REGISTRY.relative_to(ROOT)):registry_hash,str(RESULTS.relative_to(ROOT)):sha(RESULTS),
             str(PROTOCOL.relative_to(ROOT)):sha(PROTOCOL),str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    for relative in ('scripts/real_video_development/finalize_horizon10_paper.slurm','tests/test_horizon10_paper_finalization.py'):
        sources[relative]=sha(ROOT/relative)
    for name in ('horizon10_schedule_amendment.json','horizon10_schedule_amendment_seed2.json'):
        amendment=ROOT/'reports/evidence'/name
        if amendment.exists():sources[str(amendment.relative_to(ROOT))]=sha(amendment)
    for relative,expected_hash in sources.items():
        if sha(ROOT/relative)!=expected_hash:raise ValueError('Registered source changed: '+relative)
    manifest=read(ROOT/'data/features/droid_selected_v1/manifest.json')
    populations={h:{} for h in (5,10)}
    for episode in manifest['episodes']:
        if episode['split'] in ('train','val'):
            item=episode['cameras']['exterior_image_1_left']
            sources['data/features/droid_selected_v1/'+item['file']]=item['sha256']
        if episode['split']!='val':continue
        count=episode['cameras']['exterior_image_1_left']['frames']
        for h in (5,10):
            starts=list(range(0,count-(3+h)+1,5))
            if starts:populations[h][episode['episode_id']]={'session_id':episode['session_id'],'starts':starts}
    # This loader opens original validation payloads only and verifies their hashes.
    data=RealVideoDataset(ROOT/'data/features/droid_selected_v1','val',horizon=10,stride=5,verify=True)
    sample=data[0];support=sample['features'][None,:3].float();actions=sample['actions'][None].float()
    records=[]
    for row in registry['runs']:
        path=ROOT/row['config']
        if sha(path)!=row['config_sha256']:raise ValueError('Registered configuration changed')
        sources[row['config']]=sha(path);config=read(path)
        directory=(ROOT/config['output_dir']).resolve()
        directory.relative_to((ROOT/'runs/real_video_development/horizon10_v1').resolve())
        if (config['mode'],config['seed'],config['epochs'],config['train_horizon'],config['validation_horizon'])!=(row['mode'],row['seed'],30,10,10):
            raise ValueError('Incorrect registered training identity')
        if trainer.scientific_config(read(directory/'training_config.json'))!=trainer.scientific_config(config):
            raise ValueError('Run differs from registered scientific configuration')
        summary=trainer.validate_completed(directory);receipt=read(directory/'development_receipt.json')
        if (receipt.get('status')!='completed' or receipt.get('training')!=summary
                or receipt.get('offline_cpu_reload_max_error')!=0.0 or receipt.get('registration_sha256')!=registry_hash
                or not math.isfinite(receipt.get('elapsed_seconds',-1)) or receipt.get('elapsed_seconds',-1)<=0
                or set(receipt['evaluations'])!=set(EVALUATIONS)):
            raise ValueError('Incomplete/stale training and evaluation receipt')
        checkpoint=directory/'best';model,state=trainer.load_package(checkpoint,'cpu')
        second,_=trainer.load_package(checkpoint,'cpu')
        with torch.inference_mode():
            first=model.predict(support,actions[:,:2],actions[:,2:])
            reloaded=second.predict(support,actions[:,:2],actions[:,2:])
        if first.shape!=(1,10,1536) or not torch.isfinite(first).all() or not torch.equal(first,reloaded):
            raise ValueError('Independent CPU offline reload parity failed')
        if model.config.mode!=row['mode'] or state['epoch']!=summary['best_epoch']:
            raise ValueError('Loaded model identity/selected epoch mismatch')
        original_checkpoint=ROOT/row['original_h5_checkpoint']
        current={'checkpoint_sha256':sha(checkpoint/'model.pt'),'original_checkpoint_sha256':sha(original_checkpoint/'model.pt')}
        record={**row,**current,'training':summary,'elapsed_seconds':receipt['elapsed_seconds'],
                'checkpoint_bytes':(checkpoint/'model.pt').stat().st_size,
                'cpu_prediction_sha256':hashlib.sha256(first.numpy().tobytes()).hexdigest(),'evaluations':{}}
        registered_result=next(r for r in official['runs'] if r['name']==row['name'])
        for key in EVALUATIONS:
            relative=receipt['evaluations'][key];eval_path=ROOT/relative
            if eval_path.resolve()!=(directory/(key+'.json')).resolve():raise ValueError('Unexpected evaluation path')
            expected_hash=current['original_checkpoint_sha256' if key.startswith('matched_original') else 'checkpoint_sha256']
            population=populations[5] if key=='validation_h5' else populations[10]
            document=read(eval_path)
            episodes=check_evaluation(document,row,key,expected_hash,population,registry_hash)
            saved=registered_result['evaluations'][key]
            if saved['source']!=relative or saved['source_sha256']!=sha(eval_path):raise ValueError('Registered report source hash differs')
            for method,metrics in document['result']['summary'].items():
                for metric,value in metrics.items():close(value,saved['result']['summary'][method][metric],'registered report mean')
            record['evaluations'][key]={'episodes':episodes,'summary':document['result']['summary']}
            sources[relative]=sha(eval_path)
        for name in ('training_config.json','training_summary.json','development_receipt.json','metrics.jsonl',
                     'best/model.pt','best/config.json','best/package_manifest.json',
                     'last/model.pt','last/config.json','last/package_manifest.json'):
            sources[str((directory/name).relative_to(ROOT))]=sha(directory/name)
        records.append(record);del model,second,state
    for relative,expected_hash in sources.items():
        if sha(ROOT/relative)!=expected_hash:raise ValueError('Source/data changed during audit: '+relative)
    return records,official,sources,populations


def matrix(records,key,metric,mode):
    rows=sorted((r for r in records if r['mode']==mode),key=lambda r:r['seed'])
    if [r['seed'] for r in rows]!=[0,1,2]:raise ValueError('Three seeds required')
    population=None;values=[]
    for row in rows:
        episodes=row['evaluations'][key]['episodes']
        current=[(e['episode_id'],e['session_id'],e['window_starts']) for e in episodes]
        if population is not None and current!=population:raise ValueError('Paired seed population differs')
        population=current;values.append([e['errors']['model'][metric] for e in episodes])
    return np.asarray(values,dtype=np.float64),population


def compare(records,first_key,second_key,metric,first_mode,second_mode):
    first,population=matrix(records,first_key,metric,first_mode)
    second,other=matrix(records,second_key,metric,second_mode)
    if population!=other:raise ValueError('Comparison prediction starts differ')
    if second.mean()<=0:raise ValueError('Relative gain denominator must be positive')
    return {'first_mean':float(first.mean()),'second_mean':float(second.mean()),
            'first_seed_sd':float(first.mean(1).std(ddof=1)),'second_seed_sd':float(second.mean(1).std(ddof=1)),
            'gain_percent':float(100*(second.mean()-first.mean())/second.mean()),
            'paired':bootstrap(first-second,[p[1] for p in population]),
            'episodes':len(population),'windows':sum(len(p[2]) for p in population)}


def aggregate(records,official):
    native,comparisons=[],[]
    for h in (5,10):
        for metric in (f'h{h}_standardized_mse','mean_standardized_mse'):
            methods={}
            for mode in MODES:
                values,population=matrix(records,f'validation_h{h}',metric,mode)
                methods[mode]={'mean':float(values.mean()),'seed_sd':float(values.mean(1).std(ddof=1)),
                               'per_seed':values.mean(1).tolist()}
            baseline=methods['framewise']['mean']
            for value in methods.values():value['gain_percent']=100*(baseline-value['mean'])/baseline
            native.append({'horizon':h,'metric':metric,'methods':methods})
            comparisons.append({'type':'matched_h10_training_modes','horizon':h,'metric':metric,
                                **compare(records,f'validation_h{h}',f'validation_h{h}',metric,'factorized','framewise')})
    for mode in MODES:
        for h in (5,10):
            for metric in (f'h{h}_standardized_mse','mean_standardized_mse'):
                comparisons.append({'type':'horizon_training_control','mode':mode,'horizon':h,'metric':metric,
                                    **compare(records,'matched_new_h5' if h==5 else 'validation_h10',
                                              f'matched_original_h{h}',metric,mode,mode)})
    if len(official['comparisons'])!=20:raise ValueError('All 20 registered comparisons must be present')
    for row in comparisons:
        matches=[x for x in official['comparisons'] if (x['type'],x.get('mode'),x['horizon'],x['metric'])==(row['type'],row.get('mode'),row['horizon'],row['metric'])]
        if len(matches)!=1:raise ValueError('Missing/duplicate registered comparison')
        expected=matches[0]
        for key in ('first_mean','second_mean'):close(row[key],expected[key],key)
        close(row['gain_percent'],expected['first_reduction_vs_second_percent'],'relative gain')
        pair=expected['paired_first_minus_second']
        for key in ('mean_difference','ci95'):
            if not np.allclose(row['paired'][key],pair[key],rtol=1e-11,atol=1e-13):raise ValueError('Independent bootstrap mismatch')
        for key in ('draws','bootstrap_seed','training_seed_count','session_count','episode_count'):
            if row['paired'][key]!=pair[key]:raise ValueError('Bootstrap sampling metadata differs')
        if row['type']=='matched_h10_training_modes':
            means=next(n for n in native if n['horizon']==row['horizon'] and n['metric']==row['metric'])['methods']
            for mode in MODES:close(means[mode]['mean'],expected['means'][mode],'native method mean')
    return native,comparisons


def gain(value,ci=None):
    text=f'{value:+.2f}'
    if value>0:text=r'\positivegain{'+text+'}'
    if ci is not None and ci[0]<=0<=ci[1]:text+=r'$^{\dagger}$'
    return text


def interval(row):
    lower,upper=row['paired']['ci95']
    return f'[{lower:+.5f}, {upper:+.5f}]'


def table(header,columns):
    return [r'\begin{table}[p]\centering\footnotesize',
            r'\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.13}',
            r'\begin{tabular}{'+columns+r'}\toprule',header+r'\\\midrule']


def end(lines,caption,label):
    return '\n'.join([*lines,r'\bottomrule\end{tabular}',r'\caption{'+caption+r'}',r'\label{'+label+r'}\end{table}'])+'\n'


def tables(records,native,comparisons):
    files={}
    for measure,tag in (('endpoint','endpoint'),('mean','mean')):
        lookup={h:next(n for n in native if n['horizon']==h and n['metric']==(f'h{h}_standardized_mse' if measure=='endpoint' else 'mean_standardized_mse')) for h in (5,10)}
        title='Endpoint MSE' if measure=='endpoint' else 'Mean query MSE'
        lines=table('Method & '+title+r'@5 & '+title+r'@10 & Gain@5 (\%) & Gain@10 (\%)','lrrrr')
        for mode in MODES:
            a,b=lookup[5]['methods'][mode],lookup[10]['methods'][mode]
            if mode=='factorized':lines.append(r'\rowcolor{orange!9}')
            values=' & '.join(f"{x['mean']:.5f}" for x in (a,b))
            gains='reference & reference' if mode=='framewise' else gain(a['gain_percent'])+' & '+gain(b['gain_percent'])
            lines.append(NAMES[mode]+' & '+values+' & '+gains+r'\\')
        caption=(r'\textbf{Matched h10-trained methods: '+title.lower()+r'.} Every method completed 30 epochs with width 192, depth 4 and three support frames, at seeds 0/1/2. '
                 r'Values are means across seeds and equally weighted validation episodes; all four methods use identical populations within each horizon. '
                 r'The standard h5 and h10 populations may differ because each requires a complete query window. '
                 r'Gains use Framewise as the reference within the same horizon. \positivegain{Bold green} marks positive point reductions, not significance. '
                 r'All methods and regressions are retained; these are development results, not a new test evaluation.')
        files[f'native_{tag}.tex']=end(lines,caption,'tab:h10-native-'+tag)
    lines=table(r'Scoring & Ours & Framewise & Gain (\%) & Paired 95\% interval','lrrrl')
    for row in (r for r in comparisons if r['type']=='matched_h10_training_modes'):
        scoring=('Mean 1--' if row['metric']=='mean_standardized_mse' else 'Endpoint ')+str(row['horizon'])
        lines.append(f"{scoring} & {row['first_mean']:.5f} & {row['second_mean']:.5f} & {gain(row['gain_percent'],row['paired']['ci95'])} & {interval(row)}"+r'\\')
    files['ours_intervals.tex']=end(lines,r'\textbf{ShiftWM (ours) versus equally h10-trained Framewise.} Intervals are for ours minus Framewise in standardized feature MSE, so negative favors ours. '
        r'The primary control metric averages all ten query predictions. Endpoint metrics and mean-five errors are secondary. '
        r'Intervals jointly resample the three training seeds and recording-session clusters (10,000 draws); they are exploratory and unadjusted for multiple comparisons. '
        r'$\dagger$ marks intervals including zero. Green bold point gains alone do not establish a difference.','tab:h10-ours-intervals')
    for measure in ('mean','endpoint'):
        lines=table(r'Method & $h$ & Original h5 & Trained h10 & Gain (\%) & Paired 95\% interval','lrrrrl')
        for mode in MODES:
            for h in (5,10):
                row=next(r for r in comparisons if r['type']=='horizon_training_control' and r['mode']==mode and r['horizon']==h and r['metric']==('mean_standardized_mse' if measure=='mean' else f'h{h}_standardized_mse'))
                if mode=='factorized':lines.append(r'\rowcolor{orange!9}')
                lines.append(f"{NAMES[mode]} & {h} & {row['second_mean']:.5f} & {row['first_mean']:.5f} & {gain(row['gain_percent'],row['paired']['ci95'])} & {interval(row)}"+r'\\')
            if mode!=MODES[-1]:lines.append(r'\addlinespace[2pt]')
        detail='average all query steps' if measure=='mean' else 'score the final query step'
        caption=(r'\textbf{Matched training-horizon control: '+measure+r' errors.} Original h5-trained and new h10-trained checkpoints use the exact same h10-eligible episode/session identities and query starts. '
                 r'The h5 rows use the five-step prefix of those windows; h10 uses all ten. These rows '+detail+r'. '
                 r'Gains are $100(\mathrm{original}-\mathrm{h10})/\mathrm{original}$; paired intervals are h10 minus original. '
                 r'Both selection objectives retain window weighting but use their respective training horizons. '
                 r'$\dagger$ marks intervals including zero; green bold identifies only positive point gains. '
                 r'All four methods are shown, with no independent-test claim.')
        files[f'matched_{measure}.tex']=end(lines,caption,'tab:h10-matched-'+measure)
    lines=table(r'Method & Best epochs & Total (M) & Trained (M) & Seconds/run','llrrl')
    for mode in MODES:
        subset=sorted((r for r in records if r['mode']==mode),key=lambda r:r['seed'])
        counts=[r['training']['parameter_counts'] for r in subset]
        if any(c!=counts[0] for c in counts):raise ValueError('Parameter counts differ across seeds')
        epochs='/'.join(str(r['training']['best_epoch']) for r in subset)
        seconds=[r['elapsed_seconds'] for r in subset]
        if mode=='factorized':lines.append(r'\rowcolor{orange!9}')
        lines.append(f"{NAMES[mode]} & {epochs} & {counts[0]['total']/1e6:.3f} & {counts[0]['trainable']/1e6:.3f} & {np.median(seconds):.0f} [{min(seconds):.0f}, {max(seconds):.0f}]"+r'\\')
    files['checkpoints.tex']=end(lines,r'\textbf{Horizon-ten checkpoints and computation.} Best epochs follow seeds 0/1/2, selected by window-weighted FP32 validation MSE over all ten recursive predictions after complete 30-epoch training. '
        r'Total parameters include inactive frozen modules; trained parameters exclude them. Seconds/run is the median [minimum, maximum] successful runner invocation, including training, five evaluation passes and CPU reload checks, excluding prior interrupted work. '
        r'Hardware partitions differ, so invocation times are operational records, not a matched speed comparison. All 12 selected packages pass registered and independently repeated weights-only CPU reload checks.','tab:h10-checkpoints')
    return files


def proof(stage,files):
    for filename,contents in files.items():(stage/filename).write_text(contents)
    preamble=r'''\documentclass{article}
\usepackage{iclr2027_conference,times}
\usepackage{booktabs,amsmath,xcolor,colortbl,hyperref}
\definecolor{gainpositive}{HTML}{166534}
\newcommand{\positivegain}[1]{\textcolor{gainpositive}{\textbf{#1}}}
\begin{document}
'''
    (stage/'proof.tex').write_text(preamble+'\n'.join(r'\input{'+name+'}' for name in files)+r'\end{document}'+'\n')
    environment=dict(os.environ,TEXINPUTS=str(ROOT/'paper/template/official/iclr2027')+'//:'+os.environ.get('TEXINPUTS',''))
    for iteration in range(2):
        result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','proof.tex'],cwd=stage,env=environment,capture_output=True,text=True)
        (stage/f'pass{iteration+1}.stdout').write_text(result.stdout+result.stderr)
        if result.returncode:raise ValueError('ICLR table proof failed: '+result.stdout[-1800:])
    log=(stage/'proof.log').read_text()
    if any(word in log for word in ('Overfull','undefined','multiply defined')):
        raise ValueError('ICLR-width proof contains overflow or unresolved references')
    return {'status':'passed','compile_passes':2,'style':'official ICLR 2027','overflow_or_reference_warnings':0,
            'proof_sha256':sha(stage/'proof.pdf'),'visual_review':'pending; automated TeX gates do not certify visual quality'}


def finalize(build_paper=False):
    torch.set_num_threads(2)
    records,official,sources,populations=collect()
    native,comparisons=aggregate(records,official)
    files=tables(records,native,comparisons)
    section=r'''\section{Matched ten-step training on recorded video}
\label{app:horizon10-development}
This registered control tests whether training at the evaluation horizon improves
long-horizon prediction. It reuses the unchanged width-192, depth-4 LeWM-based
predictor with Framewise, Constant dynamics, ShiftWM (ours), and Action-free
variants. All twelve runs train for 30 epochs at seeds 0/1/2, using three
observed support frames and ten recursively predicted query targets. The
optimizer, training-only normalization, action blocks, camera, training stride,
and validation stride match the original five-step campaign.

No query image conditions forecasts or inferred contexts. Model selection uses
FP32 standardized MSE averaged over all ten query steps and windows, preserving
the original window weighting. An equal-episode metric is logged only as an
auxiliary diagnostic. All evaluation tables equally weight episodes and then
training seeds, as in the original evaluation. Training and checkpoint-selection
horizons change together; this ordinary horizon-mismatch control is not a new
architecture or evidence of additional method novelty.

Tables~\ref{tab:h10-native-endpoint} and~\ref{tab:h10-native-mean} compare all
h10-trained methods on each standard validation population.
Table~\ref{tab:h10-ours-intervals} qualifies ours--Framewise differences using
paired seed/session intervals. Tables~\ref{tab:h10-matched-mean}
and~\ref{tab:h10-matched-endpoint} compare original and new checkpoints on
identical h10-eligible query starts, including their five-step prefixes. The
primary diagnostic is mean error across all ten query predictions; endpoint
and five-step effects are secondary. Positive point gains, regressions and
intervals including zero are all retained. These exploratory validation results
require independent evaluation before any new confirmatory claim. Neither
original nor fresh test payloads enter this study.

Table~\ref{tab:h10-checkpoints} documents the selected reusable checkpoints and
measured invocation times. The accompanying evidence ledger records every
configuration, checkpoint, metric source and independent numerical check.
'''
    section+='\n'.join(r'\input{generated/real_video/horizon10/'+name+'}' for name in files)+'\n'
    with tempfile.TemporaryDirectory(prefix='shiftwm-horizon10-proof-') as name:
        stage=Path(name);review=proof(stage,files)
        for relative,expected in sources.items():
            if sha(ROOT/relative)!=expected:raise ValueError('Source changed before table publication')
        GENERATED.mkdir(parents=True,exist_ok=True)
        for filename,contents in files.items():
            temporary=GENERATED/(filename+'.tmp');temporary.write_text(contents);temporary.replace(GENERATED/filename)
        for filename in ('proof.pdf','proof.log','proof.tex','pass1.stdout','pass2.stdout'):
            shutil.copyfile(stage/filename,GENERATED/filename)
        trainer.atomic_json(review,GENERATED/'review.json')
        trainer.atomic_json({'sources':sources,'registration_sha256':sha(REGISTRY),'native':native,'comparisons':comparisons,
                             'units':'fixed train-standardized feature MSE','scope':'original validation only'},GENERATED/'evidence.json')
        # No manuscript include exists until every numeric, provenance and TeX gate passes.
        temporary=SECTION.with_suffix('.tex.tmp');temporary.write_text(section);temporary.replace(SECTION)
    report={'status':'completed','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'completed_training_runs':12,'completed_validation_evaluations':60,'registered_checkpoint_parity_passes':12,
            'independent_cpu_reload_parity_passes':12,'independent_paired_comparisons':20,
            'registration_sha256':sha(REGISTRY),'sources':sources,'review':review,'native':native,'comparisons':comparisons,
            'runs':[{k:v for k,v in r.items() if k!='evaluations'} for r in records],
            'manuscript_files':{str(p.relative_to(ROOT)):sha(p) for p in [SECTION,*sorted(GENERATED.glob('*.tex'))]}}
    trainer.atomic_json(report,REPORT)
    if build_paper:
        subprocess.run(['bash','paper/build.sh'],cwd=ROOT,check=True)
        report['paper_build']={'status':'passed','pdf_sha256':sha(ROOT/'paper/world_model_draft.pdf')}
        trainer.atomic_json(report,REPORT)
    print(json.dumps({'status':'completed','runs':12,'evaluations':60,'independent_comparisons':20,'paper_built':build_paper}))
    return report


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--build-paper',action='store_true');args=parser.parse_args()
    lock=ROOT/'artifacts/development/horizon10_paper_finalization.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);finalize(args.build_paper)


if __name__=='__main__':main()
