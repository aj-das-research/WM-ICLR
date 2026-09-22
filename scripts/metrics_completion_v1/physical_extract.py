#!/usr/bin/env python3
"""Secondary physical-outcome summaries of completed historical simulator ledgers.

Reads JSON only; does not load models, raw datasets, or run simulators. All
conditions and methods remain visible. New continuous-error contrasts are
post-hoc descriptive analyses, not replacement primary endpoints.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODES = {'frozen':'Frozen LeWM', 'framewise':'Framewise', 'single':'Shared context',
         'factorized_unpaired':'Unpaired contexts', 'factorized':'ShiftWM (ours)',
         'plain':'Unaligned diagnostic', 'constant_dynamics':'Constant dynamics',
         'random':'Random reference', 'replay_oracle':'Recorded-command oracle'}
CORE_METRICS = {'pusht':{'block_translation_error_px':'px', 'agent_position_error_px':'px',
                         'block_angle_error_rad':'rad', 'position_error_px':'px'},
                'reacher':{'wrapped_joint_error_rad':'rad'}}

def require(ok, why):
    if not ok: raise ValueError(why)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def finite(value):
    x = float(value)
    require(math.isfinite(x), 'Nonfinite measurement')
    return x

class Sources:
    def __init__(self, root): self.root, self.hashes = Path(root).resolve(), {}
    def read(self, name, expected=None):
        path = (self.root / name).resolve()
        require(path.is_relative_to(self.root), 'Source escapes project')
        raw = path.read_bytes(); h = hashlib.sha256(raw).hexdigest()
        require(expected is None or h == expected, f'Stale source: {name}')
        self.hashes[str(path.relative_to(self.root))] = h
        return json.loads(raw)
    def unchanged(self):
        for p,h in self.hashes.items(): require(sha(self.root/p)==h, f'Source changed: {p}')

def validate_core(d, env, mode, seed, split, source):
    require(d['status']==d['planning']['status']=='complete', 'Incomplete core study')
    require((d['environment'],d['model_mode'],d['training_seed']) ==
            (env,mode,None if mode=='frozen' else seed), 'Wrong core identity')
    for k in ('checkpoint_sha256','evaluator_sha256','data_manifest_sha256'):
        require(d[k]==source[k], f'Core source identity: {k}')
    p=d['planning']; rows=p['records']; planner=p['planner']
    require(p['split']==split and p['policy']=='world_model', 'Wrong core split/policy')
    require(planner['native_budget']==50 and planner['history_steps_charged']==10 and
            planner['action_block']==5, 'Core native budget changed')
    conditions={(o,d) for o in range(3) for d in range(3)} if split=='test' else {(3,0),(0,3),(3,3)}
    keys=[(r['seed'],r['observation_id'],r['dynamics_id']) for r in rows]
    seeds={r['seed'] for r in rows}
    require(len(seeds)==64 and len(set(keys))==len(keys)==64*len(conditions), 'Core missing/duplicate rows')
    require(set(keys)=={(s,o,d) for s in seeds for o,d in conditions}, 'Core condition grid changed')
    out=[]
    for r in rows:
        require(r['goal_index']==7, 'Wrong core goal index')
        for k in ('success','final_success','success_during_context','policy_eligible'):
            require(r[k] in (0,1), 'Invalid success flag')
        require(r['policy_eligible']==1-r['success_during_context'] and
                r['success']>=r['final_success'] and r['success']>=r['success_during_context'], 'Success flags disagree')
        calls=r['native_steps']; t=r['steps_to_success_or_budget']; success=bool(r['success'])
        require(1<=calls<=50 and 1<=t<=50, 'Invalid core command accounting')
        require((success and t<=calls) or (not success and t==50 and calls==50), 'Failure time/censoring differs')
        vals={k:finite(r[k]) for k in CORE_METRICS[env] if k!='position_error_px'}
        require(all(v>=0 for v in vals.values()), 'Negative physical error')
        if env=='pusht': vals['position_error_px']=math.hypot(vals['block_translation_error_px'],vals['agent_position_error_px'])
        if env=='pusht': require(bool(r['final_success']) == (vals['position_error_px']<20 and vals['block_angle_error_rad']<np.pi/9), 'PushT final success differs from exact native criterion')
        out.append(dict(task_seed=r['seed'],observation_id=r['observation_id'],dynamics_id=r['dynamics_id'],
                        trajectory_id=r['trajectory_id'],success=success,final_success=bool(r['final_success']),
                        support_success=bool(r['success_during_context']),eligible=bool(r['policy_eligible']),
                        first_success=t if success else None,native_calls=calls,budget=50,
                        budget_penalized_commands=t,errors=vals,
                        support_identity={k:r[k] for k in r if k.startswith('initial_')},
                        failure_flags={'budget_exhausted':not success,'early_termination':False},
                        requires_manipulation=r.get('requires_manipulation')))
    return out

def valid_extension_success(m):
    return bool(m['success'] and m.get('valid_action',True) and m.get('stable_deformation',True)
                and not m.get('crash',False) and not m.get('workspace_escape',False))

def validate_extension(d, domain):
    require(d['status']=='completed' and len(d['records'])==d['tasks']==8, 'Incomplete extension study')
    out=[]; keys=[]
    for r in d['records']:
        keys.append((r['seed'],r['observation_id'],r['dynamics_id']))
        require((r['observation_id'],r['dynamics_id'])==(1,1), 'Extension development population changed')
        require(r['native_budget']==200 and r['support_budget']==10, 'Extension budget changed')
        trace=r['metrics_per_native_step']; calls=r['native_calls']
        require(len(trace)==calls and 1<=calls<=200, 'Extension trace accounting changed')
        hits=[i+1 for i,m in enumerate(trace) if valid_extension_success(m)]
        require(bool(hits)==r['success'] and (hits[0] if hits else None)==r['first_success_native_step'], 'Extension success ledger mismatch')
        require(not r['success_during_support'], 'Unexpected support-only success')
        field='goal_distance_m' if domain=='drone' else 'distance_m'
        require(math.isclose(r['final_distance_m'],trace[-1][field],rel_tol=1e-12,abs_tol=1e-14), 'Final physical error mismatch')
        vals={'goal_distance_mm':1000*finite(r['final_distance_m'])}
        # All unit-bearing final criteria, not only the favorable distance.
        if domain=='drone':
            vals.update(speed_mm_s=1000*finite(r['final_metrics']['speed_m_s']),
                        altitude_error_mm=1000*finite(r['final_metrics']['altitude_error_m']))
        flags={'budget_exhausted':not hits and calls==200,'early_termination':not hits and calls<200,
               'invalid_action':any(not m.get('valid_action',True) for m in trace),
               'unstable_deformation':any(not m.get('stable_deformation',True) for m in trace),
               'crash':any(m.get('crash',False) for m in trace),
               'workspace_escape':any(m.get('workspace_escape',False) for m in trace)}
        out.append(dict(task_seed=r['seed'],observation_id=1,dynamics_id=1,trajectory_id=r['trajectory_id'],
                        success=bool(hits),final_success=valid_extension_success(r['final_metrics']),
                        support_success=False,eligible=True,first_success=hits[0] if hits else None,
                        native_calls=calls,budget=200,budget_penalized_commands=hits[0] if hits else 200,
                        errors=vals,support_identity={'initial_distance_m':r['initial_distance_m']},
                        failure_flags=flags,stop_reason=r['stop_reason']))
    require(len(set(keys))==8 and sum(r['success'] for r in out)==d['successes'], 'Extension duplicate/count mismatch')
    return out

def moments(values):
    a=np.asarray(values,dtype=float)
    if not len(a): return None
    return dict(mean=float(a.mean()),median=float(np.median(a)),q25=float(np.quantile(a,.25)),
                q75=float(np.quantile(a,.75)),min=float(a.min()),max=float(a.max()))

def summarize(runs, condition=None):
    rows=[r for run in runs for r in run['records'] if condition is None or (r['observation_id'],r['dynamics_id'])==condition]
    require(bool(rows),'Empty summary population')
    first=runs[0]; metrics=rows[0]['errors'].keys()
    per_seed=[]
    for run in runs:
        rr=[r for r in run['records'] if condition is None or (r['observation_id'],r['dynamics_id'])==condition]
        per_seed.append(dict(seed=run['training_seed'],n=len(rr),successes=sum(r['success'] for r in rr)))
    n=len(rows); successes=sum(r['success'] for r in rows); eligible=[r for r in rows if r['eligible']]
    return dict(family=first['family'],task=first['task'],backbone=first['backbone'],split=first['split'],
                mode=first['mode'],label=MODES[first['mode']],condition=list(condition) if condition else 'all',
                seeds=[r['training_seed'] for r in runs],n=n,physical_task_seeds=len({r['task_seed'] for r in rows}),
                per_seed=per_seed,successes=successes,success_percent=100*successes/n,
                final_success_percent=100*sum(r['final_success'] for r in rows)/n,
                support_successes=sum(r['support_success'] for r in rows),eligible_n=len(eligible),
                eligible_success_percent=100*sum(r['success'] for r in eligible)/len(eligible) if eligible else None,
                errors={k:moments([r['errors'][k] for r in rows]) for k in metrics},
                native_calls=moments([r['native_calls'] for r in rows]),
                success_calls=moments([r['first_success'] for r in rows if r['success']]),
                budget_penalized_commands=moments([r['budget_penalized_commands'] for r in rows]),
                failures=n-successes,failure_counts=dict(Counter(k for r in rows for k,v in r['failure_flags'].items() if v)),
                success_by_command=[100*sum(r['success'] and r['first_success']<=t for r in rows)/n for t in range(rows[0]['budget']+1)])

def paired_effect(left, right, metric, *, crossed=False, condition=None):
    require([r['training_seed'] for r in left]==[0,1,2] and [r['training_seed'] for r in right]==[0,1,2], 'Paired effects need exact three seeds')
    key=lambda r:(r['task_seed'],r['observation_id'],r['dynamics_id'])
    arrays=[]; task_order=None
    for a,b in zip(left,right):
        aa={key(r):r for r in a['records'] if condition is None or key(r)[1:]==condition}
        bb={key(r):r for r in b['records'] if condition is None or key(r)[1:]==condition}
        require(aa.keys()==bb.keys() and aa, 'Paired populations differ')
        for k in aa:
            require(aa[k]['support_success']==bb[k]['support_success'], 'Paired support differs')
            require(aa[k]['support_identity'].keys()==bb[k]['support_identity'].keys(), 'Support fields differ')
            for f in aa[k]['support_identity']:
                require(math.isclose(aa[k]['support_identity'][f],bb[k]['support_identity'][f],rel_tol=1e-9,abs_tol=1e-8), 'Initial conditions differ')
        task_ids=sorted({k[0] for k in aa})
        require(task_order is None or task_order==task_ids,'Task seeds differ')
        task_order=task_ids
        def val(r):
            if metric=='success':return float(r['success'])*100
            if metric=='budget_penalized_commands':return float(r[metric])
            return r['errors'][metric]
        # Positive always favors ours: success ours-base; errors/budget base-ours.
        sign=1 if metric=='success' else -1
        arrays.append([np.mean([sign*(val(aa[k])-val(bb[k])) for k in aa if k[0]==s]) for s in task_ids])
    values=np.asarray(arrays); rng=np.random.default_rng(417 if crossed else 1701)
    if crossed:
        distribution=[]
        for _ in range(5000):
            si=rng.integers(3,size=3); ti=rng.integers(len(task_order),size=len(task_order))
            distribution.append(values[np.ix_(si,ti)].mean())
    else:
        ti=rng.integers(len(task_order),size=(20000,len(task_order)))
        distribution=values.mean(0)[ti].mean(1)
    return dict(mean=float(values.mean()),ci95=np.quantile(distribution,[.025,.975]).tolist(),
                per_seed=values.mean(1).tolist(),training_seed_sd=float(values.mean(1).std(ddof=1)),
                task_clusters=len(task_order),method='crossed_training_task_5000_rng417' if crossed else 'fixed_training_task_cluster_20000_rng1701',
                favorable_direction='positive',metric=metric,comparator=right[0]['mode'])

def extract(root):
    sources=Sources(root); primary=sources.read('paper/generated/primary_results.json')
    ext=sources.read('reports/completed_extension_results.json')
    require(ext['status']=='verified_complete' and len(ext['original_runs'])==36 and len(ext['geometry_runs'])==6,'Incomplete extension finalization')
    runs=[]
    for env in ('pusht','reacher'):
        for mode in ('frozen','framewise','single','factorized_unpaired','factorized','plain'):
            for seed in ([0] if mode=='frozen' else [0,1,2]):
                for split in ('test','extrapolation'):
                    path=f'results/world/{env}_{mode}_s{seed}/planning_{split}.json'
                    binding=primary['sources'][path]; d=sources.read(path,binding['sha256'])
                    rr=validate_core(d,env,mode,seed,split,binding)
                    runs.append(dict(family='core',task=env,backbone='transformer',mode=mode,training_seed=seed,
                                     split=split,source=path,checkpoint_sha256=d['checkpoint_sha256'],records=rr))
    for collection,family in [('original_runs','original'),('geometry_runs','wide_gain')]:
        for run in ext[collection]:
            require(run['training_summary']['completed_epochs']==30 and run['training_summary']['status']=='completed','Incomplete extension training')
            path=run['planning_result']; d=sources.read(path,ext['source_sha256'][path])
            require(d['identity']['arguments']['split']=='development','Unexpected extension split')
            runs.append(dict(family=family,task=run['domain'],backbone=run['architecture'],mode=run['mode'],training_seed=run['training_seed'],
                             split='development',source=path,checkpoint_sha256=run['checkpoint_sha256'],records=validate_extension(d,run['domain'])))
    for ref in ext['reference_policies']:
        path=ref['source']; d=sources.read(path,ext['source_sha256'][path])
        runs.append(dict(family='reference',task=ref['domain'],backbone='none',mode=ref['policy'],training_seed=None,
                         split='development',source=path,checkpoint_sha256=None,records=validate_extension(d,ref['domain'])))
    groups=defaultdict(list)
    for r in runs: groups[(r['family'],r['task'],r['backbone'],r['split'],r['mode'])].append(r)
    summaries=[]; contrasts=[]
    for key,rr in sorted(groups.items()):
        rr.sort(key=lambda r:-1 if r['training_seed'] is None else r['training_seed'])
        conditions=sorted({(r['observation_id'],r['dynamics_id']) for r in rr[0]['records']})
        for condition in [None]+conditions: summaries.append(summarize(rr,condition))
        if key[-1]!='factorized':continue
        comparators=('framewise','single','factorized_unpaired','plain') if key[0]=='core' else (('framewise','constant_dynamics') if key[0]=='original' else ('constant_dynamics',))
        for comparator in comparators:
            bb=groups[(*key[:-1],comparator)]
            for condition in [None]+conditions:
                for metric in ['success','budget_penalized_commands',*rr[0]['records'][0]['errors']]:
                    effect=paired_effect(rr,bb,metric,crossed=key[0]!='core',condition=condition)
                    contrasts.append(dict(family=key[0],task=key[1],backbone=key[2],split=key[3],condition=list(condition) if condition else 'all',**effect))
    sources.unchanged()
    return dict(schema='historical_physical_metrics_v1',status='complete_saved_ledgers_only',
                scope='Post-hoc physical-outcome summaries; historical context models, not the current spatial decoder. Core test/extrapolation remain separate; extensions development only.',
                conventions={'core_uncertainty':'20,000 paired initial-state-cluster draws, RNG1701; all conditions and three observed training seeds stay together. Conditional on these training seeds, not population training uncertainty.',
                             'extension_uncertainty':'5,000 crossed matched training-seed and task-seed draws, RNG417; eight tasks, exploratory unadjusted intervals.',
                             'commands':'Native calls include ten support commands. Success curves are empirical cumulative incidence. Failures never become successes; budget-penalized commands assign full budget even for early terminal failures, not an estimated time-to-success. Actual calls and success-only calls are reported separately.',
                             'endpoints':'Physical errors are measured at actual termination/budget. Core success is any-call and may differ from final-success. Reacher wrapped L2 is diagnostic, not its per-joint unwrapped success criterion.',
                             'comparators':'Fixed matched Framewise and all available within-family learned controls; no outcome-selected comparator. Frozen/random/oracle have no training-seed population intervals.',
                             'units':'px=512-canvas native PushT coordinates; radians for joint/block orientation; extension metres converted exactly to millimetres. No across-unit/backbone aggregate.',
                             'missing':'No current spatial-decoder closed-loop outcomes, robot outcomes, IoU, RGB fidelity, or new simulator trajectories are inferred. Historical dense core physical paths unavailable in these summary ledgers.'},
                source_sha256=sources.hashes,definition_source_sha256={p:sha(Path(root)/p) for p in (
                    'src/shiftwm/evaluate.py','src/shiftwm/upstream.py','src/shiftwm/extensions/evaluate.py',
                    'src/shiftwm/extensions/drone.py','src/shiftwm/extensions/surgery.py',
                    'scripts/summarize_paired_planning.py','scripts/extensions/summarize.py')},
                extractor_sha256=sha(__file__),runs=runs,summaries=summaries,contrasts=contrasts)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,default=ROOT/'reports/metrics_completion_v1/physical/data.json');a=p.parse_args()
    data=extract(a.root); a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps({'status':'complete','runs':len(data['runs']),'records':sum(len(r['records']) for r in data['runs']),'summary_rows':len(data['summaries']),'contrasts':len(data['contrasts']),'output_sha256':sha(a.output)}))
if __name__=='__main__':main()
