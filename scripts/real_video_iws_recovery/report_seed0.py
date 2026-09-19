#!/usr/bin/env python3
"""Source-bound operational status and all seed0 development scores, not study finalization."""
from pathlib import Path
import argparse,hashlib,json,subprocess,math
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
J=lambda p:json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output
 if out.exists() or out.with_suffix('.md').exists():raise ValueError('Refusing to overwrite prior operational report')
 config='configs/real_video_iws/training_v1.json';registry='configs/real_video_iws/training_registration_v1.json';sources={config:H(ROOT/config),registry:H(ROOT/registry)};records={}
 for task in ['pusht','bimanual_box','bimanual_rope']:
  records[task]={}
  for mode in ['autoregressive','anchored_additive','bounded_spatial_mix']:
   name=task+'_'+mode+'_s0';ep='reports/real_video_iws/evaluations/'+name+'.json';cp='reports/real_video_iws/training_completions/'+name+'.json';sp='runs/real_video_iws/v1/'+name+'/training_summary.json';d=J(ROOT/ep);c=J(ROOT/cp);s=J(ROOT/sp)
   assert d['status']=='passed' and d['scope']=='internal_development' and d['seed']==0 and d['mode']==mode and d['task']==task
   assert d['official_validation_payloads_read']==0 and d['completed_epochs']==30 and s['completed_epochs']==30 and s['status']=='completed'
   assert d['selected_epoch']==s['best_epoch'] and d['registration_sha256']==sources[registry] and d['config_sha256']==sources[config]
   assert c['evaluation_sha256']==H(ROOT/ep) and c['summary_sha256']==H(ROOT/sp)
   npz=ROOT/d['window_ledger_path'];assert H(npz)==d['window_ledger_sha256']
   with np.load(npz,allow_pickle=False) as a:
    expected=[(r['episode_index'],start) for r in d['population_audit']['records'] for start in range(0,r['frames']-60,5)]
    assert list(zip(a['episode_index'].tolist(),a['window_start'].tolist()))==expected
    errors=a['standardized_mse'];assert errors.shape==(len(expected),59) and np.isfinite(errors).all() and (errors>=0).all()
    means=[]
    for r in d['population_audit']['records']:
     if not r['windows']:continue
     mask=a['episode_index']==r['episode_index'];assert int(mask.sum())==r['windows'];value=float(errors[mask,-1].mean(dtype=np.float64));means.append(value)
     saved=next(x for x in d['episodes'] if x['episode_id']==r['episode_id'])['standardized_mse_by_offset'][-1];assert math.isclose(value,saved,rel_tol=1e-12,abs_tol=1e-12)
   value=float(np.mean(means));assert len(means)==d['eligible_trajectories'] and len(expected)==d['total_windows'];assert math.isclose(value,d['h60_standardized_mse'],rel_tol=1e-12,abs_tol=1e-12)
   records[task][mode]={'H60_standardized_feature_MSE':value,'selected_epoch':d['selected_epoch'],'full_training_epochs':30,'trajectories':len(means),'windows':len(expected),'prefix_maximum_absolute_difference':d['prefix_maximum_absolute_difference'],'checkpoint_sha256':d['selected_checkpoint_sha256'],'evaluation_path':ep,'primitive_H60_arithmetic_independently_recomputed':True}
   for q in [ep,cp,sp,d['window_ledger_path']]:sources[q]=H(ROOT/q)
 rows=[]
 for task,arms in records.items():
  v={mode:d['H60_standardized_feature_MSE'] for mode,d in arms.items()};ours=v['bounded_spatial_mix'];rows.append({'task':task,**v,'mixing_vs_additive_gain_percent':100*(v['anchored_additive']-ours)/v['anchored_additive'],'mixing_vs_AR_gain_percent':100*(v['autoregressive']-ours)/v['autoregressive']})
 extras=['scripts/real_video_iws_recovery/report_seed0.py','scripts/real_video_iws_recovery/evaluate_completed_cpu.py','scripts/real_video_iws_recovery/prefix_probe.py','scripts/real_video_iws/evaluate.py','src/shiftwm/real_video_iws/model.py','reports/real_video_iws/recovery/seed0_cpu_evaluation.json','reports/real_video_iws/recovery/prefix_cpu_probe_v1.json','reports/real_video_iws/recovery/submission_v1.json','logs/iws-single-200640_0.log','logs/iws-single-200640_3.log','logs/iws-single-200640_6.log','paper/sections/experiment_alignment.tex','paper/generated/experiment_alignment/status.tex','paper/generated/experiment_alignment/ingestion_status.tex']
 for q in extras:sources[q]=H(ROOT/q)
 recovery=J(ROOT/'reports/real_video_iws/recovery/seed0_cpu_evaluation.json');assert recovery['status']=='passed' and len(recovery['runs'])==3
 summaries=[J(p) for p in (ROOT/'runs/real_video_iws/v1').glob('*/training_summary.json')];evaluations=[J(p) for p in (ROOT/'reports/real_video_iws/evaluations').glob('*.json')]
 queue=subprocess.check_output(['squeue','-h','-u',__import__('getpass').getuser(),'-r','-o','%i|%j|%T|%M|%R'],text=True).strip().splitlines();jobs=[dict(zip(['job_id','name','state','elapsed','reason_or_node'],line.split('|'))) for line in queue if line]
 count={'planned_training_runs':27,'full30_training_summaries':sum(s.get('status')=='completed' and s.get('completed_epochs')==30 for s in summaries),'passed_development_evaluations':sum(e.get('status')=='passed' for e in evaluations),'running_training_jobs':sum(j['state']=='RUNNING' and j['name']=='iws-single-observation' for j in jobs),'pending_training_jobs':sum(j['state']=='PENDING' and j['name']=='iws-single-observation' for j in jobs)}
 d={'schema':'shiftwm_iws_seed0_operational_review_v1','status':'preliminary_single_seed_not_study_finalization','created_utc':datetime.now(timezone.utc).isoformat(),'counts':count,'records':records,'all_three_task_rows':rows,'source_sha256':sources,'scheduler_snapshot':jobs,'CPU_recovery_job':200834,'future_seed12_CPU_job':200835,'future_dependency':'afterany:200640:200645','scope':'Internal-development seed0 only, all three tasks and all three trained arms retained. No confidence intervals, multi-seed aggregation, reserved evaluation, SOTA or universal superiority claim. Lower MSE is better; positive gain favors mixing.','failure_and_recovery':{'GPU_failure':'After30epochs all three seed0 AR GPU evaluations failed the existing H15/30/H45 causal-prefix consistency test; original logs preserved. No OOM is evidenced.','CPU_proof':'All three exact frozen CPU evaluations completed all windows, passed identical prefix tolerances and reproduced the already-selected training score within the existing gate.','unchanged':'Registered evaluator/model/config, selected epochs, microbatch64, all59offset metrics, population, denominators and every tolerance unchanged. Only existing --device cpu option used.','limits':'Actual GPU discrepancy magnitude and earliest differing operation remain unmeasured; CPU success supports a backend-sensitive numerical explanation but does not prove CUDA correctness.'},'paper_status':'Manuscript protocol/setup and completed DROID evidence are updated. Numerical IWS table/plot ingestion remains pending until the complete27run study finalizer; these preliminary values have not been inserted into paper result tables.'}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(d,indent=2)+'\n')
 lines=['# IWS operational recovery and preliminary seed0 scores','',f"Snapshot: {d['created_utc']}",'', '**Scope: one completed seed per task; incomplete study, not a finalized comparison.** All values are internal-development H60 feature MSE; lower is better. Positive signed gain favors bounded mixing. No uncertainty claim is made.','', '| Task | AR | Additive anchor | Bounded mixing | Gain vs anchor | Gain vs AR |','|---|---:|---:|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['task']} | {r['autoregressive']:.9f} | {r['anchored_additive']:.9f} | {r['bounded_spatial_mix']:.9f} | {r['mixing_vs_additive_gain_percent']:+.3f}% | {r['mixing_vs_AR_gain_percent']:+.3f}% |")
 lines+=['',f"Current counts: {count['full30_training_summaries']}/27 trained for all30epochs; {count['passed_development_evaluations']}/27 development receipts; {count['running_training_jobs']} training jobs running and {count['pending_training_jobs']} pending.",'','All nine seed0 scores were independently reconstructed from hash-bound primitive window ledgers using equal-trajectory aggregation. Existing selected checkpoint epochs are retained. The source-bound JSON provides exact denominators, hashes, and job identities.','',d['failure_and_recovery']['GPU_failure'],d['failure_and_recovery']['CPU_proof'],d['failure_and_recovery']['unchanged'],d['failure_and_recovery']['limits'],'',d['paper_status'],'','Future CPU continuation200835 waits for both training arrays200640/200645 (afterany), preserving all frozen gates. It does not preempt training.']
 out.with_suffix('.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'report':str(out),'sha256':H(out),'counts':count,'rows':rows},indent=2))
if __name__=='__main__':main()
