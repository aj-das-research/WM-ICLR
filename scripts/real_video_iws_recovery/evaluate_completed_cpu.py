#!/usr/bin/env python3
"""Run the unchanged frozen IWS evaluator on CPU for completed missing AR receipts.

Operational continuation only. No recipe, selected checkpoint, metric, batching,
normalization, source-registration or consistency tolerance is changed.
"""
from pathlib import Path
import argparse,fcntl,importlib.util,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
sp=importlib.util.spec_from_file_location('frozen_iws_campaign',ROOT/'scripts/real_video_iws/campaign.py');campaign=importlib.util.module_from_spec(sp);sp.loader.exec_module(campaign)
def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--report',type=Path,required=True);args=p.parse_args()
 if args.report.exists():raise ValueError('Refusing to replace recovery provenance')
 reg=campaign.check_registration(campaign.CONFIG);rows=[r for r in reg['runs'] if r['mode']=='autoregressive' and r['seed'] in args.seeds]
 if not rows or len(rows)!=3*len(set(args.seeds)):raise ValueError('Expected all three tasks for each requested registered seed')
 recovery=ROOT/'reports/real_video_iws/recovery';recovery.mkdir(parents=True,exist_ok=True)
 with (recovery/'evaluation.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  report={'schema':'shiftwm_iws_cpu_evaluation_recovery_v1','status':'running','started_utc':campaign.now(),'device':'cpu','slurm_job_id':os.environ.get('SLURM_JOB_ID'),'gpu_requested':False,'registration_sha256':campaign.sha(campaign.REGISTRATION),'evaluator_sha256':campaign.sha(ROOT/'scripts/real_video_iws/evaluate.py'),'launcher_sha256':campaign.sha(__file__),'seeds':args.seeds,'scope':'Exact frozen evaluator on unchanged selected completed checkpoints. Original GPU prefix-check failures retained in logs. CPU success does not establish GPU numerical parity. No reserved upstream payload access.','runs':[]}
  campaign.atomic_json(report,args.report)
  tr=campaign.trainer();failed=False
  for row in rows:
   name=row['name'];directory=ROOT/row['output'];summary_path=directory/'training_summary.json';evpath=ROOT/'reports/real_video_iws/evaluations'/f'{name}.json';record={'name':name,'started_utc':campaign.now()}
   try:
    summary=tr.validate_completed(directory)
    record.update({'completed_epochs':summary['completed_epochs'],'selected_epoch':summary['best_epoch'],'training_summary_sha256':campaign.sha(summary_path)})
    if evpath.exists():
     receipt=campaign.read(evpath)
     if receipt.get('status')!='passed' or receipt.get('evaluator_sha256')!=report['evaluator_sha256'] or receipt.get('registration_sha256')!=report['registration_sha256']:raise ValueError('Existing evaluation identity/status mismatch')
     record['execution']='existing completed receipt preserved; device not inferred'
    else:
     command=[sys.executable,str(ROOT/'scripts/real_video_iws/evaluate.py'),'--config',str(campaign.CONFIG),'--name',name,'--output',str(evpath),'--device','cpu'];record['command']=command
     log=recovery/(name+'.cpu_evaluation.log')
     with log.open('x') as stream:run=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,text=True)
     record.update({'returncode':run.returncode,'log':str(log.relative_to(ROOT)),'log_sha256':campaign.sha(log),'execution':'frozen evaluator CPU invocation'})
     if run.returncode:raise RuntimeError(f'Frozen CPU evaluator returned{run.returncode}; gate unchanged; inspect '+str(log))
     receipt=campaign.read(evpath)
    if receipt['scope']!='internal_development' or receipt['official_validation_payloads_read']!=0 or receipt['completed_epochs']!=30 or receipt['selected_epoch']!=summary['best_epoch']:raise ValueError('Unexpected evaluation scope or selection')
    if campaign.sha(ROOT/receipt['window_ledger_path'])!=receipt['window_ledger_sha256']:raise ValueError('Primitive ledger identity mismatch')
    completion=ROOT/'reports/real_video_iws/training_completions'/f'{name}.json'
    if not completion.exists():campaign.atomic_json({'status':'training_and_development_evaluation_completed','utc':campaign.now(),**row,'summary_sha256':campaign.sha(summary_path),'evaluation_sha256':campaign.sha(evpath)},completion)
    record.update({'status':'passed','evaluation_sha256':campaign.sha(evpath),'window_ledger_sha256':receipt['window_ledger_sha256'],'completion_sha256':campaign.sha(completion),'prefix_maximum_absolute_difference':receipt['prefix_maximum_absolute_difference'],'h60_standardized_mse':receipt['h60_standardized_mse']})
   except Exception as exc:
    failed=True;record.update({'status':'failed_preserving_gate','error':repr(exc)})
   record['finished_utc']=campaign.now();report['runs'].append(record);campaign.atomic_json(report,args.report);print(json.dumps(record),flush=True)
  campaign.check_registration(campaign.CONFIG)
  report.update({'status':'failed_preserving_gate' if failed else 'passed','completed_utc':campaign.now()});campaign.atomic_json(report,args.report)
  if failed:raise SystemExit(1)
  subprocess.run([sys.executable,str(ROOT/'scripts/real_video_iws/finalize.py'),'--config',str(campaign.CONFIG),'--if-ready'],cwd=ROOT,check=True)
if __name__=='__main__':main()
