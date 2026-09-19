#!/usr/bin/env python3
"""Operational CPU re-evaluation; registered science and selected weights stay frozen."""
from __future__ import annotations
import argparse,fcntl,importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/real_video_iws/recovery/common_cpu_v1'
REG=OUT/'registration.json'
SP=importlib.util.spec_from_file_location('_common_cpu_campaign',ROOT/'scripts/real_video_iws/campaign.py')
campaign=importlib.util.module_from_spec(SP);SP.loader.exec_module(campaign)

def rel(p):return str(Path(p).relative_to(ROOT))
def record(p):return {'path':rel(p),'sha256':campaign.sha(p)}
def immutable_json(value,path):
 path=Path(path)
 if path.exists():
  if campaign.read(path)!=value:raise ValueError('Refusing to replace immutable receipt: '+str(path))
 else:campaign.atomic_json(value,path)

def archive_copy(original,destination,expected):
 original,destination=Path(original),Path(destination)
 if campaign.sha(original)!=expected:raise ValueError('Original changed before archival: '+str(original))
 destination.parent.mkdir(parents=True,exist_ok=True)
 if destination.exists():
  if campaign.sha(destination)!=expected:raise ValueError('Existing archive differs')
 else:
  temp=destination.with_name(destination.name+'.tmp')
  with original.open('rb') as src,temp.open('xb') as dst:
   shutil.copyfileobj(src,dst,8<<20);dst.flush();os.fsync(dst.fileno())
  if campaign.sha(temp)!=expected:raise ValueError('Archive copy differs')
  os.replace(temp,destination)
 if campaign.sha(original)!=expected or campaign.sha(destination)!=expected:raise ValueError('Archive identity verification failed')

def retire_original(original,archive,expected):
 """Remove only the authorized canonical copy after its immutable twin exists."""
 original,archive=Path(original),Path(archive)
 if not archive.is_file() or campaign.sha(archive)!=expected:raise ValueError('No exact archive; canonical file retained')
 if original.exists():
  if campaign.sha(original)!=expected:raise ValueError('Canonical evidence differs; refusing removal')
  original.unlink()

def baseline(receipt):
 return [{k:v for k,v in ep.items() if k.startswith('persistence_') or k in ('episode_id','windows')} for ep in receipt['episodes']]

def frozen_check(reg):
 scientific=campaign.check_registration()
 if campaign.sha(campaign.REGISTRATION)!=reg['scientific_registration_sha256']:raise ValueError('Scientific registration changed')
 if campaign.sha(ROOT/reg['archive_manifest']['path'])!=reg['archive_manifest']['sha256']:raise ValueError('Archive manifest changed')
 for p,d in reg['operational_dependencies'].items():
  if campaign.sha(ROOT/p)!=d:raise ValueError('Operational dependency changed: '+p)
 for r in reg['retained_autoregressive']:
  for b in r['files']:
   if campaign.sha(ROOT/b['path'])!=b['sha256']:raise ValueError('Retained CPU AR evidence changed')
 return scientific

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'.prepare.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if REG.exists():raise ValueError('Operational registration already exists')
  if (ROOT/'reports/real_video_iws/development_finalization.json').exists():raise ValueError('Study already finalized')
  scientific=campaign.check_registration();runs=[];keep=[];sources={}
  for row in scientific['runs']:
   name=row['name'];directory=ROOT/row['output'];summary=campaign.read(directory/'training_summary.json')
   if summary.get('completed_epochs')!=30 or summary.get('status')!='completed':raise ValueError('Incomplete training '+name)
   ev=ROOT/'reports/real_video_iws/evaluations'/f'{name}.json';receipt=campaign.read(ev)
   if receipt['status']!='passed' or receipt['scope']!='internal_development' or receipt['official_validation_payloads_read']!=0:raise ValueError('Invalid existing evaluation')
   files=[record(ev),record(ev.with_suffix('.npz')),record(ROOT/'reports/real_video_iws/training_completions'/f'{name}.json')]
   if files[1]['sha256']!=receipt['window_ledger_sha256']:raise ValueError('Original ledger mismatch')
   config=campaign.read(directory/'best/package_manifest.json') if (directory/'best/package_manifest.json').is_file() else None
   # Package directory may be a pointer; selected model path is resolved by frozen reader.
   tr=campaign.trainer();package,state=tr.read_package(directory/'best')
   checkpoint=record(package/'model.pt')
   if checkpoint['sha256']!=receipt['selected_checkpoint_sha256'] or summary['best_epoch']!=receipt['selected_epoch']:raise ValueError('Selected checkpoint identity mismatch')
   base={**row,'selected_epoch':summary['best_epoch'],'checkpoint':checkpoint,'summary':record(directory/'training_summary.json'),'files':files}
   if row['mode']=='autoregressive':keep.append(base);continue
   if row['mode'] not in ('anchored_additive','bounded_spatial_mix'):raise ValueError('Unregistered repair arm')
   base['repair_index']=len(runs)
   for b in files:
    original=ROOT/b['path'];dest=OUT/'archive'/name/original.parent.name/original.name
    archive_copy(original,dest,b['sha256']);b['archive_path']=rel(dest)
   runs.append(base)
  if len(runs)!=18 or len(keep)!=9:raise ValueError('Expected 18 replacement and 9 retained runs')
  for task in campaign.TASKS:
   copies=[campaign.read(ROOT/r['files'][0]['path']) for r in keep if r['task']==task]
   if any(baseline(v)!=baseline(copies[0]) for v in copies):raise ValueError('CPU AR references already differ')
  operational=['scripts/real_video_iws_recovery/common_cpu_v1.py','scripts/real_video_iws_recovery/common_cpu_v1.slurm','paper/scripts/render_iws_results.py']
  archive={'schema':'iws_common_cpu_archive_v1','status':'verified_exact_copies_before_canonical_replacement','files':[b for r in runs for b in r['files']]}
  immutable_json(archive,OUT/'archive_manifest.json')
  reg={'schema':'iws_common_cpu_operational_registration_v1','created_utc':campaign.now(),'scientific_registration_sha256':campaign.sha(campaign.REGISTRATION),'scientific_config_sha256':scientific['config_sha256'],'scope':'Internal-development evaluations only; unchanged frozen evaluator --device cpu, all 59 offsets and original selected checkpoints. No training or official-validation access.','cause':'Frozen finalizer rejects exact persistence inequality between CPU AR and GPU additive/mix metrics; preserve original GPU evidence and recompute on one backend.','runs':runs,'retained_autoregressive':keep,'archive_manifest':record(OUT/'archive_manifest.json'),'operational_dependencies':{p:campaign.sha(ROOT/p) for p in operational},'resources':{'device':'cpu','gpus':0,'cpus_per_task':8,'memory_gib':24,'wall_time':'01:00:00','ws_ia_max_parallel':2,'gpu_partition_max_parallel':2,'gpu_partition_cpu_total':16},'frozen_scientific_dependencies':scientific['dependencies'],'repair_invariants':['No model/config/epochs/checkpoint-selection/metric/prefix-tolerance changes','Original18 JSON+NPZ+completion triplets archived before their removal','Existing9 CPU AR JSON+NPZ+completion triplets immutable','Full27 finalizer and its exact persistence check remain unchanged']}
  frozen_check(reg);immutable_json(reg,REG)
  print(json.dumps({'status':'prepared','registration':rel(REG),'sha256':campaign.sha(REG),'archived_files':54,'runs':18}),flush=True)

def validate_cpu_result(row,reg):
 ev=ROOT/row['files'][0]['path'];r=campaign.read(ev)
 if (r['status']!='passed' or r['scope']!='internal_development' or r['completed_epochs']!=30 or r['selected_epoch']!=row['selected_epoch'] or r['selected_checkpoint_sha256']!=row['checkpoint']['sha256'] or r['official_validation_payloads_read']!=0 or r['evaluator_sha256']!=reg['frozen_scientific_dependencies']['scripts/real_video_iws/evaluate.py']):raise ValueError('CPU result identity changed')
 if campaign.sha(ev.with_suffix('.npz'))!=r['window_ledger_sha256']:raise ValueError('CPU result ledger differs')
 reference=next(x for x in reg['retained_autoregressive'] if x['task']==row['task'] and x['seed']==0)
 if baseline(r)!=baseline(campaign.read(ROOT/reference['files'][0]['path'])):raise ValueError('CPU persistence still not exactly equal to retained AR reference')
 return r

def run(index):
 reg=campaign.read(REG);frozen_check(reg)
 if not 0<=index<len(reg['runs']):raise ValueError('Unknown repair index')
 row=reg['runs'][index];name=row['name'];folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
 with (folder/'execution.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  finished=folder/'completion.json'
  if finished.exists():
   for b in campaign.read(finished)['new_files']:
    if campaign.sha(ROOT/b['path'])!=b['sha256']:raise ValueError('Completed CPU evidence changed')
   validate_cpu_result(row,reg);print(json.dumps({'status':'already_complete','name':name}),flush=True);return
  if campaign.sha(ROOT/row['checkpoint']['path'])!=row['checkpoint']['sha256'] or campaign.sha(ROOT/row['summary']['path'])!=row['summary']['sha256']:raise ValueError('Training output changed')
  for b in row['files']:
   if campaign.sha(ROOT/b['archive_path'])!=b['sha256']:raise ValueError('Original archive changed')
  started=folder/'started.json'
  if not started.exists():
   for b in row['files']:
    if campaign.sha(ROOT/b['path'])!=b['sha256']:raise ValueError('Unexpected canonical evidence before repair')
   immutable_json({'started_utc':campaign.now(),'registration_sha256':campaign.sha(REG),'name':name,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'device':'cpu','old_files':row['files']},started)
  ev=ROOT/row['files'][0]['path']
  # Resume only known original files or a complete, independently valid CPU result.
  is_new=ev.exists() and campaign.sha(ev)!=row['files'][0]['sha256']
  if not is_new:
   for b in row['files']:retire_original(ROOT/b['path'],ROOT/b['archive_path'],b['sha256'])
   cmd=[sys.executable,str(ROOT/'scripts/real_video_iws/evaluate.py'),'--config',str(campaign.CONFIG),'--name',name,'--output',str(ev),'--device','cpu']
   log=folder/'evaluator.log'
   if log.exists():raise ValueError('Existing failed attempt requires inspection before retry; original archive retained')
   with log.open('x') as stream:rc=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode
   immutable_json({'command':cmd,'returncode':rc,'log':record(log),'finished_utc':campaign.now()},folder/'invocation.json')
   if rc:raise RuntimeError('Frozen CPU evaluator failed; archive remains intact')
  r=validate_cpu_result(row,reg);frozen_check(reg)
  completion=ROOT/'reports/real_video_iws/training_completions'/f'{name}.json'
  value={'status':'training_and_development_evaluation_completed','utc':campaign.now(),**{k:row[k] for k in ('mode','name','output','seed','task')},'summary_sha256':row['summary']['sha256'],'evaluation_sha256':campaign.sha(ev)}
  if not completion.exists():campaign.atomic_json(value,completion)
  elif campaign.read(completion).get('evaluation_sha256')!=campaign.sha(ev):raise ValueError('Completion marker belongs to another evaluation')
  result={'status':'passed_common_cpu_replay','name':name,'finished_utc':campaign.now(),'registration_sha256':campaign.sha(REG),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'device':'cpu','selected_epoch':row['selected_epoch'],'checkpoint_sha256':row['checkpoint']['sha256'],'prefix_maximum_absolute_difference':r['prefix_maximum_absolute_difference'],'official_validation_payloads_read':0,'persistence_exactly_matches_cpu_ar':True,'new_files':[record(ev),record(ev.with_suffix('.npz')),record(completion)],'original_files':row['files']}
  immutable_json(result,finished);print(json.dumps(result),flush=True)

def finalize():
 reg=campaign.read(REG);frozen_check(reg)
 for row in reg['runs']:
  p=OUT/'runs'/row['name']/'completion.json'
  if not p.is_file() or campaign.read(p)['status']!='passed_common_cpu_replay':raise ValueError('Incomplete common CPU replay')
  validate_cpu_result(row,reg)
 for b in campaign.read(OUT/'archive_manifest.json')['files']:
  if campaign.sha(ROOT/b['archive_path'])!=b['sha256']:raise ValueError('Archive changed')
 log=OUT/'finalizer.log'
 with log.open('x') as stream:
  subprocess.run([sys.executable,str(ROOT/'scripts/real_video_iws/finalize.py'),'--config',str(campaign.CONFIG)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
  subprocess.run([sys.executable,str(ROOT/'paper/scripts/render_iws_results.py'),'--if-ready'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
 frozen_check(reg)
 result={'status':'passed_full27_common_cpu_finalization','completed_utc':campaign.now(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'registration':record(REG),'archive_manifest':record(OUT/'archive_manifest.json'),'scientific_finalization':record(ROOT/'reports/real_video_iws/development_finalization.json'),'log':record(log),'retained_autoregressive_runs':9,'recomputed_additive_mix_runs':18,'official_validation_payloads_read':0,'main_paper_build_performed':False,'plot_review':'Actual PDF and integrated manuscript visual review remain for parent.'}
 immutable_json(result,OUT/'completion.json');print(json.dumps(result),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['prepare','run','finalize']);p.add_argument('--index',type=int);args=p.parse_args()
 if args.action=='prepare':prepare()
 elif args.action=='run':run(args.index)
 else:finalize()
