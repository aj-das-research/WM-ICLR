"""Operational afterok chain; scientific sources/parameters are never changed.

Dry-run is default. Root executes --submit after inspecting cache/profile jobs.
"""
import argparse
import datetime
import fcntl
import json
import subprocess
from planning_common import REG, REPORT, ROOT, atomic_json, checked_registration, require, sha


def submit(cache_jobs, execute=False):
    checked_registration()
    require(len(cache_jobs)==2 and all(x.isdigit() for x in cache_jobs),'Exactly two numeric cache job IDs required')
    ws=','.join(str(i)for i in range(24)if i%3!=2)+'%2'
    gpu=','.join(str(i)for i in range(24)if i%3==2)+'%1'
    stages=[('train_ws','planning_train.slurm',['--partition=ws-ia','--array='+ws],('cache',)),
            ('train_gpu','planning_train.slurm',['--partition=gpu','--array='+gpu],('cache',)),
            ('evaluate_ws','planning_evaluate.slurm',['--partition=ws-ia','--array='+ws],('train_ws','train_gpu')),
            ('evaluate_gpu','planning_evaluate.slurm',['--partition=gpu','--array='+gpu],('train_ws','train_gpu')),
            ('finalize','planning_finalize.slurm',[],('evaluate_ws','evaluate_gpu'))]
    if not execute:
        result={'registration_sha256':sha(REG),'cache_jobs':cache_jobs,'stages':[
            {'name':name,'command':['sbatch','--parsable','--dependency=afterok:<'+','.join(deps)+'>',*args,
                                  'scripts/metrics_completion_v1/'+script]}for name,script,args,deps in stages],
                'gate':'all24 complete models are independently rechecked before test payloads; finalizer checks all2400 complete traces',
                'resource_limits':'at most two ws-ia GPU allocations plus one gpu allocation; root may reserve slots by delaying submission'}
        atomic_json(result,REPORT/'submission_plan.json');return result
    path=REPORT/'campaign_submission.json'
    with path.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not path.exists(),'Existing submission requires explicit inspection; duplicate submissions refused')
        receipt={'status':'submitting','registration_sha256':sha(REG),'source_sha256':sha(__file__),
                 'cache_jobs':cache_jobs,'jobs':{},'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        atomic_json(receipt,path)
        try:
            for name,script,args,deps in stages:
                ids=cache_jobs if deps==('cache',) else[receipt['jobs'][d]['job_id']for d in deps]
                command=['sbatch','--parsable','--dependency=afterok:'+':'.join(ids),*args,
                         str(ROOT/'scripts/metrics_completion_v1'/script)]
                receipt['next_command']=command;atomic_json(receipt,path)
                p=subprocess.run(command,check=True,capture_output=True,text=True)
                job=p.stdout.strip().split(';')[0];require(job.isdigit(),'Unrecognized sbatch identity')
                receipt['jobs'][name]={'job_id':job,'command':command};atomic_json(receipt,path)
            receipt.pop('next_command',None);receipt['status']='submitted';atomic_json(receipt,path)
        except BaseException as error:
            receipt.update(status='submission_failed_inspect_saved_jobs_before_retry',error=str(error));atomic_json(receipt,path);raise
        return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-jobs',nargs=2,required=True);p.add_argument('--submit',action='store_true')
    a=p.parse_args();print(json.dumps(submit(a.cache_jobs,a.submit),indent=2))
