"""Reviewed single-GPU cache/profile and full-epoch training entry points."""
import argparse
import datetime
import fcntl
import json
import os
import socket
import subprocess
import time
import torch
from planning_common import REPORT, REG, ROOT, atomic_json, checked_registration, module, read, require, select_run, sha


def request_continuation():
    job=os.environ.get('SLURM_JOB_ID','')
    require(job.isdigit() and int(os.environ.get('SLURM_RESTART_COUNT','0'))<5,'Incomplete work and continuation allowance exhausted')
    subprocess.run(['scontrol','requeue',job],check=True,capture_output=True,text=True)
    return job


def run_evaluation(index,split):
    reg=checked_registration();row=select_run(reg,index=index)
    require(torch.cuda.is_available(),'Allocated GPU required')
    path=REPORT/'executions'/f'{row["name"]}_{split}_{time.time_ns()}.json'
    execution={'schema':'current_spatial_planning_execution_v1','status':'started','name':row['name'],'split':split,
               'registration_sha256':sha(REG),'hostname':socket.gethostname(),'gpu':torch.cuda.get_device_name(),
               'torch':str(torch.__version__),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
               'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    atomic_json(execution,path);began=time.monotonic()
    try:
        result=module('planning_evaluate').evaluate(row,split)
        execution.update(status=result['status'],completed_cases=len(result['records']),elapsed_seconds=time.monotonic()-began)
        if result['status']!='complete':
            execution['status']='case_boundary_continuation_requested';atomic_json(execution,path)
            execution['requeued_job_id']=request_continuation();execution['status']='case_boundary_continuation_submitted'
        atomic_json(execution,path);return result
    except BaseException as error:
        execution.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-began)
        atomic_json(execution,path);raise


def run_training(index):
    reg=checked_registration();row=select_run(reg,index=index);directory=ROOT/row['output'];directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.training.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(torch.cuda.is_available(),'Allocated GPU required')
        execution={'schema':'current_spatial_training_execution_v1','status':'started','name':row['name'],
                   'registration_sha256':sha(REG),'hostname':socket.gethostname(),'gpu':torch.cuda.get_device_name(),
                   'torch':str(torch.__version__),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
                   'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        path=directory/f'execution_{time.time_ns()}.json';atomic_json(execution,path);began=time.monotonic()
        try:
            result=module('planning_train').train(row)
            execution.update(status=result['status'],completed_epochs=result['completed_epochs'],elapsed_seconds=time.monotonic()-began)
            if result['status']!='completed':
                execution['status']='epoch_boundary_continuation_requested';atomic_json(execution,path)
                execution['requeued_job_id']=request_continuation()
                execution['status']='epoch_boundary_continuation_submitted'
            else:
                checked_registration()
                execution['selected_model_sha256']=sha(directory/'best/model.pt')
            atomic_json(execution,path);return result
        except BaseException as error:
            execution.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-began)
            atomic_json(execution,path);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('cache-profile');q.add_argument('--task',choices=('pusht','reacher'),required=True)
    q=sub.add_parser('train');q.add_argument('--index',type=int,required=True)
    q=sub.add_parser('evaluate');q.add_argument('--index',type=int,required=True);q.add_argument('--split',choices=('development','test'),default='test')
    a=p.parse_args()
    if a.command=='cache-profile':
        module('planning_cache').prepare(a.task);module('planning_profile').profile(a.task)
    elif a.command=='train':run_training(a.index)
    else:run_evaluation(a.index,a.split)
