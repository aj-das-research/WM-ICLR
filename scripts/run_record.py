"""Execute a task and preserve its command, allocation and exit status."""
import datetime, json, os, pathlib, subprocess, sys, time
argv = sys.argv[1:]
if argv and argv[0] == '--': argv = argv[1:]
if not argv: raise SystemExit('Missing command')
job=os.environ.get('SLURM_JOB_ID',f'local-{os.getpid()}')
out=pathlib.Path('runs/jobs')/job; out.mkdir(parents=True,exist_ok=True)
record={'command':argv,'job_id':job,'partition':os.getenv('SLURM_JOB_PARTITION'),'node':os.uname().nodename,'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running'}
try:
 record['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).splitlines()
except Exception as e: record['gpu_query_error']=str(e)
p=out/'record.json';p.write_text(json.dumps(record,indent=2));start=time.monotonic()
try: result=subprocess.run(argv);record.update(exit_code=result.returncode,status='complete' if result.returncode==0 else 'failed')
except BaseException as e: record.update(exit_code=1,status='failed',error=repr(e))
record.update(elapsed_seconds=time.monotonic()-start,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(record,indent=2));raise SystemExit(record['exit_code'])
