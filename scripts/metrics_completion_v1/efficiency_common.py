"""Source-bound, train-input-only encoder-inclusive resource measurements."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from torch.utils._python_dispatch import TorchDispatchMode

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
REPORT=ROOT/'reports/metrics_completion_v1/efficiency'
REG=REPORT/'registration.json'
CONFIG=ROOT/'configs/metrics_completion_v1/efficiency.json'
POLICY={
 'schema':'encoder_inclusive_efficiency_v1','batch_size':1,'warmup_calls':10,'timed_calls':30,
 'predictor_dtype':'float32','encoder_dtype':'BF16 autocast; FP32 preprocessing, pooling and storage',
 'tf32':False,'cpu_threads':2,'interop_threads':1,'required_gpu':'NVIDIA RTX 5000 Ada Generation',
 'input_scope':'fixed training inputs only; no accuracy, future RGB targets, reserved inputs or model selection',
 'droid':{'observed_frames':3,'native_frame_indices':[0,5,10],'query_steps':10,'command_blocks':12},
 'iws':{'training_episode':'000011','observed_frames':1,'native_frame_indices':[0],'command_rows':60,'predicted_offsets':59},
 'rows':73,'learned_checkpoints':69,'persistence_rows':4,
 'encoder_call':'existing DinoSpatialEncoder; CPU RGB resident -> H2D -> preprocess -> BF16 DINOv2 -> FP32 4x4 pool -> CPU numpy',
 'full_call':'encoder_call + feature/command H2D + FP32 predictor; output stays on GPU; no RGB decoder',
 'excluded':'video/file decoding, model loading, metric scoring, RGB decoder, planning/controller, network and allocator initialization',
 'timing':'perf_counter_ns, CUDA synchronize before/after each full call; resident model and inputs; 10 warmups then 30 calls',
 'stages':['encoder','predictor','full_call'],
 'reuse':'IWS cached predictor-only timings reused from completed iws_predictor_resources_v1; no rerun of those timings',
 'flops':'torch.utils.flop_counter registered covered-operator subtotal, multiply-add counts as two; unsupported operations explicitly listed; not complete total FLOPs',
 'flop_dispatch':'Counter may decompose operations. Separate dispatch-observed inventory and unsupported post-decomposition operations recorded. Instrumented probes excluded from latency.',
 'memory':'PyTorch allocated/reserved peak and incremental allocated peak above warmed resident baseline; excludes CUDA context/non-PyTorch allocators',
 'aggregation':'Per fixed checkpoint and input; seed-level runtime variation is not a statistical confidence interval or dataset accuracy',
}

def require(ok,message):
 if not ok:raise ValueError(message)
def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def relative(path):return str(Path(path).resolve().relative_to(ROOT))
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def atomic_json(value,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 text=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
 if path.exists():require(path.read_text()==text,'Refusing changed existing artifact: '+str(path));return
 fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
 try:
  with os.fdopen(fd,'w') as f:f.write(text);f.flush();os.fsync(f.fileno())
  os.link(tmp,path)
 finally:Path(tmp).unlink(missing_ok=True)

def validate_cases(rows):
 require(len(rows)==73 and len({r['name'] for r in rows})==73,'Incomplete or duplicate case roster')
 iws={(r['task'],r['mode'],r['seed']) for r in rows if r['family']=='iws' and r['mode']!='persistence'}
 require(iws=={(t,m,s) for t in ('pusht','bimanual_box','bimanual_rope') for m in ('autoregressive','anchored_additive','bounded_spatial_mix','unbounded_spatial_mix') for s in range(3)},'IWS arm/seed grid differs')
 droid={(r['mode'],r['seed']) for r in rows if r['family']=='droid' and r['mode']!='persistence'}
 modes=('autoregressive','anchored_additive','transport','context_off','action_free','bounded_additive','unbounded_transport','official_one_step_shifted','matched_recursive_h10','official_raw_one_step','official_raw_recursive_h10')
 require(droid=={(m,s) for m in modes for s in range(3)},'DROID arm/seed grid differs')
 require({r['task'] for r in rows if r['mode']=='persistence'}=={'droid','pusht','bimanual_box','bimanual_rope'},'Persistence grid differs')

def verify_registration(review=True,full=True):
 r=read(REG);require(r['policy']==POLICY and r['status']=='frozen_before_resource_measurement','Changed efficiency contract')
 validate_cases(r['cases'])
 if review:
  v=read(REPORT/'source_review.json');require(v['status']=='passed' and v['registration_sha256']==sha(REG),'Missing independent source review')
 if full:
  for p,h in r['dependencies'].items():
   path=(ROOT/p).resolve();require(path.is_relative_to(ROOT) and sha(path)==h,'Changed bound source: '+p)
 return r

def configure():
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def summarize(values):
 x=np.asarray(values,dtype=np.float64)
 require(x.shape==(30,) and np.isfinite(x).all() and (x>0).all(),'Need thirty finite positive latency samples')
 return {'samples_ms':x.tolist(),'mean_ms':float(x.mean()),'median_ms':float(np.median(x)),
         'p90_ms':float(np.quantile(x,.9)),'minimum_ms':float(x.min()),'maximum_ms':float(x.max())}

class Inventory(TorchDispatchMode):
 def __init__(self):super().__init__();self.calls=Counter()
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  self.calls[str(func._overloadpacket)]+=1
  return func(*args,**(kwargs or {}))

class AuditedCounter(FlopCounterMode):
 def __init__(self):super().__init__(display=False);self.observed=Counter()
 def _count_flops(self,func_packet,out,args,kwargs):
  self.observed[str(func_packet)]+=1
  return super()._count_flops(func_packet,out,args,kwargs)

def flop_audit(call):
 with torch.inference_mode(),Inventory() as native:reference=call()
 with torch.inference_mode(),AuditedCounter() as counter:result=call()
 # Decomposed dispatch can change rounding; this is a resource audit, not scoring.
 def tensor(x):return torch.from_numpy(x) if isinstance(x,np.ndarray) else x.detach().cpu()
 a,b=tensor(reference),tensor(result)
 require(a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all(),'Invalid FLOP probe result')
 counted={str(k):int(v) for k,v in counter.get_flop_counts()['Global'].items()}
 registered={str(k) for k in counter.flop_registry}
 missing={k:n for k,n in counter.observed.items() if k not in registered}
 return {'covered_operator_flops':int(counter.get_total_flops()),'complete_total_flops_claimed':False,
         'counted_flops_by_operator':counted,'dispatch_probe_operator_calls':dict(native.calls),
         'post_decomposition_operator_calls':dict(counter.observed),'unsupported_post_decomposition_calls':missing,
         'unsupported_recurrent_or_attention_ops':{k:n for k,n in missing.items() if any(s in k for s in ('rnn','gru','lstm','attention'))},
         'decomposition_vs_dispatch_probe_maxabs':float((a.float()-b.float()).abs().max()),
         'interpretation':POLICY['flops'],'probe_scope':POLICY['flop_dispatch']}
