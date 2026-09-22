"""Matched-active-parameter, patch-local bounded-additive DROID control."""
from pathlib import Path
import sys, json, argparse, os, importlib.util
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import torch
from torch import nn
from shiftwm.real_video_spatial_components.model import ComponentWorldModel
from shiftwm.real_video_spatial.model import SpatialWorldModel
from shiftwm.real_video.data import sha256
OUT=ROOT/'reports/iclr_review_2026-09-22/capacity_control'
KIND='shiftwm_capacity_matched_bounded_additive_v1'

def private(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

class PatchCapacity(nn.Module):
 def __init__(self,d):
  super().__init__();self.a=nn.Linear(d,d,bias=False);self.b=nn.Linear(d,d,bias=False);self.gate=nn.Linear(d,1)
  nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,-3.)
 def forward(self,x):return x+torch.sigmoid(self.gate(x))*self.b(torch.nn.functional.gelu(self.a(x)))

class CapacityModel(ComponentWorldModel):
 MODES=SpatialWorldModel.MODES+('bounded_additive',)
 def __init__(self,config,**statistics):
  super().__init__(config,**statistics)
  self.output_projection=nn.Sequential(PatchCapacity(self.config.hidden_dim),self.output_projection)
 @property
 def package_config(self):return {**super().package_config,'capacity_control_schema':KIND}

def from_config(c):
 if c.get('capacity_control_schema')!=KIND:raise ValueError('Wrong capacity package')
 return CapacityModel(c['model_config'],**{k:c[k]for k in ('feature_mean','feature_std','action_mean','action_std')})

legacy=private(ROOT/'scripts/real_video_spatial/train.py','_capacity_spatial_trainer')
legacy.SpatialWorldModel=CapacityModel;legacy.base.RealVideoWorldModel=CapacityModel
legacy.base.from_config=from_config;legacy.base.PACKAGE_KIND=KIND
original_sources=legacy.source_files

def sources():
 return {**original_sources(),str(Path(__file__)):sha256(__file__),str(OUT/'protocol.md'):sha256(OUT/'protocol.md'),str(OUT/'registration.json'):sha256(OUT/'registration.json')}
legacy.base.source_files=sources

def config(seed):
 c=json.loads((ROOT/f'configs/real_video_spatial_components/v1/bounded_additive_s{seed}.json').read_text())
 c.update(output_dir=str(ROOT/f'runs/iclr_capacity_control_v1/s{seed}'),protocol_path=str(OUT/'protocol.md'),resume_if_present=True,max_runtime_seconds=5400)
 c.pop('component_schema',None);return c

def register():
 OUT.mkdir(parents=True,exist_ok=True)
 text='''# Capacity-matched DROID control\n\nPost-hoc original-development study; not held-out confirmation. Retain the\ncompleted three-seed bounded-additive baseline and bounded spatial mixer.\nAdd a patch-local gated two-layer MLP before the existing residual projection:\nh + sigmoid(g(h)) B GELU(Ah). A/B are bias-free96x96; g is affine96-to1.\nExactly18,529 active parameters are added, equal to the mixer's two projections\nand gate. The new head never mixes patch positions. The shared upstream spatial\nencoder remains unchanged. All old active tensors initialize identically to the\nbounded-additive arm at each seed; the added head consumes subsequent RNG draws.\nThe correction stays tanh-bounded. No dropout or regularization is added.\nThree seeds0/1/2,30full epochs, all other original DROID training, normalization,\nselection, horizons and development scoring settings are retained. No test\npayload is read. This controls parameter count, not every capacity/optimization\nproperty. Report either outcome against BOTH original bounded-additive and\nbounded-mixing comparators; no model is promoted based on this experiment.\n'''
 if (OUT/'protocol.md').exists():assert (OUT/'protocol.md').read_text()==text
 else:(OUT/'protocol.md').write_text(text)
 stats=json.loads((ROOT/'data/features/droid_spatial_v1/training_statistics.json').read_text());st={k:stats[k]for k in ('feature_mean','feature_std','action_mean','action_std')}
 c=config(0)['model_config']
 torch.manual_seed(0);control=CapacityModel({**c,'mode':'bounded_additive'},**st)
 torch.manual_seed(0);mix=SpatialWorldModel({**c,'mode':'transport'},**st)
 torch.manual_seed(0);base=ComponentWorldModel({**c,'mode':'bounded_additive'},**st)
 counts={n:sum(p.numel()for p in m.parameters()if p.requires_grad)for n,m in [('control',control),('mixing',mix),('bounded_additive',base)]}
 assert counts['control']==counts['mixing']and counts['control']-counts['bounded_additive']==18529
 # Old shared modules and output head retain exact initialization.
 for n,v in base.state_dict().items():
  key='output_projection.1.'+n[len('output_projection.'):]if n.startswith('output_projection.')else n
  assert torch.equal(v,control.state_dict()[key]),n
 # Trainable added weights influence the output after the original zero head learns.
 head=control.output_projection[0];x=torch.randn(2,16,96,requires_grad=True);head(x).square().mean().backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in head.parameters())
 d={'status':'frozen_before_capacity_training','schema':KIND,'seeds':[0,1,2],'epochs':30,'parameter_counts':counts,
    'initialization_and_gradient_checks':'passed','configs':[config(s)for s in range(3)],
    'source_sha256':{**original_sources(),str(Path(__file__)):sha256(__file__),str(OUT/'protocol.md'):sha256(OUT/'protocol.md')},
    'comparators':{str(p.relative_to(ROOT)):sha256(p)for p in [ROOT/'reports/real_video_spatial/finalization.json',ROOT/'reports/real_video_spatial_components/finalization.json']}}
 if (OUT/'registration.json').exists():assert json.loads((OUT/'registration.json').read_text())==d
 else:legacy.atomic_json(d,OUT/'registration.json')
 print(json.dumps(counts))

def run(seed):
 d=json.loads((OUT/'registration.json').read_text())
 for p,digest in d['source_sha256'].items():assert sha256(p)==digest,p
 assert os.environ.get('SLURM_JOB_ID')and torch.cuda.is_available()
 c=d['configs'][seed];result=legacy.train(c)
 if result['status']!='completed':raise RuntimeError('Full30epochs not complete; epoch-boundary resume required')
 ev=private(ROOT/'scripts/real_video_spatial/evaluate.py','_capacity_eval');ev.train=legacy
 result=ev.evaluate(c,str(OUT/f's{seed}_evaluation.json'))
 legacy.atomic_json({'status':'complete','seed':seed,'registration_sha256':sha256(OUT/'registration.json'),
    'evaluation_sha256':sha256(OUT/f's{seed}_evaluation.json'),'job_id':os.environ['SLURM_JOB_ID'],'gpu':torch.cuda.get_device_name()},OUT/f's{seed}_completion.json')
 print(json.dumps(result['summary']))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');p.add_argument('--seed',type=int,choices=[0,1,2]);a=p.parse_args()
 if a.register:register()
 else:run(a.seed)
