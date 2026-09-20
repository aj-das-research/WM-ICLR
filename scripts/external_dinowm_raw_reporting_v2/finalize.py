#!/usr/bin/env python3
"""Complete-only raw-coordinate follow-up; frozen v1 arithmetic reused privately."""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V1 = Path('scripts/external_dinowm_reporting_v1/finalize.py')
V1_SHA = '0367ff732b2febd66c47bb90390294dd6b2c4a2f89e1604f4ad03ac704c9ac52'
LEDGER = Path('scripts/real_video_spatial/validate_ledger.py')
LEDGER_SHA = 'ecb4bbfdf1e0c7d7916f0a64f224f344c79b63844ce544b726501e8920cb79a2'
if hashlib.sha256((ROOT/V1).read_bytes()).hexdigest() != V1_SHA:
    raise ValueError('Frozen v1 reporting helper changed')
spec = importlib.util.spec_from_file_location('_raw_v2_private_reporting', ROOT/V1)
_core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_core)
TRAIN_DIR = Path('scripts/external_dinowm_raw_v2')
TRAIN_REPORT = Path('reports/external_dinowm_raw_v2')
REPORT = Path('reports/external_dinowm_raw_reporting_v2')
REG = REPORT/'registration.json'
BASE_REG = _core.BASE_REG
OBJECTIVES = ('official_raw_one_step', 'official_raw_recursive_h10')
INTERNAL = _core.INTERNAL
MODES = (*INTERNAL, *OBJECTIVES)
METRICS = _core.METRICS
COORDINATES = 'raw_channel_major_384x4x4_dino_features_normalized_actions'
PACKAGE_KIND = 'adapted_official_dinowm_raw_droid_v2'
FAMILY = 'adapted_official_dinowm_raw_v2'
SELECTION = 'window_mean_all10_shared_channel_standardized_mse'
PRECISION = 'float32 validation; autocast disabled; CUDA TF32 disabled'
LABELS = {k: v for k,v in _core.LABELS.items() if k in INTERNAL} | {
 'official_raw_one_step':'Adapted DINO-WM: raw shifted one-step (100 epochs)',
 'official_raw_recursive_h10':'Adapted DINO-WM: raw recursive H10 (100 epochs)'}
POLICY = {**deepcopy(_core.POLICY), 'schema':'external_dinowm_raw_reporting_policy_v2',
 'external_epochs':100, 'internal_epochs':30, 'objectives':list(OBJECTIVES),
 'visual_training_coordinates':'raw pooled DINO features; raw visual MSE objective',
 'common_selection':SELECTION,
 'evaluation_coordinates':'unchanged native4x4 and original2x2 standardized MSE with exact cached original targets',
 'scope':'posthoc raw-coordinate/recipe follow-up on DROID development with an adapted official DINO-WM baseline; not an official benchmark/SOTA reproduction; not a single-factor intervention'}
POLICY.pop('epochs')
# These assignments affect only this new private instance, never imported v1 globals.
for _name in ('TRAIN_DIR','TRAIN_REPORT','REPORT','REG','OBJECTIVES','MODES','LABELS','POLICY'):
    setattr(_core, _name, globals()[_name])
require, sha, read, local, module = (_core.require,_core.sha,_core.read,_core.local,_core.module)
atomic, write_text = _core.atomic, _core.write_text
roster, pending_inputs, reporting_gate = _core.roster, _core.pending_inputs, _core.reporting_gate
aggregate = _core.aggregate
common_population_and_persistence = _core.common_population_and_persistence


def ledger_module(root=ROOT, external=False):
    """Private exact metric code; only external epoch upper bound is100."""
    source=local(str(LEDGER),root).read_bytes()
    require(hashlib.sha256(source).hexdigest()==LEDGER_SHA,'Frozen ledger source changed')
    if not external:
        return module('_raw_v2_internal_ledger',root/LEDGER)
    tree=ast.parse(source,filename=str(root/LEDGER)); changed=0
    target="1 <= result['selected_epoch'] <= 30"
    for function in tree.body:
        if not isinstance(function,ast.FunctionDef) or function.name!='validate_ledger':continue
        for node in ast.walk(function):
            if isinstance(node,ast.Compare) and ast.unparse(node)==target:
                require(isinstance(node.comparators[-1],ast.Constant),'Epoch AST shape changed')
                node.comparators[-1].value=100;changed+=1
    require(changed==1,'Expected exactly one external selected-epoch bound')
    ast.fix_missing_locations(tree)
    value=types.ModuleType('_raw_v2_external_ledger');value.__file__=str(root/LEDGER)
    exec(compile(tree,str(root/LEDGER),'exec'),value.__dict__)
    value.MODES=OBJECTIVES
    return value


def validate_package_metadata(state):
    package=state['config']
    require(package.get('package_kind')==PACKAGE_KIND and package.get('model_family')==FAMILY
            and package.get('coordinate_layout')==COORDINATES,'Raw v2 package/coordinate identity differs')
    require(package.get('metadata',{}).get('selection')==SELECTION
            and package['metadata'].get('validation_precision')==PRECISION,'Common standardized selection differs')
    require(type(state.get('epoch')) is int and 1<=state['epoch']<=100,'Raw v2 selected epoch differs')

def validate_execution(value,name,registration_sha256):
    require(value.get('schema')=='external_dinowm_raw_training_execution_v2'
            and value.get('status')=='completed' and value.get('completed_epochs')==100
            and value.get('name')==name and value.get('registration_sha256')==registration_sha256,
            'External execution receipt identity differs')


def freeze(root=ROOT):
 """Freeze implementation and source identities before reading new results."""
 external=read(root/TRAIN_REPORT/'registration.json');roster(external,True)
 baseline=read(root/BASE_REG);roster(baseline,False)
 deps={}
 paths=[str(TRAIN_REPORT/'registration.json'),str(TRAIN_REPORT/'source_review.json'),str(BASE_REG),'reports/real_video_spatial/finalization.json',
  'scripts/real_video_spatial/validate_ledger.py','scripts/real_video_spatial/train.py',
  'scripts/real_video_spatial/evaluate.py','scripts/real_video_spatial/campaign.py']
 paths += [str(TRAIN_DIR/n) for n in ('registry.py','train.py','evaluate.py','model.py')]
 paths += [str(Path('scripts/external_dinowm_raw_reporting_v2')/n) for n in ('finalize.py','test_finalize.py','README.md','run.slurm')]
 paths += [str(V1), 'tests/test_spatial_ledger_validation.py']
 paths += [r['config'] for reg in (external,baseline) for r in reg['runs']]
 # Hash completed comparator evidence now; new external results are not opened.
 paths += [f"reports/real_video_spatial/{r['name']}_validation.json" for r in baseline['runs']]
 for rel in sorted(set(paths)):deps[rel]=sha(local(rel,root))
 for reg in (external,baseline):
  for rel,expected in reg['dependencies'].items():
   require(sha(local(rel,root))==expected,'Changed registered scientific dependency: '+rel);deps[rel]=expected
 value={'schema':'external_dinowm_raw_reporting_registration_v2','status':'frozen_before_external_results_read',
  'policy':POLICY,'dependencies':deps,'external_results_read':0}
 p=root/REG
 if p.exists():require(read(p)==value,'Reporting registration is immutable')
 else:atomic(value,p)
 return {'status':value['status'],'registration_sha256':sha(p),'source_count':len(deps)}

def validate_external(result,config,state,config_path,registration_path,root=ROOT,ledger=None):
 validate_package_metadata(state)
 require(config.get('epochs')==100,'External budget must be100')
 ledger=ledger or ledger_module(root,external=True)
 expected={'schema':'adapted_official_dinowm_raw_droid_validation_v2','model_family':'adapted_official_dinowm_raw_v2',
  'run_name':f"{config['mode']}_s{config['seed']}",'training_objective':config['mode'],
  'registration_sha256':sha(registration_path),'config_sha256':sha(config_path),
  'training_identity':state['config']['metadata']['training_identity'],
  'normalization_sha256':sha(local(config['cache_root']+'/training_statistics.json',root)),
  'selection_metric':state['config']['metadata']['selection'],
  'evaluation_precision':state['config']['metadata']['validation_precision'],
  'source_sha256':sha(root/TRAIN_DIR/'evaluate.py'),
  'upstream_evaluation_source_sha256':sha(root/'scripts/real_video_spatial/evaluate.py')}
 require(config['mode'] in OBJECTIVES,'Unknown external objective')
 for k,v in expected.items():require(result.get(k)==v,'External evaluator identity differs: '+k)
 normalized=deepcopy(result);normalized['source_sha256']=result['upstream_evaluation_source_sha256']
 return ledger.validate_ledger(normalized,config,root,state)

def collect(root=ROOT):
 reportreg=reporting_gate(root)
 external_api=module('_external_report_registry',root/TRAIN_DIR/'registry.py')
 external=external_api.verify();roster(external,True)
 base_api=module('_external_report_base_campaign',root/'scripts/real_video_spatial/campaign.py')
 baseline=base_api.verify();roster(baseline,False)
 require(not pending_inputs(external,root),'All six completed training/evaluation markers are required')
 exttrain=module('_external_report_train',root/TRAIN_DIR/'train.py')
 basetrain=module('_external_report_base_train',root/'scripts/real_video_spatial/train.py')
 ledger=ledger_module(root);external_ledger=ledger_module(root,external=True);sources=dict(reportreg['dependencies']);prepared=[]
 def bind(rel):
  path=local(rel,root);sources[str(path.relative_to(root))]=sha(path);return path
 # FIRST gate all six100-epoch external packages and fifteen30-epoch comparators.
 # No new validation ledger values are read until every package gate passes.
 for is_external,registry,trainer,report_dir in ((True,external,exttrain,TRAIN_REPORT),(False,baseline,basetrain,Path('reports/real_video_spatial'))):
  budget=100 if is_external else 30
  for row in roster(registry,is_external):
   config_path=local(row['config'],root);require(sha(config_path)==row['sha256'],'Frozen run config changed')
   config=read(config_path)
   require(config['mode']==row['mode'] and config['seed']==row['seed'] and config['epochs']==budget,'Wrong mode/seed/training budget')
   directory=local(config['output_dir'],root);summary=trainer.validate_completed(directory)
   package,state=trainer.read_package(directory/'best')
   # Retain only metadata, not model tensors, across the21-run validation pass.
   selected={'epoch':state['epoch'],'config':state['config']};del state
   if is_external:validate_package_metadata(selected)
   require(summary['completed_epochs']==budget and summary['best_epoch']==selected['epoch'],'Incomplete/incorrect selected checkpoint')
   marker_rel=str(report_dir/(row['name']+'_completed.json'));marker=read(bind(marker_rel))
   require(marker.get('status')=='completed' and marker.get('epochs')==budget and marker.get('name')==row['name'],'Missing complete-run marker')
   if is_external:
    require(marker.get('registration_sha256')==sha(root/TRAIN_REPORT/'registration.json')
            and marker.get('training_summary_sha256')==sha(directory/'training_summary.json')
            and marker.get('checkpoint_sha256')==sha(directory/'best/model.pt')
            and marker.get('selected_epoch')==selected['epoch'],'External completion identity differs')
    execution=bind(marker.get('execution_receipt',''))
    require(execution.parent==directory.resolve() and execution.name.startswith('execution_')
            and marker.get('execution_receipt_sha256')==sha(execution),'External execution receipt binding differs')
    validate_execution(read(execution),row['name'],sha(root/TRAIN_REPORT/'registration.json'))
   for name in ('training_summary.json','training_config.json','metrics.jsonl'):bind(str((directory/name).relative_to(root)))
   for kind in ('best','last'):
    package_path=(directory/kind).resolve();manifest=read(bind(str((package_path/'package_manifest.json').relative_to(root))))
    for name,digest in manifest['files'].items():
     path=bind(str((package_path/name).relative_to(root)));require(sources[str(path.relative_to(root))]==digest,'Changed selected/completed package payload')
   prepared.append((is_external,row,config,config_path,summary,selected,marker,str(report_dir/(row['name']+'_validation.json'))))
 rows=[];run_proofs=[]
 # THEN validate every primitive window, episode mean and scope/source identity.
 for is_external,row,config,config_path,summary,selected,marker,relative in prepared:
  result=read(bind(relative))
  if is_external:require(marker.get('validation_sha256')==sha(local(relative,root)),'External completion/validation binding differs')
  computed=(validate_external(result,config,selected,config_path,root/TRAIN_REPORT/'registration.json',root,external_ledger)
            if is_external else ledger.validate_ledger(result,config,root,selected))
  result['summary']=computed;rows.append(result)
  run_proofs.append({'name':row['name'],'mode':row['mode'],'seed':row['seed'],'external':is_external,
   'completed_epochs':100 if is_external else 30,'selected_epoch':summary['best_epoch'],'selected_checkpoint_sha256':result['checkpoint_sha256'],
   'evaluation_path':relative,'evaluation_sha256':sources[str(local(relative,root).relative_to(root))],
   'parameter_counts':result['parameter_counts'],'windows':len(result['windows']),'episodes':len(result['episodes'])})
 # ONLY NOW are any cross-run means or intervals constructed.
 result=aggregate(rows,ledger)
 for rel,digest in sources.items():require(sha(local(rel,root))==digest,'Evidence changed during finalization: '+rel)
 return {'schema':'external_dinowm_raw_complete_development_comparison_v2','status':'passed','policy':POLICY,
  'completed_external_runs':6,'completed_internal_runs':15,'epochs_per_model':{'external':100,'internal':30},'results':result,'per_run':run_proofs,
  'reporting_registration_sha256':sha(root/REG),'training_registration_sha256':sha(root/TRAIN_REPORT/'registration.json'),
  'source_dependencies':sources,'new_inference_performed':False,'reserved_or_test_payloads_read':0}

def markdown(value):
    text=_core.markdown(value)
    text=text.replace('# Adapted DINO-WM: complete DROID development comparison',
                      '# Adapted DINO-WM: complete raw-coordinate DROID follow-up')
    text=text.replace('All six external and15 internal runs completed30 epochs.',
                      'All six raw-coordinate external runs completed100 epochs; all15 fixed internal runs completed30 epochs.')
    return text

def main(if_ready=False,root=ROOT):
 training_reg=root/TRAIN_REPORT/'registration.json'
 if not training_reg.exists():
  if if_ready:return {'status':'pending','reason':'external_registration_absent'}
  raise ValueError('External registration absent')
 ext=read(training_reg);missing=pending_inputs(ext,root)
 if missing:
  if if_ready:return {'status':'pending','missing':missing}
  raise ValueError('External campaign incomplete: '+str(len(missing))+' inputs missing')
 reporting_gate(root)
 out=root/REPORT;out.mkdir(parents=True,exist_ok=True)
 with (out/'.finalization.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  final=out/'finalization.json';completion=out/'completion.json'
  if final.exists():
   require(completion.is_file(),'Existing finalization lacks completion binding')
   receipt=read(completion)
   require(receipt.get('status')=='passed' and receipt.get('finalization_sha256')==sha(final)
           and (out/'results.md').is_file() and receipt.get('results_md_sha256')==sha(out/'results.md'),
           'Existing finalization lacks valid completion binding')
   value=read(final);require(value.get('status')=='passed' and value.get('policy')==POLICY,'Invalid previous finalization')
   require(value.get('schema')=='external_dinowm_raw_complete_development_comparison_v2'
           and value.get('epochs_per_model')=={'external':100,'internal':30}
           and value.get('completed_external_runs')==6 and value.get('completed_internal_runs')==15
           and value.get('reporting_registration_sha256')==sha(root/REG),'Previous finalization scope differs')
   for rel,digest in value['source_dependencies'].items():require(sha(local(rel,root))==digest,'Changed completed evidence')
  else:
   value=collect(root);atomic(value,final)
   write_text(markdown(value),out/'results.md')
   atomic({'status':'passed','finalization_sha256':sha(final),'results_md_sha256':sha(out/'results.md')},completion)
  return {'status':'passed','completed_external_runs':6,'completed_internal_runs':15,'finalization_sha256':sha(final)}

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--freeze',action='store_true');p.add_argument('--if-ready',action='store_true')
    a=p.parse_args();print(json.dumps(freeze() if a.freeze else main(a.if_ready),sort_keys=True))
