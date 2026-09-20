"""Synthetic complete/partial/adverse reporting cases; no external results read."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('external_report_test',Path(__file__).with_name('finalize.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
spec=importlib.util.spec_from_file_location('frozen_spatial_fixture',ROOT/'tests/test_spatial_ledger_validation.py')
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)


def write(root,name,value):
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value if isinstance(value,str) else json.dumps(value));return p


@pytest.fixture
def full_rows(tmp_path):
 original,_,_,_=fixtures.fixture.__wrapped__(tmp_path)
 offsets={'transport':0.,'autoregressive':.2,'anchored_additive':.3,'context_off':.4,'action_free':.5,
          'official_one_step_shifted':-1.,'matched_recursive_h10':1.}
 rows=[]
 for mode in f.MODES:
  for seed in range(3):
   row=deepcopy(original);row.update(mode=mode,seed=seed)
   for metric in ('native_mse','original_2x2_mse'):
    for e in row['windows']+row['episodes']:
     e[metric]=[value+offsets[mode]+seed*.01 for value in e[metric]]
    row['summary'][metric]=np.mean([e[metric] for e in row['episodes']],axis=0).tolist()
   rows.append(row)
 return rows


def test_all21_population_metrics_and_adverse_primary_retained(full_rows):
 v=f.aggregate(full_rows,f.ledger_module(),draws=80)
 assert set(v['aggregate'])==set(f.MODES) and len(v['comparisons'])==10
 assert v['population']=={'windows_per_run':3,'eligible_episodes':2,'sessions':2}
 assert len(v['per_seed'])==7 and all(set(v['per_seed'][m])=={'0','1','2'} for m in f.MODES)
 ours=v['aggregate']['transport']['native_mse'][9]
 assert ours==pytest.approx(5.01) # equal episodes, not the unequal-window mean4.01
 adverse=v['comparisons']['transport_vs_official_one_step_shifted']['metrics']['native_mse'][9]
 assert adverse['mean_difference']==pytest.approx(1.) and adverse['relative_error_reduction_percent']<0
 np.testing.assert_allclose(adverse['ci95'],[1,1])
 positive=v['comparisons']['transport_vs_matched_recursive_h10']['metrics']['native_mse'][9]
 assert positive['mean_difference']==pytest.approx(-1.) and positive['relative_error_reduction_percent']>0
 for comparison in v['comparisons'].values():
  assert set(comparison['metrics'])==set(f.METRICS)
  assert all(len(points)==10 for points in comparison['metrics'].values())
  for metric in ('native_persistence_mse','original_2x2_persistence_mse'):
   assert all(point['mean_difference']==0 and point['ci95']==[0,0] for point in comparison['metrics'][metric])
 text=f.markdown({'results':v})
 assert 'not an official benchmark/SOTA reproduction' in text and '-24.938%' in text
 assert '+1.000000 [+1.000000, +1.000000]' in text


@pytest.mark.parametrize('damage',['missing','duplicate','population','persistence','nonfinite'])
def test_incomplete_or_incompatible_comparison_rejected(full_rows,damage):
 rows=deepcopy(full_rows)
 if damage=='missing':rows.pop()
 elif damage=='duplicate':rows[-1]=deepcopy(rows[-2])
 elif damage=='population':rows[-1]['windows'].pop()
 elif damage=='persistence':rows[-1]['windows'][0]['native_persistence_mse'][9]+=.01
 elif damage=='nonfinite':rows[-1]['summary']['native_mse'][0]=float('nan')
 with pytest.raises(ValueError):f.aggregate(rows,f.ledger_module(),draws=10)


def test_zero_comparator_stays_undefined(full_rows):
 for row in full_rows:
  if row['mode']=='official_one_step_shifted':
   row['summary']['native_mse']=[0.]*10
   for e in row['episodes']:e['native_mse']=[0.]*10
 value=f.aggregate(full_rows,f.ledger_module(),draws=10)
 assert value['comparisons']['transport_vs_official_one_step_shifted']['metrics']['native_mse'][9]['relative_error_reduction_percent'] is None
 assert 'undefined (zero comparator)' in f.markdown({'results':value})


def test_missing_campaign_writes_nothing(tmp_path):
 before=list(tmp_path.iterdir())
 assert f.main(if_ready=True,root=tmp_path)['status']=='pending'
 assert list(tmp_path.iterdir())==before


def synthetic_registry(root):
 rows=[]
 for mode in f.OBJECTIVES:
  for seed in range(3):
   name=f'{mode}_s{seed}';rel=str(f.TRAIN_DIR/'configs'/(name+'.json'))
   path=write(root,rel,{'mode':mode,'seed':seed,'epochs':30,'output_dir':'runs/'+name})
   rows.append({'name':name,'mode':mode,'seed':seed,'config':rel,'sha256':f.sha(path)})
 reg={'runs':rows,'dependencies':{}};write(root,str(f.TRAIN_REPORT/'registration.json'),reg);return reg


def test_partial_six_run_gate_never_calls_aggregation(tmp_path,monkeypatch):
 reg=synthetic_registry(tmp_path)
 for row in reg['runs'][:-1]:
  for suffix in ('validation','completed'):write(tmp_path,str(f.TRAIN_REPORT/(row['name']+'_'+suffix+'.json')),{'synthetic':True})
  write(tmp_path,'runs/'+row['name']+'/training_summary.json',{'synthetic':True})
 monkeypatch.setattr(f,'aggregate',lambda *a,**kw:pytest.fail('Partial campaign reached aggregation'))
 assert f.main(True,tmp_path)['status']=='pending'
 assert not (tmp_path/f.REPORT).exists()
 reg['runs'][-1]=deepcopy(reg['runs'][0]);write(tmp_path,str(f.TRAIN_REPORT/'registration.json'),reg)
 with pytest.raises(ValueError,match='roster'):f.main(True,tmp_path)


def test_complete_presence_still_requires_source_review(tmp_path):
 reg=synthetic_registry(tmp_path)
 for row in reg['runs']:
  for suffix in ('validation','completed'):write(tmp_path,str(f.TRAIN_REPORT/(row['name']+'_'+suffix+'.json')),{'synthetic':True})
  write(tmp_path,'runs/'+row['name']+'/training_summary.json',{'synthetic':True})
 with pytest.raises(ValueError,match='source review'):f.main(True,tmp_path)
 assert not (tmp_path/f.REPORT).exists()


def test_sixth_incomplete_training_stops_before_reading_any_external_errors(tmp_path,monkeypatch):
 from types import SimpleNamespace
 reg=synthetic_registry(tmp_path);calls=[]
 for row in reg['runs']:
  directory='runs/'+row['name']
  for name in ('training_summary.json','training_config.json','metrics.jsonl'):write(tmp_path,directory+'/'+name,{})
  for which in ('best','last'):
   model=write(tmp_path,directory+'/'+which+'/model.pt','Synthetic bytes; no tensor data')
   write(tmp_path,directory+'/'+which+'/package_manifest.json',{'files':{'model.pt':f.sha(model)}})
  write(tmp_path,str(f.TRAIN_REPORT/(row['name']+'_validation.json')),{'synthetic_errors_must_not_be_read':True})
  write(tmp_path,str(f.TRAIN_REPORT/(row['name']+'_completed.json')),{'status':'completed','epochs':30,'name':row['name'],
   'registration_sha256':f.sha(tmp_path/f.TRAIN_REPORT/'registration.json'),
   'training_summary_sha256':f.sha(tmp_path/directory/'training_summary.json'),
   'checkpoint_sha256':f.sha(tmp_path/directory/'best/model.pt'),'selected_epoch':1})
 baseline={'runs':[{'name':f'{m}_s{s}','mode':m,'seed':s,'config':f'base/{m}_s{s}.json','sha256':'unused'} for m in f.INTERNAL for s in range(3)]}
 def validate(directory):
  calls.append(Path(directory).name)
  return {'completed_epochs':29 if len(calls)==6 else 30,'best_epoch':1}
 trainer=SimpleNamespace(validate_completed=validate,read_package=lambda directory:(directory,{'epoch':1,'config':{}}))
 def loader(name,path):
  if name=='_external_report_registry':return SimpleNamespace(verify=lambda:reg)
  if name=='_external_report_base_campaign':return SimpleNamespace(verify=lambda:baseline)
  return trainer
 original_read=f.read
 def guarded_read(path):
  assert not str(path).endswith('_validation.json'),'Incomplete sixth training run exposed evaluation outcomes'
  return original_read(path)
 monkeypatch.setattr(f,'reporting_gate',lambda root:{'dependencies':{}})
 monkeypatch.setattr(f,'module',loader);monkeypatch.setattr(f,'ledger_module',lambda root:fixtures.ledger)
 monkeypatch.setattr(f,'read',guarded_read)
 monkeypatch.setattr(f,'aggregate',lambda *args,**kwargs:pytest.fail('Incomplete training reached aggregate'))
 with pytest.raises(ValueError,match='Incomplete/incorrect selected'):f.collect(tmp_path)
 assert len(calls)==6
 assert not (tmp_path/f.REPORT).exists()


def test_frozen_reporter_and_review_binding(tmp_path):
 path=write(tmp_path,'frozen.py','immutable')
 registry={'policy':f.POLICY,'dependencies':{'frozen.py':f.sha(path)}}
 p=write(tmp_path,str(f.REG),registry)
 review=write(tmp_path,str(f.REPORT/'source_review.json'),{'status':'passed','registration_sha256':f.sha(p)})
 assert f.reporting_gate(tmp_path)==registry
 path.write_text('modified')
 with pytest.raises(ValueError,match='Changed reporting dependency'):f.reporting_gate(tmp_path)
 path.write_text('immutable');review.write_text('{"status":"passed","registration_sha256":"bad"}')
 with pytest.raises(ValueError,match='approval'):f.reporting_gate(tmp_path)


@pytest.fixture
def external_fixture(tmp_path):
 result,config,root,state=fixtures.fixture.__wrapped__(tmp_path)
 mode=f.OBJECTIVES[0];config.update(mode=mode,epochs=30);state['config']['model_config']['mode']=mode
 state['config']['metadata'].update(training_identity='identity',selection='selection',validation_precision='float32')
 config_path=write(root,'config.json',config);registration=write(root,'registration.json',{'registered':True})
 write(root,str(f.TRAIN_DIR/'evaluate.py'),'synthetic external evaluator')
 stats=write(root,'cache/training_statistics.json',{'fit_split':'train'})
 result.update(mode=mode,schema='adapted_official_dinowm_droid_validation_v1',model_family='adapted_official_dinowm_v1',
  run_name=mode+'_s0',training_objective=mode,registration_sha256=f.sha(registration),config_sha256=f.sha(config_path),
  training_identity='identity',normalization_sha256=f.sha(stats),selection_metric='selection',evaluation_precision='float32',
  upstream_evaluation_source_sha256=result['source_sha256'],source_sha256=f.sha(root/f.TRAIN_DIR/'evaluate.py'))
 return result,config,state,config_path,registration,root


def test_external_wrapper_reuses_full_frozen_population_validator(external_fixture):
 summary=f.validate_external(*external_fixture,ledger=f.ledger_module())
 assert summary['native_mse']==[5.]*10
 # Loading an expanded private namespace did not mutate the frozen fixture one.
 assert set(fixtures.ledger.MODES)==set(f.INTERNAL)


@pytest.mark.parametrize('damage',['source','upstream_source','identity','statistics','config','registration','objective','selection','missing_window','wrong_target'])
def test_external_provenance_or_primitive_damage_rejected(external_fixture,damage):
 args=list(deepcopy(external_fixture));r=args[0]
 fields={'source':'source_sha256','upstream_source':'upstream_evaluation_source_sha256','identity':'training_identity',
  'statistics':'normalization_sha256','config':'config_sha256','registration':'registration_sha256',
  'objective':'training_objective','selection':'selection_metric','wrong_target':'original_target_coordinates'}
 if damage=='missing_window':r['windows'].pop()
 else:r[fields[damage]]='corrupt'
 with pytest.raises(ValueError):f.validate_external(*args,ledger=f.ledger_module())
