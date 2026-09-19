"""Release gates and portability contracts; synthetic inputs, no new study outputs."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import types

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/publishing' / (name+'.py'))
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


release = load('spatial_release')
runtime = load('spatial_release_runtime')


@pytest.fixture
def completed():
    registry = {'expected_runs':15,'expected_epochs_per_run':30,'runs':[]}
    report = {'registration_sha256':release.EXPECTED_REGISTRY,'status':'passed',
              'scope':'original_validation_development_only','completed_models':15,'epochs_per_model':30,
              'runs':[],'offline_cpu_parity':[],'aggregate':{},'paired_effects':[]}
    for index, mode in enumerate(release.MODES):
        # Include improvements and regressions; neither may suppress a release.
        report['aggregate'][mode] = {key:[float(index+1)]*10 for key in release.METRICS}
        for seed in range(3):
            name = f'{mode}_s{seed}'
            registry['runs'].append({'name':name,'mode':mode,'seed':seed,'config':f'configs/{name}.json'})
            report['runs'].append({'name':name,'selected_epoch':12,'validation':f'reports/real_video_spatial/{name}_validation.json'})
            report['offline_cpu_parity'].append({'name':name,'status':'passed','max_abs_error':0.,
                'relocated_isolated_process':True,'device':'cpu','input_split':'original_validation'})
    for mode in release.MODES:
        if mode == 'transport': continue
        for metric in ('native_mse','original_2x2_mse'):
            for horizon in (5,10):
                ours = report['aggregate']['transport'][metric][horizon-1]
                base = report['aggregate'][mode][metric][horizon-1]
                difference = ours-base
                report['paired_effects'].append({'method':'transport','comparator':mode,'metric':metric,'horizon':horizon,
                    'method_mean':ours,'comparator_mean':base,'method_minus_comparator':difference,
                    'relative_error_reduction_percent':100*(base-ours)/base,
                    'paired_95_percent_interval':[difference-.5,difference+.5],
                    'interval_includes_zero':difference-.5 <= 0 <= difference+.5,
                    'bootstrap_draws':10000,'bootstrap_seed':173,
                    'scope':'exploratory original validation; no multiple-comparison adjustment'})
    return registry, report


def test_complete_gate_retains_negative_outcomes(completed):
    registry, report = completed
    assert any(r['relative_error_reduction_percent'] < 0 for r in report['paired_effects'])
    assert any(r['relative_error_reduction_percent'] > 0 for r in report['paired_effects'])
    release.validate_gate(report,registry,release.EXPECTED_REGISTRY)
    text = release.readme(report)
    assert all(mode in text for mode in release.MODES)
    assert '-200.0000%' in text
    assert 'not RGB' in text and 'without multiplicity adjustment' in text


@pytest.mark.parametrize('mutation',[
    lambda r,g:r.update(status='running'),
    lambda r,g:r.update(completed_models=14),
    lambda r,g:r.update(epochs_per_model=29),
    lambda r,g:r.update(scope='fresh_test'),
    lambda r,g:r.update(registration_sha256='0'*64),
    lambda r,g:r['runs'].pop(),
    lambda r,g:r['runs'].__setitem__(1,deepcopy(r['runs'][0])),
    lambda r,g:g['runs'][0].update(seed=2),
    lambda r,g:g['runs'][0].update(config='../outside.json'),
    lambda r,g:r['runs'][0].update(selected_epoch=True),
    lambda r,g:r['runs'][0].update(selected_epoch=31),
    lambda r,g:r['runs'][0].update(validation='/tmp/substitute.json'),
    lambda r,g:r['offline_cpu_parity'].pop(),
    lambda r,g:r['offline_cpu_parity'][0].update(max_abs_error=1e-8),
    lambda r,g:r['offline_cpu_parity'][0].update(relocated_isolated_process=False),
    lambda r,g:r['aggregate']['transport']['native_mse'].__setitem__(4,float('nan')),
    lambda r,g:r['aggregate']['transport']['native_mse'].__setitem__(4,-1),
    lambda r,g:r['aggregate']['transport']['native_mse'].pop(),
    lambda r,g:r['paired_effects'].pop(),
    lambda r,g:r['paired_effects'][0].update(bootstrap_draws=9999),
    lambda r,g:r['paired_effects'][0].update(paired_95_percent_interval=[2.,1.]),
    lambda r,g:r['paired_effects'][0].update(interval_includes_zero=True),
    lambda r,g:r['paired_effects'][0].update(relative_error_reduction_percent=1),
    lambda r,g:r['paired_effects'][0].update(method_minus_comparator=1),
])
def test_failed_or_altered_completion_rejected(completed, mutation):
    registry, report = completed; mutation(report,registry)
    with pytest.raises(ValueError): release.validate_gate(report,registry,release.EXPECTED_REGISTRY)


def test_zero_baseline_is_not_fabricated_gain(completed):
    registry, report = completed
    for key in release.METRICS: report['aggregate']['autoregressive'][key]=[0.]*10
    for row in report['paired_effects']:
        if row['comparator']=='autoregressive':
            row.update(comparator_mean=0.,method_minus_comparator=3.,relative_error_reduction_percent=None,
                       paired_95_percent_interval=[2.5,3.5])
    release.validate_gate(report,registry,release.EXPECTED_REGISTRY)


def test_relocation_proof_requires_selected_epoch():
    epochs={name:19 for name in release.EXPECTED_NAMES}
    proof={'status':'passed','relocated':True,'device':'cpu','network_attempts':0,
           'models':[{'name':name,'status':'passed','epoch':19,'max_abs_error':0.} for name in epochs]}
    release.validate_relocated_proof(proof,epochs)
    proof['models'][0]['epoch']=18
    with pytest.raises(ValueError): release.validate_relocated_proof(proof,epochs)


def cache_fixture(tmp_path):
    cache=tmp_path/'cache';cache.mkdir()
    release.write(cache/'identity.json',{'encoder':'test-only'})
    release.write(cache/'manifest.json',{'identity':{'encoder':'test-only'}})
    release.write(cache/'training_statistics.json',{'fit_split':'train',
        'normalization':'shared_per_channel_over_train_frames_and_patches',
        'cache_manifest_sha256':release.sha(cache/'manifest.json')})
    throughput=tmp_path/'reports/real_video_spatial/throughput.json'
    release.write(throughput,{'status':'measured','batch_size':128,
        'rows':[{'mode':m,'estimated_seconds_per_30_epoch_run_without_io':60.} for m in release.MODES],
        'estimated_15_run_total_gpu_hours_without_io':.25})
    gate=tmp_path/'reports/real_video_spatial/cache_and_budget_gate.json'
    release.write(gate,{'status':'passed','registration_sha256':release.EXPECTED_REGISTRY,
        'cache_manifest_sha256':release.sha(cache/'manifest.json'),
        'statistics_sha256':release.sha(cache/'training_statistics.json'),'throughput_sha256':release.sha(throughput)})
    return {'cache_root':'cache'},gate,throughput


def test_cache_gate_checks_training_only_stats_and_budget(tmp_path):
    config,gate,throughput=cache_fixture(tmp_path)
    assert len(release.validate_cache_gate(tmp_path,config))==5
    changed=release.read(throughput);changed['estimated_15_run_total_gpu_hours_without_io']=81
    release.write(throughput,changed)
    gate_json=release.read(gate);gate_json['throughput_sha256']=release.sha(throughput);release.write(gate,gate_json)
    with pytest.raises(ValueError,match='budget'):release.validate_cache_gate(tmp_path,config)


def test_cache_identity_change_fails(tmp_path):
    config,_,_=cache_fixture(tmp_path)
    release.write(tmp_path/'cache/identity.json',{'encoder':'altered'})
    with pytest.raises(ValueError,match='identity'):release.validate_cache_gate(tmp_path,config)


@pytest.mark.parametrize('name',['/abs','../up','a/../b','a//b','a/./b','a\\b','a\x00b',''])
def test_unsafe_paths_rejected(name):
    with pytest.raises(ValueError):release.safe_relative(name)


def scanner():
    return types.SimpleNamespace(TOKEN_RULES={'test-token':re.compile(b'unittest-secret-[A-Z]{12}')})


def test_scan_boundary_and_symlink_rejected(tmp_path):
    file=tmp_path/'example.bin';token=b'unittest-'+b'secret-'+b'Z'*12
    file.write_bytes(b'x'*((1<<20)-9)+token)
    with pytest.raises(ValueError,match='value redacted'):release.inventory(tmp_path,scanner())
    file.write_bytes(b'normal');(tmp_path/'linked').symlink_to(file)
    with pytest.raises(ValueError,match='Forbidden'):release.inventory(tmp_path,scanner())


@pytest.mark.parametrize('name',['training_state.pt','.env','.git/config','private.credentials'])
def test_disallowed_payload_names(tmp_path,name):
    path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'not public')
    with pytest.raises(ValueError):release.inventory(tmp_path,scanner())


def bundle_fixture(tmp_path):
    bundle=tmp_path/release.BUNDLE;bundle.mkdir()
    for name in ('README.md','MODEL_CARD.md','verification/offline_packages.json'):
        path=bundle/name;path.parent.mkdir(exist_ok=True);path.write_text('{}\n')
    manifest={'files':release.inventory(bundle,scanner())}
    release.write(bundle/'manifest.json',manifest)
    return bundle,manifest


def test_archive_is_exact_deterministic_and_detects_changed_bytes(tmp_path):
    bundle,manifest=bundle_fixture(tmp_path)
    names=release.archive(bundle,manifest,tmp_path)
    digest=release.sha(tmp_path/names[0]);(tmp_path/names[0]).unlink()
    assert release.archive(bundle,manifest,tmp_path)==names
    assert release.sha(tmp_path/names[0])==digest
    (bundle/'README.md').write_text('changed')
    with pytest.raises(ValueError,match='bytes differ'):release.archive(bundle,manifest,tmp_path)


@pytest.mark.parametrize('bad_name',[release.BUNDLE+'/../outside','other/file',release.BUNDLE+'/unlisted'])
def test_existing_unsafe_archive_is_rejected(tmp_path,bad_name):
    bundle,manifest=bundle_fixture(tmp_path)
    with tarfile.open(tmp_path/(release.BUNDLE+'.tar.gz'),'w:gz') as output:
        info=tarfile.TarInfo(bad_name);info.size=4;output.addfile(info,io.BytesIO(b'bad!'))
    with pytest.raises(ValueError):release.archive(bundle,manifest,tmp_path)


def test_omitted_archive_file_is_rejected(tmp_path):
    bundle,manifest=bundle_fixture(tmp_path)
    with tarfile.open(tmp_path/(release.BUNDLE+'.tar.gz'),'w:gz'): pass
    with pytest.raises(ValueError,match='incomplete'):release.archive(bundle,manifest,tmp_path)


def package_fixture(tmp_path):
    from shiftwm.real_video_spatial.model import SpatialWorldModel
    torch.set_num_threads(1);torch.manual_seed(817)
    model=SpatialWorldModel({'mode':'transport','hidden_dim':6,'depth':1,'context_dim':2,'context_hidden':6},
        [0.]*6144,[1.]*6144,[0.]*35,[1.]*35).eval()
    with torch.no_grad():
        model.output_projection[-1].weight.normal_(0,.05)
        model.output_projection[-1].bias.normal_(0,.01)
    config={**model.package_config,'package_kind':runtime.PACKAGE_KIND,'metadata':{'parameter_counts':{
        'total':sum(p.numel() for p in model.parameters()),
        'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}}}
    release.write(tmp_path/'config.json',config)
    torch.save({'config':config,'state_dict':model.state_dict(),'epoch':23},tmp_path/'model.pt')
    release.write(tmp_path/'package_manifest.json',{'format_version':1,'package_kind':runtime.PACKAGE_KIND,
        'files':{name:release.sha(tmp_path/name) for name in ('model.pt','config.json')}})
    return model


def test_portable_loader_exact_and_weights_tampering_rejected(tmp_path):
    original=package_fixture(tmp_path);loaded,state=runtime.load_package(tmp_path)
    inputs=(torch.randn(1,3,6144),torch.randn(1,2,35),torch.randn(1,10,35))
    with torch.inference_mode():
        expected=original.predict(*inputs);actual=loaded.predict(*inputs)
    assert state['epoch']==23 and torch.equal(expected,actual)
    assert not torch.equal(expected,inputs[0][:,-1:].expand_as(expected))
    with (tmp_path/'model.pt').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='identity'):runtime.load_package(tmp_path)


def test_portable_loader_rejects_optimizer_and_false_counts(tmp_path):
    package_fixture(tmp_path);manifest=release.read(tmp_path/'package_manifest.json')
    manifest['files']['training_state.pt']='0'*64;release.write(tmp_path/'package_manifest.json',manifest)
    with pytest.raises(ValueError,match='inference-only'):runtime.load_package(tmp_path)
    del manifest['files']['training_state.pt'];config=release.read(tmp_path/'config.json')
    config['metadata']['parameter_counts']['trainable']=1
    state=torch.load(tmp_path/'model.pt',weights_only=True);state['config']=config
    torch.save(state,tmp_path/'model.pt');release.write(tmp_path/'config.json',config)
    manifest['files']={name:release.sha(tmp_path/name) for name in ('model.pt','config.json')}
    release.write(tmp_path/'package_manifest.json',manifest)
    with pytest.raises(ValueError,match='counts'):runtime.load_package(tmp_path)


def encoder_fixture(tmp_path,monkeypatch):
    calls=[]
    class FakeEncoder:
        @classmethod
        def from_pretrained(cls,path,local_files_only):
            assert Path(path)==tmp_path and local_files_only is True
            return cls()
        def to(self,device):return self
        def eval(self):return self
        def requires_grad_(self,value):assert value is False;return self
        def __call__(self,pixel_values):
            calls.append(pixel_values.clone())
            spatial=torch.nn.functional.adaptive_avg_pool2d(pixel_values.mean(1,keepdim=True),(16,16))
            patches=spatial.flatten(2).transpose(1,2).expand(-1,-1,384)
            return types.SimpleNamespace(last_hidden_state=torch.cat((patches[:,:1],patches),1))
    fake=types.ModuleType('transformers');fake.Dinov2Model=FakeEncoder
    monkeypatch.setitem(sys.modules,'transformers',fake)
    names=('config.json','preprocessor_config.json','model.safetensors','README.md')
    for name in names:(tmp_path/name).write_bytes(name.encode())
    provenance={'repo':'facebook/dinov2-small','revision':'ed25f3a31f01632728cabb09d1542f84ab7b0056',
        'files':[{'file':name,'bytes':(tmp_path/name).stat().st_size,'sha256':release.sha(tmp_path/name)} for name in names]}
    release.write(tmp_path/'provenance.json',provenance)
    return calls


def test_encoder_resize_then_normalize_and_channel_major_pool(tmp_path,monkeypatch):
    calls=encoder_fixture(tmp_path,monkeypatch)
    images=torch.linspace(0,1,2*3*17*23).reshape(2,3,17,23)
    encoded=runtime.encode_rgb(images,tmp_path)
    resized=torch.nn.functional.interpolate(images,(224,224),mode='bilinear',align_corners=False,antialias=True)
    expected=(resized-resized.new_tensor([.485,.456,.406])[None,:,None,None])/resized.new_tensor([.229,.224,.225])[None,:,None,None]
    assert torch.equal(calls[0],expected)
    pooled=torch.nn.functional.adaptive_avg_pool2d(torch.nn.functional.adaptive_avg_pool2d(expected.mean(1,keepdim=True),(16,16)),(4,4))
    assert torch.equal(encoded,pooled.expand(-1,384,-1,-1).flatten(1))
    assert encoded.shape==(2,6144)


def test_encoder_empty_provenance_and_out_of_range_inputs_rejected(tmp_path,monkeypatch):
    calls=encoder_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='finite float'):runtime.encode_rgb(torch.full((1,3,16,16),2.),tmp_path)
    provenance=release.read(tmp_path/'provenance.json');provenance['files']=[];release.write(tmp_path/'provenance.json',provenance)
    with pytest.raises(ValueError,match='inventory'):runtime.encode_rgb(torch.zeros(1,3,16,16),tmp_path)
    assert calls==[]
