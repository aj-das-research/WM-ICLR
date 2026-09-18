"""Causal/coordinate/checkpoint contracts for the new spatial development family."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import torch
import pytest
from shiftwm.real_video_spatial.model import SpatialWorldModel, from_config, pool_to_original_2x2
from shiftwm.real_video_spatial.data import validate_spatial_manifest, SpatialDataset

ROOT=Path(__file__).resolve().parents[1]


def make(mode='transport'):
    torch.manual_seed(42)
    mean=torch.arange(384).float().repeat_interleave(16)/1000
    std=torch.linspace(.5,2,384).repeat_interleave(16)
    model=SpatialWorldModel({'mode':mode,'depth':1},mean,std,torch.zeros(35),torch.ones(35)).eval()
    # Nonzero learned-like heads make causal checks informative for additive
    # arms too; their production zero-head initialization would be vacuous.
    with torch.no_grad():
        model.output_projection[-1].weight.normal_(0,.02)
        model.output_projection[-1].bias.normal_(0,.01)
        model.context_adapter.affine.weight.normal_(0,.01)
        model.context_adapter.affine.bias.normal_(0,.01)
    return model


def inputs():
    torch.manual_seed(7)
    return torch.randn(2,3,6144),torch.randn(2,2,35),torch.randn(2,4,35)


@pytest.mark.parametrize('mode',SpatialWorldModel.MODES)
def test_prefix_causality_and_query_independence(mode):
    torch.set_num_threads(2); m=make(mode); support,past,future=inputs()
    with torch.no_grad():
        prediction=m.predict(support,past,future)
        changed=future.clone(); changed[:,2:]+=100
        torch.testing.assert_close(prediction[:,:2],m.predict(support,past,changed)[:,:2],rtol=0,atol=0)
        torch.testing.assert_close(prediction[:,:2],m.predict(support,past,future[:,:2]),rtol=0,atol=0)
        frames=torch.cat((support,torch.randn(2,4,6144)),1); actions=torch.cat((past,future),1)
        before=m({'features':frames,'actions':actions})['predictions']; frames[:,3:]+=200
        torch.testing.assert_close(before,m({'features':frames,'actions':actions})['predictions'],rtol=0,atol=0)


def test_action_free_invariance():
    m=make('action_free'); s,p,a=inputs()
    with torch.no_grad(): torch.testing.assert_close(m.predict(s,p,a),m.predict(s,p+200,a-200),rtol=0,atol=0)


def test_transport_simplex_bound_and_nonzero_gate_gradient():
    m=make(); s,p,a=inputs(); z=m.normalize_features(s)
    result,details=m._predict_normalized(z,p,a,True)
    anchor=m.tokens(z)[:,-1]; bound=anchor.abs().amax((1,2))[:,None,None]+m.config.innovation_bound
    assert torch.all(result.abs()<=bound+1e-5)
    for d in details:
        torch.testing.assert_close(d['transport'].sum(-1),torch.ones(2,16))
        assert (d['transport']>=0).all() and ((d['gate']>0)&(d['gate']<1)).all()
    loss=(result-torch.randn_like(result)).square().mean(); loss.backward()
    assert m.gate.bias.grad is not None and m.gate.bias.grad.abs().sum()>0
    assert m.transport_query.weight.grad.abs().sum()>0
    assert m.output_projection[-1].weight.grad.abs().sum()>0


def test_grid_layout_and_pooling_are_channel_major():
    m=make(); x=torch.arange(6144).float()[None,None]
    assert m.tokens(x)[0,0,1,2]==x[0,0,2*16+1]
    torch.testing.assert_close(m.flatten(m.tokens(x)),x,rtol=0,atol=0)
    expected=torch.nn.functional.avg_pool2d(x.reshape(1,384,4,4),2).flatten(1)
    torch.testing.assert_close(pool_to_original_2x2(x)[:,0],expected,rtol=0,atol=0)


def test_reject_position_specific_statistics():
    m=make(); cfg=m.package_config; cfg['feature_mean'][1]+=.1
    with pytest.raises(ValueError,match='shared per-channel'): from_config(cfg)


def test_package_relocated_cpu_parity(tmp_path):
    torch.set_num_threads(2); m=make(); s,p,a=inputs(); initial=m.predict(s,p,a).detach()
    config=m.package_config; src=tmp_path/'original'; src.mkdir()
    (src/'config.json').write_text(json.dumps(config)); torch.save(m.state_dict(),src/'model.pt')
    relocated=tmp_path/'different'/'model'; relocated.parent.mkdir(); shutil.copytree(src,relocated); shutil.rmtree(src)
    restored=from_config(json.loads((relocated/'config.json').read_text()))
    restored.load_state_dict(torch.load(relocated/'model.pt',map_location='cpu',weights_only=True)); restored.eval()
    torch.testing.assert_close(initial,restored.predict(s,p,a),rtol=0,atol=0)


def test_test_split_rejected_before_reading_files(tmp_path):
    with pytest.raises(ValueError,match='rejects test'): SpatialDataset(tmp_path,'test')


def test_shared_training_statistics_do_not_treat_patch_positions_as_separate_features():
    from shiftwm.real_video_spatial.data import statistics_from_episodes
    values=torch.arange(2*6144).reshape(2,6144).float().numpy()
    actions=torch.ones(1,35).numpy()
    actual=statistics_from_episodes([{'features':values,'actions':actions},{'features':values+1,'actions':actions+1}])
    import numpy as np
    points=np.concatenate([values.reshape(2,384,16).transpose(0,2,1).reshape(-1,384),
                           (values+1).reshape(2,384,16).transpose(0,2,1).reshape(-1,384)])
    np.testing.assert_allclose(np.array(actual['feature_mean']).reshape(384,16)[:,0],points.mean(0))
    np.testing.assert_allclose(np.array(actual['feature_std']).reshape(384,16)[:,0],points.astype(np.float64).std(0,ddof=1))
    assert actual['counts']=={'feature':64,'action':2}


def test_manifest_rejects_any_test_row_even_if_not_selected():
    manifest={'status':'complete','feature_dim':6144,'action_dim':35,'episodes':[
        {'episode_id':s,'session_id':s,'split':s,'cameras':{'exterior_image_1_left':{}}} for s in ('train','val','test')]}
    with pytest.raises(ValueError,match='held-out'): validate_spatial_manifest(manifest)


def trainer():
    spec=importlib.util.spec_from_file_location('test_spatial_trainer',ROOT/'scripts/real_video_spatial/train.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def test_actual_atomic_checkpoint_package_relocated_cpu(tmp_path):
    t=trainer();m=make();optimizer=torch.optim.AdamW(m.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,30)
    counts={'total':sum(p.numel() for p in m.parameters()),'trainable':sum(p.numel() for p in m.parameters() if p.requires_grad)}
    t.save_package(m,tmp_path/'best',optimizer=optimizer,scheduler=scheduler,epoch=1,step=2,best_metric=1.,
        metadata={'training_identity':'fixture','parameter_counts':counts},generator=torch.Generator(),history=[])
    shutil.copytree(tmp_path/'best',tmp_path/'relocated');(tmp_path/'best').unlink()
    loaded,state=t.load_package(tmp_path/'relocated','cpu')
    assert state['config']['package_kind']=='shiftwm_real_video_spatial_v1'
    s,p,a=inputs()
    with torch.no_grad(): torch.testing.assert_close(m.predict(s,p,a),loaded.predict(s,p,a),rtol=0,atol=0)


def test_window_weighted_validation_selection_records_episode_diagnostic():
    t=trainer()
    class Fixture(torch.nn.Module):
        def forward(self,b):
            target=torch.zeros(len(b['features']),10,6144)
            pred=b['value'][:,None,None].expand_as(target)
            return {'loss':-torch.ones(()),'standardized_predictions':pred,'standardized_targets':target}
    batch={'features':torch.zeros(3,13,6144),'actions':torch.zeros(3,12,35),'value':torch.tensor([1.,1.,3.]),'episode_index':torch.tensor([0,0,1])}
    result=t.epoch_pass(Fixture(),[batch],torch.device('cpu'))
    assert result['standardized_mse']==pytest.approx(11/3)
    assert result['equal_episode_diagnostic_mse']==5


def test_full_thirty_epoch_new_model_lifecycle_and_resume(tmp_path):
    t=trainer();torch.set_num_threads(1)
    class Fixture(torch.utils.data.Dataset):
        def __init__(self):
            generator=torch.Generator().manual_seed(67)
            self.x=torch.randn(1,13,6144,generator=generator);self.a=torch.randn(1,12,35,generator=generator)
        def __len__(self):return 1
        def __getitem__(self,i):return {'features':self.x[i],'actions':self.a[i],'episode_index':i}
    def small():
        t.base.seed_everything(0)
        return SpatialWorldModel({'mode':'anchored_additive','hidden_dim':6,'depth':1,'context_dim':4,'context_hidden':8},
                                 [0.]*6144,[1.]*6144,[0.]*35,[1.]*35)
    def config(path):return {'epochs':30,'seed':0,'mode':'anchored_additive','output_dir':str(path),'device':'cpu','batch_size':1,
        'lr':.001,'min_lr':.00001,'weight_decay':.01,'grad_clip':1.,'bf16':False,'train_horizon':10,'validation_horizon':10}
    d=Fixture();cfg=config(tmp_path/'complete');identity={'scientific_config':t.base.scientific_config(cfg),'dependencies':{}}
    result=t.base.fit(small(),cfg,d,d,identity);assert result['completed_epochs']==30
    other=config(tmp_path/'resumed');other['max_runtime_seconds']=1e-12
    interrupted=t.base.fit(small(),other,d,d,{'scientific_config':t.base.scientific_config(other),'dependencies':{}})
    assert interrupted['completed_epochs']==1
    other.pop('max_runtime_seconds');other['resume_if_present']=True
    resumed=t.base.fit(small(),other,d,d,{'scientific_config':t.base.scientific_config(other),'dependencies':{}})
    assert resumed['completed_epochs']==30
    first=t.read_package(tmp_path/'complete/last')[1];second=t.read_package(tmp_path/'resumed/last')[1]
    assert first['history']==second['history']
    for k,v in first['state_dict'].items():torch.testing.assert_close(v,second['state_dict'][k],rtol=0,atol=0)


def test_exporter_isolated_process_offline_reload(tmp_path,monkeypatch):
    t=trainer();m=make();optimizer=torch.optim.AdamW(m.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,30)
    counts={'total':sum(p.numel() for p in m.parameters()),'trainable':sum(p.numel() for p in m.parameters() if p.requires_grad)}
    source=tmp_path/'source'
    t.save_package(m,source/'best',optimizer=optimizer,scheduler=scheduler,epoch=30,step=30,best_metric=1.,
        metadata={'training_identity':'fixture','parameter_counts':counts},generator=torch.Generator(),history=[])
    spec=importlib.util.spec_from_file_location('test_spatial_campaign',ROOT/'scripts/real_video_spatial/campaign.py')
    campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)
    class Data:
        def __init__(self,*args,**kwargs): pass
        def __getitem__(self,index):
            torch.manual_seed(2)
            return {'features':torch.randn(13,6144),'actions':torch.randn(12,35)}
    monkeypatch.setattr(campaign,'SpatialDataset',Data)
    destination=tmp_path/'exported'
    proof=campaign.offline_parity({'output_dir':str(source),'cache_root':'unused'},destination)
    assert proof['relocated_isolated_process'] and proof['max_abs_error']==0
    assert set(p.name for p in destination.iterdir())=={'model.pt','config.json','package_manifest.json'}


def test_campaign_frozen_dependency_rejects_changes(tmp_path,monkeypatch):
    from shiftwm.real_video.data import sha256
    spec=importlib.util.spec_from_file_location('test_spatial_registry',ROOT/'scripts/real_video_spatial/campaign.py')
    campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)
    source=tmp_path/'source.py';source.write_text('original')
    rows=[]
    for mode in SpatialWorldModel.MODES:
        for seed in (0,1,2):
            name=f'{mode}_s{seed}';config=tmp_path/(name+'.json');config.write_text(json.dumps({'mode':mode,'seed':seed}))
            rows.append({'name':name,'mode':mode,'seed':seed,'config':config.name,'sha256':sha256(config)})
    registration={'expected_runs':15,'runs':rows,'dependencies':{'source.py':sha256(source)}}
    registry=tmp_path/'registration.json';registry.write_text(json.dumps(registration))
    monkeypatch.setattr(campaign,'ROOT',tmp_path);monkeypatch.setattr(campaign,'REG',registry)
    assert campaign.verify()['expected_runs']==15
    duplicate={**registration,'runs':[rows[0]]*15};registry.write_text(json.dumps(duplicate))
    with pytest.raises(ValueError,match='each mode/seed'):campaign.verify()
    registry.write_text(json.dumps(registration))
    source.write_text('modified')
    with pytest.raises(ValueError,match='Frozen spatial dependency'):campaign.verify()


def test_extractor_never_opens_test_payload_and_preserves_original_coordinates(tmp_path,monkeypatch):
    import numpy as np
    import transformers
    from types import SimpleNamespace
    from shiftwm.real_video.data import sha256
    from shiftwm.real_video_spatial.features import extract
    processed=tmp_path/'processed';old=tmp_path/'old';encoder=tmp_path/'encoder';out=tmp_path/'spatial'
    for p in (processed,old,encoder):p.mkdir()
    grid=torch.arange(256*384).float().reshape(1,256,384)/1000
    old_features=torch.nn.functional.adaptive_avg_pool2d(grid.transpose(1,2).reshape(1,384,16,16),(2,2)).flatten(1).repeat(3,1).numpy()
    episodes=[];old_episodes=[]
    for split in ('train','val','test'):
        name=split+'.npz';actions=np.zeros((2,35),np.float32);indices=np.array([0,5,10])
        if split!='test':
            np.savez(processed/name,images=np.zeros((3,8,8,3),np.uint8),actions=actions,frame_indices=indices)
            np.savez(old/name,features=old_features,actions=actions,frame_indices=indices)
        episodes.append({'episode_id':split,'session_id':split,'split':split,'steps':3,
                         'cameras':{'exterior_image_1_left':{'file':name,'sha256':sha256(processed/name) if split!='test' else 'must-never-be-opened'}}})
        if split!='test':old_episodes.append({'episode_id':split,'session_id':split,'split':split,'steps':3,
            'cameras':{'exterior_image_1_left':{'file':name,'sha256':sha256(old/name)}}})
    (processed/'manifest.json').write_text(json.dumps({'status':'complete','episodes':episodes}))
    digest=sha256(processed/'manifest.json')
    (processed/'data_audit.json').write_text(json.dumps({'status':'passed','dataset_manifest_sha256':digest}))
    (old/'manifest.json').write_text(json.dumps({'identity':{'dataset_manifest_sha256':digest},'episodes':old_episodes}))
    (encoder/'config.json').write_text('{}')
    (encoder/'provenance.json').write_text(json.dumps({'files':[{'file':'config.json','sha256':sha256(encoder/'config.json')}]}))
    class Frozen(torch.nn.Module):
        @classmethod
        def from_pretrained(cls,*args,**kwargs): return cls()
        def forward(self,pixel_values):
            patches=grid.repeat(len(pixel_values),1,1)
            return SimpleNamespace(last_hidden_state=torch.cat((torch.zeros(len(pixel_values),1,384),patches),1))
    from types import ModuleType
    fake=ModuleType('transformers');fake.__version__=transformers.__version__;fake.Dinov2Model=Frozen
    monkeypatch.setitem(sys.modules,'transformers',fake)
    extract(processed,out,encoder,batch_size=32,device='cpu',original_cache=old)
    manifest=json.loads((out/'manifest.json').read_text())
    assert {e['split'] for e in manifest['episodes']}=={'train','val'}
    assert not (processed/'test.npz').exists() and not (out/'episodes/exterior_image_1_left/test.npz').exists()
    assert manifest['feature_dim']==6144
    assert max(e['cameras']['exterior_image_1_left']['original_2x2_parity_max_abs'] for e in manifest['episodes'])<2e-5


def test_cross_resolution_metric_uses_exact_original_targets(tmp_path,monkeypatch):
    from shiftwm.real_video.data import sha256
    spec=importlib.util.spec_from_file_location('test_spatial_evaluator',ROOT/'scripts/real_video_spatial/evaluate.py')
    evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
    old=tmp_path/'old';cache=tmp_path/'spatial';run=tmp_path/'run'
    for p in (old,cache,run/'best'):p.mkdir(parents=True)
    for p in (old,cache):(p/'manifest.json').write_text('{}')
    (old/'training_statistics.json').write_text(json.dumps({'fit_split':'train','feature_std':[1e-5]*1536,
                                                           'cache_manifest_sha256':sha256(old/'manifest.json')}))
    (run/'best/model.pt').write_text('fixture')
    class Data(torch.utils.data.Dataset):
        def __init__(self,path,*args,**kwargs):
            d=1536 if Path(path).name=='old' else 6144
            value=1e-5 if d==1536 else 0.
            x=torch.full((13,d),value).numpy()
            self.episodes=[{'episode_id':'e','session_id':'s','features':x}];self.windows=[(0,0)]
        def __len__(self):return 1
        def __getitem__(self,i):return {'features':torch.from_numpy(self.episodes[0]['features']),
                                      'actions':torch.zeros(12,35),'episode_index':0,'window_start':0}
    class Model:
        feature_std=torch.ones(6144)
        def predict(self,support,past,future):return torch.zeros(len(support),10,6144)
    state={'epoch':30,'config':{'metadata':{'parameter_counts':{'total':1,'trainable':1}}}}
    monkeypatch.setattr(evaluation,'SpatialDataset',Data);monkeypatch.setattr(evaluation,'RealVideoDataset',Data)
    monkeypatch.setattr(evaluation.train,'validate_completed',lambda *a,**k: None)
    monkeypatch.setattr(evaluation.train,'load_package',lambda *a,**k:(Model(),state))
    result=evaluation.evaluate({'device':'cpu','cpu_threads':1,'cache_root':str(cache),'original_cache':str(old),
                               'output_dir':str(run),'batch_size':1,'mode':'transport','seed':0},tmp_path/'result.json')
    assert result['summary']['native_mse']==[0.]*10
    assert result['summary']['original_2x2_mse']==[1.]*10
    assert result['summary']['original_2x2_persistence_mse']==[0.]*10
    assert len(result['windows'])==1
