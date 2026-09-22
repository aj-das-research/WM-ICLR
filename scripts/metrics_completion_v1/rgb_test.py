"""Synthetic-only tests: no IWS payload access, training or metrics."""
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset
import rgb_core as core
import rgb_train as train
import rgb_data as data
import rgb_finalize as finish


class Tiny(Dataset):
    def __init__(self):
        self.episode_ids=['a','b'];self.x=torch.arange(15,dtype=torch.float32).reshape(5,3)/15
        self.y=(self.x*.3+.1).reshape(5,3,1,1)
    def __len__(self):return 5
    def __getitem__(self,i):return self.x[i],self.y[i],int(i>1)


def tiny_model():return torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.Unflatten(1,(3,1,1)))


def test_official_adapter_forward_and_backward():
    torch.set_num_threads(2)
    model=core.RGBDecoder({'feature_mean':[0.]*6144,'feature_std':[1.]*6144})
    x=torch.zeros(1,6144);out=model(x)
    assert out.shape==(1,3,224,224) and torch.isfinite(out).all()
    out.square().mean().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_invalid_feature_stats_and_shape():
    with pytest.raises(ValueError):core.RGBDecoder({'feature_mean':[0.]*6144,'feature_std':[0.]*6144})


def test_target_resize_and_stride():
    rgb=np.full((2,480,640,3),73,np.uint8)
    x=core.target_pixels(rgb)
    assert x.shape==(2,224,224,3) and x.dtype==np.uint8 and (x==73).all()
    assert core.selected_indices(199).tolist()==list(range(0,199,5))
    assert core.selected_indices(200).tolist()==list(range(0,200,5))
    with pytest.raises(ValueError):core.target_pixels(rgb.astype(np.float32))


def test_reconstruction_trajectory_weighting():
    d=Tiny();m=tiny_model()
    with torch.no_grad():m[0].weight.zero_();m[0].bias.zero_()
    result=train.validate(m,d,2,'cpu')
    errors=d.y.square().mean((1,2,3)).numpy()
    assert result['equal_trajectory_unclipped_mse']==pytest.approx((errors[:2].mean()+errors[2:].mean())/2)
    assert [r['frames'] for r in result['episodes']]==[2,3]


def test_epoch_resume_matches_uninterrupted(tmp_path):
    torch.set_num_threads(2);d=Tiny();identity={'seed':173,'task':'synthetic'}
    train.seed_all(173);first=tiny_model()
    a=train.fit(first,d,d,tmp_path/'full',identity,'cpu',epochs=3,batch_size=2)
    train.seed_all(173);second=tiny_model()
    assert train.fit(second,d,d,tmp_path/'resume',identity,'cpu',epochs=3,batch_size=2,max_seconds=0)['status']=='requeue'
    train.seed_all(173);third=tiny_model()
    b=train.fit(third,d,d,tmp_path/'resume',identity,'cpu',epochs=3,batch_size=2)
    x=torch.load(tmp_path/'full'/'selected.pt',weights_only=False)
    y=torch.load(tmp_path/'resume'/'selected.pt',weights_only=False)
    assert x['selected_epoch']==y['selected_epoch']
    assert all(torch.equal(x['model'][k],y['model'][k]) for k in x['model'])
    assert x['validation']==y['validation']
    assert a['epochs']==b['epochs']==3


def test_resume_changed_identity_rejected(tmp_path):
    d=Tiny();train.fit(tiny_model(),d,d,tmp_path,{'seed':173},'cpu',epochs=2,max_seconds=0)
    with pytest.raises(ValueError,match='changed decoder identity'):
        train.fit(tiny_model(),d,d,tmp_path,{'seed':174},'cpu',epochs=2)


def test_reserved_split_denied_before_files(tmp_path,monkeypatch):
    monkeypatch.setattr(data,'REPORT',tmp_path)
    with pytest.raises(ValueError,match='Reserved'):
        data.RGBPairs('pusht','reserved_official_validation',{'tasks':{} })
    with pytest.raises(ValueError,match='Resource profile'):
        data.RGBPairs('pusht','internal_development',{'tasks':{'pusht':{}}},first_train_frames=1)


def test_login_gpu_denied(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(ValueError,match='Slurm'):train.allocated()


def test_pending_finalizer_writes_nothing(tmp_path,monkeypatch):
    monkeypatch.setattr(finish,'REPORT',tmp_path);monkeypatch.setattr(finish,'checked_registration',lambda:{})
    assert finish.finalize(True)['status']=='pending'
    assert list(tmp_path.iterdir())==[]
    with pytest.raises(ValueError,match='Incomplete'):finish.finalize(False)


def test_registration_review_before_hashing(tmp_path,monkeypatch):
    reg=tmp_path/'registration.json'
    core.atomic_json({'schema':'iws_shared_rgb_decoder_registration_v1','status':'frozen_before_decoder_training',
                     'recipe':core.RECIPE,'dependencies':{'nonexistent':'0'*64}},reg)
    monkeypatch.setattr(core,'REG',reg);monkeypatch.setattr(core,'REPORT',tmp_path)
    with pytest.raises(FileNotFoundError):core.checked_registration()
    core.atomic_json({'status':'passed','registration_sha256':'wrong'},tmp_path/'source_review.json')
    with pytest.raises(ValueError,match='review'):core.checked_registration()


def synthetic_full_grid(tmp_path,monkeypatch):
    regpath=tmp_path/'registration.json';regpath.write_text('synthetic only')
    reg={'tasks':{}}
    for task in core.TASKS:
        target=tmp_path/'targets'/task/'manifest.json';core.atomic_json({'synthetic':True},target)
        path=tmp_path/'runs'/f'{task}_s173';path.mkdir(parents=True)
        identity={'registration_sha256':core.sha(regpath),'task':task,'seed':173,'recipe':core.RECIPE,
                  'statistics_sha256':'a'*64,'target_manifest_sha256':core.sha(target)}
        reg['tasks'][task]={'statistics_sha256':'a'*64,'counts':{'internal_train':{'frames':2},'internal_development':{'frames':2}},
                            'records':[{'episode_id':'dev','split':'internal_development','selected_indices':[0,5]}]}
        history=[]
        for epoch in range(1,31):
            # Tie ensures earliest selector is tested.
            score=1./min(epoch,20)
            val={'equal_trajectory_unclipped_mse':score,'equal_trajectory_clipped_mse':score,
                 'episodes':[{'episode_id':'dev','frames':2,'unclipped_mse':score,'clipped_mse':score}]}
            history.append({'epoch':epoch,'train_frames':2,'development_frames':2,'validation':val})
        package={'kind':'iws_shared_rgb_decoder_v1','identity':identity,'epochs':30,'selected_epoch':20,
                 'validation':history[19]['validation'],'model':{'synthetic':torch.zeros(1)}}
        train.atomic_torch(package,path/'selected.pt')
        summary={'status':'completed','identity':identity,'epochs':30,'selected_epoch':20,
                 'history':history,'reconstruction_quality':history[19]['validation'],'selected_sha256':core.sha(path/'selected.pt')}
        core.atomic_json(summary,path/'training_summary.json')
        marker={'status':'completed','identity':identity,'epochs':30,'selected_epoch':20,
                'summary_sha256':core.sha(path/'training_summary.json'),'selected_sha256':core.sha(path/'selected.pt')}
        core.atomic_json(marker,path/'completion.json')
    monkeypatch.setattr(finish,'REPORT',tmp_path);monkeypatch.setattr(finish,'REG',regpath)
    monkeypatch.setattr(finish,'checked_registration',lambda:reg)
    return reg


def test_complete_grid_and_idempotence(tmp_path,monkeypatch):
    synthetic_full_grid(tmp_path,monkeypatch)
    assert finish.finalize()['status']=='three_decoders_complete'
    before=(tmp_path/'decoder_finalization.json').read_bytes()
    finish.finalize();assert (tmp_path/'decoder_finalization.json').read_bytes()==before
    result=core.read(tmp_path/'decoder_finalization.json')
    assert result['forecast_metrics']['status']=='not_run'
    assert result['LPIPS']['status']=='not_measured'


@pytest.mark.parametrize('mutation',['epochs','frames','selection','identity','payload'])
def test_corrupt_complete_grid_rejected(tmp_path,monkeypatch,mutation):
    synthetic_full_grid(tmp_path,monkeypatch)
    path=tmp_path/'runs'/'pusht_s173';summary=core.read(path/'training_summary.json')
    if mutation=='epochs':summary['history'].pop()
    if mutation=='frames':summary['history'][0]['train_frames']=1
    if mutation=='selection':summary['selected_epoch']=21
    if mutation=='identity':summary['identity']['seed']=0
    if mutation=='payload':(path/'selected.pt').write_bytes(b'corrupt')
    core.atomic_json(summary,path/'training_summary.json')
    marker=core.read(path/'completion.json');marker['summary_sha256']=core.sha(path/'training_summary.json')
    core.atomic_json(marker,path/'completion.json')
    with pytest.raises(ValueError):finish.finalize()
    assert not (tmp_path/'decoder_finalization.json').exists()
