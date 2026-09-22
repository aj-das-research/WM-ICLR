"""Synthetic preaccess contracts; no simulator datasets or learned weights read."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from planning_common import require
from planning_data import PlanningDataset,grouped_rows,statistics
from planning_train import make_model,from_config,base


def stats():
    return {'feature_mean':[0.]*6144,'feature_std':[1.]*6144,'action_mean':[0.]*10,'action_std':[1.]*10}


def config(mode):
    return {'mode':mode,'action_dim':10,'depth':1}


def test_grouped_native_action_alignment():
    row={'offset':100,'length':42};local,global_rows=grouped_rows(row)
    assert local.tolist()==list(range(0,42,5))
    assert global_rows.tolist()==list(range(100,142,5))
    native=np.arange(82).reshape(41,2)[:local[-1]]
    actions=native.reshape(-1,10)
    assert np.array_equal(actions[2].reshape(5,2),native[10:15])


def test_windows_never_cross_episode():
    episodes=[{'features':np.full((10,6144),i,dtype=np.float32),'actions':np.full((9,10),i,dtype=np.float32)}for i in range(2)]
    data=PlanningDataset(episodes)
    assert len(data)==6
    for row in data:
        assert row['features'].shape==(8,6144) and row['actions'].shape==(7,10)
        assert torch.unique(row['features']).numel()==1


def test_training_statistics_shared_channels():
    rng=np.random.default_rng(2)
    episode={'features':rng.normal(size=(8,6144)).astype(np.float32),'actions':rng.normal(size=(7,10)).astype(np.float32)}
    result=statistics([episode]);means=np.asarray(result['feature_mean']).reshape(384,16)
    assert np.array_equal(means,means[:,:1].repeat(16,1))
    expected=episode['features'].reshape(-1,384,16).transpose(0,2,1).reshape(-1,384).mean(0,dtype=np.float64)
    np.testing.assert_allclose(means[:,0],expected,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('mode',['transport','autoregressive','bounded_additive','unbounded_transport'])
def test_current_model_raw_shapes_and_strict_reload(mode):
    torch.set_num_threads(2);torch.manual_seed(1)
    model=make_model(config(mode),stats()).eval()
    package={**model.package_config,'package_kind':'shiftwm_current_spatial_simulator_v1'}
    restored=from_config(package);restored.load_state_dict(model.state_dict(),strict=True);restored.eval()
    x=torch.randn(2,3,6144);past=torch.randn(2,2,10);future=torch.randn(2,5,10)
    with torch.inference_mode():a=model.predict(x,past,future);b=restored.predict(x,past,future)
    assert a.shape==(2,5,6144) and torch.isfinite(a).all() and torch.equal(a,b)
    with pytest.raises(ValueError):from_config({**package,'package_kind':'historical_simulator'})


def test_causal_command_prefix():
    torch.set_num_threads(2)
    model=make_model(config('transport'),stats()).eval()
    with torch.no_grad():model.output_projection[-1].weight.normal_(std=.01)
    x=torch.randn(1,3,6144);past=torch.randn(1,2,10);future=torch.randn(1,5,10)
    with torch.inference_mode():
        a=model.predict(x,past,future);changed=future.clone();changed[:,2:]+=50;b=model.predict(x,past,changed)
    torch.testing.assert_close(a[:,:2],b[:,:2],rtol=1e-6,atol=1e-6)


def test_modes_match_initialization():
    states={}
    for mode in('transport','autoregressive','bounded_additive','unbounded_transport'):
        torch.manual_seed(3);states[mode]=make_model(config(mode),stats()).state_dict()
    for mode in states:
        assert states[mode].keys()==states['transport'].keys()
        assert all(torch.equal(v,states['transport'][k])for k,v in states[mode].items())


def test_goal_cost_has_no_state_input():
    from planning_evaluate import SpatialGoalCost
    model=make_model(config('transport'),stats()).eval();cost=SpatialGoalCost(model)
    actions=torch.randn(1,2,5,10)
    info={'history_features':torch.randn(1,2,3,6144),'past_actions':torch.randn(1,2,2,10),'goal_features':torch.randn(1,2,6144)}
    with torch.inference_mode():out=cost.get_cost(info,actions)
    assert out.shape==(1,2) and cost.candidate_evaluations==2
    assert cost.to_native(torch.full((1,10),20.)).max()==1
    assert cost.to_native(torch.full((1,10),-20.)).min()==-1


def test_planner_goal_does_not_exceed_remaining_budget():
    from planning_evaluate import planning_horizon
    assert planning_horizon(10)==5
    assert planning_horizon(35)==3
    assert planning_horizon(45)==1
    for n in(0,9,11,50):
        with pytest.raises(ValueError):planning_horizon(n)


def test_bootstrap_preserves_adverse_differences():
    from planning_finalize import ci
    value=ci(np.full((3,100),-.25),draws=100)
    assert value['mean']==-.25 and value['ci95']==[-.25,-.25]
    with pytest.raises(ValueError):ci(np.zeros((2,100)))


def test_no_eligible_cases_are_undefined_not_zero_success():
    from planning_finalize import eligible_success_ci
    result=eligible_success_ci(np.ones((3,100)),np.zeros((3,100),dtype=bool),draws=100)
    assert result['mean'] is None and result['ci95'] is None
    mask=np.zeros((3,100),dtype=bool);mask[:,:10]=True
    result=eligible_success_ci(np.ones((3,100)),mask,draws=100)
    assert result['mean']==1. and result['ci95']==[1.,1.]


def test_cache_training_rejects_test_before_read(monkeypatch):
    import planning_data
    monkeypatch.setattr(planning_data,'read',lambda unused:pytest.fail('Read before membership gate'))
    with pytest.raises(ValueError):planning_data.load_cache('pusht','test','x')


def test_development_selection_equal_episode_not_window_weighted():
    from planning_train import epoch_pass
    class Fake(torch.nn.Module):
        def forward(self,batch):
            y=batch['features'][:,3:]
            return {'standardized_predictions':y,'standardized_targets':torch.zeros_like(y)}
    rows=[]
    for eid,value in[(0,1.),(0,1.),(0,1.),(1,3.)]:
        rows.append({'features':torch.full((8,6144),value),'actions':torch.zeros(7,10),'episode_index':eid})
    class Data(torch.utils.data.Dataset):
        episodes=[0,1]
        def __len__(self):return len(rows)
        def __getitem__(self,i):return rows[i]
    out=epoch_pass(Fake(),torch.utils.data.DataLoader(Data(),batch_size=4),torch.device('cpu'))
    assert out['standardized_mse']==5. and out['window_weighted_mse']==3.


def test_exact_model_sources_not_copied_or_replaced():
    from shiftwm.real_video_spatial.model import SpatialWorldModel
    from shiftwm.real_video_spatial_components.model import ComponentWorldModel
    assert type(make_model(config('transport'),stats()))is SpatialWorldModel
    assert type(make_model(config('autoregressive'),stats()))is SpatialWorldModel
    assert type(make_model(config('bounded_additive'),stats()))is ComponentWorldModel
    assert type(make_model(config('unbounded_transport'),stats()))is ComponentWorldModel


def test_epoch_and_case_continuation_fail_closed(monkeypatch):
    import planning_campaign as c
    calls=[]
    monkeypatch.setattr(c.subprocess,'run',lambda args,**kw:calls.append(args))
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setenv('SLURM_RESTART_COUNT','4')
    assert c.request_continuation()=='123' and calls==[['scontrol','requeue','123']]
    monkeypatch.setenv('SLURM_RESTART_COUNT','5')
    with pytest.raises(ValueError):c.request_continuation()
    assert len(calls)==1
    monkeypatch.delenv('SLURM_JOB_ID')
    with pytest.raises(ValueError):c.request_continuation()
