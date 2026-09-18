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
    return SpatialWorldModel({'mode':mode,'depth':1},mean,std,torch.zeros(35),torch.ones(35)).eval()


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
