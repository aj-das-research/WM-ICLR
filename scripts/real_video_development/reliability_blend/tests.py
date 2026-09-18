import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import core
spec=importlib.util.spec_from_file_location("reliability_runner",HERE/"run.py")
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
torch.set_num_threads(1)


class Toy(torch.nn.Module):
    def __init__(self,slope):
        super().__init__();self.slope=slope
        self.register_buffer("feature_std",torch.ones(4))
        self.seen=[]
    def predict(self,support,past,future):
        self.seen.append((support.clone(),past.clone(),future.clone()))
        assert support.shape[1]==3 and past.shape[1]==2
        return support[:,-1:]+self.slope*torch.cumsum(future[:,:,:1],1).expand(-1,-1,4)


def inputs():
    return torch.arange(23.).view(1,23,1).repeat(1,1,4),torch.ones(1,22,7)


def test_prefix_forecasts_cannot_use_backtest_targets():
    z,u=inputs();donor=Toy(.8)
    first=core.prefix_forecast(donor,z[:,:13],u[:,:12])
    changed=z.clone();changed[:,3:13]+=1000
    second=core.prefix_forecast(donor,changed[:,:13],u[:,:12])
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    assert all(torch.equal(call[0],z[:,:3]) for call in donor.seen)


def test_one_step_backtests_use_only_preceding_observations():
    z,u=inputs();donor=Toy(.8)
    core.prefix_forecast(donor,z[:,:13],u[:,:12],True)
    received=donor.seen[0][0]
    for h in range(10):torch.testing.assert_close(received[h],z[0,h:h+3],rtol=0,atol=0)


def test_query_target_invariance_and_final_three_observations():
    z,u=inputs();first,second=Toy(.8),Toy(1.2)
    model=core.ReliabilityBlend(first,second,.3,.01)
    a=model.predict(z[:,:13],u[:,:12],u[:,12:])
    changed=z.clone();changed[:,13:]=torch.nan
    b=model.predict(changed[:,:13],u[:,:12],u[:,12:])
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    for donor in (first,second):torch.testing.assert_close(donor.seen[-1][0],z[:,10:13],rtol=0,atol=0)
    alpha=model.infer_gate(z[:,:13],u[:,:12])
    model.predict(z[:,:13],u[:,:12],u[:,12:]*100)
    torch.testing.assert_close(alpha,model.infer_gate(z[:,:13],u[:,:12]),rtol=0,atol=0)


def test_prior_fallback_clipping_and_invalid_moments():
    np.testing.assert_array_equal(core.gate(np.zeros(3),np.zeros(3),.37,0),[.37]*3)
    assert core.gate(1,3,.4,0)==1 and core.gate(1,-1,.4,0)==0
    assert core.gate(0,0,.37,2)==.37
    with pytest.raises(ValueError):core.gate(-1,0,.2,0)
    with pytest.raises(ValueError):core.gate(1,np.nan,.2,0)


def test_invalid_prefix_and_test_split_rejected_before_payload_access():
    z,u=inputs()
    with pytest.raises(ValueError):core.prefix_forecast(Toy(1),z[:,:3],u[:,:2])
    with pytest.raises(ValueError,match="forbidden"):runner.open_dataset({},"test")
    with pytest.raises(ValueError,match="training-only"):runner.fit_gate({}, {},"prefix","val")


def test_quadratic_exact_and_equal_episode_weighting():
    rng=np.random.default_rng(41)
    d,e=rng.normal(size=(7,10,4)),rng.normal(size=(7,10,4))
    data={"query_a":np.mean(d*d,-1),"query_b":np.mean(e*d,-1),"query_c":np.mean(e*e,-1)}
    alpha=np.linspace(0,1,7)
    for h in (5,10):np.testing.assert_allclose(runner.quadratic(data,alpha,h),np.mean((e[:,h-1]-alpha[:,None]*d[:,h-1])**2,-1),rtol=1e-12,atol=1e-12)
    assert runner.episode_mean([0,0,12],[0,0,1])==6


def test_shuffled_gates_preserve_distribution_and_cross_sessions():
    alpha=np.arange(8.)
    sessions=np.array(["a","a","a","b","b","c","d","e"])
    shuffled=runner.shuffle_across_sessions(alpha,sessions,np.array(list("abcdefgh")),np.zeros(8))
    np.testing.assert_array_equal(np.sort(shuffled),alpha)
    assert np.all(sessions!=sessions[shuffled.astype(int)])
    assert runner.shuffle_across_sessions(alpha,np.array(["a"]*7+["b"]),np.array(list("abcdefgh")),np.zeros(8)) is None


def test_regularized_prefix_loss_no_worse_than_prior():
    rng=np.random.default_rng(78)
    d,e=rng.normal(size=(13,10,4)),rng.normal(size=(13,10,4))
    a,b=np.mean(d*d,(1,2)),np.mean(d*e,(1,2))
    alpha=core.gate(a,b,.47,.13)
    assert np.all(np.mean((e-alpha[:,None,None]*d)**2,(1,2))<=np.mean((e-.47*d)**2,(1,2))+1e-12)


def test_train_session_cross_validation_ties_choose_larger_shrinkage():
    sessions=np.array([f"session-{i}" for i in range(30)])
    data={"episode":np.arange(30),"session":sessions,"prefix_a":np.ones(30),"prefix_b":np.full(30,.5),
          "query_a":np.ones((30,10)),"query_b":np.full((30,10),.5),"query_c":np.ones((30,10))}
    result=runner.fit_gate(data,{"fold_salt":"support-reliability-v1:","kappa_grid":[0,1,10,100]},"prefix","train")
    assert result["kappa"]==100 and result["prior"]==.5
    for candidate in result["candidates"]:
        assert candidate["out_of_fold_h5"]==.75
        for fold in candidate["fold_parameters"]:
            assert fold["fitting_sessions"]+fold["heldout_sessions"]==30


def write_fixture_donor(directory,mode):
    from shiftwm.real_video.model import RealVideoWorldModel
    directory.mkdir()
    base=RealVideoWorldModel({"mode":mode,"feature_dim":4,"action_dim":7,"hidden_dim":12,"context_dim":4,"context_hidden":8,"depth":1},[0.]*4,[1.]*4,[0.]*7,[1.]*7).eval()
    torch.nn.init.normal_(base.output_projection[-1].weight,std=.03)
    config={**base.package_config,"package_kind":runner.training.PACKAGE_KIND,"metadata":{"parameter_counts":{"total":sum(p.numel() for p in base.parameters()),"trainable":sum(p.numel() for p in base.parameters() if p.requires_grad)}}}
    torch.save({"config":config,"epoch":1,"step":1,"state_dict":base.state_dict()},directory/"model.pt")
    (directory/"config.json").write_text(json.dumps(config))
    runner.atomic_json({"format_version":1,"package_kind":runner.training.PACKAGE_KIND,"files":{name:runner.sha256(directory/name) for name in ("model.pt","config.json")}},directory/"package_manifest.json")
    return {"format_version":1,"package_kind":"development_train_only_residual_calibration","fit":{"fit_split":"train","displacement_energy":1.,"displacement_alignment":.6,"scale":.6},"base_checkpoint":directory.name,"base_checkpoint_sha256":runner.sha256(directory/"model.pt"),"wrapper_sha256":runner.sha256(runner.ROOT/"src/shiftwm/real_video_development.py")}


def test_exact_offline_reload_after_complete_artifact_relocation(tmp_path):
    original=tmp_path/"original";original.mkdir()
    donors={}
    for mode in ("framewise","factorized"):
        calibration=write_fixture_donor(original/mode,mode)
        path=original/(mode+".json");runner.atomic_json(calibration,path)
        donors[mode]={"calibration":path.name,"sha256":runner.sha256(path)}
    runner.atomic_json({"format_version":1,"kind":"development_causal_reliability_blend","fit_split":"train","inference_source_sha256":runner.sha256(HERE/"core.py"),"parameters":{"prior":.6,"regularization":.1},"donors":donors},original/"gate.json")
    first=core.load_gate(original/"gate.json")
    moved=tmp_path/"moved";shutil.copytree(original,moved);shutil.rmtree(original)
    second=core.load_gate(moved/"gate.json")
    z,u=inputs()
    with torch.inference_mode():torch.testing.assert_close(first.predict(z[:,:13],u[:,:12],u[:,12:]),second.predict(z[:,:13],u[:,:12],u[:,12:]),rtol=0,atol=0)
