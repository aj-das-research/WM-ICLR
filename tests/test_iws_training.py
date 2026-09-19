"""Behavioral checks for full-horizon accumulation, selection and continuation."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from shiftwm.real_video_iws import training as trainer


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


class ToyModel(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        self.config = SimpleNamespace(feature_dim=4, action_dim=4, mode="anchored_additive")
        self.weight = nn.Parameter(torch.full((4,), .03))
        self.register_buffer("feature_std", torch.tensor([1.,2.,3.,4.]))
        self.register_buffer("feature_mean",torch.zeros(4))
        self.register_buffer("command_mean",torch.zeros(4))
        self.register_buffer("command_std",torch.ones(4))
        self.dropout = nn.Dropout(.2 if dropout else 0.)
        self.calls = 0
        self.fail_call = None

    @property
    def package_config(self):
        return {"format_version":1,"package_kind":trainer.PACKAGE_KIND,"test_fixture":"linear_full_H60",
                "model_config":vars(self.config),**{k:getattr(self,k).tolist() for k in
                    ("feature_mean","feature_std","command_mean","command_std")}}

    def predict(self, initial_features, native_commands):
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError("injected interruption")
        return initial_features[:,None,:] + self.dropout(native_commands[:,1:,:] * self.weight)


class Windows(Dataset):
    def __init__(self,n=70,unequal=False):
        generator=torch.Generator().manual_seed(42)
        self.initial=torch.randn(n,4,generator=generator)*.1
        self.commands=torch.randn(n,60,4,generator=generator)
        self.targets=self.initial[:,None]+self.commands[:,1:]*.4
        self.ids=[i%3 for i in range(n)]
        if unequal:
            assert n==4
            self.initial.zero_();self.commands.zero_();self.ids=[0,1,1,1]
            self.targets=torch.ones(n,59,4)
            self.targets[1:]*=3
        self.audit={"windows":n,"records":[{"episode_index":i,"windows":self.ids.count(i)} for i in sorted(set(self.ids))]}

    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        return {"initial_features":self.initial[i],"commands":self.commands[i],"targets":self.targets[i],
                "episode_index":self.ids[i],"window_start":i*5,"episode_id":str(self.ids[i])}


def fit_inputs(tmp_path, *, dropout=True):
    train, val=Windows(7),Windows(4)
    recipe={"seed":0,"mode":"anchored_additive","task":"pusht","model":{"feature_dim":4},"task_config":{"action_dim":4},
        "training":{"epochs":30,"horizon":60,"batch_size":4,"microbatch_size":2,"accumulation_steps":2,
                    "bf16":False,"lr":.01,"min_lr":1e-6,"weight_decay":.01,"grad_clip":1.}}
    model=ToyModel(dropout)
    identity={"scientific_config":recipe,"dependencies":{},"populations":{"train":train.audit,"val":val.audit},
              "normalization":{k:getattr(model,k).tolist() for k in ("feature_mean","feature_std","command_mean","command_std")}}
    config={"scientific_config":recipe,"output_dir":str(tmp_path),"device":"cpu","resume_if_present":False}
    return model,config,train,val,identity


def assert_tree_equal(first,second):
    assert type(first)==type(second)
    if torch.is_tensor(first):assert torch.equal(first,second)
    elif isinstance(first,dict):
        assert first.keys()==second.keys()
        for k in first:assert_tree_equal(first[k],second[k])
    elif isinstance(first,(tuple,list)):
        assert len(first)==len(second)
        for a,b in zip(first,second):assert_tree_equal(a,b)
    else:assert first==second


def test_accumulation_equals_effective_batches_including_short_tail():
    data=Windows(70);first=ToyModel();second=deepcopy(first)
    outputs=[]
    for model,micro in [(first,16),(second,64)]:
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        outputs.append(trainer.epoch_pass(model,DataLoader(data,batch_size=64,shuffle=False),torch.device("cpu"),
                                        optimizer=optimizer,microbatch_size=micro,bf16=False,grad_clip=1.))
    torch.testing.assert_close(first.weight,second.weight,rtol=1e-6,atol=1e-8)
    assert outputs[0]["optimizer_updates"]==outputs[1]["optimizer_updates"]==2
    assert outputs[0]["microbatches"]==5 and outputs[1]["microbatches"]==2
    assert outputs[0]["windows"]==70 and outputs[0]["elements"]==70*59*4


def test_selector_equal_trajectory_not_equal_window_and_float32():
    model=ToyModel();model.feature_std.fill_(1)
    with torch.autocast("cpu",dtype=torch.bfloat16):
        result=trainer.epoch_pass(model,DataLoader(Windows(4,unequal=True),batch_size=3),torch.device("cpu"),microbatch_size=2)
    assert result["standardized_mse"]==5.
    assert result["window_mean_all59_standardized_mse"]==7.
    assert result["episodes"]==2 and result["aggregation"]==trainer.SELECTION
    assert result["diagnostic_endpoints"]=={str(h):5. for h in (15,30,45,60)}
    assert result["optimizer_updates"]==0


def test_no_loss_only_targets_are_passed_to_predictor():
    model=ToyModel();batch=next(iter(DataLoader(Windows(2),batch_size=2)))
    called=[];original=model.predict
    def predict(initial_features,native_commands):
        called.append((tuple(initial_features.shape),tuple(native_commands.shape)))
        return original(initial_features,native_commands)
    model.predict=predict
    trainer.epoch_pass(model,[batch],torch.device("cpu"),microbatch_size=1)
    assert called==[((1,4),(1,60,4))]*2


@pytest.mark.parametrize("field,shape",[("commands",(2,59,4)),("targets",(2,58,4)),("initial_features",(2,2,4))])
def test_rejects_short_horizon_or_fabricated_observed_history(field,shape):
    batch=next(iter(DataLoader(Windows(2),batch_size=2)));batch[field]=torch.zeros(shape)
    with pytest.raises(ValueError):trainer.epoch_pass(ToyModel(),[batch],torch.device("cpu"))


def test_rejects_empty_population():
    with pytest.raises(ValueError,match="Empty"):
        trainer.epoch_pass(ToyModel(),[],torch.device("cpu"))


def test_full30_resume_is_bit_exact_including_optimizer_rng_loader(tmp_path):
    m,c,t,v,i=fit_inputs(tmp_path/"continuous");full=trainer.fit(m,c,t,v,i)
    n,d,t,v,i=fit_inputs(tmp_path/"resumed");d["max_runtime_seconds"]=0
    interrupted=trainer.fit(n,d,t,v,i);assert interrupted["completed_epochs"]==1
    d.update(resume_if_present=True,max_runtime_seconds=float("inf"))
    # JSON serializability of operational config must remain strict.
    d.pop("max_runtime_seconds")
    trainer.fit(ToyModel(True),d,t,v,i)
    for part in ["last","best"]:
        _,a=trainer.read_package(tmp_path/"continuous"/part,True)
        _,b=trainer.read_package(tmp_path/"resumed"/part,True)
        assert_tree_equal(a,b)
        sa=torch.load((tmp_path/"continuous"/part/"training_state.pt"),weights_only=True)
        sb=torch.load((tmp_path/"resumed"/part/"training_state.pt"),weights_only=True)
        assert_tree_equal(sa,sb)
    assert full["completed_epochs"]==30


def test_interrupted_first_epoch_replays_from_initial_generation(tmp_path):
    model,config,train,val,identity=fit_inputs(tmp_path/"baseline")
    trainer.fit(model,config,train,val,identity)
    model,config,train,val,identity=fit_inputs(tmp_path/"replay")
    model.fail_call=3
    with pytest.raises(RuntimeError,match="injected"):
        trainer.fit(model,config,train,val,identity)
    _,state=trainer.read_package(tmp_path/"replay"/"last",True);assert state["epoch"]==0
    config["resume_if_present"]=True
    trainer.fit(ToyModel(True),config,train,val,identity)
    _,a=trainer.read_package(tmp_path/"baseline"/"last")
    _,b=trainer.read_package(tmp_path/"replay"/"last")
    assert_tree_equal(a,b)


def test_repairs_best_swap_ahead_of_authoritative_last(tmp_path,monkeypatch):
    args=fit_inputs(tmp_path/"reference",dropout=False);trainer.fit(*args)
    model,config,train,val,identity=fit_inputs(tmp_path/"crash",dropout=False)
    real=trainer.save_package
    def fail_after_best(model,directory,**kw):
        result=real(model,directory,**kw)
        if Path(directory).name=="best" and kw["epoch"]==2:raise RuntimeError("after best swap")
        return result
    monkeypatch.setattr(trainer,"save_package",fail_after_best)
    with pytest.raises(RuntimeError,match="after best"):
        trainer.fit(model,config,train,val,identity)
    assert trainer.read_package(tmp_path/"crash"/"best")[1]["epoch"]==2
    assert trainer.read_package(tmp_path/"crash"/"last")[1]["epoch"]==1
    monkeypatch.setattr(trainer,"save_package",real);config["resume_if_present"]=True
    trainer.fit(ToyModel(False),config,train,val,identity)
    assert_tree_equal(trainer.read_package(tmp_path/"reference"/"last")[1],trainer.read_package(tmp_path/"crash"/"last")[1])


def test_resume_rejects_changed_recipe(tmp_path):
    model,config,train,val,identity=fit_inputs(tmp_path);config["max_runtime_seconds"]=0
    trainer.fit(model,config,train,val,identity)
    config=deepcopy(config);identity=deepcopy(identity);config["resume_if_present"]=True
    config["scientific_config"]["training"]["lr"]*=2
    identity["scientific_config"]=config["scientific_config"]
    with pytest.raises(ValueError,match="identity or metadata"):
        trainer.fit(ToyModel(True),config,train,val,identity)


def test_cross_package_kind_rejected_before_tensor_load(tmp_path):
    (tmp_path/"package_manifest.json").write_text(json.dumps({"format_version":1,"package_kind":"shiftwm_real_video_droid_v1","files":{}}))
    with pytest.raises(ValueError,match="Unsupported"):
        trainer.read_package(tmp_path)


def test_real_model_package_reloads_offline_after_physical_relocation(tmp_path):
    from shiftwm.real_video_iws.model import SingleObservationWorldModel
    model=SingleObservationWorldModel({"hidden_dim":12,"depth":1,"mode":"bounded_spatial_mix"},
          np.zeros(6144),np.ones(6144),np.zeros(4),np.ones(4)).eval()
    generator=torch.Generator().manual_seed(5)
    initial=torch.randn(1,6144,generator=generator);commands=torch.randn(1,3,4,generator=generator)
    with torch.inference_mode():expected=model.predict(initial,commands)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30)
    identity={"scientific_config":{"fixture":"relocated_inference"},"dependencies":{}}
    trainer.save_package(model,tmp_path/"original"/"best",optimizer=opt,scheduler=scheduler,epoch=1,step=1,
                         best_metric=1.,metadata=trainer._metadata(model,identity),generator=generator,history=[])
    shutil.copytree((tmp_path/"original"/"best").resolve(),tmp_path/"relocated")
    shutil.rmtree(tmp_path/"original")
    loaded,_=trainer.load_package(tmp_path/"relocated")
    with torch.inference_mode():actual=loaded.predict(initial,commands)
    assert torch.equal(expected,actual)


@pytest.mark.parametrize("what",["architecture","normalization"])
def test_fit_rejects_model_not_equal_to_registered_contract(tmp_path,what):
    model,config,train,val,identity=fit_inputs(tmp_path)
    if what=="architecture":identity["scientific_config"]["model"]["unregistered_field"]=4
    else:model.feature_std[0]*=2
    with pytest.raises(ValueError,match="architecture|normalization"):
        trainer.fit(model,config,train,val,identity)
    assert not (tmp_path/"last").exists()


@pytest.mark.parametrize("what",["renamed_episode","redistributed_windows","updates"])
def test_completion_rejects_self_consistent_wrong_population(tmp_path,what):
    trainer.fit(*fit_inputs(tmp_path,dropout=False))
    # Change the hash-consistent journal in all selected/last copies. This tests
    # semantic validation, beyond the package checksum check.
    for name in ("last","best"):
        directory,state=trainer.read_package(tmp_path/name)
        row=state["history"][0]
        if what=="renamed_episode":row["val"]["episode_metrics"][0]["episode_index"]=99
        elif what=="redistributed_windows":
            records=row["val"]["episode_metrics"];records[0]["windows"]-=1;records[1]["windows"]+=1
        else:row["train"]["optimizer_updates"]+=1
        torch.save(state,directory/"model.pt")
        manifest=json.loads((directory/"package_manifest.json").read_text())
        manifest["files"]["model.pt"]=trainer.base.sha256(directory/"model.pt")
        (directory/"package_manifest.json").write_text(json.dumps(manifest))
    _,last=trainer.read_package(tmp_path/"last")
    trainer.base.write_history(last["history"],tmp_path)
    with pytest.raises(ValueError,match="population|episode ledger"):
        trainer.validate_completed(tmp_path)


def test_inference_export_excludes_optimizer_and_preserves_tensors(tmp_path):
    run=tmp_path/"run";trainer.fit(*fit_inputs(run,dropout=False))
    result=trainer.export_inference_package(run,tmp_path/"export")
    assert result["optimizer_or_rng_included"] is False
    assert {p.name for p in (tmp_path/"export").iterdir()}=={"model.pt","config.json","package_manifest.json"}
    assert_tree_equal(trainer.read_package(run/"best")[1],trainer.read_package(tmp_path/"export")[1])
    with pytest.raises(ValueError,match="replace"):
        trainer.export_inference_package(run,tmp_path/"export")


@pytest.mark.parametrize("task,width",[("pusht",4),("bimanual_box",14),("bimanual_rope",8)])
def test_training_statistics_uses_real_task_specific_helper(task,width):
    """The Box/Rope helper requires the task STRING, never its numeric width."""
    path=Path(__file__).resolve().parents[1]/"scripts/real_video_iws/train.py"
    spec=importlib.util.spec_from_file_location("_iws_stats_cli_fixture",path)
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    eid="000011";n=3;rng=np.random.default_rng(42)
    receipt={"episode_id":eid,"split":"internal_train","frames":n,"task":task,
             "command_width":width,"payload_sha256":"synthetic_test_payload"}
    arrays={"features":rng.normal(size=(n,6144)).astype(np.float32),
            "commands":rng.normal(size=(n,width)).astype(np.float32),
            "frame_indices":np.arange(n,dtype=np.int64),"command_row_indices":np.arange(n,dtype=np.int64)}
    if task=="pusht":
        from shiftwm.real_video_iws.data import training_statistics
        expected=training_statistics([(receipt,arrays)],[eid])
    else:
        from shiftwm.real_video_iws_tasks.data import training_statistics
        expected=training_statistics([(receipt,arrays)],[eid],task)
    dataset=SimpleNamespace(split="internal_train",task=task,command_width=width,episodes=[{**receipt,**arrays}])
    cache=SimpleNamespace(index={eid:receipt},statistics=expected,inventory=SimpleNamespace(partitions={"internal_train":[eid]}))
    cli.verify_training_statistics(dataset,cache)
    cache.statistics["command_mean"][0]+=1
    with pytest.raises(ValueError,match="normalization"):
        cli.verify_training_statistics(dataset,cache)
