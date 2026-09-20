"""New-namespace contracts on synthetic tensors; no dataset payload reads."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location("external_train_test_" + name, HERE / (name + ".py"))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


@pytest.fixture(scope="module")
def trainer():
    torch.set_num_threads(2)
    return module("train")


@pytest.fixture(scope="module")
def model(trainer):
    return trainer.models.ExternalDinoWM(
        {**trainer.models.ARCHITECTURE, "mode": "official_raw_one_step"},
        feature_mean=torch.zeros(6144), feature_std=torch.ones(6144),
        action_mean=torch.zeros(35), action_std=torch.ones(35)).eval()


def test_exact_package_reload_preserves_forecasts(trainer, model, tmp_path):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30)
    counts = {"total": 19412420, "trainable": 19412420}
    metadata = {"training_identity": "synthetic-only", "parameter_counts": counts}
    trainer.base.save_package(model, tmp_path / "best", optimizer=optimizer,
        scheduler=scheduler, epoch=1, step=1, best_metric=1., metadata=metadata,
        generator=torch.Generator().manual_seed(0), history=[])
    restored, state = trainer.load_package(tmp_path / "best", "cpu")
    assert state["config"]["package_kind"] == "adapted_official_dinowm_raw_droid_v2"
    assert restored.package_config == model.package_config
    batch = trainer.models.adapter.synthetic_batch(1)
    with torch.inference_mode():
        for objective in model.MODES:
            assert torch.equal(model(batch, objective)["standardized_predictions"],
                               restored(batch, objective)["standardized_predictions"])


class RoutingModel(torch.nn.Module):
    def __init__(self, mode):
        super().__init__(); self.config = SimpleNamespace(mode=mode)
        self.scale = torch.nn.Parameter(torch.tensor(1.)); self.calls = []

    def forward(self, batch, objective):
        self.calls.append((objective, torch.is_grad_enabled(), self.training))
        target = batch["features"][:, 1:4] if objective == "official_raw_one_step" else batch["features"][:, 3:13]
        return {"standardized_predictions": torch.ones_like(target) * self.scale,
                "standardized_targets": target.detach(),
                "raw_predictions": torch.ones_like(target) * self.scale * 3,
                "raw_targets": target.detach() * 3}


@pytest.mark.parametrize("mode,steps", [("official_raw_one_step", 3), ("official_raw_recursive_h10", 10)])
def test_training_routes_declared_objective_but_validation_always_h10(trainer, mode, steps):
    model = RoutingModel(mode)
    batches = [{"features": torch.zeros(n, 13, 6144), "episode_index": torch.tensor(ids)}
               for n, ids in ((2, [0, 0]), (1, [1]))]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.)
    training = trainer.epoch_pass(model, batches, torch.device("cpu"), optimizer)
    validation = trainer.epoch_pass(model, batches, torch.device("cpu"))
    assert training["raw_mse"] == 9. and training["standardized_mse"] == 1.
    assert training["optimized_loss"] == "raw_visual_mse"
    assert training["elements"] == 3 * steps * 6144
    assert training["target_grid_indices"] == ([1, 2, 3] if steps == 3 else list(range(3, 13)))
    assert validation["elements"] == 3 * 10 * 6144 and validation["episodes"] == 2
    assert validation["standardized_mse"] == 1.
    assert model.calls == [(mode, True, True)] * 2 + [("official_raw_recursive_h10", False, False)] * 2


def test_validation_is_window_weighted_not_equal_episode(trainer):
    model = RoutingModel("official_raw_one_step")
    batches = [{"features": torch.zeros(2, 13, 6144), "episode_index": torch.tensor([0, 0])},
               {"features": torch.full((1, 13, 6144), 4.), "episode_index": torch.tensor([1])}]
    value = trainer.epoch_pass(model, batches, torch.device("cpu"))
    assert value["standardized_mse"] == pytest.approx(11 / 3)
    assert value["equal_episode_diagnostic_mse"] == 5.


@pytest.mark.parametrize("field,value", [("epochs", 29), ("batch_size", 64), ("train_horizon", 5),
                                         ("validation_stride", 2), ("cache_root", "reserved"), ("seed", True)])
def test_recipe_and_population_changes_rejected(field, value):
    registry = module("registry")
    config = registry.expected_config("official_raw_one_step", 0); config[field] = value
    with pytest.raises(ValueError): registry.validate_config(config)


def test_review_rejection_precedes_source_and_data_access(monkeypatch, tmp_path):
    registry = module("registry")
    registration = tmp_path / "registration.json"; review = tmp_path / "review.json"
    registration.write_text(json.dumps({"dependencies": {"forbidden-payload": "bad"}}))
    review.write_text(json.dumps({"status": "passed", "registration_sha256": "stale"}))
    monkeypatch.setattr(registry, "REG", registration); monkeypatch.setattr(registry, "REVIEW", review)
    monkeypatch.setattr(registry, "sources", lambda: pytest.fail("Sources touched before review rejection"))
    monkeypatch.setattr(registry, "profile_gate", lambda: pytest.fail("Profile touched before review rejection"))
    with pytest.raises(ValueError, match="review missing or stale"): registry.verify()


def test_completion_rejects_partial_population_and_wrong_objective(trainer, monkeypatch, tmp_path):
    registry = module("registry")
    (tmp_path / "training_config.json").write_text(json.dumps(registry.expected_config("official_raw_one_step", 0)))
    def part(steps, n, objective):
        return {"supervised_grids": steps, "windows": n, "elements": steps * n * 6144,
                "objective": objective, "batches": (n + 31) // 32,
                "target_grid_indices": [1, 2, 3] if steps == 3 else list(range(3, 13)), "episodes": 141, "raw_mse": 1., "optimized_loss": "raw_visual_mse", "precision": "float32"}
    row = {"lr": 5e-4, "train": part(3, 18660, "official_raw_one_step"), "val": part(10, 1631, "official_raw_recursive_h10")}
    monkeypatch.setattr(trainer.base, "validate_completed", lambda *a: {"parameter_counts": {"total": 19412420, "trainable": 19412420}})
    rows = [row]
    monkeypatch.setattr(trainer.base, "metric_rows", lambda _: rows)
    trainer.validate_completed(tmp_path)
    rows = [copy.deepcopy(row)]; rows[0]["val"]["windows"] -= 1
    with pytest.raises(ValueError, match="population"): trainer.validate_completed(tmp_path)
    rows = [copy.deepcopy(row)]; rows[0]["val"]["objective"] = "official_raw_one_step"
    with pytest.raises(ValueError, match="population"): trainer.validate_completed(tmp_path)


def test_duplicate_submission_cannot_enter_training(monkeypatch, tmp_path):
    import fcntl
    campaign = module("campaign")
    directory = tmp_path / "run"; directory.mkdir()
    fake = SimpleNamespace(verify=lambda: {"runs": [{"name": "run", "config": "config.json"}]},
                           read=lambda _: {"output_dir": str(directory)})
    monkeypatch.setattr(campaign, "module", lambda name: fake if name == "registry" else pytest.fail("Training entered while locked"))
    with (directory / ".training.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): campaign.run(index=0)


def test_execution_receipt_binds_actual_visible_hardware_and_scheduler(monkeypatch):
    campaign = module("campaign")
    props = SimpleNamespace(name="synthetic device", total_memory=123456, major=8, minor=9)
    fake = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda="test-cuda"),
                           cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                                get_device_properties=lambda _: props))
    monkeypatch.setenv("SLURM_JOB_ID", "123"); monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "4")
    value = campaign.execution_identity(fake)
    assert value["slurm"]["SLURM_JOB_ID"] == "123" and value["slurm"]["SLURM_ARRAY_TASK_ID"] == "4"
    assert value["visible_gpus"] == [{"visible_index": 0, "name": "synthetic device",
                                      "total_memory_bytes": 123456, "compute_capability": [8, 9]}]
    assert value["hostname"] and value["torch_version"] == "test" and value["torch_cuda_build"] == "test-cuda"


def test_raw_prediction_is_independent_of_feature_standardization(model):
    batch = model.adapter.synthetic_batch(1) if hasattr(model, 'adapter') else module('model').adapter.synthetic_batch(1)
    old_mean, old_std = model.feature_mean.clone(), model.feature_std.clone()
    try:
        with torch.inference_mode():
            first = model(batch, 'official_raw_one_step')
            model.feature_mean.fill_(7.); model.feature_std.fill_(3.)
            changed = model(batch, 'official_raw_one_step')
        assert torch.equal(first['raw_predictions'], changed['raw_predictions'])
        assert torch.equal(first['raw_targets'], changed['raw_targets'])
        assert not torch.equal(first['standardized_predictions'], changed['standardized_predictions'])
        assert torch.equal(changed['raw_targets'], batch['features'][:, 1:4])
    finally:
        model.feature_mean.copy_(old_mean); model.feature_std.copy_(old_std)


def test_raw_native_exactly_matches_official_predictor_tensor_path(model):
    adapter = module('model').adapter
    batch = adapter.synthetic_batch(1, seed=43)
    with torch.inference_mode():
        visual = adapter.tokens(batch['features'][:, :3])
        actions = model.action_encoder(model.normalized_actions(batch['actions'][:, :3]))
        embedded = torch.cat((visual, actions.unsqueeze(2).expand(-1, -1, 16, -1)), -1)
        expected = adapter.flatten(model.predictor(embedded.reshape(1, 48, 394)).reshape(1, 3, 16, 394)[..., :384])
        actual = model(batch, 'official_raw_one_step')['raw_predictions']
    assert torch.equal(actual, expected)


def test_raw_recursion_uses_matching_actions_and_full_graph(model, monkeypatch):
    calls=[]
    def predictor(history, actions):
        calls.append((history.detach().clone(), actions.detach().clone()))
        return history * 2 + actions[..., :1]
    monkeypatch.setattr(model, 'predict_three_slots', predictor)
    support=torch.ones(1,3,6144,requires_grad=True)
    past=torch.zeros(1,2,35); future=torch.arange(10.).view(1,10,1).expand(1,10,35)
    value=model.predict(support,past,future)
    assert len(calls)==10
    joined=model.normalized_actions(torch.cat((past,future),1))
    for step,(_,actions) in enumerate(calls):
        assert torch.equal(actions,joined[:,step:step+3])
    value[:,-1].sum().backward()
    assert torch.equal(support.grad[:,-1],torch.full((1,6144),2.**10))
    assert torch.equal(support.grad[:,:2],torch.zeros(1,2,6144))


def test_training_optimizes_raw_loss_not_standardized_diagnostic(trainer):
    model=RoutingModel('official_raw_one_step')
    batch={'features':torch.zeros(1,13,6144),'episode_index':torch.tensor([0])}
    trainer.epoch_pass(model,[batch],torch.device('cpu'),torch.optim.SGD(model.parameters(),lr=.01))
    assert model.scale.item()==pytest.approx(.82,abs=1e-6) # d(3*scale)^2/dscale =18


def test_epoch_rejects_precision_or_clipping_recipe_change(trainer):
    with pytest.raises(ValueError,match='FP32'):
        trainer.epoch_pass(RoutingModel('official_raw_one_step'),[],torch.device('cpu'),bf16=True)
    with pytest.raises(ValueError,match='FP32'):
        trainer.epoch_pass(RoutingModel('official_raw_one_step'),[],torch.device('cpu'),grad_clip=1.)


def test_hundred_epoch_resume_is_exact_and_constant_lr(monkeypatch,tmp_path):
    engine=module('engine').load()
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor([.3,-.7]))
            self.config=SimpleNamespace(mode='synthetic',history_length=3)
        @property
        def package_config(self):return {'format_version':1,'kind':'synthetic-only'}
    def epoch_pass(model,loader,device,optimizer=None,*args):
        total=0.;n=0;batches=0
        for data in loader:
            if optimizer is not None:
                optimizer.zero_grad();noise=torch.rand_like(data)*.02
                prediction=(data+noise)*model.weight
            else:prediction=data*model.weight
            loss=prediction.square().mean()
            if optimizer is not None:loss.backward();optimizer.step()
            total+=float(loss.detach())*data.numel();n+=data.numel();batches+=1
        return {'standardized_mse':total/n,'elements':n,'batches':batches}
    monkeypatch.setattr(engine,'epoch_pass',epoch_pass)
    monkeypatch.setattr(engine,'from_config',lambda _:Tiny())
    dataset=torch.arange(10,dtype=torch.float32).reshape(5,2)/10
    config={'epochs':100,'seed':0,'mode':'synthetic','output_dir':str(tmp_path/'whole'),'device':'cpu',
            'lr':5e-4,'min_lr':5e-4,'weight_decay':.01,'batch_size':2,'num_workers':0,'bf16':False,'grad_clip':None}
    identity={'scientific_config':engine.scientific_config(config),'dependencies':{}}
    whole=engine.fit(Tiny(),config,dataset,dataset,identity)
    split={**config,'output_dir':str(tmp_path/'split'),'max_runtime_seconds':0}
    partial=engine.fit(Tiny(),split,dataset,dataset,identity)
    assert partial['completed_epochs']==1 and partial['status']=='interrupted'
    split.update(resume_if_present=True,max_runtime_seconds=1e9)
    completed=engine.fit(Tiny(),split,dataset,dataset,identity)
    assert whole['completed_epochs']==completed['completed_epochs']==100
    wholepath,wholestate=engine.read_package(tmp_path/'whole'/'last',require_training=True)
    splitpath,splitstate=engine.read_package(tmp_path/'split'/'last',require_training=True)
    assert wholestate['history']==splitstate['history']
    assert all(r['lr']==5e-4 for r in splitstate['history'])
    assert torch.equal(wholestate['state_dict']['weight'],splitstate['state_dict']['weight'])
    a=torch.load(wholepath/'training_state.pt',weights_only=True);b=torch.load(splitpath/'training_state.pt',weights_only=True)
    assert torch.equal(a['loader_generator'],b['loader_generator'])
    for key in a['optimizer']['state'][0]:
        assert torch.equal(a['optimizer']['state'][0][key],b['optimizer']['state'][0][key])
    assert a['scheduler']==b['scheduler']
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])


@pytest.mark.parametrize('case',['submitted','scheduler_failure','restart_exhausted'])
def test_epoch_boundary_continuation_cannot_evaluate_or_complete(case,monkeypatch,tmp_path):
    import subprocess
    campaign=module('campaign');directory=tmp_path/'run';writes={};calls=[]
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    def atomic(value,path):writes[str(path)]=copy.deepcopy(value)
    train=SimpleNamespace(torch=SimpleNamespace(),atomic_json=atomic,
                          train=lambda _: {'status':'interrupted','completed_epochs':42})
    registry=SimpleNamespace(REG=tmp_path/'registration.json',REPORT=tmp_path/'reports',sha=lambda _: 'synthetic',
                             verify=lambda: {'runs':[{'name':'synthetic','config':'synthetic.json'}]},
                             read=lambda _: {'output_dir':str(directory)})
    def load(name):
        if name=='registry':return registry
        if name=='train':return train
        pytest.fail('Evaluation entered before100 epochs')
    monkeypatch.setattr(campaign,'module',load)
    monkeypatch.setattr(campaign,'execution_identity',lambda _: {'synthetic_only':True})
    monkeypatch.setenv('SLURM_JOB_ID','1234')
    monkeypatch.setenv('SLURM_RESTART_COUNT','5' if case=='restart_exhausted' else '0')
    def requeue(command,**kwargs):
        calls.append(command)
        if case=='scheduler_failure':raise subprocess.CalledProcessError(1,command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(campaign.subprocess,'run',requeue)
    if case=='submitted':
        assert campaign.run(index=0)=={'status':'interrupted','completed_epochs':42}
        assert calls==[['scontrol','requeue','1234']]
    else:
        with pytest.raises((RuntimeError,subprocess.CalledProcessError)):campaign.run(index=0)
    assert len(writes)==1
    receipt=next(iter(writes.values()))
    assert receipt['completed_epochs']==42
    assert receipt['status']==('epoch_boundary_continuation_submitted' if case=='submitted' else 'continuation_failed')
    assert not any(path.endswith('_completed.json') for path in writes)
    assert (not calls) if case=='restart_exhausted' else len(calls)==1
