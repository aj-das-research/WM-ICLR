"""Horizon-ten matched control: full30 lifecycle, causal access and selection checks."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("horizon10_training_tests", ROOT / "scripts/real_video_development/horizon10_train.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(67)
        self.features = torch.randn(5, 13, 8, generator=generator)
        self.actions = torch.randn(5, 12, 10, generator=generator)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return {"features": self.features[index], "actions": self.actions[index], "episode_index": (0 if index < 4 else 1)}


def model(mode="factorized"):
    trainer.seed_everything(0)
    return trainer.RealVideoWorldModel({"feature_dim": 8, "action_dim": 10, "hidden_dim": 12,
        "context_dim": 4, "context_hidden": 8, "depth": 1, "mode": mode},
        [0.] * 8, [2.] * 8, [0.] * 10, [1.] * 10)


def settings(directory, mode="factorized"):
    return {"epochs": 30, "seed": 0, "mode": mode, "output_dir": str(directory),
            "device": "cpu", "batch_size": 3, "train_horizon": 10, "validation_horizon": 10, "lr": .001, "min_lr": .00001,
            "weight_decay": .01, "grad_clip": 1., "bf16": True}


def identity(config):
    return {"scientific_config": trainer.scientific_config(config), "dependencies": {}}


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    directory = tmp_path_factory.mktemp("realvideo-completed")
    cfg = settings(directory)
    result = trainer.fit(model(), cfg, TinyDataset(), TinyDataset(), identity(cfg))
    torch.set_num_threads(previous)
    return directory, result


def test_all30_epochs_and_offline_best_selection(completed, monkeypatch):
    directory, result = completed
    assert result["completed_epochs"] == 30 and result["step"] == 60
    rows = trainer.metric_rows(directory)
    best = min(rows, key=lambda row: row["val"]["standardized_mse"])
    loaded, state = trainer.load_package(directory / "best")
    assert state["epoch"] == best["epoch"]
    assert state["best_metric"] == best["val"]["standardized_mse"]
    assert result["parameter_counts"]["total"] > result["parameter_counts"]["trainable"] > 0
    assert trainer.validate_completed(directory) == result
    assert not loaded.training
    assert torch.load(directory / "best/training_state.pt", weights_only=True)["epoch"] == state["epoch"]


def test_validation_is_fp32_all10_window_weighted_with_equal_episode_auxiliary():
    m = model()
    original = m.forward
    expected = {0: [], 1: []}
    window_batches = []
    def forward(batch):
        assert not m.training and torch.is_inference_mode_enabled()
        assert not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda")
        out = original(batch)
        assert out["standardized_targets"].shape[1:] == (10, 8)
        assert out["standardized_predictions"].dtype == torch.float32
        errors = (out["standardized_predictions"] - out["standardized_targets"]).square()
        window_batches.append((float(errors.mean()), errors.numel()))
        values = errors.mean((1,2))
        for index, value in zip(batch['episode_index'].tolist(), values.tolist()):
            expected[index].append(value)
        out["loss"] = -torch.ones(())
        return out
    m.forward = forward
    actual = trainer.epoch_pass(m, torch.utils.data.DataLoader(TinyDataset(), batch_size=3), torch.device("cpu"))
    equal_episode = np.mean([np.mean(v) for v in expected.values()])
    window_weighted = np.mean(sum(expected.values(),[]))
    assert actual["auxiliary_equal_episode_standardized_mse"] == equal_episode
    assert actual["standardized_mse"] == sum(v*n for v,n in window_batches)/sum(n for v,n in window_batches)
    assert actual["standardized_mse"] == pytest.approx(window_weighted)
    assert abs(equal_episode-window_weighted) > 1e-3
    assert actual["elements"] == 5*10*8 and actual['episodes'] == 2
    assert actual['query_steps'] == 10 and actual['aggregation'] == trainer.AGGREGATION


def test_tenth_target_changes_selection_metric_even_when_first_nine_perfect():
    m=model()
    data=TinyDataset()
    data.features.zero_()
    with torch.no_grad():
        data.features[:,12,:] = 10
    result=trainer.epoch_pass(m,torch.utils.data.DataLoader(data,batch_size=2),torch.device('cpu'))
    # Feature std=2: final target squared normalized error=25; all-ten average=2.5.
    assert result['standardized_mse'] == 2.5


@pytest.mark.parametrize('mode',trainer.RealVideoWorldModel.MODES)
def test_query_images_never_condition_predictions_and_future_actions_are_causal(mode):
    m=model(mode).eval()
    with torch.no_grad():
        m.output_projection[-1].weight.normal_(std=.03)
        # Exercise learned action paths rather than AdaLN-zero initialization.
        for block in m.predictor.transformer.layers:
            block.adaLN_modulation[-1].weight.normal_(std=.03)
    row=TinyDataset()[0]
    features=row['features'][None].clone().requires_grad_()
    actions=row['actions'][None].clone()
    first=m({'features':features,'actions':actions})['predictions']
    changed=features.detach().clone();changed[:,3:]+=100
    second=m({'features':changed,'actions':actions})['predictions']
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    gradient=torch.autograd.grad(first.sum(),features)[0]
    assert torch.equal(gradient[:,3:],torch.zeros_like(gradient[:,3:]))
    assert torch.count_nonzero(gradient[:,:3]) > 0
    changed_actions=actions.clone();changed_actions[:,7:]+=50
    third=m({'features':features,'actions':changed_actions})['predictions']
    torch.testing.assert_close(first[:,:5],third[:,:5],rtol=0,atol=0)
    if mode=='action_free':
        torch.testing.assert_close(first,third,rtol=0,atol=0)
    else:
        assert not torch.allclose(first[:,5:],third[:,5:])


def test_rejects_h5_windows_in_h10_selection():
    data=TinyDataset();data.features=data.features[:,:8];data.actions=data.actions[:,:7]
    with pytest.raises(ValueError,match='ten query'):
        trainer.epoch_pass(model(),torch.utils.data.DataLoader(data,batch_size=2),torch.device('cpu'))


def test_interrupted_completed_epoch_resume_is_exact(completed, tmp_path):
    cfg = settings(tmp_path)
    cfg["max_runtime_seconds"] = 1e-12
    result = trainer.fit(model(), cfg, TinyDataset(), TinyDataset(), identity(cfg))
    assert result["status"] == "interrupted" and result["completed_epochs"] == 1
    # Simulate stale external journal: committed model history wins on resume.
    (tmp_path / "metrics.jsonl").write_text("")
    cfg.pop("max_runtime_seconds")
    cfg["resume_if_present"] = True
    result = trainer.fit(model(), cfg, TinyDataset(), TinyDataset(), identity(cfg))
    assert result["status"] == "completed"
    expected = trainer.read_package(completed[0] / "last")[1]
    actual = trainer.read_package(tmp_path / "last")[1]
    assert expected["history"] == actual["history"]
    for key, value in expected["state_dict"].items():
        torch.testing.assert_close(actual["state_dict"][key], value, rtol=0, atol=0)


def copy_completed(source, destination):
    shutil.copytree(source, destination, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".*.generations", ".training.lock"))


@pytest.mark.parametrize("damage", ["history", "summary", "selection", "identity", "file", "epochs", "mode"])
def test_completion_rejects_changed_evidence(completed, tmp_path, damage):
    copy_completed(completed[0], tmp_path)
    if damage == "history":
        rows = trainer.metric_rows(tmp_path)
        trainer.write_history(rows[:-1], tmp_path)
    elif damage in ("summary", "selection", "identity"):
        path = tmp_path / "training_summary.json"
        value = json.loads(path.read_text())
        key, new = {"summary": ("completed_epochs", 29), "selection": ("validation_metric", "train_loss"),
                    "identity": ("training_identity", "bad")}[damage]
        value[key] = new
        trainer.atomic_json(value, path)
    elif damage == "file":
        with (tmp_path / "best/model.pt").open("ab") as handle:
            handle.write(b"corrupt")
    else:
        path = tmp_path / "training_config.json"
        value = json.loads(path.read_text())
        value["epochs" if damage == "epochs" else "mode"] = 29 if damage == "epochs" else "framewise"
        trainer.atomic_json(value, path)
    with pytest.raises(ValueError):
        trainer.validate_completed(tmp_path)


def test_resume_rejects_changed_hyperparameters(completed, tmp_path):
    copy_completed(completed[0], tmp_path)
    # Copied packages are intentionally immutable; mismatch is rejected before save.
    cfg = settings(tmp_path)
    cfg.update(resume_if_present=True, lr=.002)
    with pytest.raises(ValueError, match="identity"):
        trainer.fit(model(), cfg, TinyDataset(), TinyDataset(), identity(cfg))


def test_final_checkpoint_repairs_interrupted_completion_marker(completed, tmp_path):
    copy_completed(completed[0], tmp_path)
    (tmp_path / "training_summary.json").unlink()
    (tmp_path / "metrics.jsonl").write_text("")
    cfg = settings(tmp_path)
    cfg["resume_if_present"] = True
    result = trainer.fit(model(), cfg, TinyDataset(), TinyDataset(), identity(cfg))
    assert result["completed_epochs"] == 30 and result["status"] == "completed"
    assert trainer.metric_rows(tmp_path) == trainer.metric_rows(completed[0])


def audited_fixture(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    cache = tmp_path / "cache"
    processed.mkdir(); cache.mkdir()
    trainer.atomic_json({"source": "real fixture manifest"}, processed / "manifest.json")
    source_sha = trainer.sha256(processed / "manifest.json")
    trainer.atomic_json({"status": "passed", "dataset_manifest_sha256": source_sha}, processed / "data_audit.json")
    manifest = {"status": "complete", "feature_dim": 8, "action_dim": 10,
                "identity": {"dataset_manifest_sha256": source_sha},
                "episodes": [{"episode_id": str(i), "session_id": str(i), "split": split, "cameras": {}}
                             for i, split in enumerate(("train", "val", "test"))]}
    trainer.atomic_json(manifest, cache / "manifest.json")
    stats = {"fit_split": "train", "camera": trainer.PRIMARY_CAMERA,
             "cache_manifest_sha256": trainer.sha256(cache / "manifest.json"), "ddof": 1, "std_floor": 1e-5,
             "feature_mean": [0.]*8, "feature_std": [1.]*8, "action_mean": [0.]*10, "action_std": [1.]*10}
    trainer.atomic_json(stats, cache / "training_statistics.json")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Registered fixture protocol")
    monkeypatch.setattr(trainer, "source_files", lambda: {})
    return {"cache_root": str(cache), "metadata_audit": str(processed / "data_audit.json"),
            "protocol_path": str(protocol)}


@pytest.mark.parametrize("damage", ["audit", "dataset_hash", "crossed_session", "statistics", "statistics_hash"])
def test_audit_rejects_unverified_or_leaking_data(tmp_path, monkeypatch, damage):
    cfg = audited_fixture(tmp_path, monkeypatch)
    trainer.audit_inputs(cfg)
    if damage in ("audit", "dataset_hash"):
        path = Path(cfg["metadata_audit"])
        value = json.loads(path.read_text())
        value["status" if damage == "audit" else "dataset_manifest_sha256"] = "bad"
    elif damage == "crossed_session":
        path = Path(cfg["cache_root"]) / "manifest.json"
        value = json.loads(path.read_text())
        value["episodes"][1]["session_id"] = value["episodes"][0]["session_id"]
    else:
        path = Path(cfg["cache_root"]) / "training_statistics.json"
        value = json.loads(path.read_text())
        value["fit_split" if damage == "statistics" else "cache_manifest_sha256"] = "test"
    trainer.atomic_json(value, path)
    with pytest.raises(ValueError):
        trainer.audit_inputs(cfg)


def test_statistics_are_recomputed_only_from_training_episode_arrays():
    arrays = np.arange(24, dtype=np.float32).reshape(3,8)
    dataset = type("Data", (), {"episodes": [{"features": arrays, "actions": arrays[:2]}]})()
    stats = {"counts": {"feature": 3, "action": 2}, "feature_mean": arrays.mean(0),
             "feature_std": arrays.std(0,ddof=1), "action_mean": arrays[:2].mean(0),
             "action_std": arrays[:2].astype(np.float64).std(0,ddof=1)}
    trainer.verify_training_statistics(dataset, stats)
    stats["feature_mean"] = arrays.mean(0) + .1
    with pytest.raises(ValueError):
        trainer.verify_training_statistics(dataset, stats)


def test_matched_h5_prefix_keeps_exact_h10_start_population():
    sys.path.insert(0,str(ROOT/'scripts/real_video_development'))
    import horizon10_campaign as campaign
    class Data(TinyDataset):
        horizon=10
        episodes=[{'episode_id':'a','session_id':'a','split':'val'},{'episode_id':'b','session_id':'b','split':'val'}]
        windows=[(0,0),(0,5),(0,10),(0,15),(1,0)]
        def __getitem__(self,index):
            return {**super().__getitem__(index),'window_start':self.windows[index][1]}
    data=Data();prefix=campaign.PrefixDataset(data)
    assert prefix.windows is data.windows and prefix.episodes is data.episodes
    assert len(prefix)==len(data) and prefix.horizon==5
    for i in range(len(data)):
        assert prefix[i]['window_start']==data[i]['window_start']
        assert prefix[i]['episode_index']==data[i]['episode_index']
        torch.testing.assert_close(prefix[i]['features'],data[i]['features'][:8],rtol=0,atol=0)
        torch.testing.assert_close(prefix[i]['actions'],data[i]['actions'][:7],rtol=0,atol=0)
    data.episodes=[{'split':'test'}]
    with pytest.raises(ValueError,match='original validation'):
        campaign.PrefixDataset(data)


def test_package_namespace_cannot_be_loaded_as_original_h5(completed):
    sys.path.insert(0,str(ROOT/'scripts/real_video'))
    import train as original
    with pytest.raises(ValueError,match='Unsupported'):
        original.load_package(completed[0]/'best')
    loaded,state=trainer.load_package(completed[0]/'best')
    assert state['config']['package_kind']=='shiftwm_real_video_droid_horizon10_v1'
    row=TinyDataset()[0]
    with torch.inference_mode():
        forecast=loaded.predict(row['features'][None,:3],row['actions'][None,:2],row['actions'][None,2:])
    assert forecast.shape==(1,10,8) and torch.isfinite(forecast).all()


def test_registered_cli_rejects_architecture_changes_before_loading_data(tmp_path):
    cfg=settings(tmp_path)
    cfg.update(model_config={'hidden_dim':96,'depth':2,'context_dim':16,'context_hidden':64})
    with pytest.raises(ValueError,match='matched-control architecture'):
        trainer.train(cfg)


@pytest.mark.parametrize('mode',['framewise','constant_dynamics','action_free'])
def test_each_other_registered_mode_completes_all30_and_reloads(tmp_path,mode):
    cfg=settings(tmp_path,mode)
    result=trainer.fit(model(mode),cfg,TinyDataset(),TinyDataset(),identity(cfg))
    loaded,state=trainer.load_package(tmp_path/'best')
    assert result['completed_epochs']==30 and len(trainer.metric_rows(tmp_path))==30
    assert loaded.config.mode==mode and state['config']['metadata']['selection']==trainer.AGGREGATION
    assert all(row['val']['episodes']==2 and row['val']['query_steps']==10 for row in trainer.metric_rows(tmp_path))


def test_training_entrypoint_requests_only_train_and_validation_horizon10(monkeypatch,tmp_path):
    cfg=settings(tmp_path)
    cfg.update(model_config={'hidden_dim':192,'depth':4,'context_dim':32,'context_hidden':128},
               lr=1e-4,min_lr=1e-6,weight_decay=.01,batch_size=128,stride=2,cache_root='fixture')
    stats={'feature_mean':[0.]*8,'feature_std':[1.]*8,'action_mean':[0.]*10,'action_std':[1.]*10}
    monkeypatch.setattr(trainer,'audit_inputs',lambda config:({'feature_dim':8,'action_dim':10},stats,identity(config)))
    seen=[]
    def dataset(root,split,horizon,stride,verify):
        assert split in ('train','val') and horizon==10 and verify
        seen.append((split,horizon,stride))
        return TinyDataset()
    monkeypatch.setattr(trainer,'RealVideoDataset',dataset)
    monkeypatch.setattr(trainer,'verify_training_statistics',lambda data,values:None)
    monkeypatch.setattr(trainer,'fit',lambda *args:{'status':'fixture'})
    assert trainer.train(cfg)=={'status':'fixture'}
    assert seen==[('train',10,2),('val',10,5)]


def test_paired_diagnostic_rejects_mismatched_window_population():
    sys.path.insert(0,str(ROOT/'scripts/real_video_development'))
    import horizon10_campaign as campaign
    records=[]
    for seed in (0,1,2):
        episode={'episode_id':'one','session_id':'session','window_starts':[0,5],
                 'errors':{'model':{'mean_standardized_mse':.1}}}
        values={'result':{'episodes':[episode]}}
        records.append({'seed':seed,'mode':'factorized','evaluations':{'new':deepcopy(values),'old':deepcopy(values)}})
    records[2]['evaluations']['old']['result']['episodes'][0]['window_starts']=[0,10]
    with pytest.raises(ValueError,match='windows/sessions/episodes'):
        campaign.paired(records,'new','old','mean_standardized_mse')
