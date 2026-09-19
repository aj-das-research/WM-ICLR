"""Meaningful causal, decoder, identity and full-30 continuation tests."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from shiftwm.real_video_spatial.model import SpatialWorldModel
from shiftwm.real_video_spatial_components.model import ComponentWorldModel, MODES, CONTROL_FOR, PACKAGE_KIND, SCHEMA, from_config

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def module(name):
    spec = importlib.util.spec_from_file_location("test_component_private_" + name, ROOT / "scripts/real_video_spatial_components" / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def make(mode, learned=False, seed=42, full=False):
    torch.manual_seed(seed)
    config = {"mode": mode} if full else {"mode": mode, "hidden_dim": 12, "depth": 1, "context_dim": 4, "context_hidden": 8}
    cls = ComponentWorldModel if mode in MODES else SpatialWorldModel
    m = cls(config, torch.zeros(6144), torch.ones(6144), torch.zeros(35), torch.ones(35)).eval()
    if learned:
        with torch.no_grad():
            m.output_projection[-1].weight.normal_(0, .05)
            m.output_projection[-1].bias.normal_(0, .03)
            m.context_adapter.affine.weight.normal_(0, .03)
            m.context_adapter.affine.bias.normal_(0, .02)
            # LeWM's action-conditioned AdaLN is also zero initialized. Make
            # only this learned-like fixture nonzero so action tests exercise
            # real computation instead of passing with a disconnected gate.
            for layer in m.predictor.transformer.layers:
                layer.adaLN_modulation[-1].weight.normal_(0, .03)
                layer.adaLN_modulation[-1].bias.normal_(0, .02)
    return m


def inputs():
    g = torch.Generator().manual_seed(73)
    return torch.randn(2, 3, 6144, generator=g), torch.randn(2, 2, 35, generator=g), torch.randn(2, 4, 35, generator=g)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", (0, 1, 2))
def test_exact_matched_initial_tensors_rng_masks_and_zero_head_predictions(mode, seed):
    new = make(mode, seed=seed)
    new_rng = torch.get_rng_state().clone()
    old = make(CONTROL_FOR[mode], seed=seed)
    torch.testing.assert_close(torch.get_rng_state(), new_rng, rtol=0, atol=0)
    assert list(new.state_dict()) == list(old.state_dict())
    for name, tensor in new.state_dict().items():
        torch.testing.assert_close(tensor, old.state_dict()[name], rtol=0, atol=0)
    assert {k: p.requires_grad for k, p in new.named_parameters()} == {k: p.requires_grad for k, p in old.named_parameters()}
    with torch.inference_mode():
        torch.testing.assert_close(new.predict(*inputs()), old.predict(*inputs()), rtol=0, atol=0)


@pytest.mark.parametrize("mode,expected", (("bounded_additive", 1007776), ("unbounded_transport", 1026305)))
def test_full_registered_active_capacity(mode, expected):
    m = make(mode, full=True)
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == expected
    assert sum(p.numel() for p in m.parameters()) == 1026305


@pytest.mark.parametrize("mode", MODES)
def test_learned_heads_prefix_causality_query_invariance_and_action_sensitivity(mode):
    m = make(mode, learned=True)
    support, past, future = inputs()
    with torch.no_grad():
        expected = m.predict(support, past, future)
        changed = future.clone(); changed[:, 2:] += 100
        torch.testing.assert_close(expected[:, :2], m.predict(support, past, changed)[:, :2], rtol=0, atol=0)
        torch.testing.assert_close(expected[:, :2], m.predict(support, past, future[:, :2]), rtol=1e-6, atol=1e-6)
        assert not torch.equal(expected[:, 2:], m.predict(support, past, changed)[:, 2:])
        frames = torch.cat((support, torch.randn(2, 4, 6144)), 1)
        batch = {"features": frames, "actions": torch.cat((past, future), 1)}
        before = m(batch)["predictions"]
        batch["features"][:, 3:] += 500
        torch.testing.assert_close(before, m(batch)["predictions"], rtol=0, atol=0)
    features = torch.cat((support, torch.randn(2, 4, 6144)), 1).requires_grad_(True)
    actions = torch.cat((past, future), 1).requires_grad_(True)
    pred = m({"features": features, "actions": actions})["predictions"][:, :2].square().mean()
    gx, ga = torch.autograd.grad(pred, (features, actions))
    assert torch.count_nonzero(gx[:, 3:]) == 0 and torch.count_nonzero(ga[:, 4:]) == 0
    assert gx[:, :3].abs().sum() > 0 and ga[:, :4].abs().sum() > 0


@pytest.mark.parametrize("mode", MODES)
def test_real_parameter_gradients_and_inactive_heads(mode):
    m = make(mode, learned=True)
    y = m.predict(*inputs())
    (y-torch.randn_like(y)).square().mean().backward()
    for name in ("input_projection.1.weight", "action_prefix.weight_ih_l0", "context_adapter.affine.weight", "output_projection.1.weight"):
        gradient = dict(m.named_parameters())[name].grad
        assert gradient is not None and gradient.isfinite().all() and gradient.abs().sum() > 0, name
    for block in (m.transport_query, m.transport_key, m.gate):
        for p in block.parameters():
            if mode == "unbounded_transport": assert p.grad is not None and p.grad.abs().sum() > 0
            else: assert p.grad is None and not p.requires_grad


def test_bounded_innovation_and_unbounded_counterexample():
    s, p, a = inputs()
    for mode in MODES:
        model = make(mode)
        with torch.no_grad(): model.output_projection[-1].bias.fill_(3.)
        pred, details = model._predict_normalized(s, p, a, True)
        if mode == "bounded_additive":
            assert all(d["innovation"].abs().max() <= 1 for d in details)
            assert (pred-s[:, -1:]).abs().max() <= 1.000001
        else:
            assert all(d["innovation"].min() == 3 for d in details)
            zero = torch.zeros_like(s)
            assert model._predict_normalized(zero, p, a).abs().min() > 1
            for d in details:
                torch.testing.assert_close(d["transport"].sum(-1), torch.ones(2, 16))
                assert (d["transport"] >= 0).all() and ((d["gate"] > 0) & (d["gate"] < 1)).all()


@pytest.mark.parametrize("mode", MODES)
def test_package_schema_refuses_other_kind_and_original_modes(mode):
    c = make(mode).package_config
    from_config(c)
    for key, value in (("component_schema", "wrong"), ("package_kind", "shiftwm_real_video_spatial_v1"), ("coordinate_layout", "1536")):
        with pytest.raises(ValueError): from_config({**c, key: value})
    c["model_config"]["mode"] = "transport"
    with pytest.raises(ValueError): from_config(c)


def save_fixture(t, m, destination):
    counts = {"total": sum(p.numel() for p in m.parameters()), "trainable": sum(p.numel() for p in m.parameters() if p.requires_grad)}
    optimizer = torch.optim.AdamW((p for p in m.parameters() if p.requires_grad), lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30)
    t.save_package(m, destination, optimizer=optimizer, scheduler=scheduler, epoch=30, step=30, best_metric=1.,
                   metadata={"training_identity": "fixture", "parameter_counts": counts}, generator=torch.Generator(), history=[])


@pytest.mark.parametrize("mode", MODES)
def test_atomic_package_relocation_distinct_kind_and_offline_source_copy(tmp_path, monkeypatch, mode):
    t = module("train")
    m = make(mode, learned=True)
    source = tmp_path / "source"
    save_fixture(t, m, source / "best")
    shutil.copytree(source / "best", tmp_path / "copy")
    restored, state = t.load_package(tmp_path / "copy", "cpu")
    assert state["config"]["package_kind"] == PACKAGE_KIND
    with torch.no_grad(): torch.testing.assert_close(m.predict(*inputs()), restored.predict(*inputs()), rtol=0, atol=0)
    with pytest.raises(ValueError, match="Unsupported real-video package"):
        module("campaign").module("train", old=True).read_package(tmp_path / "copy")
    inference = module("inference")
    class Data:
        def __init__(self, *args, **kwargs): pass
        def __getitem__(self, index):
            g = torch.Generator().manual_seed(12)
            return {"features": torch.randn(13, 6144, generator=g), "actions": torch.randn(12, 35, generator=g)}
    monkeypatch.setattr(inference, "SpatialDataset", Data)
    proof = inference.export_and_check({"output_dir": str(source), "cache_root": "fixture"}, tmp_path / "export")
    assert proof["status"] == "passed" and proof["max_abs_error"] == 0
    assert set(p.name for p in (tmp_path / "export").iterdir()) == {"model.pt", "config.json", "package_manifest.json"}


@pytest.mark.parametrize("mode", MODES)
def test_full_thirty_epoch_atomic_resume_exact(tmp_path, mode):
    t = module("train")
    torch.set_num_threads(1)
    class Data(torch.utils.data.Dataset):
        def __init__(self):
            g = torch.Generator().manual_seed(67)
            self.x = torch.randn(1, 13, 6144, generator=g); self.a = torch.randn(1, 12, 35, generator=g)
        def __len__(self): return 1
        def __getitem__(self, i): return {"features": self.x[i], "actions": self.a[i], "episode_index": i}
    def fresh():
        t.base.seed_everything(0)
        return ComponentWorldModel({"mode": mode, "hidden_dim": 6, "depth": 1, "context_dim": 4, "context_hidden": 8},
                                   [0.]*6144, [1.]*6144, [0.]*35, [1.]*35)
    def config(path):
        return {"epochs": 30, "seed": 0, "mode": mode, "output_dir": str(path), "device": "cpu", "batch_size": 1,
                "lr": .001, "min_lr": .00001, "weight_decay": .01, "grad_clip": 1., "bf16": False,
                "train_horizon": 10, "validation_horizon": 10, "component_schema": SCHEMA}
    data = Data(); cfg = config(tmp_path / "full")
    identity = lambda c: {"scientific_config": t.base.scientific_config(c), "dependencies": {}}
    result = t.base.fit(fresh(), cfg, data, data, identity(cfg))
    assert result["completed_epochs"] == 30
    resumed = config(tmp_path / "resumed"); resumed["max_runtime_seconds"] = 1e-12
    partial = t.base.fit(fresh(), resumed, data, data, identity(resumed))
    assert partial["completed_epochs"] == 1
    with pytest.raises(ValueError): t.validate_completed(tmp_path / "resumed")
    resumed.pop("max_runtime_seconds"); resumed["resume_if_present"] = True
    result = t.base.fit(fresh(), resumed, data, data, identity(resumed))
    assert result["completed_epochs"] == 30
    a = t.read_package(tmp_path / "full/last")[1]; b = t.read_package(tmp_path / "resumed/last")[1]
    assert a["history"] == b["history"]
    for key, value in a["state_dict"].items(): torch.testing.assert_close(value, b["state_dict"][key], rtol=0, atol=0)
    assert t.validate_completed(tmp_path / "resumed")["completed_epochs"] == 30


def analysis_fixture():
    a = module("analysis")
    levels = {"anchored_additive": 5., "bounded_additive": 4., "unbounded_transport": 4., "transport": 2.}
    rows = []
    for mode in a.MODES:
        for seed in (0, 1, 2):
            episodes = []
            for i, session in enumerate(("session0", "session0", "session1")):
                episodes.append({"episode_id": str(i), "session_id": session,
                    **{metric: [levels[mode]+seed+i/10]*10 for metric in a.helper().METRICS}})
            rows.append({"mode": mode, "seed": seed, "episodes": episodes,
                         "windows": [{"episode_id": str(i), "session_id": s, "window_start": 0} for i, s in enumerate(("session0", "session0", "session1"))]})
    return rows


def test_factorial_twenty_effects_and_paired_noise_cancellation():
    a = module("analysis")
    result = a.contrasts(analysis_fixture(), draws=200)
    assert len(result["reported_effects"]) == 20
    for effect in result["reported_effects"]:
        if effect["contrast"] == "mixing_x_bounding_interaction":
            assert effect["difference_of_differences"] == pytest.approx(-1)
            np.testing.assert_allclose(effect["paired_95_percent_interval"], [-1, -1], atol=1e-14)
    actual = result["interaction_all_horizons"]
    assert actual["episode_count"] == 3 and actual["session_count"] == 2 and actual["seed_count"] == 3


@pytest.mark.parametrize("change", ("missing_seed", "duplicate_seed", "session", "window", "nonfinite", "negative"))
def test_interaction_fails_closed_for_bad_population_or_errors(change):
    rows = analysis_fixture()
    if change == "missing_seed": rows.pop()
    elif change == "duplicate_seed": rows[1] = copy.deepcopy(rows[0])
    elif change == "session": rows[-1]["episodes"][0]["session_id"] = "wrong"
    elif change == "window": rows[-1]["windows"].pop()
    elif change == "nonfinite": rows[-1]["episodes"][0]["native_mse"][0] = float("nan")
    else: rows[-1]["episodes"][0]["native_mse"][0] = -1
    with pytest.raises(ValueError): module("analysis").interaction(rows, draws=10)


def test_private_rebinding_does_not_change_original_family():
    new = module("train")
    old = module("campaign").module("train", old=True)
    assert new.base.PACKAGE_KIND == PACKAGE_KIND
    assert old.base.PACKAGE_KIND == "shiftwm_real_video_spatial_v1"
    assert old.SpatialWorldModel is SpatialWorldModel
    assert new.legacy.SpatialWorldModel is ComponentWorldModel


def test_all_six_exact_matched_configurations_and_ws_only_scripts():
    c = module("campaign")
    for mode in MODES:
        for seed in (0, 1, 2):
            cfg = c.expected_config(mode, seed)
            assert cfg["epochs"] == 30 and cfg["batch_size"] == 128 and cfg["train_horizon"] == cfg["validation_horizon"] == 10
            assert cfg["component_schema"] == SCHEMA and cfg["output_dir"] == f"runs/real_video_spatial_components/v1/{mode}_s{seed}"
    training = (ROOT / "scripts/real_video_spatial_components/train.slurm").read_text()
    finalizer = (ROOT / "scripts/real_video_spatial_components/finalize.slurm").read_text()
    assert "--partition=ws-ia" in training and "--gres=gpu:1" in training
    assert "--partition=ws-ia" in finalizer and "--gres" not in finalizer


def registry_fixture(tmp_path, monkeypatch):
    c = module("campaign")
    from shiftwm.real_video.data import sha256
    old = tmp_path / "old.json"; old.write_text("original registry")
    source = tmp_path / "source.py"; source.write_text("unchanged")
    monkeypatch.setattr(c, "ROOT", tmp_path); monkeypatch.setattr(c, "OLD_REG", old)
    monkeypatch.setattr(c, "OLD_REG_SHA", sha256(old)); monkeypatch.setattr(c, "REG", tmp_path / "registry.json")
    monkeypatch.setattr(c, "expected_config", lambda mode, seed: {"mode": mode, "seed": seed, "full_recipe": 30})
    registry = {"status": "registered", "component_schema": SCHEMA, "package_kind": PACKAGE_KIND,
                "expected_new_runs": 6, "expected_control_runs": 6, "expected_epochs_per_run": 30,
                "old_registration_sha256": sha256(old), "dependencies": {"source.py": sha256(source)}}
    for key, modes in (("runs", MODES), ("controls", c.CONTROLS)):
        registry[key] = []
        for mode in modes:
            for seed in (0, 1, 2):
                name = f"{mode}_s{seed}"; path = tmp_path / (name + ".json")
                path.write_text(json.dumps(c.expected_config(mode, seed)))
                registry[key].append({"name": name, "mode": mode, "seed": seed, "config": path.name, "sha256": sha256(path)})
    c.REG.write_text(json.dumps(registry))
    return c, registry


@pytest.mark.parametrize("mutation", ("source", "duplicate", "package", "recipe", "control"))
def test_registration_fail_closed(tmp_path, monkeypatch, mutation):
    from shiftwm.real_video.data import sha256
    c, registry = registry_fixture(tmp_path, monkeypatch)
    assert len(c.verify()["runs"]) == 6
    if mutation == "source": (tmp_path / "source.py").write_text("changed")
    elif mutation == "duplicate": registry["runs"][1] = registry["runs"][0]
    elif mutation == "package": registry["package_kind"] = "old"
    elif mutation == "control": registry["controls"].pop()
    else:
        row = registry["runs"][0]; path = tmp_path / row["config"]
        cfg = json.loads(path.read_text()); cfg["full_recipe"] = 3
        path.write_text(json.dumps(cfg)); row["sha256"] = sha256(path)
    c.REG.write_text(json.dumps(registry))
    with pytest.raises(ValueError): c.verify()


def test_complete_executed_recipe_must_match_registered_not_only_self_consistent(tmp_path):
    campaign, trainer = module("campaign"), module("train")
    registered = {"mode": "bounded_additive", "seed": 0, "epochs": 30, "lr": .0001, "batch_size": 128,
                  "output_dir": "registered", "device": "cuda"}
    operational = {**registered, "output_dir": "relocated", "device": "cpu", "resume_if_present": True}
    path = tmp_path / "training_config.json"
    path.write_text(json.dumps(operational))
    campaign.require_executed_recipe(trainer, tmp_path, registered)
    for key, value in (("lr", .1), ("batch_size", 1), ("epochs", 3), ("mode", "unbounded_transport")):
        path.write_text(json.dumps({**operational, key: value}))
        with pytest.raises(ValueError, match="Executed scientific training recipe"):
            campaign.require_executed_recipe(trainer, tmp_path, registered)
