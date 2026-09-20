"""Synthetic portability/boundary contracts; never load dataset examples."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location("iws_release_v2_test_" + name, HERE / (name + ".py"))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


runtime = module("runtime")


@pytest.fixture
def bundle(tmp_path):
    rows = [{"name": f"{t}_{m}_s{s}", "directory": f"models/{t}_{m}_s{s}",
             "task": t, "mode": m, "seed": s, "completed_epochs": 30,
             "package_kind": runtime.KINDS[m == "unbounded_spatial_mix"]}
            for t in runtime.TASKS for m in runtime.MODES for s in range(3)]
    files = set(runtime.RUNTIME_FILES)
    for row in rows:
        files.update(row["directory"] + "/" + n for n in ("model.pt", "config.json", "package_manifest.json"))
        files.add("model_cards/" + row["name"] + ".md")
        files.add("fixtures/" + row["name"] + "_b1_prediction.npz")
    for name in files:
        target = tmp_path / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(b"synthetic-fixture")
    doc = {"schema": runtime.SCHEMA, "backend": runtime.BACKEND, "device": "cpu", "precision": "float32",
           "threads": 8, "interop_threads": 1, "models": rows,
           "fixture_scope": "deterministic synthetic inputs; no dataset examples or targets",
           "files": {name: runtime.sha(tmp_path / name) for name in files}}
    def commit():
        (tmp_path / "manifest.json").write_text(json.dumps(doc))
        (tmp_path / "SHA256SUMS").write_text("".join(f"{runtime.sha(tmp_path / n)}  {n}\n" for n in sorted(set(doc["files"]) | {"manifest.json"})))
    commit(); return tmp_path, doc, commit


def test_complete_36_grid_and_public_checksums(bundle):
    root, _, _ = bundle
    assert len(runtime.verify_bundle(root)["models"]) == 36


@pytest.mark.parametrize("field,value", [("backend", "native"), ("threads", 2), ("device", "cuda"),
                                         ("precision", "bfloat16"), ("fixture_scope", "real training example")])
def test_runtime_or_fixture_contract_changes_rejected(bundle, field, value):
    root, doc, commit = bundle; doc[field] = value; commit()
    with pytest.raises(ValueError): runtime.verify_bundle(root)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_kind", "wrong_epoch_budget", "traversing_directory"])
def test_roster_and_selected_package_identity_rejected(bundle, mutation):
    root, doc, commit = bundle
    if mutation == "missing": doc["models"].pop()
    elif mutation == "duplicate": doc["models"][-1] = doc["models"][0]
    elif mutation == "wrong_kind": doc["models"][0]["package_kind"] = runtime.KINDS[True]
    elif mutation == "wrong_epoch_budget": doc["models"][0]["completed_epochs"] = 29
    else: doc["models"][0]["directory"] = "../outside"
    commit()
    with pytest.raises(ValueError): runtime.verify_bundle(root)


def test_changed_payload_and_unlisted_state_rejected(bundle):
    root, _, _ = bundle
    (root / "README.md").write_text("changed")
    with pytest.raises(ValueError, match="digest"): runtime.verify_bundle(root)
    (root / "README.md").write_bytes(b"synthetic-fixture")
    (root / "training_state.pt").write_bytes(b"optimizer")
    with pytest.raises(ValueError, match="unlisted"): runtime.verify_bundle(root)


def test_symlinks_and_unsafe_paths_rejected(bundle):
    root, _, _ = bundle
    for name in ("../escape", "/tmp/escape", "a\\b", "a/../b", "", None):
        with pytest.raises(ValueError): runtime.regular(root, name)
    (root / "alias").symlink_to(root / "README.md")
    with pytest.raises(ValueError): runtime.verify_bundle(root)


def test_public_load_always_verifies_inventory_before_import(monkeypatch, bundle):
    root, _, _ = bundle
    (root / "README.md").write_text("changed")
    monkeypatch.setattr(runtime, "configure_runtime", lambda: pytest.fail("Imported runtime before inventory validation"))
    with pytest.raises(ValueError, match="digest"): runtime.load_model("pusht_autoregressive_s0", root)
    with pytest.raises(TypeError): runtime.load_model("pusht_autoregressive_s0", root, verified_manifest={})


def test_synthetic_inputs_are_deterministic_batch_prefixes_without_statistics():
    for task, width in runtime.TASKS.items():
        big_x, big_u = runtime.synthetic_inputs(task, 64)
        for batch in (1, 8):
            x, u = runtime.synthetic_inputs(task, batch)
            assert np.array_equal(big_x[:batch], x) and np.array_equal(big_u[:batch], u)
            assert x.dtype == u.dtype == np.float32 and u.shape == (batch, 60, width)
        assert abs(big_x).max() <= .5 and abs(big_u).max() <= .25
    with pytest.raises(ValueError): runtime.synthetic_inputs("unknown", 1)


def test_predictor_rejects_wrong_precision_and_thread_contract(monkeypatch):
    class Tiny(torch.nn.Module):
        def __init__(self): super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(1.))
        def predict(self, x, u): return x[:, None] + u[:, 1:, :1]
    model = runtime.Predictor(Tiny().eval(), {})
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    x = torch.zeros(1, 6144); u = torch.zeros(1, 60, 4)
    assert model.predict(x, u).shape == (1, 59, 6144)
    with pytest.raises(ValueError): model.predict(x.double(), u)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    with pytest.raises(ValueError): model.predict(x, u)


def test_prefix_proof_checks_all_offsets_not_only_endpoint():
    class Good:
        def predict(self, x, u): return torch.zeros(len(x), u.shape[1] - 1, 6144)
    class Bad:
        def predict(self, x, u):
            out = torch.zeros(len(x), u.shape[1] - 1, 6144); out[:, 0, 0] = .01; return out
    x = torch.zeros(1, 6144); u = torch.zeros(1, 60, 4); full = torch.zeros(1, 59, 6144)
    assert len(runtime.prefix_checks(Good(), x, u, full)) == 3
    with pytest.raises(ValueError, match="prefix"): runtime.prefix_checks(Bad(), x, u, full)


def test_builder_requires_review_before_hashing_or_loading(monkeypatch, tmp_path):
    builder = module("build")
    reg = tmp_path / "registration.json"; review = tmp_path / "source_review.json"
    reg.write_text(json.dumps({"dependencies": {"forbidden": "bad"}}))
    review.write_text(json.dumps({"status": "passed", "registration_sha256": "stale"}))
    monkeypatch.setattr(builder, "REG", reg); monkeypatch.setattr(builder, "REVIEW", review)
    monkeypatch.setattr(builder, "authorized_models", lambda: pytest.fail("Source model touched before review"))
    with pytest.raises(ValueError, match="review missing or stale"): builder.verify()


def test_builder_source_closure_excludes_real_fixtures_and_cached_payloads():
    builder = module("build")
    paths = builder.sources([{"source_package": "artifacts/releases/synthetic/models/test"}])
    assert not any(p.suffix in (".npz", ".h5", ".hdf5", ".mp4", ".safetensors") for p in paths)
    assert not any("/fixtures/" in str(p) or "/data/features/" in str(p) for p in paths)


def test_destination_guard_preserves_existing_and_external_paths(tmp_path, monkeypatch):
    builder = module("build")
    with pytest.raises(ValueError): builder.destination_guard(builder.OLD[0])
    with pytest.raises(ValueError): builder.destination_guard(tmp_path / "outside")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    dest = tmp_path / "artifacts/releases/new"; dest.parent.mkdir(parents=True)
    dest.with_suffix(".tar.gz").symlink_to(tmp_path / "nonexistent")
    with pytest.raises(ValueError): builder.destination_guard(dest)


def test_cached_external_vendor_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "shiftwm_upstream_module", SimpleNamespace(__file__="/outside/module.py"))
    with pytest.raises(ValueError, match="cached source"): runtime.verify_import_locations(tmp_path)
