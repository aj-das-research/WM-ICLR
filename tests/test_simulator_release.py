import importlib.util
import json
from pathlib import Path

import pytest
import torch

from shiftwm import simulator_release as release


@pytest.fixture
def package(tmp_path):
    (tmp_path / "shared").mkdir()
    (tmp_path / "models").mkdir()
    torch.save({"shared": torch.tensor([1.25, 2.5])}, tmp_path / "shared/common.pt")
    config = {"extension_format_version": 1, "mode": "factorized"}
    (tmp_path / "models/config.json").write_text(json.dumps(config))
    torch.save({"state_dict": {"delta": torch.tensor([3.75])}, "config": config,
                "epoch": 25, "step": 250, "best_metric": .01}, tmp_path / "models/delta.pt")
    spec = lambda name: {"path": name, "sha256": release.sha256(tmp_path / name)}
    registry = {"kind": release.KIND, "format_version": 1,
        "shared": {**spec("shared/common.pt"), "tensor_count": 1}, "models": [{
            "id": "original/example", "family": "original", "selected_epoch": 25,
            "state_tensor_count": 2, "config": spec("models/config.json"), "delta": spec("models/delta.pt")}]}
    (tmp_path / "models.json").write_text(json.dumps(registry))
    return tmp_path


def update_registry(root, edit):
    value = json.loads((root / "models.json").read_text())
    edit(value)
    (root / "models.json").write_text(json.dumps(value))


def test_exact_roundtrip_and_relocation(package, tmp_path_factory):
    import shutil
    moved = tmp_path_factory.mktemp("elsewhere") / "relocated"
    shutil.copytree(package, moved)
    shutil.rmtree(package)
    state, record = release.read_package(moved, "original/example")
    assert state["epoch"] == 25
    assert torch.equal(state["state_dict"]["shared"], torch.tensor([1.25, 2.5]))
    assert torch.equal(state["state_dict"]["delta"], torch.tensor([3.75]))


@pytest.mark.parametrize("name", ["shared/common.pt", "models/delta.pt", "models/config.json"])
def test_tampered_payload_rejected(package, name):
    with (package / name).open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        release.read_package(package, "original/example")


@pytest.mark.parametrize("kind", ["unknown", "duplicate", "wrong_count", "wrong_epoch", "path_escape"])
def test_manifest_contract_rejected(package, kind):
    def edit(value):
        if kind == "unknown":
            value["kind"] = "different"
        elif kind == "duplicate":
            value["models"].append(dict(value["models"][0]))
        elif kind == "wrong_count":
            value["shared"]["tensor_count"] = 7
        elif kind == "wrong_epoch":
            value["models"][0]["selected_epoch"] = 30
        else:
            value["shared"]["path"] = "../outside.pt"
    update_registry(package, edit)
    with pytest.raises(ValueError):
        release.read_package(package, "original/example")


def test_missing_model_rejected(package):
    with pytest.raises(ValueError, match="Unknown"):
        release.read_package(package, "original/missing")


def test_symlink_rejected(package):
    path = package / "shared/common.pt"
    target = package / "actual.pt"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="Symlink"):
        release.read_package(package, "original/example")


@pytest.mark.parametrize("delta", [{"shared": torch.tensor([1.])}, {"bad": "not a tensor"}])
def test_overlapping_or_nontensor_states_rejected(delta):
    with pytest.raises(ValueError):
        release.merge_tensors({"shared": torch.tensor([1.])}, delta)


def test_payload_cannot_contain_optimizer(package):
    path = package / "models/delta.pt"
    value = torch.load(path, weights_only=True)
    value["optimizer"] = {}
    torch.save(value, path)
    update_registry(package, lambda v: v["models"][0]["delta"].update(sha256=release.sha256(path)))
    with pytest.raises(ValueError, match="payload fields"):
        release.read_package(package, "original/example")


def test_registry_exact_population():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("sim_export", root / "scripts/publishing/publish_simulator_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.inventory_rows(root)
    assert len(rows) == 42
    assert sum(row["family"] == "original" for row in rows) == 36
    assert sum(row["family"] == "geometry" for row in rows) == 6
