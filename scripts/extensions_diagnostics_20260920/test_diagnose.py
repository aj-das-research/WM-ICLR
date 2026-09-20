"""Synthetic fail-closed gates and exact frozen geometry-function tests."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

SPEC = importlib.util.spec_from_file_location("extension_diagnostic", Path(__file__).with_name("diagnose.py"))
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


def roster():
    return [{"family": f, "mode": m, "seed": s, "architecture": "transformer", "domain": "drone"}
            for f in ("narrow", "wide") for m in d.MODES for s in range(3)]


def test_exact_roster_and_duplicates():
    d.validate_roster(roster())
    for bad in (roster()[:-1], roster()[:-1] + [roster()[0]],
                [{**r, "architecture": "gru"} for r in roster()]):
        with pytest.raises(ValueError):
            d.validate_roster(bad)


def test_development_selection_ignores_test_metadata():
    eps = [{"split": "development", "seed": s, "dynamics_id": i,
            "file": f"episodes/development-{s}-{i}.npz", "audit_file": f"audit/development-{s}-{i}.npz"}
           for s in range(16) for i in range(3)]
    actual = d.dev_episodes({"episodes": eps + [{"split": "test", "file": "never-open"}]})
    assert actual == eps
    for bad in (eps[:-1], [*eps[:-1], eps[0]], [{**e, "file": "episodes/test.npz"} for e in eps]):
        with pytest.raises(ValueError):
            d.dev_episodes({"episodes": bad})


def fixture(tmp_path):
    (tmp_path / "bound.txt").write_text("immutable")
    registry = {"models": roster(), "sources_sha256": {"bound.txt": d.sha(tmp_path / "bound.txt")}}
    d.write_new(tmp_path / d.OUT / "registration.json", registry)
    review = {"status": "passed", "registration_sha256": d.sha(tmp_path / d.OUT / "registration.json")}
    return review


def test_review_required_before_source_access(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.setattr(d, "relative", lambda *a: pytest.fail("Sources accessed before review"))
    with pytest.raises(FileNotFoundError):
        d.checked(tmp_path)


@pytest.mark.parametrize("key,value", [("status", "pending"), ("registration_sha256", "0" * 64)])
def test_bad_review_rejected(tmp_path, key, value):
    review = fixture(tmp_path)
    review[key] = value
    d.write_new(tmp_path / d.OUT / "source_review.json", review)
    with pytest.raises(ValueError, match="review"):
        d.checked(tmp_path)


def test_source_tamper_rejected(tmp_path):
    review = fixture(tmp_path)
    d.write_new(tmp_path / d.OUT / "source_review.json", review)
    assert d.checked(tmp_path)["models"] == roster()
    (tmp_path / "bound.txt").write_text("changed")
    with pytest.raises(ValueError, match="Changed registered input"):
        d.checked(tmp_path)


def test_paths_confined_and_write_once(tmp_path):
    with pytest.raises(ValueError):
        d.relative(tmp_path, "../escape")
    with pytest.raises(ValueError):
        d.relative(tmp_path, "/tmp")
    (tmp_path / "outside").symlink_to("/etc/hosts")
    with pytest.raises(ValueError):
        d.relative(tmp_path, "outside")
    d.write_new(tmp_path / "one.json", {"a": 1})
    with pytest.raises(FileExistsError):
        d.write_new(tmp_path / "one.json", {"a": 2})


def test_frozen_helper_no_top_level_execution_and_geometry():
    x = torch.arange(16, dtype=torch.float32)[:, None].repeat(1, 3)
    tri = torch.triu_indices(16, 16, 1)
    physical = ((x[:, None, 0] - x[None, :, 0]) ** 2)[tri[0], tri[1]].numpy()
    helper = d.geometry_helper(d.ROOT, physical, 16)
    raw = helper(x)
    translated = helper(x + 10)
    scaled = helper(x * 2)
    assert raw == translated
    assert np.isclose(raw["spearman_with_squared_xy_distance"], 1)
    assert scaled["pairwise_latent_mse_mean"] == 4 * raw["pairwise_latent_mse_mean"]
    assert np.isclose(raw["pairwise_latent_mse_mean"], physical.mean())
    with pytest.raises(ValueError):
        d.geometry_helper(d.ROOT, physical[:-1], 16)


def test_extracted_tissue_success_rule_keeps_invalid_failures():
    f = d.original_function(d.ROOT, "src/shiftwm/extensions/evaluate.py", "valid_success", {})
    assert f({"success": True})
    for key, value in (("valid_action", False), ("stable_deformation", False),
                       ("crash", True), ("workspace_escape", True)):
        assert not f({"success": True, key: value})
    assert not f({"success": False})
