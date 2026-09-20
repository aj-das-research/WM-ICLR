"""Synthetic full-roster tests of the separately reviewed reserved access gate."""
import copy
import json
from pathlib import Path

import pytest

from shiftwm.real_video_iws_reserved import protocol as p


def dump(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


@pytest.fixture
def registered(tmp_path):
    root = tmp_path
    dependencies = {}
    source = root / "src/fixture.py"
    source.parent.mkdir(); source.write_text("# Synthetic protocol fixture; no model execution.\n")
    dependencies["src/fixture.py"] = p.sha(source)
    runs = []
    for task in p.TASK_WIDTHS:
        for mode in p.MODES:
            for seed in (0, 1, 2):
                name = f"{task}_{mode}_s{seed}"
                relative = f"synthetic_packages/{name}"
                row = {"name": name, "task": task, "mode": mode, "seed": seed, "package_path": relative}
                for filename, field in (("model.pt", "checkpoint_sha256"), ("config.json", "config_sha256"),
                                        ("package_manifest.json", "package_manifest_sha256")):
                    path = root / relative / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"synthetic-only:{name}:{filename}".encode())
                    row[field] = dependencies[f"{relative}/{filename}"] = p.sha(path)
                runs.append(row)
    registry = {"schema": p.SCHEMA, "status": "registered_before_reserved_access",
                "payloads_read_at_registration": 0, "expected_runs": 36, "runs": runs,
                "dependencies": dependencies,
                "tasks": {t: {"command_width": width, "expected_handles": 200,
                              "expected_trajectories": 10, "episode_ids": [f"{i:06d}" for i in range(10)]}
                          for t, width in p.TASK_WIDTHS.items()},
                "evaluation": {"device": "cpu", "dtype": "float32", "threads": 8, "batch_size": 64,
                               "horizon": 60, "prefix_horizons": [15, 30, 45], "bootstrap_draws": 10000,
                               "bootstrap_seed": 173, "primary_method": "bounded_spatial_mix",
                               "primary_reference": "anchored_additive", "primary_metric": "standardized_mse"}}
    dump(registry, root / p.REGISTRATION_PATH)
    review = {"status": "passed", "registration_sha256": p.sha(root / p.REGISTRATION_PATH),
              "reserved_payloads_read": 0, "source_sha256": dict(dependencies)}
    dump(review, root / p.REVIEW_PATH)
    return root, registry, review


def rebind(root, registry, review):
    dump(registry, root / p.REGISTRATION_PATH)
    review = {**review, "registration_sha256": p.sha(root / p.REGISTRATION_PATH),
              "source_sha256": registry["dependencies"]}
    dump(review, root / p.REVIEW_PATH)


def test_full_36_exact_roster_and_review_are_required(registered):
    root, registry, _ = registered
    assert p.checked_registration(root) == registry
    assert len(p.expected_grid()) == 36
    assert p.checked_registration(root, p.REGISTRATION_PATH) == registry


@pytest.mark.parametrize("fault", ["absent", "failed", "other_registration", "source_map", "reserved_read"])
def test_missing_or_tampered_independent_review_denies_access(registered, fault):
    root, _, review = registered
    if fault == "absent": (root / p.REVIEW_PATH).unlink()
    else:
        if fault == "failed": review["status"] = "pending"
        elif fault == "other_registration": review["registration_sha256"] = "0" * 64
        elif fault == "source_map": review["source_sha256"].pop("src/fixture.py")
        else: review["reserved_payloads_read"] = 1
        dump(review, root / p.REVIEW_PATH)
    with pytest.raises((ValueError, FileNotFoundError)): p.checked_registration(root)


def test_source_and_checkpoint_mutation_rejected(registered):
    root, registry, _ = registered
    source = root / "src/fixture.py"; original = source.read_bytes(); source.write_bytes(original + b"# changed\n")
    with pytest.raises(ValueError, match="dependency changed"): p.checked_registration(root)
    source.write_bytes(original)
    checkpoint = root / registry["runs"][0]["package_path"] / "model.pt"
    checkpoint.write_bytes(b"changed selected weights")
    with pytest.raises(ValueError, match="dependency changed"): p.checked_registration(root)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong_seed", "wrong_mode", "wrong_name", "unbound_checkpoint", "wrong_width", "partial_handles", "wrong_primary", "wrong_prefix"])
def test_scientific_grid_and_input_contract_fail_closed(registered, fault):
    _, base, _ = registered; registry = copy.deepcopy(base)
    if fault == "missing": registry["runs"].pop()
    elif fault == "duplicate": registry["runs"][-1] = copy.deepcopy(registry["runs"][0])
    elif fault == "wrong_seed": registry["runs"][0]["seed"] = 7
    elif fault == "wrong_mode": registry["runs"][0]["mode"] = "new_posthoc_model"
    elif fault == "wrong_name": registry["runs"][0]["name"] = "ambiguous"
    elif fault == "unbound_checkpoint": registry["runs"][0]["checkpoint_sha256"] = "0" * 64
    elif fault == "wrong_width": registry["tasks"]["pusht"]["command_width"] = 14
    elif fault == "partial_handles": registry["tasks"]["pusht"]["expected_handles"] = 199
    elif fault == "wrong_primary": registry["evaluation"]["primary_method"] = "unbounded_spatial_mix"
    else: registry["evaluation"]["prefix_horizons"] = [14, 29, 44]
    with pytest.raises(ValueError): p.validate_registration(registry)


@pytest.mark.parametrize("relative", ["../outside", "/tmp/outside", "src/../outside"])
def test_unsafe_dependency_paths_rejected(registered, relative):
    root, registry, review = registered
    registry["dependencies"][relative] = "0" * 64
    rebind(root, registry, review)
    with pytest.raises(ValueError): p.checked_registration(root)


def test_symlink_outside_root_and_external_registration_rejected(registered, tmp_path):
    root, registry, review = registered
    outside = root.parent / (root.name + "-external-source.py"); outside.write_text("external\n")
    link = root / "src/escape.py"; link.symlink_to(outside)
    registry["dependencies"]["src/escape.py"] = p.sha(outside); rebind(root, registry, review)
    with pytest.raises(ValueError, match="leaves"): p.checked_registration(root)
    with pytest.raises(ValueError, match="leaves"): p.checked_registration(root, outside)


def test_review_is_checked_before_any_dependency_file_read(registered, monkeypatch):
    root, _, _ = registered
    (root / p.REVIEW_PATH).unlink()
    original = p.sha; attempted = []
    def guarded(path):
        if Path(path) == root / "src/fixture.py":
            attempted.append(str(path))
            raise AssertionError("Dependency opened before required review")
        return original(path)
    monkeypatch.setattr(p, "sha", guarded)
    with pytest.raises((ValueError, FileNotFoundError)): p.checked_registration(root)
    assert not attempted


@pytest.mark.parametrize("relative", [
    "data/real_video/iws_public_v1/extracted/iws_converted/pusht/traj_000000/camera_0_rgb.mp4",
    "data/real_video/iws_public_v1/extracted/iws_converted/bimanual_box/traj_000004/metadata.h5",
    "data/features/iws_reserved_v1/bimanual_rope/episodes/000002/arrays.npz",
])
def test_reserved_payload_cannot_be_a_preaccess_dependency(registered, monkeypatch, relative):
    root, registry, review = registered
    registry["dependencies"][relative] = "0" * 64; rebind(root, registry, review)
    original = p.sha; attempted = []
    def guarded(path):
        if Path(path) == root / relative:
            attempted.append(str(path)); raise AssertionError("Reserved dependency was opened")
        return original(path)
    monkeypatch.setattr(p, "sha", guarded)
    with pytest.raises(ValueError): p.checked_registration(root, require_review=False)
    assert not attempted


def test_immutable_registration_and_duplicate_json_keys(registered):
    root, registry, _ = registered
    path = root / p.REGISTRATION_PATH; digest = p.sha(path)
    p.immutable_json(registry, path)
    assert p.sha(path) == digest
    with pytest.raises(ValueError): p.immutable_json({**registry, "expected_runs": 35}, path)
    path.write_text('{"schema":"first","schema":"second"}')
    with pytest.raises(ValueError, match="Duplicate"): p.checked_registration(root)


def test_inside_workspace_alias_cannot_hide_a_reserved_dependency(registered, monkeypatch):
    root, registry, review = registered
    target = root / "data/features/iws_reserved_v1/pusht/episodes/000000/arrays.npz"
    target.parent.mkdir(parents=True); target.write_bytes(b"synthetic forbidden target")
    alias = root / "src/alias.py"; alias.symlink_to(target)
    registry["dependencies"]["src/alias.py"] = p.sha(target)
    rebind(root, registry, review)
    original = p.sha; attempted = []
    def guarded(path):
        if Path(path).resolve() == target:
            attempted.append(str(path)); raise AssertionError("Alias opened a reserved dependency")
        return original(path)
    monkeypatch.setattr(p, "sha", guarded)
    with pytest.raises(ValueError): p.checked_registration(root, require_review=False)
    assert not attempted
