"""Independent pre-access checks; official payloads are never opened by these tests."""
from pathlib import Path
import builtins
import io
import importlib.util
import sys

import numpy as np
import pytest

from shiftwm.real_video_iws_reserved.data import ReservedInventory, validate_arrays


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"pusht": (4, 2000), "bimanual_box": (14, 1996), "bimanual_rope": (8, 2000)}


@pytest.fixture
def no_reserved_payload_open(monkeypatch):
    """Block reads even when a tested helper accidentally reaches a raw path."""
    attempts = []
    denied = (ROOT / "data/real_video/iws_public_v1/extracted/iws_converted",
              ROOT / "data/features/iws_reserved_v1")

    def guarded(original):
        def open_checked(file, *args, **kwargs):
            if isinstance(file, (str, bytes, Path)):
                path = Path(file.decode() if isinstance(file, bytes) else file).resolve()
                if any(path.is_relative_to(root) for root in denied):
                    attempts.append(str(path))
                    raise AssertionError("Independent pre-access review forbids payload reads")
            return original(file, *args, **kwargs)
        return open_checked

    monkeypatch.setattr(builtins, "open", guarded(builtins.open))
    monkeypatch.setattr(io, "open", guarded(io.open))
    yield attempts
    assert not attempts, attempts


@pytest.mark.parametrize("task", EXPECTED)
def test_exact_metadata_roster_without_payload_access(task, no_reserved_payload_open):
    inventory = ReservedInventory.metadata_only(ROOT, task)
    width, frames = EXPECTED[task]
    assert inventory.command_width == width
    assert inventory.manifest["native_frames"] == frames
    assert inventory.episode_ids == [f"{i:06d}" for i in range(10)]
    assert len(inventory.handles) == 200
    assert len({(h["episode_id"], h["start"]) for h in inventory.handles}) == 200
    partitions = inventory.document["partitions"][task]
    assert not set(inventory.episode_ids).intersection(partitions["internal_train"])
    assert not set(inventory.episode_ids).intersection(partitions["internal_development"])
    for handle in inventory.handles:
        native_length = inventory.rows[handle["episode_id"]]["shapes"]["target_qpos"][0]
        assert handle["start"] + 60 < native_length
    # Unequal window counts are real: they require trajectory-balanced metrics.
    counts = inventory.audit["handles_per_trajectory"]
    assert len(counts) == 10 and sum(counts.values()) == 200
    assert len(set(counts.values())) > 1
    assert inventory.audit["payload_paths_authorized"] == 0


@pytest.mark.parametrize("task", EXPECTED)
def test_metadata_inventory_cannot_authorize_or_return_payload_paths(task, no_reserved_payload_open):
    inventory = ReservedInventory.metadata_only(ROOT, task)
    for access in (inventory.check_access,
                   lambda: inventory.authorize("000000"),
                   lambda: inventory.paths("000000")):
        with pytest.raises(ValueError, match="Metadata-only"):
            access()
    assert inventory.audit["payload_paths_authorized"] == 0


def test_cached_native_alignment_rejects_shift_and_missing_rows():
    arrays = {"features": np.zeros((64, 6144), dtype=np.float32),
              "commands": np.zeros((64, 4), dtype=np.float32),
              "frame_indices": np.arange(64, dtype=np.int64),
              "command_row_indices": np.arange(64, dtype=np.int64)}
    validate_arrays(arrays, 64, 4)
    shifted = {**arrays, "command_row_indices": np.arange(1, 65, dtype=np.int64)}
    with pytest.raises(ValueError, match="alignment"):
        validate_arrays(shifted, 64, 4)
    with pytest.raises(ValueError, match="shape/dtype"):
        validate_arrays({**arrays, "features": arrays["features"][:-1]}, 64, 4)
    nonfinite = {**arrays, "commands": arrays["commands"].copy()}
    nonfinite["commands"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_arrays(nonfinite, 64, 4)


@pytest.fixture
def evaluator():
    path = ROOT / "scripts/real_video_iws_reserved_v1/evaluate.py"
    spec = importlib.util.spec_from_file_location("independent_reserved_evaluation_review", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_unequal_window_counts_do_not_change_primary_trajectory_weight(evaluator):
    ids = [f"{i:06d}" for i in range(10)]
    handles = ([{"episode_id": ids[0], "start": i} for i in range(191)]
               + [{"episode_id": eid, "start": 0} for eid in ids[1:]])
    population = evaluator.population_contract("pusht", handles, ids)
    indices = np.array([ids.index(h["episode_id"]) for h in handles], dtype=np.int64)
    arrays = {"episode_index": indices,
              "window_start": np.array([h["start"] for h in handles], dtype=np.int64)}
    for key in evaluator.metric_keys():
        width = 3 if key.startswith("prefix_") else 59
        arrays[key] = np.repeat((indices == 0).astype(np.float64)[:, None], width, axis=1)
    result = evaluator.aggregate_arrays(arrays, population)
    for key in evaluator.metric_keys():
        np.testing.assert_allclose(result["equal_trajectory"][key], .1)
        np.testing.assert_allclose(result["equal_handle"][key], .955)
    # The same outcomes in a different handle order must not acquire another label.
    changed = {key: value[::-1].copy() for key, value in arrays.items()}
    with pytest.raises(ValueError, match="reordered"):
        evaluator.aggregate_arrays(changed, population)


def test_prefix_targets_use_native_hminus1_and_never_enter_predict(evaluator):
    import torch

    class InputOnlyPredictor:
        feature_std = torch.ones(6144)

        def __init__(self):
            self.command_rows = []

        def predict(self, initial, commands):
            assert initial.shape == (1, 6144)
            self.command_rows.append(commands.shape[1])
            return initial[:, None].expand(-1, commands.shape[1] - 1, -1).clone()

    model = InputOnlyPredictor()
    target = torch.arange(1, 60, dtype=torch.float32)[None, :, None].expand(1, 59, 6144).clone()
    batch = {"initial_features": torch.zeros(1, 6144), "commands": torch.zeros(1, 60, 4),
             "targets": target}
    with torch.inference_mode():
        scored, calls = evaluator.score_batch(model, batch, 0)
    assert model.command_rows == [60, 15, 30, 45]
    np.testing.assert_array_equal(scored["prefix_standardized_mse"], [[14**2, 29**2, 44**2]])
    assert [call["target_native_offset"] for call in calls[1:]] == [14, 29, 44]


def test_macro_averages_task_reductions_and_preserves_original_primary():
    path = ROOT / "scripts/real_video_iws_reserved_v1/finalize.py"
    spec = importlib.util.spec_from_file_location("independent_reserved_finalization_review", path)
    finalizer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = finalizer
    spec.loader.exec_module(finalizer)
    values, prefixes, handles, handle_prefixes = {}, {}, {}, {}
    for index, task in enumerate(finalizer.TASKS):
        additive = (.1, .5, 1.0)[index]
        bounded = additive * (.9, .8, .7)[index]
        means = {"anchored_additive": additive, "bounded_spatial_mix": bounded,
                 "unbounded_spatial_mix": .8 * bounded,
                 "autoregressive": 1.1 * additive, "persistence": 2 * additive}
        values[task] = {mode: np.full((3, 10, 59, 4), value) for mode, value in means.items()}
        # These separate-prefix values intentionally differ from H60 curve slices.
        prefixes[task] = {mode: np.full((3, 10, 3, 4), .123 * value) for mode, value in means.items()}
        handles[task] = {mode: np.full((3, 200, 59, 4), value) for mode, value in means.items()}
        handle_prefixes[task] = {mode: np.full((3, 200, 3, 4), .123 * value) for mode, value in means.items()}
    report = finalizer.numerical_report(values, prefixes, handles, handle_prefixes, draws=24)
    macro = report["macro_h60"]["standardized_mse"]["bounded_vs_additive"]
    assert macro["equal_task_relative_error_reduction_percent"] == pytest.approx(20)
    assert "bounded spatial mixing versus anchored additive" in report["primary"]
    for index, task in enumerate(finalizer.TASKS):
        cell = report["task_results"][task]["standardized_mse"]
        assert cell["h60_comparisons"]["bounded_vs_additive"]["relative_error_reduction_percent"] == pytest.approx(10 * (index + 1))
        assert cell["h60_comparisons"]["no_tanh_vs_bounded"]["status"] == "secondary_post_development_component"
        short = cell["equal_trajectory"]["horizon_means"]["15"]["anchored_additive"]
        assert short == pytest.approx(.123 * (.1, .5, 1.0)[index])
