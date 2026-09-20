"""Synthetic reserved fixtures only: access boundaries and completion transactions."""
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from shiftwm.real_video_iws.data import read_json, sha
from shiftwm.real_video_iws.features import PREPROCESSING
from shiftwm.real_video_iws_reserved import cache, protocol
from shiftwm.real_video_iws_reserved.data import (
    RAW_ROOT, REGISTRATION_PATH, SPLIT, SPLIT_PATH, TASK_WIDTHS,
    ReservedInventory, safe_path, validate_arrays,
)


def dump(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    """Ten synthetic trajectories/200 handles; no real reserved payload is read."""
    root = tmp_path
    task = "pusht"
    ids = [f"{i:06d}" for i in range(10)]
    parts = {"internal_train": ["000010"], "internal_development": ["000011"], SPLIT: ids}
    document = {"kind": "shiftwm_iws_trajectory_split_v1",
                "status": "identities_frozen_before_model_training_and_official_validation_video_decoding",
                "partitions": {t: copy.deepcopy(parts) for t in TASK_WIDTHS},
                "counts": {t: {k: len(v) for k, v in parts.items()} for t in TASK_WIDTHS},
                "metadata": {}, "reserved_handles": {}}
    for t, width in TASK_WIDTHS.items():
        rows = []
        for i in range(12):
            eid = f"{i:06d}"; relative = f"{RAW_ROOT}/{t}/traj_{eid}/metadata.h5"
            path = root / relative; path.parent.mkdir(parents=True)
            with h5py.File(path, "w") as f:
                f["target_qpos"] = np.arange(81 * width, dtype=np.float32).reshape(81, width)
            (path.parent / "camera_0_rgb.mp4").write_bytes(b"synthetic-video-not-decoded")
            rows.append({"episode_id": eid, "file": relative, "sha256": sha(path),
                         "split": "val" if i < 10 else "train",
                         "shapes": {"target_qpos": [81, width], "qpos": [81, 14]}})
        document["metadata"][t] = rows
        handles = [{"version": 1, "dataset_idx": 0, "group_name": t, "traj_id": eid,
                    "frame_id": start, "sampled_horizon": 60, "feasible_horizon": 80 - start,
                    "camera_keys": ["camera_0"], "rgb_variant": "base",
                    "interaction_frame_indices": None, "dataset_root": f"data/iws_converted/{t}"}
                   for start in range(20) for eid in reversed(ids)]
        hp = root / f"metadata/{t}.json"
        dump({"meta": {"horizons": [60], "total_handles": 200, "handles_per_horizon": 200},
              "handles_by_horizon": {"60": handles}}, hp)
        document["reserved_handles"][t] = {"path": str(hp.relative_to(root)), "sha256": sha(hp),
                                            "stored_command_rows": 60, "windows": 200, "trajectories": 10}
    dump(document, root / SPLIT_PATH)
    stats = f"data/features/iws_{task}_spatial_v1/training_statistics.json"
    dump({"fit_split": "internal_train", "episodes": ["000010"]}, root / stats)
    binding = document["reserved_handles"][task]
    registry = {"dependencies": {SPLIT_PATH: sha(root / SPLIT_PATH), stats: sha(root / stats)},
                "tasks": {task: {"command_width": 4, "episode_ids": ids, "expected_handles": 200,
                    "expected_trajectories": 10, "native_frames": 810,
                    "handles_path": binding["path"], "handles_sha256": binding["sha256"],
                    "cache_root": "data/features/iws_reserved_v1/pusht",
                    "training_statistics_path": stats, "training_statistics_sha256": sha(root / stats)}},
                "extraction": {"encoder_root": "data/pretrained/dinov2-small", "batch_size": 32,
                               "device": "cuda", "precision": "bfloat16", "preprocessing": PREPROCESSING, "tf32": False}}
    dump(registry, root / REGISTRATION_PATH)
    state = {"reviewed": True, "checks": 0, "decoded": 0}

    def checked(candidate_root, registration_path=None, require_review=True):
        assert Path(candidate_root) == root and require_review is True
        state["checks"] += 1
        if not state["reviewed"]:
            raise ValueError("Missing independent preaccess review")
        current = read_json(root / REGISTRATION_PATH)
        for p, digest in current["dependencies"].items():
            if sha(root / p) != digest:
                raise ValueError("Registered dependency changed")
        return current

    def decode(inventory, eid, expected_video_sha256, split=None):
        _, _, _, video = inventory.paths(eid, split)
        assert sha(video) == expected_video_sha256
        state["decoded"] += 1
        # Broadcast a tiny zero array: native shape without storing fixture imagery.
        return np.broadcast_to(np.zeros((1, 1, 1, 3), np.uint8), (81, 480, 640, 3))

    monkeypatch.setattr(protocol, "checked_registration", checked)
    monkeypatch.setattr(cache, "decode_native_rgb", decode)
    return root, document, state


def encoder(rgb, batch_size):
    assert batch_size == 32
    return np.broadcast_to(np.arange(len(rgb), dtype=np.float32)[:, None], (len(rgb), 6144)).copy()


def test_metadata_only_retains_original_order_without_registration_or_paths(fixture):
    root, _, state = fixture
    state["reviewed"] = False
    inventory = ReservedInventory.metadata_only(root, "pusht")
    assert inventory.handles[:2] == [{"episode_id": "000009", "start": 0}, {"episode_id": "000008", "start": 0}]
    assert state["checks"] == 0 and state["decoded"] == 0
    with pytest.raises(ValueError, match="Metadata-only"):
        inventory.paths("000000")
    with pytest.raises(ValueError, match="preaccess"):
        ReservedInventory(root, "pusht")
    assert state["decoded"] == 0


@pytest.mark.parametrize("change", ["duplicate", "missing", "wrong_episode", "bad_start", "wrong_width", "wrong_split", "reordered_bytes"])
def test_bad_metadata_rejected_before_payload_access(fixture, change):
    root, doc, state = fixture
    hp = root / doc["reserved_handles"]["pusht"]["path"]
    h = read_json(hp); rows = h["handles_by_horizon"]["60"]
    if change == "duplicate": rows[-1] = rows[0]
    elif change == "missing": rows.pop()
    elif change == "wrong_episode": rows[0]["traj_id"] = "000010"
    elif change == "bad_start": rows[0]["frame_id"] = 21
    elif change == "wrong_width": doc["metadata"]["pusht"][0]["shapes"]["target_qpos"][1] = 14
    elif change == "wrong_split": doc["metadata"]["pusht"][0]["split"] = "train"
    else: rows.reverse()
    dump(h, hp)
    if change != "reordered_bytes": doc["reserved_handles"]["pusht"]["sha256"] = sha(hp)
    dump(doc, root / SPLIT_PATH)
    with pytest.raises(ValueError): ReservedInventory.metadata_only(root, "pusht")
    assert state["checks"] == 0 and state["decoded"] == 0


def test_review_revocation_and_symlink_paths_fail_closed(fixture):
    root, _, state = fixture
    inventory = ReservedInventory(root, "pusht")
    with pytest.raises(ValueError, match="Non-reserved"): inventory.paths("000010")
    state["reviewed"] = False
    with pytest.raises(ValueError, match="preaccess"): inventory.paths("000000")
    state["reviewed"] = True
    p = root / RAW_ROOT / "pusht/traj_000000/camera_0_rgb.mp4"
    p.unlink(); p.symlink_to(root / RAW_ROOT / "pusht/traj_000001/camera_0_rgb.mp4")
    with pytest.raises(ValueError, match="Symlink"): inventory.paths("000000")
    with pytest.raises(ValueError, match="Unsafe"): safe_path(root, "data/../outside")
    assert state["decoded"] == 0


@pytest.mark.parametrize("width", [4, 14, 8])
def test_native_arrays_and_all_registered_endpoint_indices(width):
    arrays = {"features": np.repeat(np.arange(81, dtype=np.float32)[:, None], 6144, axis=1),
              "commands": np.repeat(np.arange(81, dtype=np.float32)[:, None], width, axis=1),
              "frame_indices": np.arange(81, dtype=np.int64), "command_row_indices": np.arange(81, dtype=np.int64)}
    validate_arrays(arrays, 81, width)
    for H, output_index in [(2, 0), (15, 13), (30, 28), (45, 43), (60, 58)]:
        start = 2; targets = arrays["features"][start + 1:start + H]
        assert targets[output_index, 0] == start + H - 1
        assert len(arrays["commands"][start:start + H]) == H
    with pytest.raises(ValueError): validate_arrays({**arrays, "commands": arrays["commands"][:, :-1]}, 81, width)
    with pytest.raises(ValueError): validate_arrays({**arrays, "frame_indices": arrays["frame_indices"] + 1}, 81, width)


def test_completion_last_resume_exact_cache_and_no_new_statistics(fixture, monkeypatch):
    root, _, state = fixture
    inventory = ReservedInventory(root, "pusht")
    output = root / "data/features/iws_reserved_v1/pusht"
    first = cache.build_cache(inventory, encoder, {"synthetic_fixture": True}, stop_after=3)
    assert first == {"status": "incomplete", "episodes": 3}
    assert not (output / "manifest.json").exists()
    with pytest.raises(ValueError): cache.ReservedFeatureCache(root, "pusht")
    payload = output / "episodes/000000/arrays.npz"; digest = sha(payload)
    complete = cache.build_cache(inventory, encoder, {"synthetic_fixture": True})
    assert complete["status"] == "complete" and state["decoded"] == 10 and sha(payload) == digest
    assert not (output / "training_statistics.json").exists()
    loaded = cache.ReservedFeatureCache(root, "pusht")
    arrays = loaded.episode("000003")
    np.testing.assert_array_equal(arrays["commands"], np.arange(81 * 4, dtype=np.float32).reshape(81, 4))
    np.testing.assert_array_equal(arrays["features"][:, 0], np.arange(81, dtype=np.float32))
    with pytest.raises(ValueError): loaded.episode("000010")
    # Finalizer metadata inspection never invokes np.load and cannot read arrays.
    monkeypatch.setattr(cache.np, "load", lambda *a, **k: pytest.fail("Metadata reader opened a feature payload"))
    metadata = cache.ReservedFeatureCache.metadata_only(root, "pusht")
    assert metadata.handles == inventory.handles
    with pytest.raises(ValueError, match="Metadata-only"): metadata.episode("000000")


def test_committed_payload_tamper_and_changed_statistics_rejected(fixture):
    root, _, _ = fixture
    inventory = ReservedInventory(root, "pusht")
    cache.build_cache(inventory, encoder, {"synthetic_fixture": True})
    reader = cache.ReservedFeatureCache(root, "pusht")
    payload = root / "data/features/iws_reserved_v1/pusht/episodes/000000/arrays.npz"
    with payload.open("ab") as f: f.write(b"tampered")
    with pytest.raises(ValueError, match="corrupt"): reader.episode("000000")
    stats = root / "data/features/iws_pusht_spatial_v1/training_statistics.json"
    with stats.open("a") as f: f.write(" ")
    with pytest.raises(ValueError, match="dependency"): cache.ReservedFeatureCache(root, "pusht")
