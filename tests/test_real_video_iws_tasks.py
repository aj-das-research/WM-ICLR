"""Scientific data-boundary and cache lifecycle checks; no performance claims."""
from pathlib import Path
import json
import hashlib
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from shiftwm.real_video_iws_tasks import cache as cache_module
from shiftwm.real_video_iws_tasks.data import (DATA_BASE, TASK_WIDTHS, InternalInventory, RunningMoments, canonical_hash,
    decode_native_rgb, load_commands, sha, training_statistics, validate_arrays, validate_split)
from shiftwm.real_video_iws.features import PREPROCESSING, pool_tokens, preprocess
from shiftwm.real_video_iws_tasks.cache import (IWSFeatureCache, build_cache, load_package, writer_lock)


@pytest.fixture(params=tuple(TASK_WIDTHS))
def task(request):
    return request.param


@pytest.fixture
def source(tmp_path, task):
    width = TASK_WIDTHS[task]
    data_relative = DATA_BASE / task
    partitions = {"internal_train": ["000011", "000012"], "internal_development": ["000010"], "reserved_official_validation": ["000000"]}
    rows = []; commands = {}
    for eid in ["000000", "000010", "000011", "000012"]:
        file = data_relative / f"traj_{eid}" / "metadata.h5"
        path = tmp_path / file; path.parent.mkdir(parents=True)
        value = (np.arange(4*width).reshape(4, width) + int(eid) * 10).astype(np.float32)
        commands[eid] = value
        with h5py.File(path, "w") as stream:
            stream.create_dataset("target_qpos", data=value)
        (path.parent / "camera_0_rgb.mp4").write_bytes(b"fixture-video-" + eid.encode())
        rows.append({"episode_id": eid, "split": "val" if eid == "000000" else "train", "file": str(file),
                     "sha256": sha(path), "shapes": {"target_qpos": [4, width]}})
    document = {"kind": "shiftwm_iws_trajectory_split_v1", "status": "identities_frozen_before_model_training_and_official_validation_video_decoding",
                "partitions": {task: partitions}, "counts": {task: {k: len(v) for k, v in partitions.items()}}, "metadata": {task: rows}}
    split = tmp_path / "configs/real_video_iws/split_v1.json"; split.parent.mkdir(parents=True); split.write_text(json.dumps(document))
    inventory = InternalInventory(tmp_path, task)
    records = []
    for eid in inventory.selected():
        row, assigned, metadata, video = inventory.paths(eid)
        records.append({"task":task, "command_width":width, "episode_id": eid, "split": assigned, "frames": 4, "metadata_path": str(metadata.relative_to(tmp_path)),
                        "metadata_sha256": sha(metadata), "video_path": str(video.relative_to(tmp_path)), "video_sha256": sha(video)})
    static = {"schema": "shiftwm_iws_task_extraction_identity_v1", "task":task, "command_width":width, "registration_sha256": "registered", "split_sha256": inventory.split_sha256,
              "input_manifest_sha256": "inputs", "input_records_sha256": canonical_hash(records), "config_sha256": "config",
              "preprocessing": PREPROCESSING, "pooling_and_storage_precision": "float32", "tf32": False, "batch_size": 32,
              "encoder_provenance_sha256": "encoder", "official_validation_payloads_allowed": False}
    identity = {**static, "device": "cpu", "encoder_precision": "float32", "runtime": {"scope": "engineering fixture only"}}
    return SimpleNamespace(root=tmp_path, document=document, inventory=inventory, records=records, commands=commands, static=static, identity=identity)


def synthetic_rgb(eid):
    frames = np.empty((4, 480, 640, 3), dtype=np.uint8)
    for i in range(4):
        frames[i] = [int(eid) + i, 20 + i, 80 - i]
    return frames


def encoder(images, batch_size):
    return np.repeat(images[:, 0, 0, 0:1].astype(np.float32), 6144, axis=1)


def stub_decode(monkeypatch):
    monkeypatch.setattr(cache_module, "decode_native_rgb", lambda inv, eid, digest, split: synthetic_rgb(eid))


def test_heldout_rejected_before_hdf5_or_video_open(source, monkeypatch):
    import cv2
    opened = []
    monkeypatch.setattr(h5py, "File", lambda *a, **kw: opened.append(a) or pytest.fail("HDF5 opened"))
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **kw: opened.append(a) or pytest.fail("Video opened"))
    for eid in ("000000", "../000011", "999999"):
        with pytest.raises(ValueError, match="Reserved or unknown"):
            load_commands(source.inventory, eid)
        with pytest.raises(ValueError, match="Reserved or unknown"):
            decode_native_rgb(source.inventory, eid, "unused")
    assert opened == []


def test_wrong_internal_assignment_and_overlapping_split_rejected(source):
    with pytest.raises(ValueError, match="requested internal split"):
        load_commands(source.inventory, "000010", "internal_train")
    malformed = json.loads(json.dumps(source.document))
    malformed["partitions"][source.inventory.task]["internal_train"].append("000010")
    malformed["partitions"][source.inventory.task]["internal_train"].sort()
    malformed["counts"][source.inventory.task]["internal_train"] += 1
    with pytest.raises(ValueError, match="crosses partitions"):
        validate_split(malformed, source.inventory.task)
    malformed = json.loads(json.dumps(source.document)); malformed["metadata"][source.inventory.task][1]["split"] = "val"
    with pytest.raises(ValueError, match="Upstream split"):
        validate_split(malformed, source.inventory.task)


def test_commands_keep_every_native_row_and_width(source):
    actual = load_commands(source.inventory, "000011", "internal_train")
    np.testing.assert_array_equal(actual, source.commands["000011"])
    assert actual.shape == (4, source.inventory.command_width) and actual.dtype == np.float32


@pytest.mark.parametrize("length,header,error", [(3, 4, "Decoded frame count"), (5, 4, "extra decoded frames"), (4, 5, "Container frame count")])
def test_decoder_rejects_missing_extra_or_wrong_header(source, monkeypatch, length, header, error):
    import cv2
    class Fake:
        def __init__(self): self.i = 0
        def isOpened(self): return True
        def getBackendName(self): return "FFMPEG"
        def get(self, prop): return {cv2.CAP_PROP_FRAME_COUNT: header, cv2.CAP_PROP_FRAME_HEIGHT: 480, cv2.CAP_PROP_FRAME_WIDTH: 640}[prop]
        def read(self):
            self.i += 1
            return (True, np.zeros((480, 640, 3), np.uint8)) if self.i <= length else (False, None)
        def release(self): pass
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: Fake())
    record = next(r for r in source.records if r["episode_id"] == "000011")
    with pytest.raises(ValueError, match=error):
        decode_native_rgb(source.inventory, "000011", record["video_sha256"])


def test_decoder_preserves_sequential_rgb_channels(source, monkeypatch):
    import cv2
    class Fake:
        def __init__(self): self.i = 0
        def isOpened(self): return True
        def getBackendName(self): return "FFMPEG"
        def get(self, prop): return {cv2.CAP_PROP_FRAME_COUNT: 4, cv2.CAP_PROP_FRAME_HEIGHT: 480, cv2.CAP_PROP_FRAME_WIDTH: 640}[prop]
        def read(self):
            self.i += 1
            if self.i > 4: return False, None
            return True, np.broadcast_to(np.array([10, 20, self.i], np.uint8), (480, 640, 3)).copy()
        def release(self): pass
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: Fake())
    record = next(r for r in source.records if r["episode_id"] == "000011")
    images = decode_native_rgb(source.inventory, "000011", record["video_sha256"])
    np.testing.assert_array_equal(images[:, 0, 0], [[1, 20, 10], [2, 20, 10], [3, 20, 10], [4, 20, 10]])


def arrays(task, value=1):
    width=TASK_WIDTHS[task]
    return {"features": np.full((4, 6144), value, np.float32), "commands": np.arange(4*width, dtype=np.float32).reshape(4, width),
            "frame_indices": np.arange(4, dtype=np.int64), "command_row_indices": np.arange(4, dtype=np.int64)}


@pytest.mark.parametrize("mutation", ["last_command", "frame_shuffle", "command_shuffle", "nan", "wrong_width"])
def test_native_alignment_rejections(mutation, task):
    value = arrays(task)
    if mutation == "last_command": value["commands"] = value["commands"][:-1]
    elif mutation == "frame_shuffle": value["frame_indices"] = value["frame_indices"][::-1]
    elif mutation == "command_shuffle": value["command_row_indices"] = value["command_row_indices"][[0, 2, 1, 3]]
    elif mutation == "nan": value["features"][0, 0] = np.nan
    else: value["commands"] = np.zeros((4, 4), np.float32)
    with pytest.raises(ValueError): validate_arrays(value, 4, TASK_WIDTHS[task])


def test_statistics_train_only_channel_order_and_counts(task):
    a = arrays(task); b = arrays(task)
    for i in range(384):
        a["features"][:, i * 16:(i + 1) * 16] = i + np.arange(16)[None, :]
        b["features"][:, i * 16:(i + 1) * 16] = i + 2 + np.arange(16)[None, :]
    rows = [(dict(task=task,command_width=TASK_WIDTHS[task],episode_id="a", split="internal_train", frames=4, payload_sha256="A"), a),
            (dict(task=task,command_width=TASK_WIDTHS[task],episode_id="b", split="internal_train", frames=4, payload_sha256="B"), b)]
    result = training_statistics(rows, ["a", "b"], task)
    assert result["counts"] == {"feature_patch_rows": 128, "command_rows": 8}
    np.testing.assert_allclose(result["feature_mean"], np.repeat(np.arange(384) + 8.5, 16))
    explicit = np.concatenate([v["features"].reshape(4, 384, 16).transpose(0, 2, 1).reshape(-1, 384) for _, v in rows])
    np.testing.assert_allclose(result["feature_channel_std"], explicit.astype(np.float64).std(0, ddof=1))
    sentinel = (dict(task=task,command_width=TASK_WIDTHS[task],episode_id="development", split="internal_development", frames=4, payload_sha256="dev"), arrays(task,1e9))
    with pytest.raises(ValueError, match="reject development"):
        training_statistics(rows + [sentinel], ["a", "b"], task)
    with pytest.raises(ValueError, match="incomplete"):
        training_statistics(rows[:1], ["a", "b"], task)


def test_stable_moments_large_offset():
    values = 1e12 + np.arange(1000, dtype=np.float64)[:, None] / 10
    m = RunningMoments(1)
    for value in np.array_split(values, 13): m.add(value)
    mean, std = m.finish()
    np.testing.assert_allclose(mean, values.mean(0), rtol=0, atol=1e-3)
    np.testing.assert_allclose(std, values.std(0, ddof=1), rtol=1e-5)


def test_preprocessing_rgb_recipe_and_channel_major_pooling():
    rgb = np.zeros((1, 32, 48, 3), dtype=np.uint8); rgb[:, :, :, 0] = 255
    transformed = preprocess(rgb, "cpu")
    assert transformed.shape == (1, 3, 224, 224)
    np.testing.assert_allclose(transformed[0, :, 100, 100], [(1-.485)/.229, -.456/.224, -.406/.225], rtol=1e-6)
    tokens = torch.arange(256 * 384, dtype=torch.float32).reshape(1, 256, 384)
    actual = pool_tokens(tokens).numpy().reshape(384, 4, 4)
    original = tokens.numpy()[0].reshape(16, 16, 384)
    expected = np.array([[original[r:r+4, c:c+4].mean((0, 1)) for c in range(0, 16, 4)] for r in range(0, 16, 4)]).transpose(2, 0, 1)
    np.testing.assert_array_equal(actual, expected)


def test_atomic_partial_resume_and_complete_loader(source, tmp_path, monkeypatch):
    stub_decode(monkeypatch); output = tmp_path / "cache"; calls = []
    def counted(images, batch): calls.append(int(images[0, 0, 0, 0])); return encoder(images, batch)
    result = build_cache(source.inventory, source.records, output, source.identity, counted, stop_after=1)
    assert result["status"] == "incomplete" and not (output / "manifest.json").exists()
    first = output / "episodes" / "000010" / "arrays.npz"; digest = sha(first); mtime = first.stat().st_mtime_ns
    # An orphan pending transaction cannot be mistaken for a completed episode.
    (output / "episodes" / ".000011.pending-interrupted").mkdir()
    result = build_cache(source.inventory, source.records, output, source.identity, counted)
    assert result["status"] == "complete" and len(calls) == 3 and sha(first) == digest and first.stat().st_mtime_ns == mtime
    opened = IWSFeatureCache(output, source.inventory, source.records, source.static)
    row, value = opened.episode("000011", "internal_train")
    np.testing.assert_array_equal(value["commands"], source.commands["000011"])
    assert opened.statistics["episodes"] == ["000011", "000012"] and opened.statistics["counts"]["feature_patch_rows"] == 128
    with pytest.raises(ValueError, match="Reserved or unknown"): opened.episode("000000", "internal_train")


@pytest.mark.parametrize("mutation", ["payload", "missing_receipt", "identity", "input_video", "split"])
def test_resume_rejects_changed_or_corrupt_state(source, tmp_path, monkeypatch, mutation):
    stub_decode(monkeypatch); output = tmp_path / "cache"
    build_cache(source.inventory, source.records, output, source.identity, encoder, stop_after=1)
    identity = source.identity
    if mutation == "payload": (output / "episodes/000010/arrays.npz").write_bytes(b"broken")
    elif mutation == "missing_receipt": (output / "episodes/000010/receipt.json").unlink()
    elif mutation == "identity": identity = {**identity, "encoder_provenance_sha256": "different"}
    elif mutation == "input_video": (source.root / source.records[0]["video_path"]).write_bytes(b"changed")
    else: identity = {**identity, "split_sha256": "changed"}
    with pytest.raises(ValueError): build_cache(source.inventory, source.records, output, identity, encoder)
    assert not (output / "manifest.json").exists()


def test_completion_gate_prevents_premature_manifest(source, tmp_path, monkeypatch):
    stub_decode(monkeypatch); output = tmp_path / "cache"
    def fail_guard(): raise ValueError("source changed before completion")
    with pytest.raises(ValueError, match="source changed"):
        build_cache(source.inventory, source.records, output, source.identity, encoder, before_complete=fail_guard)
    assert not (output / "manifest.json").exists()
    result = build_cache(source.inventory, source.records, output, source.identity,
                         lambda *args: pytest.fail("Completed package was re-encoded"))
    assert result["status"] == "complete"


@pytest.mark.parametrize("field", ["registration_sha256", "encoder_provenance_sha256", "config_sha256"])
def test_self_consistent_wrong_registration_cache_rejected(source, tmp_path, monkeypatch, field):
    stub_decode(monkeypatch); output = tmp_path / "cache"
    # Build a valid internally self-consistent cache under a different static contract.
    identity = {**source.identity, field: "another-registered-extractor"}
    build_cache(source.inventory, source.records, output, identity, encoder)
    with pytest.raises(ValueError, match="current registered static identity"):
        IWSFeatureCache(output, source.inventory, source.records, source.static)


def test_exclusive_writer_lock(tmp_path):
    with writer_lock(tmp_path):
        with pytest.raises(ValueError, match="Another cache writer"):
            with writer_lock(tmp_path): pass


@pytest.mark.parametrize("other", ["task", "command_width"])
def test_task_identity_rejected_before_raw_open(source, tmp_path, monkeypatch, other):
    from shiftwm.real_video_iws_tasks import cache as c
    monkeypatch.setattr(c, "load_commands", lambda *a, **kw: pytest.fail("raw opened before task authorization"))
    identity=dict(source.identity)
    identity[other]="pusht" if other=="task" else 4
    with pytest.raises(ValueError, match="task/width"):
        build_cache(source.inventory,source.records,tmp_path/"wrong",identity,encoder)


def test_wrong_task_record_rejected(source,tmp_path,monkeypatch):
    records=[dict(r) for r in source.records]
    records[0]["task"]="bimanual_rope" if source.inventory.task=="bimanual_box" else "bimanual_box"
    identity={**source.identity,"input_records_sha256":canonical_hash(records)}
    with pytest.raises(ValueError,match="record task/width"):
        build_cache(source.inventory,records,tmp_path/"wrong",identity,encoder)


def test_same_generic_functions_are_reused():
    import shiftwm.real_video_iws.data as original
    import shiftwm.real_video_iws_tasks.data as extension
    assert extension.load_commands is original.load_commands
    assert extension.decode_native_rgb is original.decode_native_rgb
    assert extension.RunningMoments is original.RunningMoments
