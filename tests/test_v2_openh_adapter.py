"""Tiny synthetic checks for the Open-H Hamlyn Stage-1 adapter (no real data needed)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prepare_openh_hamlyn", ROOT / "scripts/v2/prepare_openh_hamlyn.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_stride_matches_droid_step():
    assert mod.choose_stride(30) == 10  # 10/30 s = DROID 5/15 s
    assert mod.choose_stride(15) == 5


def test_build_blocks_shapes_and_content():
    n, A, P, s = 37, 16, 30, 10
    raw_a = np.arange(n * A, dtype=np.float64).reshape(n, A)
    raw_p = np.arange(n * P, dtype=np.float64).reshape(n, P) + 0.5
    idx, acts, prop = mod.build_blocks(raw_a, raw_p, s)
    np.testing.assert_array_equal(idx, [0, 10, 20, 30])
    assert acts.shape == (3, s * A) and acts.dtype == np.float32
    assert prop.shape == (4, P) and prop.dtype == np.float32
    np.testing.assert_array_equal(acts[1].reshape(s, A), raw_a[10:20])
    np.testing.assert_array_equal(prop[2], raw_p[20])


def test_sub_offsets_uniform_step_across_fps():
    # 30 Hz: stride 10, keep rows 1,3,5,7,9; 15 Hz: stride 5, keep rows 0..4 -> both K=5
    np.testing.assert_array_equal(mod.sub_offsets(30, 10, 15), [1, 3, 5, 7, 9])
    np.testing.assert_array_equal(mod.sub_offsets(15, 5, 15), [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(mod.sub_offsets(30, 10, None), np.arange(10))
    np.testing.assert_array_equal(mod.sub_offsets(15, 5, 30), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    raw_a = np.arange(40 * 2, dtype=float).reshape(40, 2)
    _, acts, _ = mod.build_blocks(raw_a, raw_a, 10, mod.sub_offsets(30, 10, 15))
    assert acts.shape == (3, 10)
    np.testing.assert_array_equal(acts[1].reshape(5, 2), raw_a[[11, 13, 15, 17, 19]])


def test_splits_deterministic_disjoint_and_proportional():
    ids = [f"suturing_1__{i:06d}" for i in range(100)]
    a = mod.assign_splits(ids)
    b = mod.assign_splits(list(reversed(ids)))
    assert a == b
    counts = {k: sum(v == k for v in a.values()) for k in ("train", "val", "test")}
    assert counts == {"train": 70, "val": 15, "test": 15}
    # tiny task still gets val and test
    small = mod.assign_splits([f"t__{i}" for i in range(5)])
    assert set(small.values()) == {"train", "val", "test"}


def test_resize_frame():
    rgb = np.random.default_rng(0).integers(0, 255, (480, 848, 3), dtype=np.uint8)
    out = mod.resize_frame(rgb)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8


def test_end_to_end_synthetic(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    av = pytest.importorskip("av")
    task = "peg_transfer"
    td = tmp_path / "src" / task
    (td / "meta").mkdir(parents=True)
    cam = "observation.images.color"
    info = {"fps": 30, "chunks_size": 1000, "robot_type": "dvrk",
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/"
                          "episode_{episode_index:06d}.mp4",
            "features": {"action": {"dtype": "float32", "shape": [16]},
                         cam: {"dtype": "video", "shape": [48, 64, 3],
                               "info": {"video.codec": "h264"}}}}
    (td / "meta/info.json").write_text(json.dumps(info))
    lengths = [150, 60]  # second -> T=6 < 14, excluded
    (td / "meta/episodes.jsonl").write_text("\n".join(
        json.dumps({"episode_index": i, "length": n}) for i, n in enumerate(lengths)))
    (td / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "x"}))
    rng = np.random.default_rng(0)
    for i, n in enumerate(lengths):
        p = td / info["data_path"].format(episode_chunk=0, episode_index=i)
        p.parent.mkdir(parents=True, exist_ok=True)
        cols = {"action": rng.normal(size=(n, 16)), "observation.state": rng.normal(size=(n, 16)),
                "observation.state.left_arm_joint": rng.normal(size=(n, 7)),
                "observation.state.right_arm_joint": rng.normal(size=(n, 7))}
        tbl = {k: pa.array(list(v.astype(np.float32))) for k, v in cols.items()}
        tbl["frame_index"] = pa.array(np.arange(n))
        pq.write_table(pa.table(tbl), p)
        v = td / info["video_path"].format(episode_chunk=0, episode_index=i, video_key=cam)
        v.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(v), "w") as c:
            s = c.add_stream("mpeg4", rate=30)
            s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
            for k in range(n):
                img = np.full((48, 64, 3), (k * 7) % 255, np.uint8)
                img[:, : 32] = 30
                for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                    c.mux(pkt)
            for pkt in s.encode():
                c.mux(pkt)
    out = tmp_path / "out"
    rc = mod.main(["--src", str(tmp_path / "src"), "--out", str(out), "--tasks", task,
                   "--workers", "1"])
    assert rc == 0
    m = json.loads((out / "manifest.json").read_text())
    assert m["action_dim"] == 80 and m["proprio_dim"] == 30 and m["frame_stride"] == 10
    assert [e["id"] for e in m["episodes"]] == [f"{task}__000000"]
    e = m["episodes"][0]
    z = np.load(out / e["file"])
    assert z["images"].shape == (15, 224, 224, 3)
    assert z["actions"].shape == (14, 80)
    assert z["proprio"].shape == (15, 30)
    assert any("000001" in s["id"] for s in m["skipped"])
