"""Synthetic checks for the BridgeData V2 / RT-1 (fractal) Stage-1 adapter (no real data needed)."""
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_openx", ROOT / "scripts/v2/prepare_openx.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fspec = importlib.util.spec_from_file_location("fetch_openx_subset",
                                               ROOT / "scripts/v2/fetch_openx_subset.py")
fetch = importlib.util.module_from_spec(fspec)
fspec.loader.exec_module(fetch)


class _Feat:
    """Minimal stand-in for tf.train.Feature."""

    def __init__(self, kind, values):
        self._kind = kind
        setattr(self, kind, SimpleNamespace(value=list(values)))

    def WhichOneof(self, _):  # noqa: N802
        return self._kind


def _png(rgb):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def test_steps_and_dims():
    for ds, (step, A, P) in {"bridge": (0.4, 14, 7), "fractal": (1 / 3, 7, 8)}.items():
        s = mod.SPECS[ds]
        assert abs(s["stride"] / s["fps"] - step) < 1e-9
        assert 0.33 <= step <= 0.5
        assert s["stride"] * sum(w for _, w in s["action_keys"]) == A
        assert sum(w for _, w in s["proprio_keys"]) == P


def test_build_blocks_stride2():
    n, A, P = 29, 7, 7
    raw_a = np.arange(n * A, dtype=np.float64).reshape(n, A)
    raw_p = np.arange(n * P, dtype=np.float64).reshape(n, P) + 0.5
    idx, acts, prop = mod.build_blocks(raw_a, raw_p, 2)
    np.testing.assert_array_equal(idx, np.arange(0, 29, 2))
    assert acts.shape == (14, 14) and acts.dtype == np.float32
    assert prop.shape == (15, 7) and prop.dtype == np.float32
    np.testing.assert_array_equal(acts[3].reshape(2, A), raw_a[6:8])  # frame 6 -> frame 8
    np.testing.assert_array_equal(prop[4], raw_p[8])
    idx1, acts1, _ = mod.build_blocks(raw_a, raw_p, 1)
    assert acts1.shape == (28, 7) and len(idx1) == 29
    np.testing.assert_array_equal(acts1, raw_a[:-1])


def test_min_T_threshold_for_bridge():
    # 27 raw steps -> 14 kept frames at stride 2 (kept), 26 -> 13 (dropped)
    assert len(mod.build_blocks(np.zeros((27, 7)), np.zeros((27, 7)), 2)[0]) == mod.MIN_T
    assert len(mod.build_blocks(np.zeros((26, 7)), np.zeros((26, 7)), 2)[0]) == mod.MIN_T - 1


def test_splits_deterministic_and_heldout():
    ids = [mod.episode_id("bridge", "train", s, i) for s in range(20) for i in range(25)]
    a = [mod.assign_split("bridge", e, "train") for e in ids]
    assert a == [mod.assign_split("bridge", e, "train") for e in ids]
    assert set(a) == {"train", "val"}
    assert 0.05 < a.count("val") / len(a) < 0.15
    assert all(mod.assign_split("bridge", mod.episode_id("bridge", "test", 3, i), "test") == "test"
               for i in range(10))
    f = [mod.assign_split("fractal", mod.episode_id("fractal", "train", s, i), "train")
         for s in range(20) for i in range(50)]
    fr = {k: f.count(k) / len(f) for k in ("train", "val", "test")}
    assert 0.06 < fr["test"] < 0.14 and 0.05 < fr["val"] < 0.13 and fr["train"] > 0.75


def test_shard_name_roundtrip():
    name = fetch.shard_name("fractal20220817_data", "train", 17, 1024)
    assert mod.parse_shard_name(Path(name)) == ("train", 17, 1024)
    assert mod.parse_shard_name(Path(fetch.shard_name("bridge", "test", 5, 512))) == ("test", 5, 512)


def test_select_shards_deterministic_budget():
    info = {"name": "train", "shardLengths": ["10"] * 100}
    sizes = {i: 100 + i for i in range(100)}
    a, ta = fetch.select_shards(info, sizes, 0, 2000)
    b, tb = fetch.select_shards(info, sizes, 0, 2000)
    assert a == b and ta == tb <= 2000 and ta == sum(sizes[i] for i in a)
    assert fetch.select_shards(info, sizes, 1, 2000)[0] != a


def test_parse_episode_fractal_like_and_decode():
    n = 5
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (256, 320, 3), dtype=np.uint8) for _ in range(n)]
    feats = {
        "steps/is_first": _Feat("int64_list", [1] + [0] * (n - 1)),
        "steps/action/world_vector": _Feat("float_list", np.arange(3 * n)),
        "steps/action/rotation_delta": _Feat("float_list", -np.arange(3 * n)),
        "steps/action/gripper_closedness_action": _Feat("float_list", [1.0] * n),
        "steps/action/base_displacement_vector": _Feat("float_list", [0.0] * 2 * n),
        "steps/action/base_displacement_vertical_rotation": _Feat("float_list", [0.0] * n),
        "steps/observation/base_pose_tool_reached": _Feat("float_list", np.ones(7 * n)),
        "steps/observation/gripper_closed": _Feat("float_list", [0.0] * n),
        "steps/observation/natural_language_instruction": _Feat("bytes_list", [b"open drawer"] * n),
        "steps/observation/image": _Feat("bytes_list", [_png(f) for f in frames]),
    }
    ex = mod.parse_episode(feats, mod.SPECS["fractal"])
    assert ex["n"] == n and ex["actions"].shape == (n, 7) and ex["proprio"].shape == (n, 8)
    np.testing.assert_array_equal(ex["actions"][1], [3, 4, 5, -3, -4, -5, 1])
    assert ex["zero_max"] == 0.0 and ex["instructions"] == ["open drawer"]
    img = mod.decode_resize(ex["images"][0])
    assert img.shape == (224, 224, 3) and img.dtype == np.uint8
    feats["steps/action/world_vector"] = _Feat("float_list", np.arange(3 * n - 1))
    with pytest.raises(ValueError):
        mod.parse_episode(feats, mod.SPECS["fractal"])
