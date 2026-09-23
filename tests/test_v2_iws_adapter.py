"""Synthetic checks for the IWS Stage-1 adapter (no raw data needed)."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_iws", ROOT / "scripts/v2/prepare_iws.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_handle_window_covers_exact_official_rows():
    idx = mod.handle_window(start=7, n_rows=200, stride=5)
    assert len(idx) == 13 and idx[0] == 7 and idx[-1] == 67
    cmd = np.arange(200 * 4, dtype=np.float32).reshape(200, 4)
    acts = mod.blocks(cmd, idx, 5)
    assert acts.shape == (12, 20)
    # concatenated blocks == exactly the 60 official command rows start..start+59
    np.testing.assert_array_equal(acts.reshape(60, 4), cmd[7:67])


def test_handle_window_rejects_infeasible_start():
    import pytest
    with pytest.raises(AssertionError):
        mod.handle_window(start=140, n_rows=200)  # upstream rule s+60 < N


def test_full_grid_blocks():
    idx = mod.grid_indices(199, 5)
    assert len(idx) == 40 and idx[-1] == 195
    cmd = np.random.default_rng(0).normal(size=(199, 14)).astype(np.float32)
    acts = mod.blocks(cmd, idx, 5)
    assert acts.shape == (39, 70)
    np.testing.assert_array_equal(acts[3].reshape(5, 14), cmd[15:20])


def test_train_val_split_deterministic_disjoint():
    ids = [f"box__{i:06d}" for i in range(10, 612)]
    a = mod.assign_train_val(ids)
    assert a == mod.assign_train_val(list(reversed(ids)))
    assert sum(v == "val" for v in a.values()) == round(0.15 * len(ids))
    assert set(a) == set(ids)
