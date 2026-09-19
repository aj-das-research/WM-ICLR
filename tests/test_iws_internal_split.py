"""The new internal split cannot import reserved trajectories or use outcomes."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("iws_split_test_module", ROOT / "scripts/real_video_iws/prepare_split.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def records():
    return [{"task": task, "episode_id": f"{i:06d}", "split": "val" if i < 2 else "train"}
            for task in module.TASKS for i in range(12)]


def test_split_is_order_and_outcome_independent_and_reserved_disjoint():
    rows = records()
    first = module.partition(rows)
    altered = [{**row, "success": bool(i % 2), "future_error": i * 100} for i, row in enumerate(reversed(rows))]
    assert module.partition(altered) == first
    for value in first.values():
        a, b, c = (set(value[name]) for name in ("internal_train", "internal_development", "reserved_official_validation"))
        assert (len(a), len(b), len(c)) == (8, 2, 2)
        assert not (a & b or a & c or b & c)
        assert c == {"000000", "000001"}
        assert a | b | c == {f"{i:06d}" for i in range(12)}


def test_upstream_overlap_or_duplicate_fails_closed():
    rows = records()
    with pytest.raises(ValueError, match="Duplicate"):
        module.partition(rows + [dict(rows[0], split="train")])
    with pytest.raises(ValueError, match="Duplicate"):
        module.partition(rows + [rows[-1]])
