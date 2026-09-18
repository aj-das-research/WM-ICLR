"""Reporting guards against misleading planning marks."""
import copy
import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "scripts/render_planning_comparison.py"
_spec = importlib.util.spec_from_file_location("planning_comparison", _path)
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)


@pytest.fixture
def ledger():
    rows = []
    for env in ("pusht", "reacher"):
        for split in ("test", "extrapolation"):
            for comparator, _ in module.COMPARATORS:
                rows.append({"environment": env, "split": split, "comparator": comparator,
                             "comparison": "factorized_minus_" + comparator,
                             "metric": "raw_success", "required_training_seeds": [0, 1, 2],
                             "status": "pending", "metrics": None})
    return {"results": rows}


def completed(row):
    row["status"] = "complete"
    row["availability"] = {mode: {str(seed): "complete" for seed in (0, 1, 2)}
                           for mode in ("factorized", row["comparator"])}
    row["metrics"] = {"mean_difference_pp": 10., "conditional_ci95_pp": [-10., 30.],
                      "per_training_seed": [{"training_seed": seed, "n": 10,
                          "wins": 2, "losses": 1, "both_success": 3,
                          "neither_success": 4, "paired_difference_pp": 10.}
                          for seed in (0, 1, 2)]}


def test_pending_is_not_zero(ledger):
    rows = module.select_rows(ledger)
    assert len(rows) == 12
    assert all(row["metrics"] is None for row in rows.values())


def test_complete_three_seed_contrast_preserved(ledger):
    completed(ledger["results"][0])
    rows = module.select_rows(ledger)
    assert rows["pusht", "test", "framewise"]["metrics"]["mean_difference_pp"] == 10.


def test_partial_seed_cannot_be_shown_as_complete(ledger):
    row = ledger["results"][0]
    completed(row)
    row["availability"]["framewise"]["2"] = "missing"
    with pytest.raises(ValueError, match="Incomplete sources"):
        module.select_rows(ledger)


def test_pending_with_estimate_is_rejected(ledger):
    ledger["results"][0]["metrics"] = {"mean_difference_pp": 0.}
    with pytest.raises(ValueError, match="Pending contrasts"):
        module.select_rows(ledger)


@pytest.mark.parametrize("field,value", [("mean_difference_pp", 100.),
                                          ("conditional_ci95_pp", [30., -10.]),
                                          ("conditional_ci95_pp", [float("nan"), 30.])])
def test_misleading_values_rejected(ledger, field, value):
    row = ledger["results"][0]
    completed(row)
    row["metrics"][field] = value
    with pytest.raises(ValueError):
        module.select_rows(ledger)


def test_duplicate_contrast_rejected(ledger):
    ledger["results"].append(copy.deepcopy(ledger["results"][0]))
    with pytest.raises(ValueError, match="Duplicate planning contrast"):
        module.select_rows(ledger)
