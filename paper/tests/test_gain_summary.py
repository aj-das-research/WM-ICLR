"""Check gain directions, denominators and complete-comparison reporting."""
import copy
import importlib.util
from pathlib import Path
import statistics

import pytest

_path = Path(__file__).resolve().parents[1] / "scripts/render_gain_summary.py"
_spec = importlib.util.spec_from_file_location("gain_summary", _path)
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


@pytest.fixture
def forecast():
    rows = []
    for environment in ("pusht", "reacher"):
        for split in ("test", "extrapolation"):
            for mode, values in (("factorized", [.2, 1., 1.8]),
                                 ("framewise", [1., 2., 3.]),
                                 ("factorized_unpaired", [.75, 1.5, 2.25])):
                rows.append({"environment": environment, "split": split, "mode": mode,
                    "training_seeds": [0, 1, 2], "per_seed_values": dict(zip(map(str, range(3)), values)),
                    "metric": "fixed_reference_latent_mse_h5", "direction": "lower",
                    "value": statistics.mean(values), "source_ids": [f"{environment}/{split}/{mode}/{seed}" for seed in range(3)]})
    return {"results": rows}


def test_forecast_gain_uses_ratio_of_means_not_mean_of_ratios(forecast):
    row = report.forecast_deltas(forecast)[0]
    assert row["relative_error_reduction_percent"] == pytest.approx(50.)
    assert row["ours_value"] == 1. and row["comparator_value"] == 2.
    assert len(row["source_ids"]["ours"]) == len(row["source_ids"]["comparator"]) == 3


def test_forecast_regression_is_negative_and_retained(forecast):
    forecast["results"][0].update(value=3., per_seed_values={"0": 2., "1": 3., "2": 4.})
    rows = report.forecast_deltas(forecast)
    assert rows[0]["relative_error_reduction_percent"] == -50.
    assert "-50.00" in report.render_forecast_rows({"forecast": rows})


@pytest.mark.parametrize("change", ["partial", "wrong_mean", "wrong_direction", "zero_reference", "duplicate"])
def test_forecast_cannot_report_unsupported_gain(forecast, change):
    if change == "partial":
        forecast["results"][0]["training_seeds"] = [0, 1]
    elif change == "wrong_mean":
        forecast["results"][0]["value"] = .1
    elif change == "wrong_direction":
        forecast["results"][0]["direction"] = "higher"
    elif change == "zero_reference":
        forecast["results"][1].update(value=0., per_seed_values={"0": 0., "1": 0., "2": 0.})
    else:
        forecast["results"].append(copy.deepcopy(forecast["results"][0]))
    with pytest.raises(ValueError):
        report.forecast_deltas(forecast)


@pytest.fixture
def planning():
    rows, primary = {}, []
    for environment in ("pusht", "reacher"):
        for split, metric in (("test", "heldout_success"), ("extrapolation", "extrapolation_success")):
            rows[environment, split, "single"] = {"status": "complete", "metrics": {
                "mean_difference_pp": 10., "conditional_ci95_pp": [-5., 20.]}}
            for mode, value in (("factorized", 30.), ("single", 20.)):
                primary.append({"environment": environment, "mode": mode, "metric": metric,
                                "status": "complete", "available_seeds": [0, 1, 2], "mean": value})
    return rows, {"results": primary}


def test_planning_preserves_interval_crossing_zero(planning):
    rows = report.planning_deltas(*planning)
    assert rows[0]["mean_difference_pp"] == 10.
    assert rows[0]["conditional_ci95_pp"] == [-5., 20.]
    assert "[-5.00, +20.00]" in report.render_planning_rows({"planning": rows})


def test_pending_planning_is_missing_not_zero(planning):
    planning[0]["pusht", "test", "single"] = {"status": "pending", "metrics": None}
    rows = report.planning_deltas(*planning)
    assert rows[0]["mean_difference_pp"] is None and rows[0]["conditional_ci95_pp"] is None
    assert r"\missing" in report.render_planning_rows({"planning": rows})


@pytest.mark.parametrize("change", ["partial", "mismatch"])
def test_planning_gain_must_agree_with_primary_table(planning, change):
    if change == "partial":
        planning[1]["results"][0]["available_seeds"] = [0, 1]
    else:
        planning[1]["results"][0]["mean"] = 40.
    with pytest.raises(ValueError):
        report.planning_deltas(*planning)
