"""Reporting safeguards: seed completeness, task identity, provenance and portability."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import pytest


PATH = Path(__file__).resolve().parents[1] / "paper/scripts/render_simulator_tables.py"
SPEC = importlib.util.spec_from_file_location("simulator_presentation_tables", PATH)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def core_fixture():
    results = []
    for env in ("pusht", "reacher"):
        for j, (mode, _) in enumerate(renderer.MODES):
            seeds = [0] if mode == "frozen" else [0, 1, 2]
            for metric in renderer.CORE_METRICS:
                values = [float(5 + j + seed) for seed in seeds]
                results.append({"environment": env, "mode": mode, "metric": metric,
                                "status": "complete", "available_seeds": seeds,
                                "required_seeds": seeds, "mean": statistics.mean(values),
                                "training_seed_sd": statistics.stdev(values) if len(seeds) > 1 else None,
                                "per_seed_values": {str(s): v for s, v in zip(seeds, values)}})
    return {"results": results}


def extension_fixture():
    report = {"original_runs": [], "geometry_runs": [], "geometry_planning_comparisons": []}
    for domain in ("drone", "surgery"):
        for architecture in ("transformer", "gru"):
            for mode, _ in renderer.EXTENSION_MODES:
                for seed in range(3):
                    row = {"name": f"{domain}_{architecture}_{mode}_s{seed}",
                           "domain": domain, "architecture": architecture, "mode": mode,
                           "training_seed": seed, "training_summary": {"status": "completed", "completed_epochs": 30},
                           "forecast_mse": {"h1": .001, "h3": .002, "h5": .003},
                           "forecast_trajectories": 16, "forecast_windows": 144,
                           "successes": seed + 1, "tasks": 8, "support_successes": 0}
                    report["original_runs"].append(row)
                    if domain == "drone" and architecture == "transformer" and mode != "framewise":
                        report["geometry_runs"].append(copy.deepcopy(row))
    for comparison in ("Wide ShiftWM minus wide constant", "Wide ShiftWM minus original ShiftWM",
                       "Wide constant minus original constant"):
        report["geometry_planning_comparisons"].append({"comparison": comparison, "status": "complete",
                                                       "success_difference_percentage_points": 0,
                                                       "ci95_percentage_points": [-1, 1]})
    return report


def test_all_registered_configurations_and_names_remain_visible():
    core = renderer.core_rows(core_fixture())
    ext = renderer.extension_rows(extension_fixture())
    assert len(core) == 12 and len(ext) == 14
    assert {r["mode"] for r in core} == set(dict(renderer.MODES))
    assert sum(r["mode"] == "factorized" for r in ext) == 5
    assert all(r["label"].endswith("(ours)") for r in core + ext if r["mode"] == "factorized")
    assert all(r["physical_error_mean"] is None for r in ext)


def test_missing_seed_does_not_publish_a_partial_core_mean():
    report = core_fixture()
    row = next(r for r in report["results"] if r["mode"] == "framewise")
    row["available_seeds"] = [0, 1]
    with pytest.raises(ValueError, match="Incomplete seed"):
        renderer.core_rows(report)


def test_missing_core_metric_stays_pending_and_suppresses_its_gain():
    report = core_fixture()
    report["results"] = [r for r in report["results"] if not (r["environment"] == "pusht"
                         and r["mode"] == "factorized" and r["metric"] == "heldout_mse_h5")]
    rows = renderer.core_rows(report)
    ours = next(r for r in rows if r["environment"] == "pusht" and r["mode"] == "factorized")
    assert ours["status"] == "pending" and ours["metrics"]["heldout_mse_h5"] is None
    assert "--" in renderer.core_table(rows)


@pytest.mark.parametrize("field,value", [("mean", 99), ("training_seed_sd", 9)])
def test_corrupted_core_aggregate_is_rejected(field, value):
    report = core_fixture()
    next(r for r in report["results"] if r["mode"] == "factorized")[field] = value
    with pytest.raises(ValueError, match="disagrees with seeds"):
        renderer.core_rows(report)


def test_duplicate_core_rows_are_not_double_counted():
    report = core_fixture()
    report["results"].append(copy.deepcopy(report["results"][0]))
    with pytest.raises(ValueError, match="Duplicate core"):
        renderer.core_rows(report)


def test_extension_missing_seed_never_becomes_two_seed_mean():
    report = extension_fixture()
    removed = report["original_runs"].pop()
    rows = renderer.extension_rows(report)
    row = next(r for r in rows if r["domain"] == removed["domain"] and r["backbone"] == removed["architecture"]
               and r["mode"] == removed["mode"] and r["campaign"] == "original")
    assert row["status"] == "pending" and row["metrics"] is None


@pytest.mark.parametrize("field,value", [("tasks", 7), ("support_successes", 1), ("forecast_windows", 143)])
def test_extension_population_changes_are_rejected(field, value):
    report = extension_fixture()
    report["original_runs"][0][field] = value
    with pytest.raises(ValueError, match="population changed"):
        renderer.extension_rows(report)


def test_wide_gain_is_separate_and_uses_its_own_constant_reference():
    rows = renderer.extension_rows(extension_fixture())
    assert all(r["reference"] == "constant_dynamics" for r in rows if r["campaign"] == "wide_gain")
    assert all(r["reference"] == "framewise" for r in rows if r["campaign"] == "original")
    assert len([r for r in rows if r["domain"] == "drone" and r["backbone"] == "transformer"]) == 5


def test_only_strictly_favorable_named_gains_are_green():
    assert renderer.show_gain(0) == "+0.00"
    assert renderer.show_gain(-1) == "-1.00"
    assert renderer.show_gain(None) == "--"
    assert renderer.show_gain(1) == r"\positivegain{+1.00}"
    assert renderer.gain(.5, 1, True) == 50
    assert renderer.gain(30, 20) == 10
    assert renderer.gain(1, 0, True) is None


def test_absolute_best_and_ties_are_black_not_green():
    assert renderer.best_value(2, [1, 2, 2], 1, False) == r"\textbf{2.0}"
    assert renderer.best_value(1, [1, 2], 1, True) == r"\textbf{1.0}"
    assert renderer.best_value(1.00004, [1, 1.00004], 3, True) == "1.000"


def test_changed_result_hash_is_rejected_before_use(tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"status":"completed"}')
    expected = renderer.digest(path)
    path.write_text('{"status":"failed"}')
    with pytest.raises(ValueError, match="Stale result source"):
        renderer.Sources(tmp_path).read("result.json", expected)


def test_mid_render_source_change_is_rejected(tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"status":"completed"}')
    sources = renderer.Sources(tmp_path)
    sources.read("result.json")
    path.write_text('{"status":"failed"}')
    with pytest.raises(ValueError, match="changed during render"):
        sources.unchanged()


def test_portable_build_needs_no_private_paths_and_is_idempotent(tmp_path):
    sources = renderer.Sources(tmp_path)
    renderer.portable_pack(tmp_path, core_fixture(), extension_fixture(), sources)
    output = tmp_path / "outputs"
    first = renderer.build(tmp_path, output, portable=True)
    state = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
    second = renderer.build(tmp_path, output, portable=True)
    assert first == second
    assert state == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
    assert first["input_mode"] == "portable_verified_summary"
    assert first["verification"]["upstream_report_result_hashes_checked"] is False
    assert first["verification"]["portable_summary_hashes_checked"] is True
    assert len(first["source_sha256"]) == 2
    assert "\\resizebox" not in (output / "core_complete.tex").read_text()


def test_corrupt_portable_pack_is_rejected(tmp_path):
    renderer.portable_pack(tmp_path, core_fixture(), extension_fixture(), renderer.Sources(tmp_path))
    path = tmp_path / renderer.PACK_RELATIVE / "data.json"
    text = json.loads(path.read_text())
    text["core_report"]["results"][0]["mean"] = 100
    path.write_text(json.dumps(text))
    with pytest.raises(ValueError, match="Stale result source"):
        renderer.build(tmp_path, tmp_path / "outputs", portable=True)
