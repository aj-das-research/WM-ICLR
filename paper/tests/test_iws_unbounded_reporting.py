"""Synthetic temporary reporting fixtures, never experiment or active-paper data."""
from copy import deepcopy
import functools
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("unbounded_reporting_test_module", ROOT / "paper/scripts/render_iws_unbounded_results.py")
REPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORTER)
PROTOCOL = REPORTER._protocol()  # Pure numerical functions only; no model/cache call.


@functools.lru_cache(maxsize=1)
def synthetic_numerical_fixture():
    audits = {t: {"eligible_episodes": 2, "records": [{"episode_id": f"fixture_{i}", "windows": 2} for i in range(2)]}
              for t in REPORTER.TASKS}
    rows = []
    for ti, task in enumerate(REPORTER.TASKS):
        for mode in REPORTER.MODES:
            base = ([.5, 2., 1.][ti] if mode == REPORTER.MODES[0] else 1.)
            for seed in range(3):
                episodes = []
                for e in range(2):
                    row = {"episode_id": f"fixture_{e}", "windows": 2}
                    for mi, metric in enumerate(PROTOCOL.METRICS):
                        row[metric + "_by_offset"] = [(base + .01*seed + .005*e) * (mi+1)]*59
                        row["persistence_" + metric + "_by_offset"] = [3.*(mi+1)]*59
                    episodes.append(row)
                rows.append({"task": task, "mode": mode, "seed": seed, "receipt": {"episodes": episodes}})
    arrays = PROTOCOL.arrays_from_receipts(rows, audits)
    return audits, rows, PROTOCOL.numerical_report(arrays)


@pytest.fixture
def complete(tmp_path, monkeypatch):
    """A complete miniature invented campaign confined to pytest's tmp directory."""
    m = REPORTER
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "_protocol", lambda: PROTOCOL)
    def put(path, value):
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, sort_keys=True) + "\n")
        return m.sha(p)
    audits, numerical, results = deepcopy(synthetic_numerical_fixture())
    base_registration_sha = put(m.BASE_REGISTRATION, {"fixture_only": True})
    config = {"fixture_only": True, "tasks": {t: {} for t in m.TASKS}, "seeds": [0, 1, 2],
              "modes": [m.MODES[0]], "training": {"epochs": 30, "horizon": 60},
              "evaluation": {"inference_device": "cpu", "primary_comparator": "bounded_spatial_mix",
                             "official_validation_allowed_during_training": False}}
    config_sha = put(m.CONFIG, config)
    protocol_path = "scripts/real_video_iws_unbounded/finalize.py"
    protocol_sha = put(protocol_path, {"fixture_only": "not an executable scientific source"})
    registry = {"fixture_only": True, "schema": "iws_unbounded_training_registration_v1",
                "status": "registered_before_ablation_training", "expected_runs": 9, "epochs_per_run": 30,
                "official_validation_payloads_allowed": False, "config_sha256": config_sha,
                "runs": [{k: r[k] for k in ("task", "mode", "seed")} for r in numerical if r["mode"] == m.MODES[0]],
                "dependencies": {str(m.CONFIG): config_sha, protocol_path: protocol_sha,
                                 str(m.BASE_REGISTRATION): base_registration_sha}}
    registry_sha = put(m.REGISTRATION, registry)
    sources = dict(registry["dependencies"])
    all_runs = []
    for item in numerical:
        task, mode, seed = (item[k] for k in ("task", "mode", "seed"))
        name = f"{task}_{mode}_s{seed}"
        track = "real_video_iws_unbounded" if mode == m.MODES[0] else "real_video_iws"
        path = Path(f"reports/{track}/evaluations/{name}.json")
        ledger = path.with_suffix(".npz")
        (tmp_path/ledger).parent.mkdir(parents=True, exist_ok=True)
        np.savez(tmp_path/ledger, fixture_only=np.arange(4))
        sources[str(ledger)] = m.sha(tmp_path/ledger)
        checkpoint = f"runs/fixture_only/{name}/model.pt"
        sources[checkpoint] = put(checkpoint, {"fixture_only": name})
        row = {"name": name, "task": task, "mode": mode, "seed": seed, "completed_epochs": 30,
               "selected_epoch": 30, "selected_checkpoint_sha256": sources[checkpoint],
               "window_ledger_path": str(ledger), "window_ledger_sha256": sources[str(ledger)]}
        receipt = {"fixture_only": True, "schema": "shiftwm_iws_development_evaluation_v1", "status": "passed",
                   "scope": "internal_development", "official_validation_payloads_read": 0,
                   "offsets": list(range(1, 60)), **row, **item["receipt"],
                   "registration_sha256": registry_sha if mode == m.MODES[0] else base_registration_sha}
        sources[str(path)] = put(path, receipt)
        all_runs.append({**row, "evaluation_path": str(path), "evaluation_sha256": sources[str(path)]})
    baseline = {"fixture_only": True, "schema": "shiftwm_iws_development_finalization_v1", "status": "passed",
                "expected_runs": 27, "completed_runs": 27, "scope": "internal_development",
                "official_validation_payloads_read": 0, "registration_sha256": base_registration_sha,
                "per_run": [r for r in all_runs if r["mode"] != m.MODES[0]],
                "source_dependencies": {str(m.BASE_REGISTRATION): base_registration_sha}}
    baseline_sha = put(m.BASELINE, baseline)
    sources.update({str(m.REGISTRATION): registry_sha, str(m.BASELINE): baseline_sha})
    final = {"fixture_only": True, "schema": "iws_unbounded_complete_comparison_v1", "status": "passed",
             "completed_new_runs": 9, "completed_v1_comparator_runs": 27,
             "scope": "exploratory_internal_development_after_v1", "official_validation_payloads_read": 0,
             "registration_sha256": registry_sha, "baseline_finalization_sha256": baseline_sha,
             "source_dependencies": sources, "per_run": all_runs, "populations": audits, "results": results}
    put(m.FINAL, final)
    return m, tmp_path, final, put


def test_absent_completion_writes_nothing_even_if_old_display_exists(tmp_path, monkeypatch):
    m = REPORTER
    monkeypatch.setattr(m, "ROOT", tmp_path)
    assert m.render(if_ready=True)["outputs_written"] is False
    assert list(tmp_path.iterdir()) == []
    out = tmp_path/m.OUT
    out.mkdir(parents=True)
    old = out/"unbounded_component.tex"
    old.write_text("preserved fixture")
    assert m.render(if_ready=True)["status"] == "pending"
    assert old.read_text() == "preserved fixture"
    with pytest.raises(ValueError, match="required"):
        m.render()


def test_complete_negative_results_all_means_and_idempotence(complete):
    m, root, final, put = complete
    assert m.render()["new_runs"] == 9
    tex = root/m.OUT/"unbounded_component.tex"
    before = tex.read_bytes()
    text = tex.read_text()
    for task in m.TASKS:
        means = final["results"]["task_results"][task][m.METRIC]["horizon_means"]["60"]
        line = next(line for line in text.splitlines() if line.startswith(m.NAMES[task] + " &"))
        assert all(f"{means[mode]:.6f}" in line for mode in m.DISPLAY)
    box = next(line for line in text.splitlines() if line.startswith("Box &"))
    assert "-" in box and r"\positivegain" not in box
    assert "designed after v1 development" in text and "positive values favor removing the bound" in text
    assert "stored offset 59" in text and r"\tanh" in text
    assert m.render()["status"] == "complete_validated_exploratory_development"
    assert tex.read_bytes() == before


@pytest.mark.parametrize("field,value", [("completed_new_runs", 8), ("completed_v1_comparator_runs", 26),
                                         ("official_validation_payloads_read", 1), ("status", "partial")])
def test_present_partial_or_reserved_completion_fails_closed(complete, field, value):
    m, root, final, put = complete
    final[field] = value
    put(m.FINAL, final)
    with pytest.raises(ValueError, match="Complete validated"):
        m.render(if_ready=True)
    assert not (root/m.OUT).exists()


def test_duplicate_run_is_not_complete(complete):
    m, root, final, put = complete
    final["per_run"][-1] = deepcopy(final["per_run"][0])
    put(m.FINAL, final)
    with pytest.raises(ValueError, match="Missing, duplicated"):
        m.render(if_ready=True)
    assert not (root/m.OUT).exists()


def test_changed_source_or_checkpoint_is_rejected(complete):
    m, root, final, put = complete
    path = next(p for p in final["source_dependencies"] if p.endswith("model.pt"))
    (root/path).write_text("different synthetic checkpoint")
    with pytest.raises(ValueError, match="Changed reporting evidence"):
        m.render(if_ready=True)


def test_numeric_tamper_with_valid_source_hashes_is_rejected(complete):
    m, root, final, put = complete
    final["results"]["task_results"]["pusht"][m.METRIC]["horizon_means"]["60"][m.MODES[0]] += .01
    put(m.FINAL, final)
    with pytest.raises(ValueError, match="Reported numbers"):
        m.render(if_ready=True)
    assert not (root/m.OUT).exists()


def test_escaping_evidence_path_is_rejected(complete):
    m, root, final, put = complete
    final["source_dependencies"]["../outside.json"] = "bad"
    put(m.FINAL, final)
    with pytest.raises(ValueError, match="escapes"):
        m.render(if_ready=True)
