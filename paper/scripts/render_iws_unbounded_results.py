#!/usr/bin/env python3
"""Report the complete exploratory IWS bound-removal study; never evaluate models.

Missing completion is pending and writes nothing. A present but incomplete,
changed or numerically inconsistent completion is an error, including with
--if-ready. This script does not alter the registered experiment or manuscript.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]
FINAL = Path("reports/real_video_iws_unbounded/development_finalization.json")
REGISTRATION = Path("configs/real_video_iws_unbounded/registration_v1.json")
CONFIG = Path("configs/real_video_iws_unbounded/training_v1.json")
BASELINE = Path("reports/real_video_iws/development_finalization.json")
BASE_REGISTRATION = Path("configs/real_video_iws/training_registration_v1.json")
OUT = Path("paper/generated/experiment_alignment")
TASKS = ("pusht", "bimanual_box", "bimanual_rope")
MODES = ("unbounded_spatial_mix", "bounded_spatial_mix", "anchored_additive", "autoregressive")
DISPLAY = ("persistence", "autoregressive", "anchored_additive", "bounded_spatial_mix", "unbounded_spatial_mix")
NAMES = {"pusht": "PushT", "bimanual_box": "Box", "bimanual_rope": "Rope"}
METRIC = "standardized_mse"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def local(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Evidence path escapes the workspace")
    return path


def _protocol():
    path = local("scripts/real_video_iws_unbounded/finalize.py")
    spec = importlib.util.spec_from_file_location("_unbounded_reporting_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_sources(sources):
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Missing source ledger")
    for path, digest in sources.items():
        if sha(local(path)) != digest:
            raise ValueError("Changed reporting evidence: " + path)


def keys(rows):
    return [(r.get("task"), r.get("mode"), r.get("seed")) for r in rows]


def validate_grid(rows, modes, count):
    expected = {(t, m, s) for t in TASKS for m in modes for s in range(3)}
    if len(rows) != count or set(keys(rows)) != expected:
        raise ValueError("Missing, duplicated or substituted study runs")


def load_evidence():
    """Read only immutable finalization and saved metrics; no inference/cache opens."""
    config = read(local(CONFIG))
    registry = read(local(REGISTRATION))
    if (registry.get("schema") != "iws_unbounded_training_registration_v1"
            or registry.get("status") != "registered_before_ablation_training"
            or registry.get("expected_runs") != 9 or registry.get("epochs_per_run") != 30
            or registry.get("official_validation_payloads_allowed") is not False
            or registry.get("config_sha256") != sha(local(CONFIG))):
        raise ValueError("Invalid registered nine-run study")
    validate_grid(registry.get("runs", []), (MODES[0],), 9)
    if (set(config.get("tasks", {})) != set(TASKS) or config.get("seeds") != [0, 1, 2]
            or config.get("modes") != [MODES[0]]
            or config.get("training", {}).get("epochs") != 30
            or config.get("training", {}).get("horizon") != 60
            or config.get("evaluation", {}).get("inference_device") != "cpu"
            or config["evaluation"].get("primary_comparator") != "bounded_spatial_mix"
            or config["evaluation"].get("official_validation_allowed_during_training") is not False):
        raise ValueError("Changed ablation recipe or reserved-data boundary")
    verify_sources(registry.get("dependencies"))
    final = read(local(FINAL))
    if (final.get("schema") != "iws_unbounded_complete_comparison_v1"
            or final.get("status") != "passed" or final.get("completed_new_runs") != 9
            or final.get("completed_v1_comparator_runs") != 27
            or final.get("scope") != "exploratory_internal_development_after_v1"
            or final.get("official_validation_payloads_read") != 0
            or final.get("registration_sha256") != sha(local(REGISTRATION))
            or final.get("baseline_finalization_sha256") != sha(local(BASELINE))):
        raise ValueError("Complete validated nine-plus-27 comparison required")
    sources = final.get("source_dependencies", {})
    required = {str(REGISTRATION): sha(local(REGISTRATION)), str(CONFIG): sha(local(CONFIG)),
                str(BASELINE): sha(local(BASELINE)), **registry["dependencies"]}
    if any(sources.get(p) != digest for p, digest in required.items()):
        raise ValueError("Finalization omitted a registered source or comparator binding")
    verify_sources(sources)
    baseline = read(local(BASELINE))
    if (baseline.get("schema") != "shiftwm_iws_development_finalization_v1"
            or baseline.get("status") != "passed" or baseline.get("expected_runs") != 27
            or baseline.get("completed_runs") != 27 or baseline.get("scope") != "internal_development"
            or baseline.get("official_validation_payloads_read") != 0
            or baseline.get("registration_sha256") != sha(local(BASE_REGISTRATION))):
        raise ValueError("Original common-CPU comparison is incomplete or changed")
    validate_grid(baseline.get("per_run", []), MODES[1:], 27)
    if any(sources.get(p) != digest for p, digest in baseline.get("source_dependencies", {}).items()):
        raise ValueError("Finalization does not retain all original evidence bindings")
    validate_grid(final.get("per_run", []), MODES, 36)
    baseline_rows = {tuple(keys([r])[0]): r for r in baseline["per_run"]}
    rows = []
    for run in final["per_run"]:
        key = tuple(keys([run])[0])
        if key in baseline_rows and run != baseline_rows[key]:
            raise ValueError("A v1 comparator identity was changed")
        if (run.get("completed_epochs") != 30 or not 1 <= run.get("selected_epoch", 0) <= 30
                or run.get("selected_checkpoint_sha256") not in sources.values()
                or sources.get(run.get("evaluation_path")) != run.get("evaluation_sha256")
                or sources.get(run.get("window_ledger_path")) != run.get("window_ledger_sha256")):
            raise ValueError("Unbound or incomplete selected result")
        path = local(run["evaluation_path"])
        if path.with_suffix(".npz") != local(run["window_ledger_path"]):
            raise ValueError("Result and primitive window ledger differ")
        receipt = read(path)
        expected_registration = sha(local(REGISTRATION if run["mode"] == MODES[0] else BASE_REGISTRATION))
        expected = {"schema": "shiftwm_iws_development_evaluation_v1", "status": "passed",
                    "scope": "internal_development", "completed_epochs": 30,
                    "official_validation_payloads_read": 0, "offsets": list(range(1, 60)),
                    "registration_sha256": expected_registration,
                    **{k: run[k] for k in ("task", "mode", "seed", "selected_epoch", "selected_checkpoint_sha256",
                                           "window_ledger_path", "window_ledger_sha256")}}
        if any(receipt.get(k) != value for k, value in expected.items()):
            raise ValueError("Result receipt does not match its complete-study identity")
        rows.append({**run, "receipt": receipt})
    protocol = _protocol()
    # The completed gate already reconstructs primitive ledgers. Independently
    # rebuild the reported means/paired intervals from their bound episode rows
    # as a corruption guard; this is arithmetic over saved scores, not evaluation.
    arrays = protocol.arrays_from_receipts(rows, final["populations"])
    reconstructed = protocol.numerical_report(arrays)
    if final.get("results") != reconstructed:
        raise ValueError("Reported numbers differ from bound complete-study metrics")
    sources = {**sources, str(FINAL): sha(local(FINAL))}
    return reconstructed, sources


def signed(value):
    return "undefined" if value is None else f"{value:+.2f}"


def interval(values):
    return "undefined" if values is None else f"[{values[0]:+.2f}, {values[1]:+.2f}]"


def table_text(result):
    rows = []
    for task in TASKS:
        values = result["task_results"][task][METRIC]
        means = values["horizon_means"]["60"]
        effect = values["h60_comparisons"]["bounded_spatial_mix"]
        scores = [f"{means[m]:.6f}" for m in DISPLAY]
        gain = effect["relative_error_reduction_percent"]
        gain_tex = signed(gain)
        if gain is not None and gain > 0:
            gain_tex = r"\positivegain{" + gain_tex + "}"
        rows.append(" & ".join([NAMES[task], *scores, gain_tex + " " + interval(effect["paired95"]["gain"])]) + r" \\")
    macro = result["macro_h60"][METRIC]["bounded_spatial_mix"]
    return (r"\begin{table}[!htb]\centering" + "\n"
        r"\caption{Post-development IWS innovation-bound ablation. Nine new 30-epoch runs are compared with all 27 frozen common-CPU v1 runs. Only $\tanh$ is removed from the correction. Entries are $H=60$ standardized MSE (stored offset 59), weighting windows within trajectory, trajectories and three seeds equally; lower is better. Gains and paired 95\% intervals are percentages versus bounded ShiftWM; positive values favor removing the bound. Intervals use 10,000 paired seed--trajectory draws. These unadjusted exploratory comparisons were designed after v1 development. All tasks, controls and signs remain visible; reserved validation is excluded.}" + "\n"
        r"\label{tab:iws-unbounded-component}" + "\n"
        r"\begingroup\small\setlength{\tabcolsep}{2.5pt}" + "\n"
        r"\begin{tabular}{@{}lrrrrrl@{}}\toprule" + "\n"
        r"Task & Persistence & AR & \shortstack{Additive\\anchor} & \shortstack{ShiftWM\\(ours)} & \shortstack{No $\tanh$\\(ablation)} & \shortstack[l]{Gain [95\% CI]\\vs ShiftWM} \\ \midrule" + "\n"
        + "\n".join(rows) + "\n"
        + r"\bottomrule\end{tabular}\endgroup" + "\n"
        + r"\par\smallskip\small Equal-task mean relative gain: "
        + signed(macro["equal_task_relative_error_reduction_percent"]) + r"\%; paired 95\% interval "
        + interval(macro["paired95"]) + r"\%." + "\n"
        + r"\end{table}" + "\n")


def render(if_ready=False):
    if not local(FINAL).exists():
        if if_ready:
            return {"status": "pending", "outputs_written": False, "reason": "Complete nine-plus-27 finalizer absent"}
        raise ValueError("Complete nine-plus-27 finalizer required")
    result, sources = load_evidence()
    text = table_text(result)
    source = Path(__file__).resolve()
    renderer_sha = sha(source)
    record = {"schema": "iws_unbounded_appendix_reporting_v1", "status": "complete_validated_exploratory_development",
              "finalization_sha256": sources[str(FINAL)], "source_dependencies": sources,
              "renderer_sha256": renderer_sha, "primary_comparator": "bounded_spatial_mix",
              "reported_mode": "unbounded_spatial_mix", "metric": METRIC, "H": 60, "stored_offset": 59,
              "display_means": {t: result["task_results"][t][METRIC]["horizon_means"]["60"] for t in TASKS},
              "primary_effects": {t: result["task_results"][t][METRIC]["h60_comparisons"]["bounded_spatial_mix"] for t in TASKS},
              "macro": result["macro_h60"][METRIC]["bounded_spatial_mix"], "bootstrap": result["bootstrap"],
              "tex_sha256": hashlib.sha256(text.encode()).hexdigest(), "official_validation_payloads_read": 0,
              "scope": "Exploratory component ablation designed after v1 internal development; no new algorithm or confirmatory claim."}
    verify_sources(sources)
    if sha(source) != renderer_sha:
        raise ValueError("Renderer changed during reporting")
    output = local(OUT)
    output.mkdir(parents=True, exist_ok=True)
    for name, content in (("unbounded_component.tex", text),
                          ("unbounded_component.json", json.dumps(record, indent=2, sort_keys=True) + "\n")):
        path = output / name
        if path.exists() and path.read_text() == content:
            continue
        with tempfile.NamedTemporaryFile(mode="w", dir=output, prefix=".unbounded-component-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(path)
    return {"status": record["status"], "outputs_written": True, "tasks": 3, "new_runs": 9, "v1_runs": 27}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--if-ready", action="store_true")
    args = parser.parse_args()
    print(json.dumps(render(args.if_ready), sort_keys=True))
