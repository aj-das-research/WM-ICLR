#!/usr/bin/env python3
"""Render completed predictor costs from portable timing receipts; never run a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
TASKS = {"pusht": "PushT", "bimanual_box": "Bimanual Box", "bimanual_rope": "Bimanual Rope"}
MODES = {
    "autoregressive": "Autoregressive",
    "anchored_additive": "Additive anchor",
    "bounded_spatial_mix": "ShiftWM (ours)",
    "unbounded_spatial_mix": "No tanh (ablation)",
    "persistence": "Persistence",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def checked_pack(directory):
    """Recompute displayed quantities from all 2,340 bound timing samples."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    check(manifest.get("schema") == "iws_predictor_resource_pack_manifest_v1", "Unknown pack schema")
    check(manifest.get("status") == "complete_validated_resource_measurement", "Incomplete pack")
    check(manifest.get("runtime_inputs_sha256") == {"data.json": sha(directory / "data.json")}, "Data binding differs")
    data = json.loads((directory / "data.json").read_text())
    check(data.get("schema") == "iws_predictor_resource_portable_v1" and data.get("status") == "passed", "Invalid data status")
    check(data["registration_sha256"] == manifest["source_registration_sha256"], "Registration differs")
    check(data.get("unsupported_states") == [], "Unsupported device rows")
    check(data.get("official_validation_payloads_read") == 0 and data.get("accuracy_metrics_computed") is False, "Wrong evidence scope")
    protocol = data["measurement_protocol"]
    for key, value in {"batch_size": 1, "predicted_offsets": 59, "timed_calls_per_device": 30,
                       "warmup_calls_per_device": 10, "torch_intraop_threads": 2,
                       "torch_interop_threads": 1, "dtype": "float32", "autocast": False,
                       "tf32": False}.items():
        check(protocol.get(key) == value, f"Protocol differs: {key}")
    expected = {(task, mode, seed) for task in TASKS for mode in MODES if mode != "persistence" for seed in range(3)}
    expected |= {(task, "persistence", None) for task in TASKS}
    cases = data["all39_case_measurements"]
    identity = lambda row: (row["case"]["task"], row["case"]["mode"], row["case"]["seed"])
    check(len(cases) == 39 and {identity(row) for row in cases} == expected, "Incomplete or duplicate model grid")
    for row in cases:
        check(row["status"] == "passed" and row["registration_sha256"] == data["registration_sha256"], "Invalid case status/binding")
        check(row["official_validation_payloads_read"] == 0 and row["workspace_or_network_access_attempts"] == [], "Unexpected data/network access")
        check(row["forecast_shape"] == [1, 59, 6144] and row["dtype"] == "float32", "Forecast scope differs")
        for device in ("cpu", "cuda"):
            result = row["devices"][device]
            values = result["latency"]["samples_ms"]
            check(result["status"] == "passed", "Unsupported case device")
            check(len(values) == 30 and all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values), "Invalid raw timings")
            check(statistics.median(values) == result["latency"]["median_ms"], "Case median differs")
    rows = data["rows"]
    check(len(rows) == 15 and {(row["task"], row["mode"]) for row in rows} == {(t, m) for t in TASKS for m in MODES}, "Incomplete table grid")
    for row in rows:
        subset = sorted((case for case in cases if identity(case)[:2] == (row["task"], row["mode"])), key=lambda x: x["case"]["seed"] or 0)
        check(row["measured_models"] == len(subset) and row["seeds"] == [case["case"]["seed"] for case in subset], "Seed labels differ")
        check(all(case["parameters"] == row["parameters"] for case in subset), "Parameter counts differ")
        if row["mode"] == "persistence":
            check(row["parameters"] == {"total": 0, "trainable": 0}, "Persistence is untrained")
        for device in ("cpu", "cuda"):
            aggregate = row["devices"][device]
            values = [case["devices"][device]["latency"]["median_ms"] for case in subset]
            check(aggregate["per_seed_median_ms"] == values, "Seed medians differ")
            check(aggregate["median_of_seed_medians_ms"] == statistics.median(values), "Display median differs")
            check(aggregate["mean_of_seed_medians_ms"] == statistics.mean(values), "Registered mean differs")
            check(aggregate["range_of_seed_medians_ms"] == [min(values), max(values)], "Seed range differs")
            fields = ("process_peak_rss_bytes",) if device == "cpu" else ("peak_allocated_bytes", "peak_reserved_bytes", "incremental_peak_allocated_bytes")
            for field in fields:
                peaks = [case["devices"][device]["memory"][field] for case in subset]
                check(aggregate["memory"][field + "_per_model"] == peaks, "Individual memory peaks differ")
                check(aggregate["memory"]["maximum_" + field] == max(peaks), "Memory maximum differs")
    check(manifest["learned_models"] == 36 and manifest["persistence_cases"] == 3 and manifest["timing_samples"] == 2340, "Coverage counts differ")
    return data


CAPTION = r"""Predictor-only IWS inference costs on fixed training inputs, not a control benchmark. Each call predicts all 59 offsets at batch size one in FP32. CPU: Xeon w7-2495X, two PyTorch threads; GPU: RTX 5000 Ada, synchronized wall time. After ten warmups, each model has 30 timed calls. Latency is the median of three seed-level medians; brackets give their minimum--maximum, not a confidence interval. Persistence materializes the full output but ignores commands and has one measurement per task. Parameter counts are trainable/total in thousands. Memory is the maximum across measured models: CPU whole-process peak RSS (including interpreter and loading), versus GPU PyTorch allocated peak (excluding CUDA context). Timing excludes the encoder, loading, transfers and planning. CUDA costs do not certify GPU forecast accuracy or prefix equivalence. All raw timings and mean-of-medians remain in the source pack."""


def timing(device, count):
    value = device["median_of_seed_medians_ms"]
    if count == 1:
        return f"{value:.3f}"
    low, high = device["range_of_seed_medians_ms"]
    return f"{value:.1f} [{low:.1f}--{high:.1f}]"


def tex_table(data):
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + CAPTION + "}",
             r"\label{tab:iws-predictor-resources}",
             r"\begingroup\fontsize{9}{10.8}\selectfont\setlength{\tabcolsep}{2.5pt}\renewcommand{\arraystretch}{1.12}",
             r"\begin{tabular}{@{}lrrrrr@{}}\toprule",
             r"Method & \shortstack{Parameters\\train./total ($10^3$)} & \shortstack{CPU latency\\ms [seed range]} & \shortstack{GPU latency\\ms [seed range]} & \shortstack{CPU RSS\\MiB} & \shortstack{GPU alloc.\\MiB} \\ \midrule"]
    for task, name in TASKS.items():
        lines.append(r"\multicolumn{6}{@{}l}{\textbf{" + name + r"}} \\")
        for mode, label in MODES.items():
            row = next(row for row in data["rows"] if (row["task"], row["mode"]) == (task, mode))
            cpu, gpu = row["devices"]["cpu"], row["devices"]["cuda"]
            params = row["parameters"]
            parameters = f"{params['trainable']/1000:.3f}/{params['total']/1000:.3f}" if params["total"] else "0/0"
            lines.append(" & ".join([label, parameters, timing(cpu, row["measured_models"]), timing(gpu, row["measured_models"]),
                                      f"{cpu['memory']['maximum_process_peak_rss_bytes']/1048576:.1f}",
                                      f"{gpu['memory']['maximum_peak_allocated_bytes']/1048576:.2f}"]) + r" \\")
        if task != list(TASKS)[-1]:
            lines.append(r"\addlinespace[3pt]")
    lines += [r"\bottomrule\end{tabular}\endgroup", r"\end{table}", ""]
    return "\n".join(lines)


def write_changed(path, text):
    path = Path(path)
    if path.exists() and path.read_text() == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)
    return True


def render(pack_dir, output_dir):
    pack_dir, output_dir = Path(pack_dir), Path(output_dir)
    data = checked_pack(pack_dir)
    table = output_dir / "predictor_resources.tex"
    written = write_changed(table, tex_table(data))
    evidence = {"schema": "iws_predictor_resource_table_v1", "status": "passed",
                "source_sha256": {"renderer": sha(__file__), "data.json": sha(pack_dir / "data.json"), "manifest.json": sha(pack_dir / "manifest.json")},
                "output_sha256": {table.name: sha(table)}, "rows": 15, "learned_models": 36,
                "persistence_cases": 3, "timing_samples_recomputed": 2340,
                "display_aggregation": "Median of three seed-level medians, with min/max; one case per task for persistence.",
                "memory_aggregation": "Maximum recorded peak across measured cases; CPU process RSS and GPU allocated peak are different measures.",
                "label": "tab:iws-predictor-resources", "font_size_pt": 9,
                "no_model_or_dataset_access": True, "gpu_accuracy_or_prefix_certificate": False}
    written |= write_changed(output_dir / "evidence.json", json.dumps(evidence, sort_keys=True, indent=2) + "\n")
    return {"status": "passed", "outputs_written": written, "rows": 15, "output": str(table)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pack-dir", type=Path, default=ROOT / "paper/table_sources/iws_predictor_resources_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/generated/iws_resources")
    arguments = parser.parse_args()
    print(json.dumps(render(arguments.pack_dir, arguments.output_dir), sort_keys=True))
