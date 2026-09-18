#!/usr/bin/env python3
"""Record a dated appendix snapshot; never run training or inspect final tests."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCES = {}
LABELS = {"framewise": "Framewise calibration", "factorized": "ShiftWM (ours)",
          "constant_dynamics": "Constant dynamics context"}


def digest(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    key = str(path.relative_to(ROOT))
    if key not in SOURCES:
        SOURCES[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return SOURCES[key]


def read(path):
    digest(path)
    return json.loads((ROOT / path).read_text())


def main():
    snapshot = datetime.datetime.now(datetime.timezone.utc)
    registry = read("configs/extensions/campaign.json")
    status = {d: {"training": 0, "forecast": 0, "planning": 0} for d in ("drone", "surgery")}
    for run in registry["runs"]:
        output = Path(run["output"])
        summary = read(output / "training_summary.json")
        if summary.get("status") == "completed":
            assert summary["completed_epochs"] == 30
            rows = [json.loads(line) for line in (ROOT / output / "metrics.jsonl").read_text().splitlines() if line]
            digest(output / "metrics.jsonl")
            assert [row["epoch"] for row in rows] == list(range(1, 31))
            assert np.isclose(min(row["val"]["prediction_loss"] for row in rows),
                              summary["best_validation_prediction_loss"], rtol=1e-10, atol=1e-12)
            for package in ("best", "last"):
                assert (ROOT / output / package / "model.pt").is_file()
                digest(output / package / "package_manifest.json")
            status[run["domain"]]["training"] += 1
        for kind in ("forecast", "planning"):
            path = Path("results/extensions_v1") / output.name / "development" / kind / "results.json"
            if (ROOT / path).is_file() and read(path).get("status") == "completed":
                status[run["domain"]][kind] += 1

    matched_tasks = None
    support_commands = None
    families = {}
    for architecture in ("transformer", "gru"):
        methods = {}
        for mode in LABELS:
            successes, masks = [], []
            for seed in (0, 1, 2):
                path = Path(f"results/extensions_v1/drone_{architecture}_{mode}_s{seed}/development/planning/results.json")
                result = read(path)
                args = result["identity"]["arguments"]
                assert all(args[key] == value for key, value in {
                    "domain": "drone", "split": "development", "kind": "planning",
                    "policy": "world_model", "episodes_per_gain": 8, "planner_seed": 101}.items())
                assert result["identity"]["model_config"]["mode"] == mode
                assert result["identity"]["model_config"]["architecture"] == architecture
                for source, expected in result["identity"]["sources"].items():
                    assert digest(source) == expected, source
                records = result["records"]
                assert result["status"] == "completed" and len(records) == result["tasks"] == 8
                keys = [(r["trajectory_id"], r["seed"], r["observation_id"], r["dynamics_id"]) for r in records]
                if matched_tasks is None:
                    matched_tasks = keys
                assert keys == matched_tasks and len(set(keys)) == 8
                supports = []
                for row in records:
                    assert row["policy"] == "world_model" and row["native_budget"] == 200 and row["support_budget"] == 10
                    metrics = row["metrics_per_native_step"]
                    assert len(metrics) == row["native_calls"] <= 200
                    hits = [i + 1 for i, metric in enumerate(metrics) if metric["success"] and not
                            (metric["crash"] or metric["workspace_escape"])]
                    assert row["success"] == bool(hits)
                    assert row["first_success_native_step"] == (hits[0] if hits else None)
                    assert not row["success_during_support"] and not any(i <= 10 for i in hits)
                    trace = path.parent / row["trace_file"]
                    assert digest(trace) == row["trace_sha256"]
                    with np.load(ROOT / trace, allow_pickle=False) as data:
                        assert len(data["commands"]) == row["native_calls"]
                        supports.append(data["commands"][:10].tolist())
                if support_commands is None:
                    support_commands = supports
                assert supports == support_commands
                assert result["support_successes"] == 0
                assert result["successes"] == sum(r["success"] for r in records)
                successes.append(result["successes"])
                masks.append([int(r["success"]) for r in records])
            rates = np.asarray(successes) / 8 * 100
            methods[mode] = {"successes_by_seed": successes, "tasks_per_seed": 8,
                             "mean_percent": float(rates.mean()), "sample_sd_percent": float(rates.std(ddof=1)),
                             "success_masks_by_seed": masks}
        families[architecture] = methods

    controls = {}
    for policy in ("random", "replay_oracle"):
        path = Path(f"results/extensions_v1/drone_{policy}/development/planning/results.json")
        result = read(path)
        args = result["identity"]["arguments"]
        assert args["policy"] == policy and args["planner_seed"] == 101 and args["split"] == "development"
        assert result["status"] == "completed" and result["tasks"] == 8 and result["support_successes"] == 0
        records = result["records"]
        assert [(r["trajectory_id"], r["seed"], r["observation_id"], r["dynamics_id"]) for r in records] == matched_tasks
        assert sum(r["success"] for r in records) == result["successes"]
        for source, expected in result["identity"]["sources"].items():
            assert digest(source) == expected, source
        for index, row in enumerate(records):
            assert row["policy"] == policy and row["native_budget"] == 200 and row["support_budget"] == 10
            trace = path.parent / row["trace_file"]
            assert digest(trace) == row["trace_sha256"]
            with np.load(ROOT / trace, allow_pickle=False) as data:
                assert data["commands"][:10].tolist() == support_commands[index]
        controls[policy] = {"successes": result["successes"], "tasks": 8, "policy_base_seed": 101}

    ranking = read("reports/evidence/drone_action_ranking.json")
    validation = read("reports/evidence/drone_action_ranking_validation.json")
    assert validation["status"] == "passed" and validation["report_sha256"] == digest("reports/evidence/drone_action_ranking.json")
    assert len(ranking["branches"]) == 128 and len(ranking["comparisons"]) == 24
    assert all(r["support_replay_exact"] for r in ranking["branches"])
    assert sum(not r["full_horizon_safe"] for r in ranking["branches"]) == 2
    assert len(ranking["aggregates"]) == 6 and all(r["tasks"] == r["safe_predicted_winners"] == 4 for r in ranking["aggregates"])
    for key in ("reports/extensions_mechanism_diagnosis.md", "reports/evidence/extensions_mechanism_probe.json",
                "reports/evidence/drone_action_ranking.md", "artifacts/diagnostics/goal_geometry.json",
                "paper/figures/goal_geometry.pdf", "reports/geometry_revision_protocol.md",
                "paper/scripts/record_extension_diagnostics.py"):
        digest(key)
    record = {"snapshot_utc": snapshot.isoformat(), "scope": "development-only appendix snapshot",
              "completion_check": "30 chronological metric rows, completed summary and checkpoint manifests; separate release validator checks full packages",
              "stages": status, "drone_planning": families, "controls": controls, "task_identities": matched_tasks,
              "uncertainty": "sample SD of three trained-seed rates, not a confidence interval",
              "fixed_candidate_diagnostic": ranking["aggregates"], "sources_sha256": SOURCES}
    (ROOT / "paper/evidence/extension_diagnostics.json").write_text(json.dumps(record, indent=2) + "\n")
    rows = [r"\begin{tabular}{lrrrr}\toprule",
            r"Method & Seed 0 & Seed 1 & Seed 2 & Mean $\pm$ SD (\%)\\\midrule"]
    for architecture, methods in families.items():
        rows.append(r"\multicolumn{5}{l}{\emph{" + architecture.upper() + r" predictor}}\\")
        for mode, values in methods.items():
            rows.append(LABELS[mode] + " & " + " & ".join(f"{n}/8" for n in values["successes_by_seed"]) +
                        f" & {values['mean_percent']:.2f} $\\pm$ {values['sample_sd_percent']:.2f}" + r"\\")
        rows.append(r"\addlinespace[3pt]")
    rows.append(r"\bottomrule\end{tabular}")
    (ROOT / "paper/generated/extension_drone_development.tex").write_text("\n".join(rows) + "\n")
    print(json.dumps({"snapshot_utc": record["snapshot_utc"], "stages": status,
                      "drone_means": {a: {m: x["mean_percent"] for m,x in methods.items()}
                                      for a,methods in families.items()}}, indent=2))



if __name__ == "__main__":
    main()
