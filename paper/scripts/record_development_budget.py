#!/usr/bin/env python3
"""Record the completed historical low-budget development check, without pooling."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
rows = []
for environment in ("pusht", "reacher"):
    path = root / f"results/development/{environment}_factorized_s0/both_development.json"
    raw = path.read_bytes()
    result = json.loads(raw)
    planning = result["planning"]
    assert result["status"] == "complete"
    assert result["environment"] == environment and result["model_mode"] == "factorized"
    assert result["training_seed"] == 0 and planning["split"] == "development"
    assert {key: planning["planner"][key] for key in ("samples", "iterations", "elites")} == {
        "samples": 128, "iterations": 5, "elites": 16}
    records = planning["records"]
    assert len(records) == 32 and len({x["trajectory_id"] for x in records}) == 32
    assert all(x["observation_id"] == 1 and x["dynamics_id"] == 1 for x in records)
    eligible = [x for x in records if not x["success_during_context"]]
    successes = sum(x["success"] for x in records)
    eligible_successes = sum(x["success"] for x in eligible)
    assert successes / len(records) == planning["summary"]["all"]["success"]["mean"]
    assert eligible_successes / len(eligible) == planning["eligible_summary"]["all"]["success"]["mean"]
    rows.append({"environment": environment, "source_file": str(path.relative_to(root)),
                 "source_sha256": hashlib.sha256(raw).hexdigest(),
                 "checkpoint_sha256": result["checkpoint_sha256"],
                 "data_manifest_sha256": result["data_manifest_sha256"],
                 "evaluator_sha256": result["evaluator_sha256"],
                 "checkpoint_epoch": result["checkpoint_epoch"], "training_seed": 0,
                 "planner": planning["planner"], "successes": successes, "tasks": len(records),
                 "support_successes": len(records) - len(eligible),
                 "eligible_successes": eligible_successes, "eligible_tasks": len(eligible),
                 "execution_context": result["execution_context"],
                 "efficiency_claim_eligible": False})

configuration = root / "external/le-wm/config/eval/solver/cem.yaml"
ledger = {"kind": "historical_low_budget_development_diagnostic", "results": rows,
          "main_budget": {"samples": 300, "iterations": 30, "elites": 30},
          "main_budget_source": {"path": str(configuration.relative_to(root)),
                                 "sha256": hashlib.sha256(configuration.read_bytes()).hexdigest()},
          "interpretation": "One completed trained seed per environment; no baseline or factorization comparison. Shared-GPU times are excluded. Higher search budget is not evidence of improved performance."}
(root / "paper/evidence/development_budget.json").write_text(json.dumps(ledger, indent=2) + "\n")
pusht, reacher = rows
text = f"""The first complete development check used the validation-selected checkpoint
from a completed 30-epoch seed-zero factorized run in each environment, on
all 32 prespecified development tasks at $(v_1,p_1)$, with 128 candidates,
five iterations, and 16 elites. PushT achieved {pusht['successes']}/{pusht['tasks']}
total successes; {pusht['support_successes']} occurred during common support,
leaving {pusht['eligible_successes']}/{pusht['eligible_tasks']} policy-eligible
successes. Reacher achieved {reacher['successes']}/{reacher['tasks']} total successes,
including {reacher['support_successes']} during support, with
{reacher['eligible_successes']}/{reacher['eligible_tasks']} policy-eligible successes.
These are development diagnostics from one trained seed, not full-test
results or evidence of factorization benefit. Baseline comparisons and a
matched rerun at the main sampling budget are needed to interpret them.
The official sampling-budget discrepancy was corrected before learned-policy
full-test planning began. Increasing the search budget is not itself evidence
that planning improves. Shared-GPU timings from this check are excluded
from efficiency claims.\n"""
(root / "paper/generated/development_budget.tex").write_text(text)
print(json.dumps({"source_runs": len(rows), "development_only": True}))
