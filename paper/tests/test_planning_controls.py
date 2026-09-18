"""Guard control-policy identities, source protocols and displayed counts."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "scripts/render_planning_controls.py"
_spec = importlib.util.spec_from_file_location("planning_controls", _path)
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)

IDENTITIES = {"checkpoint_sha256": "checkpoint", "data_manifest_sha256": "manifest", "evaluator_sha256": "evaluator"}
TASKS = {(f"test-{seed}", 2): {"seed": seed, "dynamics_id": 2} for seed in (100, 101)}


def sign(record):
    planning = record["planning"]
    planning["signature"] = hashlib.sha256(json.dumps(planning["protocol"], sort_keys=True).encode()).hexdigest()


def summarize(rows, metrics):
    if not rows:
        return {}
    value = {metric: {"mean": sum(row[metric] for row in rows) / len(rows),
                      "observations": len(rows), "clusters": len({row["seed"] for row in rows})}
             for metric in metrics}
    return {"all": copy.deepcopy(value), "o2_d2": copy.deepcopy(value)}


@pytest.fixture
def record():
    rows = []
    for i, seed in enumerate((100, 101)):
        rows.append({"trajectory_id": f"test-{seed}", "seed": seed, "observation_id": 2, "dynamics_id": 2,
            "goal_index": 7, "policy": "random", "success": 1 - i, "final_success": 1 - i,
            "success_during_context": 1 - i, "policy_eligible": i, "native_steps": 5 if i == 0 else 50,
            "num_replans": 0, "total_solve_seconds": 0., "initial_distance_after_context": float(i),
            "initial_block_translation_error_px": 0., "initial_block_angle_error_rad": 0.,
            "initial_agent_position_error_px": float(i)})
    budget = report.campaign.CONTROL_METADATA_BUDGET.copy()
    protocol = {key: budget[key] for key in ("samples", "iterations", "elites", "horizon", "native_budget")}
    protocol.update(policy="random", split="test", episodes_per_dynamics=64, goal_offset=5, seed=1701,
                    evaluator_sha256=IDENTITIES["evaluator_sha256"],
                    run_identity={key: IDENTITIES[key] for key in ("checkpoint_sha256", "data_manifest_sha256")})
    value = {"status": "complete", "environment": "pusht", "model_mode": "frozen", "training_seed": None,
             "checkpoint_epoch": 0, **IDENTITIES,
             "planning": {"status": "complete", "kind": "closed_loop_goal_image_planning", "policy": "random",
                          "split": "test", "planner": budget, "protocol": protocol, "records": rows,
                          "summary": summarize(rows, ("success", "final_success", "success_during_context")),
                          "eligible_summary": summarize(rows[1:], ("success", "final_success"))}}
    sign(value)
    return value


def validate(record):
    return report.validate_record(record, "pusht", "test", "random", TASKS, IDENTITIES)


def test_original_control_policy_is_preserved_by_compatibility_adapter(record):
    before = copy.deepcopy(record)
    rows = validate(record)
    assert record == before
    assert {row["policy"] for row in rows.values()} == {"random"}


@pytest.mark.parametrize("change", ["wrong_row_policy", "wrong_protocol_policy", "wrong_seed", "wrong_budget",
                                     "wrong_source", "missing_task", "duplicate_task", "wrong_goal", "cem_solve"])
def test_malformed_control_or_protocol_is_rejected(record, change):
    planning = record["planning"]
    if change == "wrong_row_policy":
        planning["records"][0]["policy"] = "world_model"
    elif change == "wrong_protocol_policy":
        planning["protocol"]["policy"] = "replay_oracle"; sign(record)
    elif change == "wrong_seed":
        planning["protocol"]["seed"] = 42; sign(record)
    elif change == "wrong_budget":
        planning["planner"]["native_budget"] = 60
    elif change == "wrong_source":
        record["checkpoint_sha256"] = "changed"
    elif change == "missing_task":
        planning["records"].pop()
    elif change == "duplicate_task":
        planning["records"][1] = copy.deepcopy(planning["records"][0])
    elif change == "wrong_goal":
        planning["records"][0]["goal_index"] = 8
    else:
        planning["records"][1]["num_replans"] = 1
    with pytest.raises((ValueError, RuntimeError)):
        validate(record)


@pytest.mark.parametrize("key,value", [("mean", .99), ("observations", 3), ("clusters", 1)])
def test_wrong_summary_statistics_are_rejected(record, key, value):
    record["planning"]["summary"]["o2_d2"]["success"][key] = value
    with pytest.raises(ValueError, match="summary count"):
        validate(record)


def test_eligible_summary_cannot_use_raw_task_denominator(record):
    record["planning"]["eligible_summary"]["o2_d2"]["success"]["observations"] = 2
    with pytest.raises(ValueError, match="summary count"):
        validate(record)


def test_shared_support_detects_changed_post_support_state(record):
    reference = validate(record)
    changed = copy.deepcopy(record)
    changed["planning"]["records"][1]["initial_distance_after_context"] += .1
    control = validate(changed)
    with pytest.raises(ValueError, match="support diagnostics"):
        report.paired.validate_shared_support(reference, control, "pusht")


def test_unused_cem_sampling_metadata_is_not_a_control_budget(record):
    record["planning"]["planner"]["samples"] = 999
    record["planning"]["protocol"]["samples"] = 999
    sign(record)
    assert validate(record)


def test_incomplete_control_is_missing_not_zero(record):
    record["status"] = "interrupted"
    assert validate(record) is None


def test_heldout_selects_o2_d2_instead_of_full_test_average():
    rows = {}
    for appearance, dynamics in ((0, 0), (2, 2)):
        for seed in range(64):
            rows[f"{dynamics}-{seed}", appearance] = {"observation_id": appearance, "dynamics_id": dynamics,
                "seed": seed, "success": int(appearance == 0 or seed < 5), "success_during_context": 0,
                "policy_eligible": 1}
    counts = report.population_counts(rows, "test")
    assert counts["raw_successes"] == 5 and counts["raw_total"] == 64
    assert counts["raw_percent"] == 7.8125


def test_extrapolation_pools_three_conditions_but_keeps_64_state_clusters():
    rows = {}
    for appearance, dynamics in ((3, 0), (0, 3), (3, 3)):
        for seed in range(64):
            rows[f"{dynamics}-{seed}", appearance] = {"observation_id": appearance, "dynamics_id": dynamics,
                "seed": seed, "success": int(seed < 32), "success_during_context": 0, "policy_eligible": 1}
    counts = report.population_counts(rows, "extrapolation")
    assert counts["raw_successes"] == 96 and counts["raw_total"] == 192
    assert counts["initial_state_clusters"] == 64 and counts["raw_percent"] == 50.
