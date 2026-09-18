"""Reporting guards for the matched development comparison; no simulator runs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "official_development", Path(__file__).parents[1] / "scripts/render_official_development.py"
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)

MANIFEST_SHA = "a" * 64
EXPECTED_TASKS = {f"development_{seed}": seed for seed in range(32)}


def summary(numerator, denominator):
    return {"mean": numerator / denominator, "clusters": denominator, "observations": denominator}


def sign(record):
    protocol = record["planning"]["protocol"]
    record["planning"]["signature"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def fixture_record(mode="single", successes=5, support=2):
    from run_evaluation_campaign import PLANNER_IDENTITY

    planner = {**report.OFFICIAL_PLANNER_BUDGET, **PLANNER_IDENTITY}
    record = {
        "status": "complete", "environment": "pusht", "model_mode": mode,
        "training_seed": None if mode == "frozen" else 0,
        "checkpoint_epoch": 4, "checkpoint_sha256": "b" * 64,
        "evaluator_sha256": "c" * 64, "data_manifest_sha256": MANIFEST_SHA,
    }
    protocol = {
        **report.OFFICIAL_PLANNER_BUDGET, "episodes_per_dynamics": 32,
        "policy": "world_model", "split": "development", "goal_offset": 5, "seed": 1701,
        "search_coordinates": PLANNER_IDENTITY["search_coordinates"],
        "evaluator_sha256": record["evaluator_sha256"],
        "run_identity": {key: record[key] for key in ("checkpoint_sha256", "data_manifest_sha256")},
    }
    rows = [
        {"trajectory_id": trajectory, "seed": seed, "observation_id": 1, "dynamics_id": 1,
         "goal_index": 7, "policy": "world_model", "success": int(seed < successes),
         "success_during_context": int(seed < support), "policy_eligible": int(seed >= support),
         "initial_distance_after_context": 0.1 + seed / 100}
        for trajectory, seed in EXPECTED_TASKS.items()
    ]
    eligible = 32 - support
    record["planning"] = {
        "status": "complete", "policy": "world_model", "split": "development",
        "protocol": protocol, "planner": planner, "records": rows,
        "summary": {"all": {"success": summary(successes, 32), "success_during_context": summary(support, 32)}},
        "eligible_summary": {"all": {"success": summary(successes - support, eligible)}} if eligible else {},
    }
    sign(record)
    return record


def validate(record):
    return report.validate_record(record, "pusht", record["model_mode"], EXPECTED_TASKS, MANIFEST_SHA)


def test_denominators_reconstructed_from_tasks():
    row = validate(fixture_record())
    assert (row["raw_successes"], row["tasks"]) == (5, 32)
    assert (row["support_successes"], row["tasks"]) == (2, 32)
    assert (row["eligible_successes"], row["eligible_tasks"]) == (3, 30)
    assert row["efficiency_claim_eligible"] is False


def test_eligible_rate_cannot_use_raw_denominator():
    record = fixture_record()
    record["planning"]["eligible_summary"]["all"]["success"] = summary(3, 32)
    with pytest.raises(ValueError, match="denominator"):
        validate(record)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_seed"])
def test_exact_32_manifest_identities_required(mutation):
    record = fixture_record()
    rows = record["planning"]["records"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    else:
        rows[-1]["seed"] = 9001
    with pytest.raises(ValueError, match="prespecified tasks|manifest seed/trajectory"):
        validate(record)


def test_pairing_checks_actual_support_mask_not_only_counts():
    reference = validate(fixture_record("frozen"))
    other = fixture_record("single")
    rows = other["planning"]["records"]
    # Both tasks are successful, so exchanging support membership preserves all
    # published counts while violating the common-support experimental design.
    rows[0]["success_during_context"], rows[2]["success_during_context"] = 0, 1
    rows[0]["policy_eligible"], rows[2]["policy_eligible"] = 1, 0
    other = validate(other)
    with pytest.raises(ValueError, match="Common support/goal mask differs"):
        report.verify_pairing([reference, other])


def test_paired_support_audit_covers_available_modes():
    rows = [validate(fixture_record(mode)) for mode in ("frozen", "single")]
    audit = report.verify_pairing(rows)
    assert audit["pusht"]["status"] == "verified_available_modes"
    assert audit["pusht"]["completed_modes"] == ["frozen", "single"]
    assert audit["pusht"]["all_four_modes_available"] is False
    assert audit["reacher"]["status"] == "pending"


def test_mixed_budget_refused_even_with_valid_signature():
    record = fixture_record()
    for container in (record["planning"]["planner"], record["planning"]["protocol"]):
        container.update(samples=128, iterations=5, elites=16)
    sign(record)
    with pytest.raises(RuntimeError, match="budget differs"):
        validate(record)


def test_no_eligible_tasks_is_undefined_not_zero_success():
    record = fixture_record(successes=32, support=32)
    row = validate(record)
    assert row["eligible_tasks"] == 0
    record["planning"]["eligible_summary"] = {"all": {"success": {"mean": 0}}}
    with pytest.raises(ValueError, match="undefined"):
        validate(record)


def test_incomplete_records_ignored_and_completed_rows_display_immediately():
    incomplete = fixture_record()
    incomplete["status"] = "partial"
    assert validate(incomplete) is None
    row = {**validate(fixture_record()), "label": "Shared context"}
    pending = {"environment": "reacher", "mode": "single", "label": "Shared context", "status": "pending"}
    ledger = {"rows": [row, pending], "completed_records": 1,
              "paired_support_verification": report.verify_pairing([row, pending])}
    tex = report.render_tex(ledger)
    assert "Shared context & 5/32 & 2/32 & 3/30" in tex
    assert r"Reacher & Shared context & \missing & \missing & \missing" in tex
    assert "1/8 completed" in tex


def test_changed_post_support_state_rejected():
    reference = validate(fixture_record("frozen"))
    other = validate(fixture_record("single"))
    other["paired_tasks"]["development_0"]["initial_distance_after_context"] += 0.1
    with pytest.raises(ValueError, match="Post-support state distance differs"):
        report.verify_pairing([reference, other])


def test_watcher_fingerprint_tracks_completed_development_only(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "training_refresh", Path(__file__).parents[1] / "scripts/refresh_training.py"
    )
    watcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watcher)
    campaign = tmp_path / "configs/world/full_campaign.json"
    campaign.parent.mkdir(parents=True)
    campaign.write_text('{"tasks": []}')
    before = watcher.fingerprint(tmp_path)
    directory = tmp_path / "results/development_official_budget/pusht_single_s0"
    directory.mkdir(parents=True)
    (directory / "planning_development.progress.json").write_text('{"status": "partial"}')
    assert watcher.fingerprint(tmp_path) == before
    (directory / "planning_development.json").write_text('{"status": "complete"}')
    assert watcher.fingerprint(tmp_path) != before
    before = watcher.fingerprint(tmp_path)
    main_directory = tmp_path / "results/world/pusht_single_s0"
    main_directory.mkdir(parents=True)
    (main_directory / "planning_test.progress.json").write_text('{"status": "partial"}')
    assert watcher.fingerprint(tmp_path) == before
    (main_directory / "planning_test.json").write_text('{"status": "interrupted"}')
    assert watcher.fingerprint(tmp_path) != before
    before = watcher.fingerprint(tmp_path)
    hybrid = tmp_path / "results/development_goal_intervention/reacher_factorized_s0_framewise_goal_s0"
    hybrid.mkdir(parents=True)
    (hybrid / "planning_development.progress.json").write_text('{"status": "partial"}')
    assert watcher.fingerprint(tmp_path) == before
    (hybrid / "planning_development.json").write_text('{"status": "complete"}')
    assert watcher.fingerprint(tmp_path) != before
    before = watcher.fingerprint(tmp_path)
    manuscript = tmp_path / "paper/main.tex"
    manuscript.parent.mkdir(parents=True)
    manuscript.write_text('Updated measured discussion')
    assert watcher.fingerprint(tmp_path) != before
