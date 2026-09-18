"""Data/selection guards for observed-frame panels; fixtures are not evidence."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load_script("qualitative_reporting_test", ROOT / "paper/scripts/render_qualitative.py")
fixtures = load_script("qualitative_source_fixtures", ROOT / "tests/test_goal_intervention_reporting.py")
SOURCES = ("paper/scripts/render_qualitative.py", "scripts/summarize_goal_intervention.py",
           "scripts/evaluate_goal_calibration_intervention.py", "src/shiftwm/evaluate.py",
           "src/shiftwm/generate.py", "src/shiftwm/data.py", "src/shiftwm/model.py",
           "src/shiftwm/checkpoint.py", "src/shiftwm/upstream.py")


def row(blocks, native_steps, eligible=1, success=1):
    return {"executed_action_blocks": blocks, "native_steps": native_steps,
            "policy_eligible": eligible, "success": success, "num_replans": len(blocks)}


def test_timeline_uses_actual_partial_terminal_native_actions():
    record = row([[0.] * 10, [0.] * 10, [0.] * 6], 23)
    assert report.frame_steps(record, [None] * 4) == [10, 15, 20, 23]
    assert report.frame_steps(row([[0.] * 10] * 8, 50, success=0), [None] * 9) == list(range(10, 51, 5))


@pytest.mark.parametrize("steps", [1, 4, 5, 6, 10])
def test_support_only_success_has_one_frame_and_no_planned_action(steps):
    assert report.frame_steps(row([], steps, eligible=0), [None]) == [steps]


@pytest.mark.parametrize("blocks", [
    [[]], [[0., 1., 2.]], [[0.] * 12], [[float("nan"), 0.]],
    [[float("inf"), 0.]], [[[0.] * 5, [0.] * 5]], [["0", "0"]],
])
def test_malformed_action_values_or_dimensions_are_rejected(blocks):
    with pytest.raises(ValueError, match="Malformed"):
        report.frame_steps(row(blocks, 11), [None] * (len(blocks) + 1))


@pytest.mark.parametrize("record,frames", [
    (row([[0.] * 10], 15), [None]),
    (row([[0.] * 10], 14), [None] * 2),
    (row([[0.] * 10] * 9, 55), [None] * 10),
    (row([[0.] * 10], 10, eligible=0), [None] * 2),
    (row([], 0, eligible=0), [None]),
    (row([], 11, eligible=0), [None]),
    (row([[0.] * 2, [0.] * 10], 16), [None] * 3),
    (row([[0.] * 2], 11, success=0), [None] * 2),
    (row([[0.] * 10], 15.), [None] * 2),
    ({**row([[0.] * 10], 15), "num_replans": 2}, [None] * 2),
])
def test_impossible_timeline_or_support_record_is_rejected(record, frames):
    with pytest.raises(ValueError):
        report.frame_steps(record, frames)


@pytest.mark.parametrize("ours,baseline,expected", [(1, 0, "ours_only"), (0, 1, "baseline_only"),
                                                     (0, 0, "neither"), (1, 1, "both")])
def test_outcome_strata_use_explicit_framewise_comparator(ours, baseline, expected):
    records = {"factorized": {"success": ours, "policy_eligible": 1},
               "framewise": {"success": baseline, "policy_eligible": 1},
               "single": {"success": 1 - baseline, "policy_eligible": 1}}
    assert report.category(records) == expected
    records["factorized"].update(policy_eligible=0, success=1)
    records["framewise"].update(policy_eligible=0, success=1)
    assert report.category(records) == "support_success"


def test_selection_is_fixed_stratified_and_independent_of_input_order():
    examples = []
    for environment in ("pusht", "reacher"):
        for category in ("ours_only", "baseline_only", "neither", "both", "support_success"):
            for trajectory, observation in (("z", 0), ("a", 2), ("a", 1)):
                examples.append({"environment": environment, "category": category,
                                 "trajectory_id": trajectory, "observation_id": observation})
    selected = report.select_examples(examples)
    assert selected == report.select_examples(list(reversed(examples)))
    assert len(selected) == 8
    assert [(ex["trajectory_id"], ex["observation_id"]) for ex in selected] == [("a", 1)] * 8
    assert [ex["category"] for ex in selected] == ["ours_only", "baseline_only", "neither", "support_success",
                                                  "ours_only", "baseline_only", "neither", "both"]
    with pytest.raises(ValueError, match="declared stratum"):
        report.select_examples([ex for ex in examples if ex["category"] != "baseline_only"])


def fixture_expectations(root, environment):
    expected = fixtures.expectations(root, environment)
    for mode, source in expected["sources"].items():
        source["path"] = str(root / f"runs/world/{environment}_{mode}_s0/best")
        source["config"] = {"provenance": {"action_stats": f"data/upstream/{environment}/action_stats.json"}}
        source["hashes"].update({"run_config_sha256": mode + "run", "training_summary_sha256": mode + "summary",
                                  "metrics_sha256": mode + "metrics", "action_stats_sha256": environment + "stats"})
    return expected


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    # Reuse the real reporting validators with their established 32-task fixture.
    # Only the expectation builder is replaced; source/protocol/support checks run.
    monkeypatch.setattr(report.contract, "build_expectations", fixture_expectations)
    for name in SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture source\n")
    fixtures.populate(tmp_path, modes=report.MODES)
    goal = np.full((224, 224, 3), 22, dtype=np.uint8)
    for count in (1, 9):
        frames = np.full((count, 224, 224, 3), 17, dtype=np.uint8)
        np.savez_compressed(tmp_path / f"frames-{count}.npz", frames=frames, goal_image=goal)
    for environment in ("pusht", "reacher"):
        for mode in report.MODES:
            path = fixtures.result_path(tmp_path, environment, mode)
            record = json.loads(path.read_text())
            for entry in record["planning"]["records"]:
                count = 1 if entry["success_during_context"] else 9
                entry["executed_action_blocks"] = [[0.] * 10 for _ in range(count - 1)]
                video = path.parent / "videos" / f"{entry['trajectory_id']}-o{entry['observation_id']}.npz"
                video.parent.mkdir(exist_ok=True)
                os.link(tmp_path / f"frames-{count}.npz", video)
            fixtures.save(path, record)
    return tmp_path


def video_path(root, mode="factorized"):
    return fixtures.result_path(root, "pusht", mode).parent / "videos/development-s0-d1-o1.npz"


def replace_video(path, change):
    with np.load(path, allow_pickle=False) as stored:
        frames, goal = stored["frames"].copy(), stored["goal_image"].copy()
    frames, goal = change(frames, goal)
    path.unlink()  # Do not modify the shared synthetic fixture through its hard link.
    np.savez_compressed(path, frames=frames, goal_image=goal)


def test_collect_retains_exact_pixels_times_all_tasks_and_validated_sources(inputs):
    examples, sources = report.collect(inputs)
    assert len(examples) == 64
    assert [ex["environment"] for ex in examples].count("pusht") == 32
    for ex in examples:
        for mode, item in ex["methods"].items():
            with np.load(inputs / item["video"], allow_pickle=False) as original:
                np.testing.assert_array_equal(item["frames"], original["frames"])
                np.testing.assert_array_equal(item["goal"], original["goal_image"])
            assert item["frame_steps"][-1] == item["record"]["native_steps"]
            assert sources[item["video"]] == report.sha(inputs / item["video"])
    for environment in ("pusht", "reacher"):
        for mode in report.MODES:
            assert sources[f"runs/world/{environment}_{mode}_s0/best/model.pt"] == mode
            assert f"runs/world/{environment}_{mode}_s0/metrics.jsonl" in sources
            assert f"runs/world/{environment}_{mode}_s0/training_summary.json" in sources
    assert set(SOURCES).issubset(sources)


@pytest.mark.parametrize("damage", ["empty", "frame_shape", "frame_dtype", "goal_shape", "goal_dtype"])
def test_invalid_saved_rgb_sequences_are_rejected(inputs, damage):
    def change(frames, goal):
        if damage == "empty":
            frames = frames[:0]
        elif damage == "frame_shape":
            frames = frames[:, :200]
        elif damage == "frame_dtype":
            frames = frames.astype(np.float32)
        elif damage == "goal_shape":
            goal = goal[..., :2]
        else:
            goal = goal.astype(np.float32)
        return frames, goal
    replace_video(video_path(inputs), change)
    with pytest.raises(ValueError, match="RGB dimensions or dtype"):
        report.collect(inputs)


@pytest.mark.parametrize("pixel", ["goal", "start"])
def test_paired_start_and_goal_require_exact_pixel_equality(inputs, pixel):
    def change(frames, goal):
        if pixel == "goal":
            goal[0, 0, 0] += 1
        else:
            frames[0, 0, 0, 0] += 1
        return frames, goal
    replace_video(video_path(inputs, "framewise"), change)
    with pytest.raises(ValueError, match="different goal/start pixels"):
        report.collect(inputs)


@pytest.mark.parametrize("damage", ["source", "split", "support", "incomplete", "task"])
def test_collect_keeps_existing_source_task_and_support_validators(inputs, damage):
    path = fixtures.result_path(inputs, "pusht", "framewise")
    record = json.loads(path.read_text())
    if damage == "source":
        record["checkpoint_sha256"] = "wrong"
    elif damage == "split":
        record["planning"]["split"] = "test"
    elif damage == "support":
        record["planning"]["records"][0]["initial_distance_after_context"] += 1
    elif damage == "incomplete":
        record["status"] = "interrupted"
    else:
        record["planning"]["records"][0]["seed"] += 1
    fixtures.save(path, record)
    with pytest.raises(ValueError):
        report.collect(inputs)


def test_dependency_ledger_reuses_prevalidated_checkpoint_hashes_and_pins_helpers(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(report, "sha", lambda path: calls.append(Path(path)) or "helper-digest")
    expected = fixture_expectations(tmp_path, "pusht")
    sources = report.validated_dependencies(tmp_path, "pusht", expected)
    assert len(calls) == len(SOURCES)
    assert not any(path.name == "model.pt" for path in calls)
    assert sources["data/world/pusht_relative/manifest.json"] == "data"
    assert sources["data/upstream/pusht/action_stats.json"] == "pushtstats"
    for mode in report.MODES:
        run = f"runs/world/pusht_{mode}_s0"
        assert sources[run + "/best/model.pt"] == mode
        assert sources[run + "/best/config.json"] == mode + "config"
        assert sources[run + "/run_config.json"] == mode + "run"
        assert sources[run + "/training_summary.json"] == mode + "summary"
        assert sources[run + "/metrics.jsonl"] == mode + "metrics"
    assert set(SOURCES).issubset(sources)
    changed = deepcopy(expected)
    changed["sources"]["single"]["hashes"]["action_stats_sha256"] = "different"
    with pytest.raises(ValueError, match="inconsistent source hashes"):
        report.validated_dependencies(tmp_path, "pusht", changed)
