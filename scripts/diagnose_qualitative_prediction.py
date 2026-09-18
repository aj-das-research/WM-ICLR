#!/usr/bin/env python3
"""Matched one-block predictions on replay-validated development executions.

Both completed checkpoints predict the SAME observed history/action transition
on each realized policy trace. This is conditional prediction error, not CEM
candidate ranking, a decoded video, a new policy evaluation or causal evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from shiftwm.checkpoint import load_package
from shiftwm.data import pixels_to_tensor
from shiftwm.evaluate import atomic_json

ROOT = Path(__file__).resolve().parents[1]
MODES = ("factorized", "framewise")
ANCHORS = (10, 15, 20)


def local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


paired = local_module("qualitative_prediction_sources", ROOT / "scripts/summarize_paired_planning.py")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read(path):
    return json.loads(Path(path).read_text())


def equal_modules(left, right, label):
    a, b = left.state_dict(), right.state_dict()
    if a.keys() != b.keys() or any(not torch.equal(a[k].cpu(), b[k].cpu()) for k in a):
        raise ValueError(f"Immutable feature coordinates differ: {label}")


def validate_coordinates(models):
    first = models[MODES[0]]
    for mode, model in models.items():
        if (model.config.mode != mode or model.config.freeze_visual is not True
                or model.config.history_length != 3 or model.latent_dim != 192 or model.action_dim != 10):
            raise ValueError("Expected original frozen-visual three-frame model configuration")
        equal_modules(model.base.encoder, model.reference_encoder, f"{mode} base/reference encoder")
        equal_modules(model.base.projector, model.reference_projector, f"{mode} base/reference projector")
        equal_modules(first.reference_encoder, model.reference_encoder, "cross-model reference encoder")
        equal_modules(first.reference_projector, model.reference_projector, "cross-model reference projector")
        for name in ("action_mean", "action_std", "pixel_mean", "pixel_std"):
            if not torch.equal(getattr(first, name).cpu(), getattr(model, name).cpu()):
                raise ValueError(f"Cross-model preprocessing differs: {name}")
        if model.base_config != first.base_config:
            raise ValueError("Cross-model backbone configuration differs")


def load_models(environment, device):
    manifest = ROOT / "data/world" / ("pusht_relative" if environment == "pusht" else environment) / "manifest.json"
    data_sha = sha(manifest)
    digests = paired.campaign.FileDigestCache()
    models, sources = {}, {}
    for mode in MODES:
        source = paired.completed_source(ROOT, environment, mode, 0, data_sha, digests)
        package = Path(source["path"])
        model, state = load_package(package, device="cpu")
        if (state["config"] != read(package / "config.json") or type(state["epoch"]) is not int
                or state["epoch"] not in source["best_epochs"]
                or not math.isclose(state["best_metric"], source["best_validation_prediction_loss"],
                                    rel_tol=1e-10, abs_tol=1e-12)
                or sha(package / "model.pt") != source["hashes"]["checkpoint_sha256"]):
            raise ValueError("Loaded checkpoint differs from completed validation-selected source")
        model.eval()
        models[mode], sources[f"{environment}_{mode}_s0"] = model, {**source, "selected_epoch": state["epoch"]}
    if sources[f"{environment}_factorized_s0"]["coordinate_identity"] != sources[f"{environment}_framewise_s0"]["coordinate_identity"]:
        raise ValueError("Checkpoint provenance/action normalization/reference coordinates differ")
    validate_coordinates(models)
    for model in models.values():
        model.float().to(device).eval()
    return models, sources


def validate_arrays(arrays, observation_id):
    required = {"canonical_frames", "shifted_frames", "native_times", "executed_native_actions"}
    if not required <= arrays.keys():
        raise ValueError("Replay archive lacks canonical support/action arrays")
    canonical, shifted = arrays["canonical_frames"], arrays["shifted_frames"]
    times, actions = arrays["native_times"], arrays["executed_native_actions"]
    if (canonical.dtype != np.uint8 or canonical.ndim != 4 or canonical.shape[1:] != (224, 224, 3)
            or shifted.shape != canonical.shape or shifted.dtype != np.uint8):
        raise ValueError("Replay frames must be matching uint8 RGB observations")
    if (times.ndim != 1 or not np.issubdtype(times.dtype, np.integer) or len(times) != len(canonical)
            or len(times) < 3 or list(times[:3]) != [0, 5, 10] or np.any(np.diff(times) <= 0)):
        raise ValueError("Replay must contain chronological support at native0,5,10")
    if (actions.ndim != 2 or actions.shape[1] != 2 or not np.issubdtype(actions.dtype, np.floating)
            or not np.isfinite(actions).all() or len(actions) != times[-1]):
        raise ValueError("Replay executed native actions do not match frame timing")
    # Planning used FLOAT transformed pixels. Quantized RGB is checked against
    # the saved videos but never used as the model input after transformation.
    expected = (pixels_to_tensor(canonical, observation_id).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    if not np.array_equal(expected, shifted):
        raise ValueError("Replay shifted observations differ from the exact original transform")
    for t in times[3:-1]:
        if t % 5:
            raise ValueError("Only the final executed block can be partial")
    return {int(t): i for i, t in enumerate(times)}


def windows(arrays, observation_id):
    index = validate_arrays(arrays, observation_id)
    result, excluded = [], []
    terminal = int(arrays["native_times"][-1])
    for anchor in ANCHORS:
        if anchor + 5 > terminal:
            excluded.append({"anchor_native_time": anchor, "reason": "no_full_next_five_native_action_block",
                             "terminal_native_time": terminal})
            continue
        times = [anchor - 10, anchor - 5, anchor, anchor + 5]
        if any(t not in index for t in times):
            raise ValueError("A complete five-step prediction lacks the required observed boundary frame")
        result.append({"anchor_native_time": anchor, "target_native_time": anchor + 5,
            "history_native_times": times[:3], "future_action_native_range": [anchor, anchor + 5],
            "history_frames": arrays["canonical_frames"][[index[t] for t in times[:3]]],
            "next_frame": arrays["canonical_frames"][index[times[-1]]],
            "past_actions": arrays["executed_native_actions"][anchor - 10:anchor].reshape(2, 10),
            "next_action": arrays["executed_native_actions"][anchor:anchor + 5].reshape(1, 10)})
    return result, excluded


def expected_population(originals):
    """Fix the outcome-conditioned population from complete original records."""
    expected = {}
    for environment in ("pusht", "reacher"):
        methods = {}
        for mode in MODES:
            record = originals[environment, mode]
            if (record.get("status") != "complete" or record.get("environment") != environment
                    or record.get("model_mode") != mode or record.get("training_seed") != 0
                    or record.get("planning", {}).get("split") != "development"):
                raise ValueError("Original source is not the completed seed-zero development controller")
            rows = record["planning"]["records"]
            keys = [(row["trajectory_id"], row["observation_id"]) for row in rows]
            if len(rows) != 32 or len(set(keys)) != 32:
                raise ValueError("Original development population is incomplete or duplicated")
            methods[mode] = dict(zip(keys, rows))
        if methods[MODES[0]].keys() != methods[MODES[1]].keys():
            raise ValueError("Original controllers have different task populations")
        for key, ours in methods["factorized"].items():
            baseline = methods["framewise"][key]
            for row in (ours, baseline):
                if (type(row.get("success")) not in (int, bool) or row["success"] not in (0, 1)
                        or row.get("policy_eligible") != 1 - row.get("success_during_context", -1)):
                    raise ValueError("Original controller outcome or support eligibility is invalid")
            if ours["policy_eligible"] != baseline["policy_eligible"]:
                raise ValueError("Original controllers disagree about support eligibility")
            if ours["policy_eligible"] and ours["success"] != baseline["success"]:
                category = "ours_only" if ours["success"] else "baseline_only"
                for mode in MODES:
                    expected[environment, *key, mode] = {"category": category, "record": methods[mode][key]}
    return expected


@torch.inference_mode()
def measure_window(models, window, observation_id, device):
    history = pixels_to_tensor(window["history_frames"], observation_id)[None].to(device)
    canonical_history = pixels_to_tensor(window["history_frames"], 0)[None].to(device)
    next_image = pixels_to_tensor(window["next_frame"], 0)[None].to(device)
    past = torch.from_numpy(window["past_actions"].astype(np.float32)).to(device)[None]
    future = torch.from_numpy(window["next_action"].astype(np.float32)).to(device)[None]
    reference = models["factorized"]
    target = reference.encode_images(next_image, reference=True)
    canonical_support = reference.encode_images(canonical_history, reference=True)
    results = {}
    for mode in MODES:
        model = models[mode]
        features = model.encode_images(history)
        obs, dyn = model.infer_context(features, past)
        predicted = model.rollout_features(features, past, future, contexts=(obs, dyn))[:, 0]
        corrected = model.correct_observations(features, obs)
        mse = (predicted - target).square().mean()
        support_mse = (corrected - canonical_support).square().mean()
        if not torch.isfinite(predicted).all() or not torch.isfinite(mse) or not torch.isfinite(support_mse):
            raise ValueError("Nonfinite diagnostic prediction/calibration measurement")
        results[mode] = {"next_prediction_mse": float(mse), "support_calibration_mse": float(support_mse),
                         "predicted_next_feature": predicted[0].cpu().tolist()}
    return {"models": results, "canonical_next_feature": target[0].cpu().tolist(),
            "difference_mse_ours_minus_framewise": results["factorized"]["next_prediction_mse"] - results["framewise"]["next_prediction_mse"],
            "input_hashes": {"canonical_history": array_sha(window["history_frames"]),
                "canonical_next_frame": array_sha(window["next_frame"]),
                "float_shifted_history_tensor": array_sha(history.cpu().numpy()),
                "past_action_tensor": array_sha(past.cpu().numpy()),
                "next_action_tensor": array_sha(future.cpu().numpy())}}


def diagnostic(replay_path, device):
    replay_path = Path(replay_path)
    evidence = read(replay_path)
    if evidence.get("status") != "complete" or not evidence.get("records") or not evidence.get("sources"):
        raise ValueError("A completed, source-pinned replay diagnostic is required")
    sources = {str(replay_path.resolve()): sha(replay_path)}
    for name, expected in evidence["sources"].items():
        if sha(resolve(name)) != expected:
            raise ValueError(f"Replay source changed: {name}")
        sources[name] = expected
    originals = {}
    resolved_sources = {resolve(name).resolve(): value for name, value in evidence["sources"].items()}
    for environment in ("pusht", "reacher"):
        for mode in MODES:
            path = ROOT / "results/development_official_budget" / f"{environment}_{mode}_s0/planning_development.json"
            if path.resolve() not in resolved_sources:
                raise ValueError("Replay did not pin the original development planning source")
            originals[environment, mode] = read(path)
    expected_records = expected_population(originals)
    by_environment = {env: [] for env in ("pusht", "reacher")}
    seen = set()
    for record in evidence["records"]:
        if record.get("behavior_mode") not in MODES:
            continue
        key = record["environment"], record["trajectory_id"], record["observation_id"], record["behavior_mode"]
        validation = record.get("validation", {})
        if (key in seen or key[0] not in by_environment or record["observation_id"] != 1
                or record.get("category") not in {"ours_only", "baseline_only"}
                or validation.get("status") != "exact" or validation.get("claim_eligible") is not True
                or key not in expected_records or record.get("original_record") != expected_records[key]["record"]
                or record["category"] != expected_records[key]["category"]
                or record["seed"] != expected_records[key]["record"]["seed"]):
            raise ValueError("Replay trace is duplicated, ineligible or not an exact discordant development execution")
        seen.add(key)
        by_environment[key[0]].append(record)
    case_keys = {(a, b, c) for a, b, c, _ in seen}
    if len(case_keys) != 13 or len(seen) != 26 or seen != expected_records.keys():
        raise ValueError("Require both realized policy traces for all13 predeclared discordant cases")
    results, skipped, checkpoints = [], [], {}
    for environment, traces in by_environment.items():
        models, identity = load_models(environment, device)
        for mode in MODES:
            original = originals[environment, mode]
            source = identity[f"{environment}_{mode}_s0"]
            if (original.get("checkpoint_sha256") != source["hashes"]["checkpoint_sha256"]
                    or original.get("checkpoint_epoch") != source["selected_epoch"]):
                raise ValueError("Measured checkpoint differs from the recorded original controller")
        checkpoints.update(identity)
        for trace in traces:
            path = resolve(trace["npz"])
            if sha(path) != trace["npz_sha256"]:
                raise ValueError("Replay archive changed")
            sources[str(path)] = trace["npz_sha256"]
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in ("canonical_frames", "shifted_frames", "native_times", "executed_native_actions")}
            if sha(path) != trace["npz_sha256"]:
                raise ValueError("Replay archive changed during reading")
            selected, omitted = windows(arrays, trace["observation_id"])
            identity = {name: trace[name] for name in ("environment", "trajectory_id", "seed", "observation_id", "behavior_mode", "category")}
            for window in selected:
                results.append({**identity, **{name: window[name] for name in ("anchor_native_time", "target_native_time", "history_native_times", "future_action_native_range")},
                    **measure_window(models, window, trace["observation_id"], device),
                    "replay_archive": str(path), "replay_archive_sha256": trace["npz_sha256"]})
            skipped.extend({**identity, **row} for row in omitted)
        del models
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    for path in (Path(__file__), ROOT / "scripts/summarize_paired_planning.py", ROOT / "src/shiftwm/model.py",
                 ROOT / "src/shiftwm/checkpoint.py", ROOT / "src/shiftwm/data.py", ROOT / "src/shiftwm/evaluate.py"):
        sources[str(path)] = sha(path)
    return {"schema_version": 1, "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "All13 outcome-discordant seed-zero development tasks, both realized behavior traces; conditional one-block prediction only.",
        "source_hashes": sources, "checkpoint_sources": checkpoints, "records": results, "excluded": skipped,
        "protocol": {"anchors_native": list(ANCHORS), "support_frames": 3, "support_action_blocks": 2,
            "next_action_block_native_steps": 5, "latent_dimensions": 192, "precision": "float32_no_autocast_tf32_disabled",
            "shift_input": "pixels_to_tensor(canonical_uint8, appearance_id); no second quantization",
            "target": "same frozen reference encoder/projector on actual canonical next observation",
            "context": "recomputed from same observed history and same executed past actions separately for each checkpoint",
            "missing": "partial terminal blocks excluded; no invented padding or terminal prediction"},
        "coordinate_validation": "Exact tensor equality of both immutable reference/base encoders and projectors; identical action/pixel buffers, backbone config and source identities.",
        "counts": {"discordant_cases": len(case_keys), "behavior_traces": len(seen), "paired_transition_records": len(results),
                   "model_predictions": 2 * len(results), "excluded_anchor_records": len(skipped)},
        "execution": {"device": str(device), "job_id": os.environ.get("SLURM_JOB_ID"), "node": os.uname().nodename},
        "limits": ["Selected development traces, not an unbiased success rate or held-out benchmark.",
                   "Different behavior traces are not paired with each other: both models predict the identical transition within each row.",
                   "Prediction/feature-calibration errors do not establish causal context usefulness, CEM ranking or policy superiority.",
                   "No decoded visual prediction, new training, CEM search or counterfactual simulator action was generated."]}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--replay", type=Path, default=ROOT / "results/qualitative_diagnostics/replay_diagnostics.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/qualitative_diagnostics/prediction_diagnostics.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = diagnostic(args.replay, args.device)
    atomic_json(result, args.output)
    print(json.dumps({"status": result["status"], **result["counts"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
