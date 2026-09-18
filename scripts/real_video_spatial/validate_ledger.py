"""Independent fail-closed validation of complete spatial evaluation ledgers.

No video/features are decoded here. The caller must first run the authoritative
full-30 checkpoint validator, then pass the loaded selected checkpoint state.
Per-window errors originate in FP32; their serialized aggregation is FP64.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

METRICS = ("native_mse", "native_persistence_mse", "original_2x2_mse", "original_2x2_persistence_mse")
MODES = ("autoregressive", "anchored_additive", "transport", "context_off", "action_free")
TARGET_COORDINATES = "exact original cached float32 features; not recomputed pooled targets"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def at_root(root, path):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def vector(value, description):
    if (not isinstance(value, list) or len(value) != 10
            or any(type(item) not in (int, float) for item in value)):
        raise ValueError("Require ten numeric horizon errors: " + description)
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("Nonfinite or negative horizon errors: " + description)
    return result


def close(actual, expected, description):
    if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
        raise ValueError("Recomputed ledger arithmetic differs: " + description)


def expected_population(cache_manifest, original_manifest):
    episodes, windows, sessions, ids = {}, set(), {}, set()
    original = {row["episode_id"]: row for row in original_manifest["episodes"] if row["split"] == "val"}
    for row in cache_manifest["episodes"]:
        eid, session, split = row["episode_id"], row["session_id"], row["split"]
        if (split not in ("train", "val") or not session or eid in ids
                or set(row["cameras"]) != {"exterior_image_1_left"}):
            raise ValueError("Invalid train/validation-only spatial population")
        ids.add(eid)
        if session in sessions and sessions[session] != split:
            raise ValueError("Recording session crosses splits")
        sessions[session] = split
        if split != "val":
            continue
        count = row["cameras"]["exterior_image_1_left"]["frames"]
        if type(count) is not int or count < 1:
            raise ValueError("Invalid stored-frame count")
        old = original.get(eid)
        if (old is None or old["session_id"] != session
                or old["cameras"]["exterior_image_1_left"]["frames"] != count):
            raise ValueError("Original target population differs")
        starts = tuple(range(0, count - 13 + 1, 5))
        if starts:
            episodes[eid] = {"session_id": session, "starts": starts}
            windows.update((eid, session, start) for start in starts)
    spatial_val = {row["episode_id"] for row in cache_manifest["episodes"] if row["split"] == "val"}
    if spatial_val != set(original) or not episodes or set(sessions.values()) != {"train", "val"}:
        raise ValueError("Incomplete original validation population")
    return episodes, windows


def validate_ledger(result, config, root, selected_state):
    """Verify all identities/populations and return the recomputed summary."""
    root = Path(root)
    if (result.get("status") != "complete" or result.get("scope") != "original_validation_development_only"
            or result.get("aggregation") != "mean windows within episode; equal episodes"
            or result.get("mode") != config["mode"] or config["mode"] not in MODES
            or type(result.get("seed")) is not int or result["seed"] != config["seed"] or result["seed"] not in (0,1,2)
            or type(result.get("selected_epoch")) is not int or not 1 <= result["selected_epoch"] <= 30
            or result["selected_epoch"] != selected_state["epoch"]
            or result.get("original_target_coordinates") != TARGET_COORDINATES):
        raise ValueError("Evaluation header/scope/selection differs")
    package = selected_state["config"]
    if package["model_config"] != {**config["model_config"], "mode": config["mode"]}:
        raise ValueError("Selected checkpoint architecture differs")
    counts = package["metadata"]["parameter_counts"]
    if (result.get("parameter_counts") != counts or set(counts) != {"total", "trainable"}
            or any(type(value) is not int for value in counts.values())
            or not 0 < counts["trainable"] <= counts["total"]):
        raise ValueError("Selected checkpoint parameter counts differ")
    cache = at_root(root, config["cache_root"])
    original = at_root(root, config["original_cache"])
    bindings = {"checkpoint_sha256": at_root(root, config["output_dir"]) / "best/model.pt",
                "cache_manifest_sha256": cache / "manifest.json",
                "original_cache_manifest_sha256": original / "manifest.json",
                "original_normalization_sha256": original / "training_statistics.json",
                "source_sha256": root / "scripts/real_video_spatial/evaluate.py"}
    for key, path in bindings.items():
        if result.get(key) != sha(path):
            raise ValueError("Evaluation identity differs: " + key)
    spatial_manifest = json.loads((cache / "manifest.json").read_text())
    old_manifest = json.loads((original / "manifest.json").read_text())
    if (spatial_manifest.get("status") != "complete" or spatial_manifest.get("feature_dim") != 6144
            or spatial_manifest.get("action_dim") != 35
            or spatial_manifest["identity"].get("original_2x2_cache_manifest_sha256") != result["original_cache_manifest_sha256"]):
        raise ValueError("Spatial cache identity differs")
    expected_episodes, expected_windows = expected_population(spatial_manifest, old_manifest)
    seen, by_episode = set(), defaultdict(list)
    for row in result.get("windows", []):
        if type(row.get("window_start")) is not int:
            raise ValueError("Noninteger window index")
        key = (row["episode_id"], row["session_id"], row["window_start"])
        if key not in expected_windows or key in seen:
            raise ValueError("Unexpected or duplicate validation window")
        seen.add(key)
        by_episode[row["episode_id"]].append(np.stack([vector(row.get(metric), metric) for metric in METRICS]))
    if seen != expected_windows:
        raise ValueError("Missing validation windows")
    recomputed = {eid: np.mean(values, axis=0, dtype=np.float64) for eid, values in by_episode.items()}
    seen_episodes = set()
    for row in result.get("episodes", []):
        eid = row["episode_id"]
        if eid not in expected_episodes or eid in seen_episodes:
            raise ValueError("Unexpected or duplicate episode aggregate")
        seen_episodes.add(eid)
        expected = expected_episodes[eid]
        if (row["session_id"] != expected["session_id"] or type(row.get("windows")) is not int
                or row["windows"] != len(expected["starts"])):
            raise ValueError("Episode session/window count differs")
        for index, metric in enumerate(METRICS):
            close(vector(row.get(metric), metric), recomputed[eid][index], eid + ":" + metric)
    if seen_episodes != set(expected_episodes) or set(result.get("summary", {})) != set(METRICS):
        raise ValueError("Missing episodes or summary metrics")
    matrix = np.stack([recomputed[eid] for eid in sorted(recomputed)])
    summary = {metric: matrix[:, index].mean(0).tolist() for index, metric in enumerate(METRICS)}
    for metric in METRICS:
        close(vector(result["summary"][metric], metric), summary[metric], "summary:" + metric)
    return summary


def paired_intervals(results, first_mode="transport", second_mode="autoregressive", draws=10000, bootstrap_seed=20260919):
    """Exploratory session×seed paired intervals at every recorded horizon.

    Call only after validate_ledger passes for every supplied result. All
    per-window keys, episodes and sessions must match across both three-seed arms.
    """
    reference_windows, reference_episodes, matrices = None, None, []
    for mode in (first_mode, second_mode):
        selected = sorted([row for row in results if row["mode"] == mode], key=lambda row: row["seed"])
        if len(selected) != 3 or [row["seed"] for row in selected] != [0,1,2]:
            raise ValueError("Paired comparison requires three unique seeds per arm")
        matrix = []
        for row in selected:
            windows = sorted((w["episode_id"], w["session_id"], w["window_start"]) for w in row["windows"])
            episodes = sorted(row["episodes"], key=lambda item: item["episode_id"])
            population = [(e["episode_id"], e["session_id"]) for e in episodes]
            if reference_windows is None:
                reference_windows, reference_episodes = windows, population
            if windows != reference_windows or population != reference_episodes:
                raise ValueError("Paired populations differ")
            matrix.append(np.stack([[vector(e[metric], metric) for metric in METRICS] for e in episodes]))
        matrices.append(np.stack(matrix))
    first, second = matrices
    difference = first - second
    sessions = np.array([session for _, session in reference_episodes])
    clusters = [np.flatnonzero(sessions == session) for session in sorted(set(sessions))]
    if len(clusters) < 2 or type(draws) is not int or draws < 1:
        raise ValueError("Need at least two sessions and positive draw count")
    rng = np.random.default_rng(bootstrap_seed)
    distribution = np.empty((draws, len(METRICS), 10))
    for i in range(draws):
        seeds = rng.integers(0,3,size=3)
        episodes = np.concatenate([clusters[j] for j in rng.integers(0,len(clusters),size=len(clusters))])
        distribution[i] = difference[seeds][:, episodes].mean((0,1))
    intervals = np.quantile(distribution,[.025,.975],axis=0)
    return {"first_mode":first_mode,"second_mode":second_mode,"scope":"exploratory_original_validation_unadjusted",
            "episode_count":len(reference_episodes),"session_count":len(clusters),"training_seed_count":3,
            "draws":draws,"bootstrap_seed":bootstrap_seed,"direction":"negative first-minus-second favors first",
            "metrics":{metric:[{"horizon":h+1,"first_mean":float(first[:,:,m,h].mean()),
                "second_mean":float(second[:,:,m,h].mean()),"mean_difference":float(difference[:,:,m,h].mean()),
                "ci95":intervals[:,m,h].tolist()} for h in range(10)] for m,metric in enumerate(METRICS)}}
