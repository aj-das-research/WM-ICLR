#!/usr/bin/env python3
"""Stage an isolated historical upstream reproduction, with no installs or GPU work.

The stable-worldmodel revision is an API-compatible historical snapshot, not a
claim about the authors' exact training environment. All sampling comes from
the released HDF5 files and the unchanged LeWM evaluation configuration.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import subprocess
import tarfile
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
LEWM_REVISION = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
HISTORICAL_SWM_REVISION = "abdced49809d5eae38e24b27dc7b635c502c4812"
DATASETS = {"pusht": ("pusht/pusht_expert_train.h5", "pusht_expert_train.h5"),
            "reacher": ("reacher/reacher.h5", "dmc/reacher_random.h5")}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024): h.update(chunk)
    return h.hexdigest()


def archive_snapshot(repository, revision, destination):
    marker = destination / ".source_revision"
    if destination.exists():
        if not marker.is_file() or marker.read_text().strip() != revision:
            raise RuntimeError(f"Refusing existing unverified snapshot: {destination}")
        return
    destination.mkdir(parents=True)
    proc = subprocess.Popen(["git", "-C", str(repository), "archive", revision], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
            archive.extractall(destination, filter="data")
        if proc.wait() != 0: raise RuntimeError("git archive failed")
        marker.write_text(revision + "\n")
    finally:
        if proc.poll() is None: proc.kill()


def link_existing(source, destination):
    if not source.is_file(): raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve(): raise RuntimeError(f"Conflicting path: {destination}")
    else: destination.symlink_to(source.resolve())


def select_official_tasks(path, config):
    """Exactly reproduce eval.py's row sampling, including its last-row exclusion."""
    with h5py.File(path, "r") as f:
        col = "episode_idx" if "episode_idx" in f else "ep_idx"
        episodes, steps = f[col][:], f["step_idx"][:]
        ep_ids, inverse = np.unique(episodes, return_inverse=True)
        # Linear-time equivalent of eval.py's per-episode max(step_idx) loop.
        lengths = np.zeros(len(ep_ids), dtype=np.int64)
        np.maximum.at(lengths, inverse, steps + 1)
        offset = config["eval"]["goal_offset_steps"]
        valid = np.flatnonzero(steps <= lengths[inverse] - offset - 1)
        rng = np.random.default_rng(config["seed"])
        selected = np.sort(valid[rng.choice(len(valid) - 1, size=config["eval"]["num_eval"], replace=False)])
        tasks = []
        for row in selected:
            goal = int(row) + offset
            if episodes[goal] != episodes[row] or steps[goal] != steps[row] + offset:
                raise ValueError("Official sampled goal crosses an episode or nonconsecutive rows")
            # Read both real images; no simulator, model inference, or GPU call.
            frames = f["pixels"][[int(row), goal]]
            if frames.shape != (2, 224, 224, 3): raise ValueError("Unexpected official frame dimensions")
            tasks.append({"row": int(row), "episode": int(episodes[row]), "start_step": int(steps[row]),
                          "goal_step": int(steps[goal]), "goal_row": goal,
                          "start_goal_pixels_sha256": hashlib.sha256(frames.tobytes()).hexdigest()})
        return {"file": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                "columns": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k,v in f.items()},
                "physical_episodes": len(ep_ids), "valid_start_rows": len(valid),
                "selected_tasks": tasks, "selected_unique_episodes": len({t['episode'] for t in tasks}),
                "sampling_note": "Exact seed-42 upstream choice excludes the last valid row, as eval.py does; not corrected here.",
                "selection_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "artifacts/upstream_planning_reproduction")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    archive_snapshot(ROOT / "external/le-wm", LEWM_REVISION, bundle / "le-wm")
    archive_snapshot(ROOT / "external/stable-worldmodel", HISTORICAL_SWM_REVISION, bundle / "stable-worldmodel")
    source_manifest = json.loads((ROOT / "references/world_artifact_sources.json").read_text())
    output = {"status": "inputs_staged", "gpu_evaluation_started": False,
              "runtime_validation_is_separate": "reports/evidence/upstream_runtime_validation.json",
              "source_revisions": {"lewm": LEWM_REVISION, "historical_stable_worldmodel": HISTORICAL_SWM_REVISION},
              "historical_snapshot_is_author_training_pin": False, "environments": {}}
    cache = bundle / "cache"
    for env, (source, alias) in DATASETS.items():
        source = ROOT / "data/upstream" / source
        link_existing(source, cache / "datasets" / alias)
        artifact = next(x for x in source_manifest["models"] if x["environment"] == env)
        hashes = {}
        for item in artifact["files"]:
            model_path = ROOT / item["path"]
            actual = digest(model_path)
            if actual != item["sha256"]: raise ValueError(f"Released model hash mismatch: {model_path}")
            link_existing(model_path, cache / "checkpoints" / env / "lewm" / item["name"])
            hashes[item["name"]] = actual
        with initialize_config_dir(version_base=None, config_dir=str(bundle / "le-wm/config/eval")):
            cfg = compose(config_name=env, overrides=[f"policy={env}/lewm", f"cache_dir={cache}",
                                                     "world.max_episode_steps=100"])
        config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
        (bundle / "configs").mkdir(exist_ok=True)
        OmegaConf.save(cfg, bundle / "configs" / f"{env}.yaml", resolve=True)
        output["environments"][env] = {"config": config, "weights": hashes,
                                     "official_data": select_official_tasks(source, config)}
    output["current_environment_imports"] = {
        name: bool(importlib.util.find_spec(name)) for name in
        ("stable_pretraining", "sklearn", "loguru", "lancedb", "lance", "pyarrow", "decord", "imageio_ffmpeg")}
    output["expected_maximum_work_per_environment"] = {
        "tasks": 50, "native_steps_per_task": 50, "replans_per_task": 2,
        "task_replans": 100, "candidate_sequences": 900000, "latent_transitions": 4500000,
        "note": "Upper bound before early success; not measured time. CEM batches one task at a time."}
    evidence = ROOT / "reports/evidence/upstream_planning_preflight.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"bundle": str(bundle), "evidence": str(evidence),
                      "tasks": {k: len(v['official_data']['selected_tasks']) for k,v in output['environments'].items()},
                      "current_environment_imports": output['current_environment_imports']}))


if __name__ == "__main__": main()
