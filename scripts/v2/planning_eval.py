#!/usr/bin/env python
"""Official-protocol MPC planning evaluation (LeWM protocol) with a pluggable predictor.

Reproduces ``external/le-wm/eval.py`` (task sampling, action StandardScaler,
ImageNet transform, dataset-driven ``World.evaluate``, CEM 300x30x30, horizon 5,
receding horizon 5, action block 5, 50 episodes, goal +25, budget 50, seed 42)
on the pinned stable-worldmodel stack, but lets any predictor implementing
``shiftwm.v2.planning.interface.PlanningPredictor`` be planned with.

Examples::

    python scripts/v2/planning_eval.py --env pusht                     # released LeWM
    python scripts/v2/planning_eval.py --env reacher --num-eval 5 --tag smoke
    python scripts/v2/planning_eval.py --env pusht --predictor mypkg.mod:build \
        --ckpt runs/x/best.pt --model shiftwm_v2_s --seed 43

Output: results/v2/planning/<env>/<model>/<seed>[_<tag>].json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # read-only access to shared 46-99 GB files

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("STABLEWM_HOME", str(ROOT / ".cache/stablewm_home"))
# hdf5plugin (Blosc filter 32001 on the released HDF5 pixels) lives in a gitignored sidecar dir:
#   uv pip install --python .venv/bin/python --target .cache/v2_planning_pydeps --no-deps hdf5plugin
if (ROOT / ".cache/v2_planning_pydeps").is_dir():
    sys.path.append(str(ROOT / ".cache/v2_planning_pydeps"))

from shiftwm.v2.planning.compat import StandardScaler, install_import_shims  # noqa: E402

install_import_shims()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision.transforms import v2 as transforms  # noqa: E402

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.data.formats.hdf5 import HDF5Dataset  # noqa: E402
from stable_worldmodel.planning.solver import CEMSolver  # noqa: E402

from shiftwm.v2.planning.compat import IMAGENET_STATS  # noqa: E402
from shiftwm.v2.planning.interface import build_cost  # noqa: E402
from shiftwm.v2.planning.protocol import get_protocol  # noqa: E402

DATA = ROOT / "data/upstream"


# --------------------------------------------------------------------------- helpers
def img_transform(size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**IMAGENET_STATS),
            transforms.Resize(size=size),
        ]
    )


def git_rev(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def load_factory(spec: str):
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn or "build")


class TimedCEM(CEMSolver):
    """Upstream CEMSolver, unchanged, with wall-clock timing of each solve() call."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.solve_log: list[dict] = []
        self.clock = None  # set to a callable returning the current env step

    def solve(self, info_dict, init_action=None):
        n = len(next(iter(info_dict.values())))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = super().solve(info_dict, init_action=init_action)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.solve_log.append(
            {"env_step": self.clock() if self.clock else None, "n_envs": n, "seconds": time.perf_counter() - t0}
        )
        return out


GOAL_ERROR_DEF = {
    "pusht": "L2 distance (px) between current and goal (agent xy, block xy), the position term of the success test (<20)",
    "tworoom": "L2 distance (px) between agent and target, the success test (<16)",
    "reacher": "max over joints of |qpos - target_qpos| (rad), the success test (<0.05)",
}


def goal_error(env_name: str, u) -> float:
    """Physical distance to the goal in the env's own success-test units (u = unwrapped env)."""
    if env_name == "pusht":
        s, g = np.asarray(u._get_obs(), np.float64), np.asarray(u.goal_state, np.float64)
        return float(np.linalg.norm(g[:4] - s[:4]))
    if env_name == "tworoom":
        return float(torch.norm(u.agent_position - u.target_position))
    if env_name == "reacher":
        return float(np.max(np.abs(u.env.physics.data.qpos - u.env.task.target_qpos)))
    raise KeyError(env_name)


def make_recording_world_cls(env_name: str | None = None):
    class RecordingWorld(swm.World):
        """swm.World that additionally records the first success step and the last physical goal error of every env."""

        def _run(self, *a, on_step=None, **kw):
            self.step_count = 0
            self.first_success = np.full(self.num_envs, -1, dtype=int)
            self.last_goal_error = np.full(self.num_envs, np.nan)
            self.goal_error_failed = None

            def wrapped(world, mask):
                if on_step is not None:
                    on_step(world, mask)
                self.step_count += 1
                newly = (world.terminateds.astype(bool)) & (self.first_success < 0)
                self.first_success[newly] = self.step_count
                if env_name is not None and self.goal_error_failed is None:
                    try:  # envs frozen after success keep their terminal error
                        for i in np.where(np.asarray(mask, bool))[0]:
                            self.last_goal_error[i] = goal_error(env_name, self.envs.envs[i].unwrapped)
                    except Exception as e:  # never break the evaluation over a diagnostic
                        self.goal_error_failed = repr(e)

            return super()._run(*a, on_step=wrapped, **kw)

    return RecordingWorld


def select_tasks(dataset, n, goal_offset, seed, selection="lewm"):
    """Identical to external/le-wm/eval.py (selection='lewm') or swm eval_wm.py ('swm')."""
    names = set(dataset.column_names)
    col = "episode_idx" if "episode_idx" in names else "ep_idx"
    ep_col = dataset.get_col_data(col)
    step_col = dataset.get_col_data("step_idx")
    ep_indices = np.unique(ep_col)
    # episode length = max(step_idx)+1 per episode (vectorised version of get_episodes_length)
    max_step = np.full(int(ep_indices.max()) + 1, -1, dtype=np.int64)
    np.maximum.at(max_step, ep_col.astype(np.int64), step_col.astype(np.int64))
    max_start_per_row = (max_step + 1 - goal_offset - 1)[ep_col.astype(np.int64)]
    valid_indices = np.nonzero(step_col <= max_start_per_row)[0]
    g = np.random.default_rng(seed)
    pool = len(valid_indices) - 1 if selection == "lewm" else len(valid_indices)
    rows = np.sort(valid_indices[g.choice(pool, size=n, replace=False)])
    return rows, ep_col[rows], step_col[rows], int(len(valid_indices)), col


# --------------------------------------------------------------------------- adapter check
@torch.no_grad()
def check_adapter(predictor, device, n_samples=64, horizon=5, act_dim=10, seed=0):
    """Compare the pluggable-interface cost with native LeWM rollout + GoalMSE."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    model = predictor.model
    res = {}
    for H in (1, 3):
        pix = torch.randn(1, 1, H, 3, 224, 224, generator=g).to(device)
        goal = torch.randn(1, 1, 1, 3, 224, 224, generator=g).to(device)
        acts = torch.randn(1, n_samples, horizon, act_dim, generator=g).to(device)
        hist = torch.randn(1, 1, H - 1, act_dim, generator=g).to(device)

        def info():
            d = {"pixels": pix.expand(1, n_samples, *pix.shape[2:]), "goal": goal.expand(1, n_samples, *goal.shape[2:])}
            d["action"] = torch.zeros(1, n_samples, 1, act_dim, device=device)
            if H > 1:
                d["action_history"] = hist.expand(1, n_samples, *hist.shape[2:])
            return d

        native = swm.planning.ShootingCostEvaluator(model, swm.planning.GoalMSE()).get_cost(info(), acts)
        adapt = build_cost(predictor).get_cost(info(), acts)
        res[f"H{H}"] = {
            "max_abs_diff": float((native - adapt).abs().max()),
            "max_rel_diff": float(((native - adapt).abs() / native.abs().clamp_min(1e-6)).max()),
            "argmin_equal": bool(native.argmin() == adapt.argmin()),
        }
    return res


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True, choices=["pusht", "tworoom", "reacher"])
    ap.add_argument("--predictor", default="shiftwm.v2.planning.lewm:build",
                    help="module:factory returning (PlanningPredictor, meta dict)")
    ap.add_argument("--ckpt", default=None, help="checkpoint passed to the factory")
    ap.add_argument("--model", default="lewm", help="name used in the results path")
    ap.add_argument("--backend", default="adapter", choices=["adapter", "native"],
                    help="adapter = pluggable interface; native = swm ShootingCostEvaluator(model, GoalMSE) "
                         "(only for predictors exposing .model with swm Dynamics API, e.g. LeWM)")
    ap.add_argument("--variant", default="released", choices=["released", "paper"])
    ap.add_argument("--seed", type=int, default=42, help="task-selection + CEM seed (official: 42)")
    ap.add_argument("--selection", default="lewm", choices=["lewm", "swm"])
    ap.add_argument("--num-eval", type=int, default=None)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--receding-horizon", type=int, default=None)
    ap.add_argument("--history-len", type=int, default=None)
    ap.add_argument("--eval-budget", type=int, default=None)
    ap.add_argument("--goal-offset", type=int, default=None)
    ap.add_argument("--dataset", default=None, help="override HDF5 path")
    ap.add_argument("--video", action="store_true", help="write per-env panel videos next to the JSON")
    ap.add_argument("--check-adapter", action="store_true")
    ap.add_argument("--policy", default="cem", choices=["cem", "random"],
                    help="cem = world-model CEM planning; random = uniform random actions from the env action "
                         "space for the same budget and tasks (floor; no predictor is loaded)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proto = get_protocol(args.env, args.variant)
    for k, sect, key in [
        ("num_eval", "eval", "num_eval"), ("eval_budget", "eval", "eval_budget"),
        ("goal_offset", "eval", "goal_offset_steps"), ("num_samples", "cem", "num_samples"),
        ("n_steps", "cem", "n_steps"), ("topk", "cem", "topk"), ("horizon", "plan", "horizon"),
        ("receding_horizon", "plan", "receding_horizon"), ("history_len", "plan", "history_len"),
    ]:
        v = getattr(args, k)
        if v is not None:
            proto[sect][key] = v
    proto["eval"]["seed"] = args.seed
    overrides = {k: getattr(args, k) for k in ["num_eval", "num_samples", "n_steps", "topk", "horizon",
                                                "receding_horizon", "history_len", "eval_budget", "goal_offset"]
                 if getattr(args, k) is not None}
    E, P, C = proto["eval"], proto["plan"], proto["cem"]
    assert P["horizon"] * P["action_block"] <= E["eval_budget"]

    out = Path(args.out) if args.out else (
        ROOT / "results/v2/planning" / args.env / args.model / f"{args.seed}{'_' + args.tag if args.tag else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- dataset + task selection (le-wm eval.py) ----
    ds_path = Path(args.dataset) if args.dataset else DATA / proto["dataset"]
    t0 = time.time()
    dataset = HDF5Dataset(path=ds_path, keys_to_cache=list(proto["keys_to_cache"]))
    process = {}
    for col in proto["keys_to_cache"]:
        if col == "pixels" or col not in dataset.column_names:
            continue
        data = dataset.get_col_data(col)
        data = data[~np.isnan(data).any(axis=1)]
        process[col] = StandardScaler().fit(data)
        if col != "action":
            process[f"goal_{col}"] = process[col]
    rows, eval_eps, eval_starts, n_valid, ep_col = select_tasks(
        dataset, E["num_eval"], E["goal_offset_steps"], args.seed, args.selection)
    t_data = time.time() - t0
    print(f"[data] {ds_path} valid_starts={n_valid} rows={rows.tolist()} ({t_data:.1f}s)", flush=True)

    callables = proto["callables"]
    if "callables_alt" in proto:
        needed = callables[0]["args"]["state"]["value"]
        if needed not in dataset.column_names:
            callables = proto["callables_alt"]
    for spec in callables:
        for a in spec["args"].values():
            key = a["value"][len("goal_"):] if a["value"].startswith("goal_") else a["value"]
            assert key in dataset.column_names, f"dataset lacks column {key!r} for {spec['method']}"

    # ---- predictor + planner ----
    if args.policy == "random":
        predictor, meta, adapter_check = None, {"policy": "uniform random over env action_space"}, None
        solver = TimedCEM.__new__(TimedCEM)  # only its (empty) solve_log is used below
        solver.solve_log, solver.clock = [], None
        policy = swm.policy.RandomPolicy(seed=args.seed)
    else:
        factory = load_factory(args.predictor)
        predictor, meta = factory(env=args.env, ckpt=args.ckpt, device=device)
        predictor = predictor.to(device).eval().requires_grad_(False)
        adapter_check = check_adapter(predictor, device) if args.check_adapter else None
        if adapter_check:
            print("[adapter-check]", adapter_check, flush=True)
        if args.backend == "native":
            cost = swm.planning.ShootingCostEvaluator(predictor.model, swm.planning.GoalMSE())
        else:
            cost = build_cost(predictor)
        solver = TimedCEM(cost=cost, device=device, seed=args.seed, **C)
        plan_cfg = swm.PlanConfig(**P)
        tf = img_transform(E["img_size"])
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=plan_cfg, process=process, transform={"pixels": tf, "goal": tf})

    World = make_recording_world_cls(args.env)
    world = World(**proto["world"], num_envs=E["num_eval"], max_episode_steps=2 * E["eval_budget"],
                  image_shape=(224, 224))
    world.set_policy(policy)
    solver.clock = lambda: getattr(world, "step_count", 0)

    video_dir = out.with_suffix("") if args.video else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    metrics = world.evaluate(
        dataset=dataset, start_steps=eval_starts.tolist(), goal_offset=E["goal_offset_steps"],
        eval_budget=E["eval_budget"], episodes_idx=eval_eps.tolist(), callables=callables, video=video_dir)
    t_eval = time.time() - t0

    succ = np.asarray(metrics["episode_successes"]).astype(bool)
    first = world.first_success
    # attribute each solve's wall time evenly over the envs it planned for
    n = E["num_eval"]
    plan_time = np.zeros(n)
    n_replans = np.zeros(n, dtype=int)
    for s in solver.solve_log:
        if s["n_envs"] == n:
            part = np.ones(n, bool)
        else:  # only still-running envs replanned
            part = (first < 0) | (first > s["env_step"])
            if part.sum() != s["n_envs"]:
                part = np.ones(n, bool)
        plan_time[part] += s["seconds"] / max(s["n_envs"], 1)
        n_replans[part] += 1
    episodes = [
        {"i": i, "row": int(rows[i]), "episode_idx": int(eval_eps[i]), "start_step": int(eval_starts[i]),
         "success": bool(succ[i]), "success_step": int(first[i]) if first[i] > 0 else None,
         "n_replans": int(n_replans[i]), "plan_time_s": float(plan_time[i]),
         "terminal_goal_error": (None if np.isnan(world.last_goal_error[i]) else float(world.last_goal_error[i]))}
        for i in range(n)
    ]
    per_env_solve = [s["seconds"] / s["n_envs"] for s in solver.solve_log]
    result = {
        "env": args.env,
        "model": args.model,
        "seed": args.seed,
        "tag": args.tag,
        "success_rate": float(succ.mean() * 100.0),
        "n_success": int(succ.sum()),
        "n_episodes": n,
        "swm_success_rate": float(metrics["success_rate"]),
        "timing": {
            "eval_wall_s": t_eval,
            "data_load_s": t_data,
            "total_wall_s": time.time() - t_start,
            "n_solve_calls": len(solver.solve_log),
            "solve_calls": solver.solve_log,
            "sec_per_plan_per_env_mean": float(np.mean(per_env_solve)) if per_env_solve else None,
            "plan_time_per_episode_mean_s": float(plan_time.mean()),
            "eval_wall_per_episode_s": t_eval / n,
            "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
            "note": "CEM solves each env sequentially (batch_size=1); per-plan time = solve wall / n_envs. "
                    "eval_wall includes simulation/rendering.",
        },
        "protocol": {**{k: v for k, v in proto.items() if k != "callables_alt"}, "callables": callables,
                     "selection": args.selection, "backend": args.backend, "overrides": overrides,
                     "is_official": not overrides and args.variant == "released" and args.selection == "lewm"},
        "tasks": {"dataset": str(ds_path), "episode_col": ep_col, "n_valid_starts": n_valid,
                  "rows_sha256": hashlib.sha256(np.asarray(rows, np.int64).tobytes()).hexdigest()},
        "predictor": {"factory": args.predictor, "ckpt": args.ckpt, "name": getattr(predictor, "name", None),
                      "meta": meta},
        "adapter_check": adapter_check,
        "policy": args.policy,
        "goal_error": {"definition": GOAL_ERROR_DEF.get(args.env), "failed": world.goal_error_failed,
                       "note": "measured after the last executed step; an env that succeeds is frozen at that step"},
        "provenance": {
            "le_wm": git_rev(ROOT / "external/le-wm"),
            "stable_worldmodel": git_rev(ROOT / "external/stable-worldmodel"),
            "repo": git_rev(ROOT),
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "argv": sys.argv,
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
        "episodes": episodes,
    }
    out.write_text(json.dumps(result, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"[result] {args.env}/{args.model}/seed{args.seed}: success {result['n_success']}/{n} "
          f"= {result['success_rate']:.1f}%  eval {t_eval:.1f}s  "
          f"plan/env {result['timing']['sec_per_plan_per_env_mean']}s -> {out}", flush=True)
    return result


if __name__ == "__main__":
    main()
