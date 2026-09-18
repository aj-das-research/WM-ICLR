# Does evaluation goal metadata change the rendered observations?

Date: 2026-09-18. This bounded audit inspected pinned source and restored three saved states from one existing development episode per environment. No training, learned-model inference, candidate search, simulator stepping, or benchmark evaluation occurred. No runtime source was modified.

**Finding: setting the task goal does not change the rendered goal marker/background in either current environment.** The source separates task-goal metadata from rendering, and a controlled before/after rendering check changed zero channel values at all six sampled states. This does not explain the poor PushT planning result.

## Exact source evidence

### Reacher

- Our collection and evaluation both construct `ReacherDMControlWrapper(task="qpos_match", seed=seed)` and reset the same density variations: `src/shiftwm/generate.py:56–63`.
- Our `set_task_goal` calls `env.set_target_qpos(goal.copy())`: `src/shiftwm/generate.py:104–108`.
- The upstream setter only assigns `self.env.task.target_qpos = np.asarray(...)`: `external/stable-worldmodel/stable_worldmodel/envs/dmcontrol/reacher.py:166–176`. It does not move the MuJoCo target geom, change colors, recompile the model, or alter rendering options.
- The qpos-match task initializes that attribute to `None` and reads it only to compute termination from joint-angle error: `external/stable-worldmodel/stable_worldmodel/envs/dmcontrol/custom_tasks/reacher.py:20–32`. Thus collection with an unset qpos target and evaluation with a target have different termination semantics, not a changed target marker.
- The render-target variation defaults to0: `external/stable-worldmodel/stable_worldmodel/envs/dmcontrol/reacher.py:112–114`. When0, material alpha is set to0: `reacher.py:294–307`. The ordinary MuJoCo finger-to-target geom is separate from the qpos task target and is invisible under the current configuration.

### PushT

- Our setter calls `_set_goal_state`: `src/shiftwm/generate.py:104–106`.
- The upstream setter only stores `self.goal_state`: `external/stable-worldmodel/stable_worldmodel/envs/pusht/env.py:546–547`.
- Rendering uses a different field, `self.goal_pose`: `env.py:439–445`. `_setup` sets this from the goal-position/angle variation values: `env.py:594–599`. The defaults are position[256,256] and angleπ/4 (`env.py:177–189`), and default sampled variations are only agent start, block start, and block angle (`env.py:14–18`).
- Consequently the visible background target remains fixed when the task switches to a recorded future state. Both our goal image and current observations contain the same fixed marker; it is not moved to the evaluated future block position. This is inherited upstream behavior, not a new goal-dependent photometric intervention.

## Relation to the pinned LeWM evaluation

`external/le-wm/config/eval/reacher.yaml:6–10,32–44` uses the same qpos-match environment and applies `set_state` plus `set_target_qpos(goal_qpos)`. PushT's config applies `_set_state` plus `_set_goal_state` (`external/le-wm/config/eval/pusht.yaml:33–43`). LeWM passes those callables to `world.evaluate` (`external/le-wm/eval.py:143–151`).

The pinned world implementation extracts a future dataset image as the desired `goal` (`external/stable-worldmodel/stable_worldmodel/world/world.py:643–683`) and applies state/goal setup calls (`world.py:556–579`). Our evaluator instead reconstructs the goal image in a separate same-seed/same-dynamics environment at the recorded goal state (`src/shiftwm/evaluate.py:228–240`).

Therefore the **task-goal setter/marker semantics match upstream**, while using a newly rendered goal instead of the stored future image is an explicit implementation difference. Changing goal metadata alone does not produce a visual difference in the current environments. A rendering/backend discrepancy remains logically separate and should not be hidden by describing every fresh render as necessarily pixel-identical to the dataset.

## Bounded rendering check

Evidence: `reports/evidence/goal_metadata_render_audit.json`.

For each environment, the check reads `development-s2031000-d1` and restores recorded states at frames0,2,7 in fresh environments using the manifest's224px resolution. It renders once with collection/default goal metadata, assigns recorded frame7 as the evaluation task goal, and renders again without stepping physics. It also compares both renders against the corresponding saved canonical image.

The Reacher check explicitly used CPU software rendering: `CUDA_VISIBLE_DEVICES=''`, `LIBGL_ALWAYS_SOFTWARE=1`, `EGL_PLATFORM=surfaceless`, `MUJOCO_GL=egl`; reported OpenGL renderer was `llvmpipe (LLVM 20.1.2, 256 bits)`. PushT rendering is the pygame/Pymunk path. This is a correctness fixture, not a latency measurement.

| Environment/frame | Channel values changed by goal setter | Spatial pixels differing: archived vs fresh | Maximum archived/fresh channel difference |
|---|---:|---:|---:|
| PushT0 | 0 | 0 | 0 |
| PushT2 | 0 | 1 | 7 |
| PushT7 | 0 | 0 | 0 |
| Reacher0 | 0 | 45,330 | 53 |
| Reacher2 | 0 | 45,379 | 43 |
| Reacher7 | 0 | 45,226 | 72 |

All archived/fresh differences above already exist **before** the goal setter. The single changed PushT pixel is not caused by goal metadata; its exact restoration/rasterization cause was not investigated here. Reacher's larger differences arise in a fresh software-rendered comparison with archived collection data. The collection renderer was not recorded in this evidence, so a backend cause is plausible but not established. These values do not demonstrate a mismatch between the actual live GPU collection/evaluation paths.

## Actionable limit

There is no evidence here for a goal-marker bug, and no goal-setting change is warranted. Before interpreting a fresh-render/cached-feature discrepancy as model failure, record `GL_RENDERER` under the actual allocated collection/evaluation environment and compare archived/fresh frames under that same backend. The goal-calibration diagnostic's per-record fresh goal/support hashes can support this check. Do not replace goals or change renderer settings during the live experiment based on a software-versus-archive comparison alone.
