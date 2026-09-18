# Development goal-calibration diagnostic

This diagnostic addresses a specific uncertainty from `reports/planning_correctness_audit.md`: a low error for canonical next-state predictions does not establish that the corrected goal embedding used by planning is calibrated to the same coordinates. Existing alignment and prediction losses average different frames/conditions and cannot answer the actual per-goal question.

Implementation: `scripts/diagnose_goal_calibration.py`. The live `src/shiftwm/model.py`, `evaluate.py`, and `generate.py` are unchanged. No training, candidate search, or future simulator-action execution occurs in this script. Parent orchestration owns GPU execution; no research diagnostic was run while implementing it.

## Fixed protocol

- Only the development split is permitted. Selection mirrors `evaluate_planning`: sort manifest episodes by `(seed,dynamics_id)`, retain development episodes and development combinations, take the first32 episodes per selected dynamics. The current split yields32 unique seeds in appearance1/dynamics1. Selection occurs before support-success status is known. The32 selected keys remain in the output.
- Only completed frozen/single/factorized/framewise `best` packages are accepted. Trained runs must have completed their configured number of epochs. Frozen packages must have explicit unchanged-upstream export metadata. The external config must match the config embedded in the weights; checkpoint family must match the dataset environment.
- The goal is recorded frame7: five grouped transitions after the end of three-frame observed support. Goal/task/environment initialization reproduces evaluator lines224–268, including a separate goal-rendering environment, chronological five-action blocks, same appearance, and immediate termination on a support success.
- The context receives only the observed shifted frames0–2 and actually executed action blocks0–1. The immutable canonical encoder sees corresponding canonical renders solely to construct diagnostic targets. Recorded future actions2–6 are used only after the context has been inferred; exactly those five grouped blocks lead to the stored goal. No future recorded image, state, canonical feature, appearance ID, or dynamics ID is passed into context inference.
- All32 cases are recorded. Support-only successes are marked `excluded_support_success` and excluded from rollout/calibration summaries because the planner never invokes a learned policy for those cases. A partially executed support block is therefore never fed into a diagnostic rollout.
- The future alternatives are the fixed recorded actions and all-zero **raw native** controls. In PushT, raw zero denotes zero target displacement; in Reacher it denotes zero torque. These are model rollouts only. The real simulator advances only through the same at-most10 support environment calls as the evaluator.

## Recorded measurements

Every measured case includes the following mean squared errors in the immutable reference feature dimension:

| Field | Meaning |
|---|---|
| `goal_calibration_mse` | Corrected shifted goal vs canonical reference goal at exactly the same physical goal state |
| `raw_goal_calibration_mse` | Uncorrected shifted goal vs canonical reference goal |
| `support_calibration_mse` | Corrected observed support vs canonical reference encodings of the same observed support renders |
| `raw_support_calibration_mse` | Raw observed support vs canonical support |
| `support_frame_calibration_mse` | Separate corrected-support error for each observed frame; not included as a scalar bootstrap metric |
| `recorded_prediction_to_canonical_goal_mse` | Predicted terminal feature under the recorded future actions vs canonical goal |
| `zero_prediction_to_canonical_goal_mse` | Predicted terminal feature under zero future actions vs the same desired canonical goal |
| `recorded_prediction_to_adapted_goal_mse` | The actual planner's terminal latent cost for the fixed recorded action sequence |
| `zero_prediction_to_adapted_goal_mse` | The same planner cost for all-zero raw controls |
| `terminal_action_sensitivity_mse` | Difference between predicted terminal features under recorded and zero controls |
| `observed_final_support_to_canonical_goal_mse` | Current corrected support feature's distance to canonical goal |
| `observed_final_support_to_adapted_goal_mse` | Current corrected support feature's distance to the corrected goal |

The zero-action error relative to the desired goal **is not prediction accuracy against a zero-action simulator outcome**. Likewise, sensitivity demonstrates dependence on action inputs, not accurate counterfactual dynamics, causal identification, or planning success. The recorded-action terminal prediction has an observed target from the recorded trajectory, but the canonical goal is diagnostic supervision and does not replace the deployable shifted goal.

Per-record scalar measurements, explicit per-seed means, and seed-cluster bootstrap summaries are saved. The output contains checkpoint/config/dataset/evaluator/diagnostic/model/generator source hashes, each episode archive's verified hash, selected keys, support/goal pixel and state hashes, support-action and recorded-future-action hashes, and context hashes. Current file hashes rather than assumed repository cleanliness determine the executed source identity. Timings are operational metadata and are marked ineligible for a primary efficiency claim.

Each policy-eligible case additionally records `archived_render_comparison`: fresh canonical support images against archived grouped frames0,1,2, and fresh canonical goal against archived frame7. Both inputs are raw uint8 RGB before any appearance transform. Measurements include mean absolute channel difference on the0–255 scale, maximum channel difference, fraction of changed channel values, and fraction of changed spatial pixels, with aggregate support and per-support-frame values. Archived images are diagnostic comparators only and never replace observed policy/context inputs.

Goal and control renderer metadata are queried only from an already existing `physics._contexts.gl` while that context is current. The lazy public `contexts` property is deliberately avoided because it can create a context. `GL_RENDERER` and `GL_VENDOR` are recorded when available; absence/failure is explicitly `unknown`, and PushT's pygame path is `not_applicable`. The output also records available `MUJOCO_GL`, `EGL_PLATFORM`, `LIBGL_ALWAYS_SOFTWARE`, `MUJOCO_EGL_DEVICE_ID`, `PYOPENGL_PLATFORM`, `CUDA_VISIBLE_DEVICES`, and `SDL_VIDEODRIVER` values. This checks actual allocated rendering conditions; the separate CPU-only audit's software/archive mismatch is not assumed to occur on the live GPU path.

## Agreement with completed planning

Pass `--planning-result` to require the completed same-checkpoint development result to have matching checkpoint, dataset and evaluator hashes; matching selected episode/appearance keys; matching goal indices, support-success masks, eligibility, initial full-state distance and initial physical-unit errors. The diagnostic refuses to write a complete output if these checks fail. It records the source result hash and maximum initial-distance discrepancy.

This runtime comparison verifies planning setup and support-state/status agreement. The planning result does not store support-pixel hashes, so pixel equality is additionally checked in the real-simulator unit test; the diagnostic records hashes for subsequent cross-mode checks.

## Validation

Command:

```bash
env SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m pytest tests/test_goal_calibration.py -q
```

All ten tests passed in4.90seconds after the renderer-metadata extension. Tests directly compare a real PushT fixture's goal pixels, observed support pixels, support success and initial distance against the unmodified evaluator. Both ordinary full support and a controlled early support-success branch are covered. Additional tests check the exact future-action slice and all-zero comparison, context independence from future actions/canonical targets, distinct canonical/adapted metric definitions, development-only selection, refusal of mismatching completed-planning results and refusal of incomplete training. Three tests cover exact archived frame indices/pixel units, no creation of GL contexts, and querying GL strings only while the existing context is current. The fixture models/trajectories are test data, not research results.

## Queuable commands

Run in the established headless simulator environment on an allocated GPU. Example:

```bash
.venv/bin/python scripts/diagnose_goal_calibration.py \
  --checkpoint runs/world/pusht_factorized_s0/best \
  --data data/world/pusht_relative \
  --device cuda \
  --planning-result results/development_official_budget/pusht_factorized_s0/planning_development.json \
  --output results/diagnostics/pusht_factorized_s0/goal_calibration.json
```

For Reacher, substitute `reacher_factorized_s0` and `data/world/reacher`; use the corresponding run names for frozen, single and framewise. All modes use the identical32-key development selection. The script refuses to overwrite an existing output, preventing accidental replacement of diagnostic evidence. It does not wait for incomplete runs or queue itself.

A completed calibration diagnostic can justify a subsequent explicitly versioned development experiment. It cannot by itself justify a changed primary evaluation protocol, a claimed planning improvement, or selecting a checkpoint on heldout test outcomes.
