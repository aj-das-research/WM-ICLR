# Matched qualitative examples: verified inventory

Sources audited on 2026-09-18. The complete machine-readable inventory is
`reports/evidence/qualitative_examples_candidates.json`. This is an evidence
inventory, not a rendered figure or a claim of representative performance.

## Available observations and verified matching

The existing `results/development_official_budget` campaign saved every one of
32 episodes for each of ShiftWM (`factorized`), Framewise calibration
(`framewise`), and Shared context (`single`) in both PushT and Reacher. There
are **192 NPZ files covering 64 matched tasks**. Each file contains actual
224-by-224 RGB uint8 simulator observations under the warm appearance shift,
plus the available goal image. These are observed executions, not model-
predicted images. The original main-test campaign did not save matched videos.

For all 64 tasks, this audit verified pixel-for-pixel equality of the first
saved observation and goal image across all three methods. It also verified
the same initial physical diagnostics, task IDs, seeds, goal index, eligibility,
native budget, action grouping, CEM search budget, evaluator source, dataset
identity, and each method's checkpoint hash. All files have the same selected
32 development keys in each environment. Source result hashes, checkpoint
identities, all NPZ hashes, and every frame's pixel hash are in the inventory.

The common condition is appearance 1 (warm RGB transform), dynamics 1: PushT
space damping 0.15 or Reacher arm/finger density 650. Every run uses training
seed zero, 300 CEM candidates, 30 iterations, 30 elites, five action blocks of
planning horizon, and a maximum of 50 native environment calls including ten
support calls. The goal is recorded frame index 7. The development combination
and these trajectories are not an untouched final generalization test.

## Deterministic balanced selections

Final paper selection rule: take the lexicographically first episode in each
outcome category **relative to Framewise only**: ShiftWM-only success,
Framewise-only success, and joint failure. The fourth case is joint success
for Reacher and explicitly labeled support-only success for PushT, where no
eligible joint success exists. The full gallery includes all 64 tasks and all
three methods. The inventory retains stricter three-method alternatives in a
separate field; those are not the paper renderer's selections.

| Environment | Paper category | Trajectory seed | ShiftWM outcome / stop | Framewise outcome / stop | Shared outcome / stop (gallery) |
|---|---|---:|---|---|---|
| PushT | ShiftWM-only success | 2031024 | Success / 21 | Failure / 50 | Failure / 50 |
| PushT | Framewise-only success | 2031000 | Failure / 50 | Success / 48 | Success / 25 |
| PushT | Both fail | 2031001 | Failure / 50 | Failure / 50 | Failure / 50 |
| PushT | Support-only success | 2031026 | Support success / 2 | Support success / 2 | Support success / 2 |
| Reacher | ShiftWM-only success | 2031004 | Success / 23 | Failure / 50 | Failure / 50 |
| Reacher | Framewise-only success | 2031001 | Failure / 50 | Success / 41 | Failure / 50 |
| Reacher | Both fail | 2031002 | Failure / 50 | Failure / 50 | Failure / 50 |
| Reacher | Both succeed | 2031000 | Success / 33 | Success / 36 | Failure / 50 |

The rule uses outcomes to make contrasting illustrations, so displayed
category proportions must not be presented as success rates or representative
prevalence. The full gallery labels the three support-success cases.

“Stop” means native environment call, not a video-frame index or elapsed
seconds. Paths follow this exact pattern:

`results/development_official_budget/{environment}_{mode}_s0/videos/development-s{seed}-d1-o1.npz`

The adjacent `planning_development.json` contains the corresponding record.
The inventory stores the exact file, record index, protocol signature,
checkpoint epoch, and all endpoint diagnostics for each method and task.

PushT has one ShiftWM-only success, one both-baselines-success case, 29 joint
failures, and no eligible joint success. One additional task succeeds during
support and cannot demonstrate planning competence. Reacher has three
ShiftWM-only successes, two both-baselines-success cases, 13 joint failures,
one joint success, 11 mixed cases, and two tasks already successful in support.
Do not fabricate a PushT joint-success planning category to make the grid
symmetric.

## Exact frame/time semantics

`frames[0]` is the first observation saved **after** the support acquisition;
for eligible tasks it is native call 10. A new frame is saved after each
executed action block. Blocks normally have five native actions, but the last
block stops immediately on success and can contain fewer actions. Consequently
the saved sequence is not a fixed-frame-rate video.

The inventory derives exact native times as:

1. Initial time = `native_steps - sum(len(block)/2 for block in executed_action_blocks)`.
2. Add `len(block)/2` for each subsequent frame.
3. Assert `len(frames) == len(executed_action_blocks) + 1`, final time equals
   `native_steps`, and initial time is ten for every eligible example.

For the PushT success at seed 2031024, ShiftWM's times are `[10,15,20,21]`;
both baselines' times are `[10,15,20,25,30,35,40,45,50]`. For Reacher seed
2031007, Shared context has only `[10,15,18]`. The optional joint-success
example at seed 2031017 has two frames per method, at `[10,11]` for ShiftWM and
Framewise and `[10,12]` for Shared. Do not invent a third distinct observation.
If a layout repeats a terminal frame to fill a column, label the repetition
as a stopped episode; preferably show fewer frames.

A concise filmstrip can show the common goal, post-support state, first
executed-block state, and endpoint, with a native-call label under each actual
frame. Endpoints may occur at different times and must carry their real times.
There are no stored intermediate simulator-state/error arrays, native-step
videos, CEM candidate trajectories, or predicted visual futures. Endpoint
numbers cannot support an interpolated error curve or intermediate-state
annotation.

## PushT goal marker: important visual distinction

The goal image's **gray T and blue pusher** show the desired episode-specific
configuration. The persistent green T is the upstream environment's reset
marker. It is **not** the new episode-specific scored goal in this evaluation.

This follows directly from the implementation. `set_task_goal` calls upstream
`_set_goal_state`, which only updates `self.goal_state`. Rendering instead
draws `self.goal_pose`, initialized from the variation-space goal position and
angle during setup. The evaluation separately restores the recorded goal
simulator state to render the available goal image. Thus the desired gray
block/pusher can lie away from the green marker. This is part of the actual
saved experiment; do not relocate or erase the green marker when preparing
figures. Add a short labeled legend or pointer and explain it in the caption.

Source locations: `src/shiftwm/generate.py:101`,
`src/shiftwm/evaluate.py:224`, and
`external/stable-worldmodel/stable_worldmodel/envs/pusht/env.py:439`, `:546`,
and `:594`.

## Physical annotations and success definitions

PushT success requires the combined Euclidean norm of pusher and block position
errors to be below 20 simulator pixels, **and** symmetry-aware block-angle
error below pi/9 radians. These pixel units refer to the 512-by-512 simulator
coordinates, not the resized 224-by-224 displayed image. Annotate block error,
pusher error, and angle with these units; avoid using the mixed-unit full-state
distance as the explanatory headline.

The PushT baseline-success case (seed 2031000) is particularly informative:
ShiftWM's block translation error is only 1.20 pixels and angle error 0.00273
radians, but its pusher remains 428.96 pixels from the target. This is a scored
failure despite the block being near its goal. Framewise has block error 1.20
pixels and pusher error 18.53 pixels; their combined norm is 18.57 pixels and
the angle test also passes. Calling this a failure to move the block would
misdescribe the observed result.

Reacher success requires **each unwrapped joint** absolute error below 0.05
radians. The saved `wrapped_joint_error_rad` is a wrapped L2 diagnostic,
not the success threshold. A success can therefore have displayed L2 above
0.05 radians: Shared context's successful seed 2031007 has L2 0.06147 radians.
Preserve the recorded success label and explain the diagnostic distinction.

## Suggested caption

“Matched development executions under appearance and dynamics shift. Each
group shows the same task and available goal image for ShiftWM (ours),
Framewise calibration, and Shared context. Images are saved simulator
observations; labels give native environment calls, including ten support
calls, under a common 50-call budget. Examples are selected by lowest seed
within declared outcome strata versus Framewise, including wins, losses,
joint failures and explicitly labeled support-only or joint successes. They illustrate behavior rather than estimate success
frequency. PushT is scored by pusher-plus-block position and block angle;
the gray block and blue pusher in the goal image specify its target, while
the persistent green marker is inherited renderer context. Reacher success
uses per-joint error; displayed joint L2 is diagnostic. All methods stop on
success; endpoint times can differ. These seed-zero development examples
are separate from the main held-out evaluation.”

No figure pixels were rendered or edited by this inventory task. The four
renderer-produced PNGs were independently inspected at their actual 1320-by-1344
resolution. Captions in `paper/sections/qualitative_rollouts.tex` match the
selected cases, times and measurements. No text/image overlap or clipping was
observed in those exports. Final manuscript placement and paper-size review
remain the renderer/integrator's responsibility under the figure skill.
