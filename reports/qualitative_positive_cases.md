# All positive paired qualitative cases

Audit date: 2026-09-19. This report covers **every saved development task on
which ShiftWM succeeds and Framewise calibration fails**. There are four:
one PushT task and three Reacher tasks. These are four individual paired
episodes, not four aggregate benchmark improvements. All four corresponding
Shared context runs also fail.

The evidence record is `reports/evidence/qualitative_positive_cases.json`.
All 198 raw result/video source hashes in the 64-task gallery were checked
against the gallery evidence. Initial observations and goal images match
exactly across the three methods. Actual timeline pixels were independently
inspected for ShiftWM and Framewise in all four cases. No frames were
generated, interpolated, retouched, or newly simulated for this audit.

## Complete positive stratum and matched-time comparison

| Environment | Episode seed | ShiftWM stops successfully | Framewise stops unsuccessfully | Last exact saved time shared by both | Terminal error: ShiftWM / Framewise |
|---|---:|---:|---:|---:|---|
| PushT | 2031024 | 21 | 50 | 20 | Block: 6.91 / 317.86 px; pusher: 14.72 / 328.38 px; angle: 0.159 / 1.384 rad |
| Reacher | 2031004 | 23 | 50 | 20 | Joint L2: 0.0478 / 0.5715 rad |
| Reacher | 2031008 | 21 | 50 | 20 | Joint L2: 0.0263 / 0.2133 rad |
| Reacher | 2031011 | 23 | 50 | 20 | Joint L2: 0.0602 / 0.3652 rad |

Times count native environment calls, including ten calls used to acquire
support. Both methods have saved observations at calls 10, 15, and 20; the
matching call-20 image is array index 2 in every case. ShiftWM's last action
block stops early at call 21 or 23. Framewise has no saved frame at those
partial-block times. Its terminal frame is at 50. A comparison should therefore
show **both actual call-20 frames**, then separately label each method's actual
endpoint. Do not interpolate a Framewise call-21 or call-23 image, relabel its
call-20 observation, or imply that terminal images show a common time.

All methods use training seed zero, the same development task, warm appearance
condition 1, physical condition 1, available goal image, planner seed, 300 CEM
candidates, 30 iterations, 30 elites, horizon five action blocks, and a maximum
50-call native budget. These matched conditions make the executions comparable;
they do not turn the selected successes into a performance-rate estimate.

## PushT, episode 2031024: reaching the configuration versus leaving the view

The goal image shows the desired gray T and blue pusher near the upper-right
corner. At call 20, both methods have moved the block toward that region, but
their block orientations differ. It would overstate the saved evidence to
assign a numerical intermediate-error advantage: no per-frame physical state
or error array was saved. ShiftWM satisfies the task at call 21. Framewise
continues moving; its later saved observations show the block leaving most of
the viewport. At call 50 only a gray segment remains at the upper-left edge,
and the blue pusher is not visible.

The measured endpoints support the success/failure distinction. ShiftWM's
combined pusher-plus-block position-error norm is 16.26 simulator pixels;
Framewise's is 457.02 pixels. Their angular errors are 0.159 and 1.384 radians.
The position criterion is a combined norm below 20 pixels, and the block-angle
criterion is below pi/9 radians. Pixel units refer to the simulator's
512-by-512 coordinates, not the 224-by-224 displayed RGB image. The mixed-unit
full-state distance includes additional state components and should not be
used as the explanatory caption metric.

Useful visual callouts:

- Identify the **gray target block pose and blue target pusher** in the goal
  image. The persistent green T is an upstream rendering marker and is not the
  episode-specific scored goal.
- Bracket the two call-20 observations as the same number of native calls.
  Do not label one as quantitatively closer without an actual state metric.
- Mark the observed call-21 success and the call-50 failure, with terminal
  block, pusher, and angular errors beside their respective endpoints.
- Point to the baseline's visible gray sliver at the upper-left image boundary:
  “Most of block outside view.” Do not invent an off-screen object location.

This is evidence of one more successful closed-loop execution under the stated
budget. It does not identify which internal ShiftWM component caused the gain.

## Reacher, episode 2031004: attaining the requested folded pose

The goal is a compact folded arm configuration. At call 20, ShiftWM's pose
visually resembles that configuration, while Framewise remains closer to the
initial nearly vertical fold. ShiftWM succeeds at call 23 with terminal joint
L2 distance 0.0478 radians. Framewise subsequently changes orientation and ends
with a nearly horizontal folded arm at call 50, at 0.5715 radians.

The visual explanation is **the requested joint configuration is reached by
one execution and missed by the other**. A useful goal-reference outline can
make the forearm-orientation difference clear, provided it is explicitly
labeled as a goal annotation and is registered at the same image scale. Such
an outline is not a predicted trajectory. Pair the exact call-20 frames and
add separate action-budget timelines, `10 support -> 23 success` and
`10 support -> 50 budget exhausted`.

Do not claim that the baseline is unstable, oscillatory in a dynamical-systems
sense, or that ShiftWM inferred the correct mass from these sparse observations.
The endpoint criterion and the visible poses are the supported observations.

## Reacher, episode 2031008: a visually near miss still fails the joint criterion

ShiftWM starts from a tightly folded pose, approaches the more open goal by
call 20, and succeeds at call 21. Framewise is flatter at call 20 and then
visits several other visible poses. Its call-50 endpoint resembles the goal
but retains a joint mismatch: 0.2133 radians versus ShiftWM's 0.0263 radians.

This case is useful because the failure can look small in an ordinary
thumbnail. Display equally scaled magnified goal and endpoint arm patches
alongside the unchanged full scenes. A goal-reference outline or short
orientation pointer can expose the residual link-pose difference. Label the
0.026/0.213-radian endpoint diagnostics beside independently sourced
Success/Failure badges; show the actual paired call-20 images as well.

The interpretation is limited to a successful pose match versus a scored
near-looking miss. Fingertip position alone and visual resemblance are not
the environment's success test, and no intermediate numerical joint errors
can be reconstructed from the saved endpoint metrics.

## Reacher, episode 2031011: distinguishing distal-link orientation

ShiftWM turns the distal link toward the goal's downward-right direction by
call 20 and reaches the goal at call 23. Framewise's arm is more open at call
20; its distal-link direction still differs from the target at call 50. The
terminal joint L2 distances are 0.0602 and 0.3652 radians respectively.

A magnified common-scale goal/endpoint inset should emphasize distal-link
orientation, not just whether the arm moved. Add exact call-20 comparison
frames and terminal call-23/call-50 labels. This is also a useful illustration
of the metric distinction: **ShiftWM succeeds with joint L2 above 0.05 radians**.
That is valid because upstream success requires each unwrapped joint's
absolute error to be below 0.05 radians; it does not require the joint L2 norm
to be below 0.05. Keep the recorded success label and avoid drawing a false
L2 threshold line.

## Presentation recommendation and limits of explanation

A positive-case figure may include all four cases with the heading
“All four ShiftWM-only successes in the saved development comparison.” For
each case use the same visual reading order: available goal, shared initial
support, paired call-20 observations, then each actual endpoint and stopping
time. Clear endpoint annotations and a small native-action timeline add useful
information without speculative arrows or invented intermediate measurements.
Any detail crop should use the same bounds, display scale, and processing for
the goal and both methods, and remain connected to a visible full-frame image.

Keep the existing failure/control panels and complete 64-task gallery. Among
31 eligible PushT tasks, this comparison has one ShiftWM-only success, one
Framewise-only success, and 29 joint failures. Among 30 eligible Reacher tasks,
there are three ShiftWM-only successes, eight Framewise-only successes, three
joint successes, and 16 joint failures. Three further tasks succeed during
shared support and do not demonstrate planning gains. The four successes
therefore cannot support an overall development superiority claim.

The images explain **what visibly happened**, and the endpoint scores explain
**why the task counted as success or failure**. They cannot establish why the
algorithm caused a gain. ShiftWM and Framewise differ in learned observation
calibration, context pathways, and trained weights. Their candidate searches
depend on those predictions and goal embeddings. Attributing the outcome to
temporal context, observation/dynamics factorization, identified physical
parameters, or more accurate candidate ranking requires interventions that
hold other components fixed, plus matched context-necessity controls. Raw
context variation or a favorable rollout image is insufficient. The new
development study can test specific hypotheses, but it must not be presented
as completed positive evidence before its outcomes are available.

These visual examples remain post-hoc, single-training-seed development
observations. They are separate from the three-seed main-test results and do
not establish real-robot, medical, arbitrary-task, or arbitrary-model behavior.
