# Observed closed-loop examples

Reader/slot: ICLR research manuscript appendix; four figures at 5.5-inch width,
5.6-inch height, plus a local all-task gallery. These are existing simulator
benchmarks used in a controlled development study, not a new benchmark claim.

Visual thesis: a shared visual goal and initial state can lead to different
closed-loop outcomes; both the beneficial cases and the remaining failure
modes must be visible alongside aggregate planning results.

Evidence: `results/development_official_budget/{environment}_{method}_s0`,
32 development episodes per environment, actual saved 224×224 RGB frames,
complete source-validated evaluations. All three methods use identical goals,
support pixels, native budget and planner settings. No generated scenes,
decoded model predictions, intermediate-state interpolation or cropped scenes.

Three composition sketches considered:

```
A. Paired temporal lanes (selected)
           goal | support | first action block | actual final state
  ours      []      []             []                    []  outcome/errors
  baseline  []      []             []                    []  outcome/errors

B. Endpoint matrix
          case1    case2    case3    case4
  goal      []       []       []       []
  ours      []       []       []       []
  baseline  []       []       []       []

C. Scene plus trajectory overlay
  [shared scene with paths]   [goal]   [endpoint error breakdown]
```

A retains the actual decision sequence and the initial correspondence at
paper size. B is more compact but hides how behavior diverges. C would be
useful with recorded physical trajectories, but the archives contain sampled
RGB frames and terminal metrics, not complete intermediate simulator states;
inferring geometric paths from them would introduce unsupported information.

Representation contract: rows are methods, columns are observations. The goal
is an input image, never an intermediate state on the time axis. The first
observation follows common support; subsequent frame labels count real native
actions, including partial final blocks. Green/red final borders duplicate
explicit success/failure labels. Terminal physical errors explain an outcome
without inventing an error curve. Successful short runs are not padded with
fabricated states. An immediate support success explicitly shows no planned
move. The gallery preserves all recorded frames and blank post-termination
space at a constant cell scale.

Selection: first lexicographic trajectory versus Framewise in each of four
strata per environment. Both use ours-only, baseline-only and neither; PushT
adds support-only success, Reacher adds both-success. This is deliberately
outcome-stratified illustration, not a random sample or win-rate estimate.
The complete 64-task gallery includes Shared context as a third method.

Critical semantic caveats: the green PushT T is the upstream renderer's
persistent marker, not the episode-specific scored goal. The gray T and blue
pusher configuration in the available goal image defines the intended state.
PushT distances are in native 512×512 simulator pixels, before display resize.
Reacher success is a per-joint threshold; the displayed joint-angle norm is
only a diagnostic. These details belong in the caption and gallery guide.

Editable source: `paper/scripts/render_qualitative.py`; exact frame times,
records and hashes: `paper/generated/qualitative/evidence.json`. New rollout
revision experiments are excluded because no outcome is yet established.
