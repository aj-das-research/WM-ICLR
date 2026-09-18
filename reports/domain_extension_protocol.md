# Domain extension protocol — registered before extension model results

Date: 2026-09-19. Status: development campaign; final test evaluation remains separate.

## Question and claim boundary

Does episode-dependent observation/dynamics inference improve closed-loop action
selection over equally trained framewise and constant-dynamics controls? Better
latent forecasting alone does not establish better control. Existing Reacher and
PushT development results have not established this mechanism or state of the art.
These extensions are new adapted tasks, not exact reproductions of a published
drone or surgery leaderboard. Clinical efficacy and autonomous surgical safety
are outside the scope of the simulations.

## Data and domains

Each domain has 384 independently recorded 200-command trajectories: 192 train,
48 validation, 48 development and 96 test, balanced over commanded-response gains
1, 0.75 and 1.25. Each learned transition groups five native two-dimensional
commands; model files contain only RGB and commanded actions. Hidden state is
reserved for collection verification, reachability checks and evaluator scoring.
Action normalization is fitted on the training split only. Dataset manifests and
payload hashes must pass complete audits before training. Original source code,
simulator versions, action interfaces and rendering conventions are recorded.

- **Drone:** pinned gym-pybullet-drones CF2X physics and upstream PID controller;
  learned policies request XY velocity through the same PID. Fixed external RGB
  camera, fixed altitude. The shift changes command gain, not wind or mass.
  Goals are recorded settled endpoints. Success requires XY distance <4 cm,
  speed <6 cm/s, altitude error <5 cm, no crash and no workspace escape.
- **Surgery:** pinned LapGym/SOFA tissue manipulation, two-dimensional gripper
  commands and deformable tissue. The shift changes command gain before native
  clipping. Forecast collection retains finite workspace-clipped transitions.
  Planning goals are separately constructed from recorded endpoints and an
  independent replay with the desired marker moved to that endpoint. Success
  follows native XZ tissue-point distance <=2 mm. Reference-marker changes must
  preserve the replayed physical trajectory. Native SOFA warnings/errors are
  retained in logs and reviewed with finite-state, stability and replay audits.

Appearance transformations and factor splits reuse the original frozen loader:
seven training/validation combinations, development (warm, gain 0.75), all nine
test combinations with (cool, gain 1.25) withheld from training. These are
photometric shifts, not camera-pose generalization. Development/test seeds never
enter training or checkpoint selection. Test-side goal eligibility and replay
verification may be audited without using model outputs or success for selection.

## Models and training

Two predictor families: released LeWM transformer initialization and an in-house
two-layer, hidden-size-256 GRU substituted into the same LeWM projected
state/action interface. Both reuse the frozen, released PushT visual encoder;
this is cross-domain transfer, not surgical or drone visual pretraining. The GRU
is a recurrent architecture control, not a reproduction of Dreamer. Parameter
counts and initialization differences are reported, and primary comparisons
are within predictor family.

Each family trains framewise correction, ShiftWM factorized episode context
(ours), and constant dynamics context retaining episode-dependent observation
correction. The constant control removes only episode information from the
dynamics network; nominal parameter count is matched, effective capacity is not
identical. Seeds 0, 1 and 2 yield 18 full training runs per domain, 36 total.

All runs: 30 complete epochs, AdamW lr 1e-4, cosine minimum 1e-6, weight decay
0.01, gradient clipping 1, batch size 128, window length 8, stride 4, support
length 3. Predict all five query targets recursively from observed support.
Identical alignment/consistency weights use the unchanged ModelConfig defaults.
BF16 training where supported; float32 validation over the complete validation
split. Checkpoints are selected solely by minimum recursive validation MSE
across completed epochs. Training and checkpoint code, source data, cached
features, normalization, optimizer and RNG identities are recorded. Complete
offline inference packages and resumable optimizer state are saved locally.

## Closed-loop development evaluation

Use the first eight eligible development trajectories in seed order at gain
0.75, under warm appearance. If fewer than eight are eligible, evaluate all and
report the shortfall. Before any model outcomes, exclude drone initial-goal XY
distance <8 cm or initially solved tasks. For surgery exclude initial XZ
distance <4 mm, initially solved tasks, or reference trajectories with any
invalid action/unstable deformation; preserve a complete exclusions registry.
The latter is a constrained reachable-goal task distribution and must be named
as such. Include invalid actions from the policy as failures, never exclusions.

Budget: 200 native commands **including** ten common recorded support commands.
Stop at first native-step success, physics/workspace failure, or budget. A
partial final block is logged exactly; never invent unexecuted support actions.
All compared learned methods share goals, support actions, seeds and CEM budget:
unchanged upstream CEM solver, horizon 5 grouped transitions, 128 candidates,
5 optimizer iterations, 16 elites, replan after up to five executed commands.
CEM acts in training-standardized coordinates and clips only converted native
commands to [-1,1]. Warm-start the remaining plan. Context uses only RGB and
already executed commands. Goal input is an actual recorded/replayed RGB image.

Controls: matched native-budget random commands and explicitly privileged
recorded-action replay to verify reachability. Replay is not a deployable
baseline. Report success, native interactions, final distance, invalid/unstable
actions, crash/escape and planning latency. Include early support successes and
identify them so they are not attributed to learned planning.

Final test evaluation is a separate command after development analysis. Do not
select final seeds, examples, checkpoints or hyperparameters for their test
outcomes. Any new hypothesis or protocol change needs a separately named run;
do not overwrite the registered campaign or retroactively replace failed runs.

## Decision and reporting

Promotion requires better closed-loop success than both within-family controls,
consistent paired evidence across training seeds and domains, with uncertainty.
Paired task/seed bootstrap estimates and actual success numerators/denominators
must accompany point gains. Describe mixed and null results. No significance or
state-of-the-art claim follows merely from this small development campaign.
Published AdaJEPA and DINO-WM/JEPA-WM implementations are separate reproduction
tracks; do not label in-house controls as independently reproduced state of art.

Diagnostics should measure action sensitivity, goal separability and agreement
between predicted candidate ranking and realized progress. Qualitative panels
must use recorded renders and exact actions, contain matched initial conditions,
annotate physical errors/support/planning steps, and show representative failures
alongside named success examples. Generated assets may explain an architecture;
they cannot stand in for experimental outcomes.

Releases remain local until requested. No surgery/drone checkpoint is described
as trained or reusable until its full training, load and inference checks pass.
