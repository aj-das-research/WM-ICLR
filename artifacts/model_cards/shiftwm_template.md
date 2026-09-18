# ShiftWM trained world model — model card template

**Release status: template; fill from the completed run and verified evaluation.**
This file describes the implemented artifact interface. It contains no performance
claims and must not be published as a completed model card with fields unfilled.

## Identity and provenance

| Field | Source to insert |
|---|---|
| Environment and action interface | Dataset manifest; PushT-relative or Reacher |
| Method variant and seed | Run configuration |
| Training completion, epochs, updates | `training_summary.json` and checkpoint |
| Base weights and SHA256 | Checkpoint `config.json` provenance |
| Official action-statistic SHA256 | Checkpoint provenance |
| Data and feature-cache manifest SHA256 | Checkpoint provenance |
| Source revision and implementation hashes | Checkpoint provenance |
| Trainable and deployment parameter counts | `reports/model_parameter_counts.json` |
| Validation selection rule | Lowest held-out validation one-step prediction MSE |
| Evaluation metrics and intervals | Immutable result JSON, with hash |

The first implementation starts from strict-loaded LeWM weights and freezes the
visual encoder/projector. It trains the action encoder, latent predictor,
prediction projector, and active context/adaptation modules. The default model
has 18,319,598 required deployment parameters and 12,025,454 trainable parameters.
The full training package also includes an immutable reference encoder and
inactive ablation modules. These counts are not performance measurements.

## Training information and privileged inputs

The observation context uses mean/variance of three past raw visual embeddings.
The dynamics context uses corrected embeddings and the two executed action blocks
between those frames. Neither module receives true simulator state, physics
parameters, intervention labels, future query images, or goal reward at deployment.

Training uses canonical renders of the same simulator trajectory to define target
latents with a frozen reference encoder. These canonical renderings are privileged
**training supervision**, not available to the deployed policy. Appearance labels
group examples for the optional observation-consistency loss; paired renders
provide the optional dynamics-consistency loss. Identical raw trajectory/render
access must be provided to matching baselines.

The implemented objective is teacher-forced one-step prediction on query frames,
plus canonical latent alignment; the factorized variant adds paired-context
consistency. It is not trained with multi-step recursive rollout losses in this
version. Deployment prediction does recurse over candidate future actions.
Reference coordinates remain fixed. There is no claimed causal identification,
uncertainty guarantee, or always-correct diagnosis of the physical change.

## Exact first-campaign scope

- Environments: upstream PushT-relative and Reacher `qpos_match`.
- Observation conditions: canonical, warm affine RGB, cool affine RGB; dim affine
  RGB is an extrapolation test. These are photometric transforms, not unseen cameras.
- Dynamics: PushT damping or Reacher density variations defined in the immutable
  dataset manifest. No arbitrary new simulator family is covered.
- Train factor pairs: `(0,0), (1,0), (2,0), (0,1), (2,1), (0,2), (1,2)`.
- Development pair `(1,1)`; designated held-out test pair `(2,2)`.
- Entire trajectory seeds stay within one split. Test labels do not select the
  checkpoint. Report interpolation, held-out combination, and extrapolation separately.

Replace this list if the completed campaign differs. Do not imply that a released
PushT checkpoint controls Reacher, medical equipment, or real robots.

## Reuse and limitations

Use `shiftwm.checkpoint.load_package(path, device)` to load the weights-only full
model. Input shapes, preprocessing, grouped actions, future-only rollout, and goal
encoding are illustrated in `frozen_lewm.md` and use the same API in every variant.
To infer a new context, supply only the latest compatible observation/action
history. Keep weights fixed for the primary deployed method.

The training package can resume with optimizer, scheduler, Python/NumPy/PyTorch
RNG state, and deterministic sampler progress. A model-only package supports
inference but is insufficient for exact training continuation.

Known limitations: context ambiguity, context collapse, visual conditions outside
training support, compounded rollout error, latent-distance/planning mismatch,
and dependence on canonical paired training supervision. The plain, matched
single-context, and unpaired-factorized controls are necessary to attribute gains.
Record negative results and all observed failure modes, not only successful demos.

No healthcare data or real-world clinical validation is included in the first
campaign. No external safety or medical claims are supported.

## Evaluation to populate before release

Report fixed-reference one/three/five-step forecast errors; persistence comparison;
closed-loop success; support-only success; manipulation versus repositioning
strata; action budget and latency; and seed-cluster uncertainty intervals. Describe
the exact matched CEM solver settings and all training data/compute costs. Published
upstream benchmark scores are not directly interchangeable with this custom shift
benchmark. Public checkpoint links remain pending actual upload and licensing review.
