# ShiftWM IWS feature forecasting checkpoints — row-wise runtime v2

This package contains 36 project-trained predictors: three tasks (PushT, bimanual
Box, bimanual Rope), four methods and three seeds. Each completed 30 epochs;
selection used internal-development equal-trajectory standardized endpoint H60
MSE, choosing the earliest epoch attaining the minimum. Exact selected epochs,
model/config hashes, parameter counts, training identities and source bundles
appear in `manifest.json` and the individual `model_cards/` files. No model was
selected anew for this release or on reserved outcomes.

The autoregressive and anchored-additive models are matched internal controls,
not external published SOTA implementations. Bounded spatial mixing is the
original proposed IWS arm. No-tanh spatial mixing is a later development-driven
bound-removal ablation; it keeps the mixing construction and removes the tanh
innovation bound. These variants differ scientifically and remain separate.

The shared compact predictor uses frozen DINOv2-small 4-by-4 pooled features,
one initial observation and native task commands. It does not use privileged
future observations, robot proprioception beyond whatever the task command
schema contains, or a hidden test-time optimizer. Model construction, parameter
counts and exact training recipe are included in each unchanged package config.
Learned predictors were trained by this project; no RLA-WM weights are included.

## Training data and evaluation populations

Training used the registered internal split of the public RLA-WM IWS task
artifacts, revision `34bd8a8cbf3fa68e09ebd69aa35cb673279f4fc2`. Internal-development
trajectories were used for checkpoint selection. The no-tanh study followed the
original development analysis; its development comparisons are exploratory.
Complete development statistics and comparator regressions are retained in
`evidence/development_summary.json` with original source hashes.

The later locally reserved upstream validation evaluation used 200 fixed handles
per task, ten trajectories per task (600 handles and 30 trajectories overall),
the same selected weights and all four methods. All 36 evaluations completed
under the explicitly registered row-wise CPU backend, after a prior attempt
stopped at a numerical prefix gate. The initial failure's cause is unresolved;
later input-only probes did not reproduce it. Neither the backend nor these
weights were chosen using reserved accuracy. The complete frozen result and
independent numerical-review hashes are recorded in the provenance.

The original bounded-versus-additive primary reserved MSE result is mixed across
tasks. The no-tanh secondary arm improves endpoint MSE versus the autoregressive
control in all three tasks, but that does not make it a universal winner or an
external-SOTA comparison. All four metrics, all methods, negative differences,
and unadjusted paired-bootstrap intervals are retained in the evidence JSONs.
Dependent task/metric/comparator contrasts are not independent experiments.

## Intended use and limitations

Intended use is research on action-conditioned prediction in this exact feature
space, reproduction of saved selected predictors, and compatible downstream
experimentation. Inputs and commands must follow `preprocessing.json`; arbitrary
image embeddings, DROID commands, or other encoders are not interchangeable.
The trained statistics are immutable and internal to the model.

This distribution returns feature vectors, not images, rewards, actions, clinical
decisions or certified physical plans. No RGB generation, closed-loop robot
control, healthcare benefit, causal identification, safety assurance or general
task transfer is demonstrated by this artifact. Errors can compound over time,
and latent forecast quality does not establish successful control. Synthetic
portability inputs are out of distribution and have no benchmark meaning.

Only CPU FP32, eight intra-op and one inter-op thread is the reviewed recovery
runtime. GPU, other precision, different threading and arbitrary Torch/platform
builds require their own verification. The older resource table measured native
sequence-GRU dispatch and must not be cited as latency of this row-wise backend.
No runtime advantage is claimed. The bundle contains no optimizer/RNG state and
cannot alone reproduce training continuation.

## Attribution and licensing

Project predictor weights and new wrapper code/configurations are distributed
under the project MIT terms; the copied LeWM code retains its MIT notice. Frozen
DINOv2-small was used for feature extraction and is Apache-2.0; its encoder
weights are not redistributed here. Dataset and model licenses are separate:
the saved RLA-WM dataset metadata does not establish redistribution terms for
genuine examples. Consequently only independent synthetic fixtures are included.
`MODEL_LICENSE.md` does not grant rights to omitted third-party data.

Primary source links: [RLA-WM data](https://huggingface.co/datasets/xyzhang368/RLA-WM),
[RLA-WM paper and code](https://github.com/mlzxy/rla-wm),
[DINOv2-small](https://huggingface.co/facebook/dinov2-small),
and the exact LeWM origin/revision in `src/shiftwm/vendor/lewm/NOTICE.json`.
The local project release is not published until the owner creates and verifies
its public distribution; consult the repository's actual release inventory.
