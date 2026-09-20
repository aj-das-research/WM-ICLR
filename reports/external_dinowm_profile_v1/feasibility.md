# External DINO-WM comparison: implementation and profile specification

20 September 2026. **Architecture, causal contracts and the full CUDA resource
profile passed. No benchmark training or evaluation was performed by this
namespace.** The completed allocation was job 201998 on an RTX 5000 Ada with
32 GiB GPU memory. Both full-batch objectives fit without a fallback.

| Training objective | Median update, batch 128 | Peak allocated GPU memory | Projected 30-epoch predictor compute per seed |
|---|---:|---:|---:|
| Official shifted one-step | 20.60 ms | 1.41 GiB | 2.57 minutes |
| Matched recursive H10 | 170.64 ms | 10.70 GiB | 13.53 minutes |

Common FP32 H10 forward medians were 164.50 and 165.39 ms per batch,
respectively. These are synthetic full-architecture measurements, not accuracy
results or end-to-end runtime promises. The projection includes the recorded
training and validation batch counts, but excludes data loading, transfers,
checkpoint writes, final evaluation and queue time. The six-run proposal
therefore remains feasible on the available GPUs; actual training gets a
separate source, selection and data registration.

The authoritative measurement is `job_201998/profile.json`. It binds the
registration, independent source review, every dependency, GPU identity,
precision, raw timing samples and exact memory counters. Temporary random
weights were discarded; this resource job produced no reusable checkpoint.

The official [DINO-WM code](https://github.com/gaoyuezhou/dino_wm/tree/0a9492fa12044b852ae9e001cc74604b79c8bb0c)
is now pinned locally at `external/official-dino-wm` with its MIT license
preserved. The checkout is unmodified. This supplies an external implementation
independent of the LeWM-derived modules already used by ShiftWM.

The adapter retains the original six-layer, sixteen-head transformer, 64-wide
attention heads, 2,048-wide MLP, learned positional embeddings, direct feature
prediction, frame-causal mask and kernel-one action encoder. The resulting
no-proprioception predictor has **19,412,420 trainable parameters**. It is not
silently reduced to ShiftWM's smaller trunk.

One hash-checked source transformation registers the existing attention mask as
a nonpersistent buffer instead of constructing a plain CUDA tensor. CPU tests
verify the exact mask and official prediction-method parity. This compatibility
change does not alter the upstream checkout or attention arithmetic.

The comparison would still be an **adapted official predictor**, not a claim to
reproduce the published benchmark scores. The local interface has sixteen
pooled DINOv2-small tokens, training-standardized features, no proprioceptive
input, three observed grids and grouped 35-dimensional commands. The official
224-image configuration resizes to 196 pixels and yields 196 native tokens;
its preprocessing, action schema, optimization recipe and data differ.

Two training objectives are kept distinct:

| Objective | Supervised grid indices per window | Forward calls | Meaning |
|---|---|---:|---|
| Official one-step shifted visual loss | 1, 2, 3 | 1 | Original three-slot temporal loss; two targets lie within observed support, one is future. |
| Matched recursive H10 loss | 3–12 | 10 | Deliberate all-ten-future-step alternative with full backpropagation through earlier predictions. |

Both resource profiles use batch128, BF16 training arithmetic, FP32 parameters,
AdamW updates and norm clipping. Both also measure common FP32 ten-step
inference. They use synthetic random tensors exclusively and save no weights or
accuracy scores. Validation memory is measured with optimizer state resident,
as during training, and is labeled accordingly.

The proposed later comparison would use existing training/development windows,
the same frozen training-only normalization, 30 epochs and three seeds, and
common FP32 all-ten-step validation selection. The official objective should be
the primary external-baseline candidate; the matched-objective alternative
is affordable as a separately declared arm. Native one-step cost is
projected over the same 18,660 H10-eligible training windows, not a larger
shorter-window dataset. Thirteen full validation batches cover the existing
1,631 development windows. The completed profile records separate per-seed and
three-seed compute projections, excluding I/O, checkpoints, extraction, final
evaluation and scheduler delay. Actual training wall time remains to be measured.

Eleven synthetic CPU tests passed in 4.42 seconds. They cover full architectural
shape/mask identity, channel-major packing, exact official prediction parity,
causal action/feature access, prefix equivalence, target isolation, all-three-slot
native loss, retained recursive gradients, bad-contract rejection, review-before-
execution and arithmetic of runtime projections.

Executable files are under `scripts/external_dinowm_profile_v1/`; the detailed
contract and commands are in its README. Registration is
`reports/external_dinowm_profile_v1/registration.json`, SHA256
`ab41bcc01561b5a024c1546497ecf7ab05a6db3491c224fa799dff1fd3e5f1ff`.
Submission followed independent source review. The allocation requested one
ws-ia GPU, eight CPU threads, 24 GiB host memory and one hour. Any full-batch OOM
is reported as a failure; there is no implicit smaller-model or precision
fallback.

This profile does not authorize a training campaign or consume test/reserved
examples. A subsequent study needs its own frozen data/source/configuration,
checkpoint-selection and complete-campaign reporting contract. Already revealed
reserved recordings can support a transparently posthoc external comparison;
they cannot become a fresh confirmatory population again.
