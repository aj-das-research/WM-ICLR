# Adapted official DINO-WM: synthetic resource profile

This isolated namespace prepares a potential external comparator. It does not
train a benchmark model, evaluate any recording or modify any frozen study.
The only optimizer steps use random synthetic features and actions, and all
temporary weights are discarded. No numerical synthetic loss is reported as an
accuracy result. No trained checkpoint is produced.

## Source and computation

The official [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) repository
was verified and cloned on 20 September 2026 at
`0a9492fa12044b852ae9e001cc74604b79c8bb0c`; its MIT license remains intact in
`external/official-dino-wm/LICENSE`. The clean checkout is not edited. This is
the authors' implementation, distinct from the already available framework
PreJEPA implementation in `external/stable-worldmodel`.

`adapter.py` imports the hash-checked official `models/vit.py` and
`models/proprio.py`. It retains the official depth-six transformer, sixteen
attention heads of width 64, MLP width 2,048, dropout 0.1, learned positional
embeddings, direct feature prediction, frame-causal full spatial attention and
kernel-one convolutional action embedding (35 inputs, ten outputs). There is no
ShiftWM residual head, anchoring, spatial mixing, GRU or FiLM in this comparator.

One explicit compatibility transformation replaces the hard-coded plain CUDA
attention mask with the identical nonpersistent registered buffer. Mask values,
attention equations and predictor parameter/state-dictionary keys are unchanged;
this permits CPU contract tests and correct device movement. The original
checkout retains its exact bytes. The mask replacement is checked for an exact
single source occurrence before execution.

## Adaptations that must appear in any eventual comparison

The adapter uses the existing DROID contract: channel-major 384×4×4 frozen
DINOv2-small features, three observations, two past and ten future action blocks,
each concatenating five seven-dimensional native commands. The resulting
predictor dimension is 384 visual + ten action = 394, with 48 tokens per call.
Proprioception is absent, not replaced by privileged state or a supervised dummy
target. Action embeddings are concatenated to every patch exactly as in the
official channel-concatenation path; predicted action channels are discarded and
each subsequent call receives the actual next supplied action block.

Our sixteen tokens are pooled from an encoder grid, not sixteen native image
patches. The official 224-image configuration instead resizes to 196 pixels
before patch-14 encoding and predicts 14×14=196 visual tokens. Its raw
representation/preprocessing and optional proprioception differ. Calling this
adapted predictor an unchanged published benchmark reproduction would be wrong.

Two separately labeled objectives are profiled:

| Profile objective | Transformer calls/window | Scored visual targets | Meaning |
|---|---:|---|---|
| `official_one_step_shifted` | 1 | Grid indices 1, 2, 3: all three shifted slots | Preserves the official one-step visual objective after the explicit feature/input adaptation; two targets are within observed support and one is future. |
| `matched_recursive_h10` | 10 | Grid indices 3–12: ten future grids | Deliberate alternative matching our recursive H10 training loss; full backpropagation through earlier predictions. |

Both consume the same synthetic thirteen-grid/twelve-action window format; the
native objective ignores the unused later grids/actions. Their supervised
element counts and computation denominators are different and recorded.
Both receive a separate FP32 ten-step recursive inference measurement. The
native objective is the first candidate for an externally authored baseline;
the matched-loss alternative requires its own declared arm if later trained.

The profile uses synthetic zero/unit normalization, never fitted statistics.
A future study would reuse the exact existing training-only per-channel and
action statistics, train/development windows and split IDs. Proposed matched
budget: 30 epochs, seeds 0/1/2, batch128, AdamW 1e-4/weight decay0.01, cosine
decay to1e-6, gradient norm1 and BF16 training. These differ from the official
100-epoch/5e-4 recipe and must remain explicit. Checkpoint selection would use
the common FP32 window-weighted recursive MSE across all ten future grids;
final development reporting would equally weight episodes and seeds, with the
existing paired session/seed analysis. Those training/evaluation adapters and
their complete-campaign gate are not implemented or authorized by this profile.

## Run and review boundary

```bash
.venv/bin/python -m pytest -q scripts/external_dinowm_profile_v1/test_contract.py
.venv/bin/python scripts/external_dinowm_profile_v1/profile_cuda.py --prepare-registration
# Independent review creates reports/external_dinowm_profile_v1/source_review.json
# with status=passed, exact registration_sha256 and exact source_sha256 map.
sbatch scripts/external_dinowm_profile_v1/run.slurm
```

Do not submit before independent source review. The script verifies the review,
specification, dependency hashes and upstream revision before constructing a
CUDA model. It repeats verification after profiling. One ws-ia GPU, eight CPU
threads, 24 GiB host RAM and one hour are requested. Full batch128 is mandatory;
OOM is a recorded failure, not permission to silently shrink the architecture,
batch or precision. Each objective uses two warmups and five timed BF16
forward/backward/clipping/AdamW steps, followed by one warmup and three timed
FP32 H10 inference calls. Timings synchronize CUDA and record every raw sample,
peak allocated/reserved CUDA memory, parameter counts and execution identity.
TF32 is disabled. CPU peak RSS includes interpreter/import overhead. Validation
memory includes resident AdamW state and earlier allocator caches, as during an
epoch; it is not a standalone inference-memory claim.

The profile reads only code/configuration and existing count-report metadata;
it has no dataset/checkpoint loader. The existing report records 18,660
training and 1,631 development windows. These are the common H10-eligible
windows, including for the native objective's projection; they are not its
potentially larger shorter-window population. Projection charges 146 training and
13 validation full batches per epoch, including partial last batches, and
reports each objective separately. These estimates exclude feature extraction,
loading/transfers, checkpoint writes, final evaluation, queue and contention.
They are predictor-only compute estimates, not end-to-end training guarantees,
RGB quality or control measurements.

Before any actual study, freeze separate source, configuration, selection and
evaluation contracts using training/development data only. Existing reserved
outcomes have already been revealed: a later evaluation there is a posthoc
external comparison, not fresh confirmatory evidence. No such evaluation or
new training is launched by this namespace.
