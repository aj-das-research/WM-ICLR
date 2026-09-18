# Frozen-donor rollout study: development protocol

Specified 2026-09-18, before any run in `runs/rollout_revision`.

The original ShiftWM campaign is unchanged. Its planning results do not establish superiority over Framewise calibration. A previous support-conditioned action adapter tied its donor on PushT and regressed on Reacher. This follow-up tests a training/inference mismatch; it is not a confirmed improvement or a matched total-training-budget comparison.

## Hypotheses and factorial controls

Eight runs: PushT and Reacher × teacher-forced and recursive prediction × inferred and constant-input context; training seed 0. Every run starts from its completed seed-zero Framewise donor. All donor weights, visual/goal calibration, buffers, dropout and batch statistics remain frozen. Only the new context network and zero-initialized residual action FiLM are trained.

Each complete eight-frame training window has observed support frames 0,1,2 and executed action blocks 0,1. Both objectives predict the same five canonical targets, frames 3–7, with action blocks 2–6. Teacher forcing uses observed rolling three-frame windows. Recursive training feeds its own latent predictions back through all five steps, without detaching. Query observations and targets never enter context inference. This corrects the missing first query target in the earlier adapter for **both** objectives, isolating recursion within the 2×2 study.

Inferred context receives calibrated support features and normalized executed actions. The constant-input control feeds zero tensors to the identical context network; its biases can learn a shared context. This removes episode information while retaining modules and nominal trainable parameter count. It does **not** equalize effective functional capacity: input weights receiving zeros cannot acquire information-dependent gradients.

## Data, training and model selection

Reuse the complete established training and validation splits, frozen visual feature caches, action statistics, donor hyperparameters and 30 epochs. No subsampling, early stopping or adaptive search over these eight conditions. Training uses the existing BF16 setting; validation always measures FP32 five-step recursive MSE from the observed support, over **all five query targets**, for every condition. Select the minimum common recursive validation loss across the 30 completed epochs. The old one-step validation number is not directly comparable.

Configs live in `configs/rollout_revision`. Portable best and last packages embed the full frozen donor and residual model; optimizer, scheduler, sampler and random-state checkpoints support exact continuation. Source, donor, manifest and cache hashes are checked. The campaign freezes eight config hashes and scientific source hashes in `runs/rollout_revision/campaign_state/protocol_identity.json` at first launch. A changed protocol must use a new study namespace.

## Development planning and paired comparisons

Evaluate every completed validation-selected arm on the same established 32 development tasks per environment, with identical initial support, planner random numbers, shifted goal input and native action budget. Use the fixed existing development protocol: horizon 5, 300 candidates, 30 CEM iterations, 30 elites, 50 native actions, 5-action blocks, planner seed 1701. Report raw successes and initially-unsuccessful eligible successes (PushT denominator 31, Reacher 30), paired wins/losses, and eligible success-rate differences versus the unchanged Framewise donor. Preserve partial progress, never present it as a completed result.

Compare inferred versus constant context within each objective and recursive versus teacher-forced training within each context condition. The principal diagnostic is whether episode information improves planning after objective matching. Report all eight arms, including negative results. Do not tune configurations from these planning outcomes within this study. Bold green values denote positive observed mean differences only; they do not imply statistical significance.

## Decision gates and paper scope

A candidate merits a subsequent study only if its inferred-context arm exceeds both its corresponding constant-input control and its frozen donor on eligible development planning success in both environments. That is a screening criterion, not proof: 32 reused development tasks and one training seed cannot establish a general effect. Any follow-up must verify multiple training seeds and use a newly specified untouched final evaluation set, since the original test results have already informed development. Cost/latency must include the extra training and support actions.

If the screen fails, retain the null/negative result and investigate failures before increasing claims or adding more search. Keep the main architecture figure tied to the established method until a revision has evidence. This study alone cannot support universal model/task compatibility, medical performance, superiority to independent published systems, or an acceptance prediction.

## Execution

Use one resumable campaign allocation per environment, four sequential arms each, in parallel across at most two authorized workstation GPUs. Preserve the original campaign's allocations and priority. Save task states, allocation IDs, logs and checkpoints locally. No project publication is authorized at this stage. Full training and development evaluation, rather than short pilot runs, are the research outputs; CPU correctness fixtures remain separate from research measurements.
