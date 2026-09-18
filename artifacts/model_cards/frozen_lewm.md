# Frozen LeWorldModel baseline: {{ENVIRONMENT}}

This package contains the released upstream LeWorldModel weights, loaded with
strict state-dictionary matching and converted to the common ShiftWM interface.
**It has not been fine-tuned by this project.** Packaging does not establish an
evaluation result. Source checkpoint, action-statistic SHA256, and pinned source
revision are recorded in `config.json` and `export_manifest.json`.

The architecture uses the existing LeWM visual encoder, action encoder,
autoregressive predictor, and projection modules. Frozen mode bypasses all
learned context adapters. The common container includes inactive ablation modules
and a reference encoder for training compatibility; these do not affect output.

## Inputs and supported use

- Images: RGB floats in `[0,1]`, shape `[batch, history, 3, height, width]`.
  The model applies ImageNet normalization and resizes to 224 internally.
- History: three observation frames, with two *executed* action blocks.
- Actions: each block concatenates five chronological two-dimensional controls;
  shape `[batch, blocks, 10]`. Raw values are normalized inside the model using
  verified official training-data statistics saved as buffers.
- PushT: relative pusher commands, `target = current_position + 100 * action`.
  Do not substitute absolute pixel target coordinates.
- Reacher: normalized torque; the upstream environment repeats each command twice.
- Goal images: visually specify a desired state in the same environment family.

This is not a general-purpose robot, medical, video-generation, or language model.
No transfer to a different action interface, environment family, or clinical use
has been established. Matched shifted-environment evaluation results, once
available, belong to this project and are not upstream paper-score reproductions.

## Loading and inference

```python
import torch
from shiftwm.checkpoint import load_package

model, metadata = load_package("path/to/package", device="cuda")
model.eval()
with torch.inference_mode():
    # images: [B,3,3,H,W]; past_actions: [B,2,10]
    history_features = model.encode_images(images)
    contexts = model.infer_context(history_features, past_actions)
    # future_actions: [B,K,10], strictly after the last observed frame
    future_latents = model.rollout_features(
        history_features, past_actions, future_actions, contexts=contexts)
    # goal_images: [B,3,H,W]
    goal_latents = model.goal_embedding(goal_images, contexts[0])
    candidate_cost = (future_latents[:, -1] - goal_latents).square().mean(-1)
```

The loader, strict-future action convention, adapted-goal API, immutable reference
targets, and checkpoint round trips are covered by tests. Placeholder variables
above must be supplied with real observations/actions; this is an API example,
not an evaluation result. `model.pt` loads with PyTorch `weights_only=True`.

## Attribution and license

Base weights: `quentinll/lewm-pusht` or `quentinll/lewm-reacher`, as recorded in
provenance. Source: https://github.com/lucas-maes/le-wm . Retain upstream notices
and verify model-weight distribution terms before public redistribution. This
card grants no additional rights beyond the source artifacts.
