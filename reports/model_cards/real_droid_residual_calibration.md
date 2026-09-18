# Real DROID residual-calibration wrappers

Twelve small reusable JSON configurations wrap the original trained real-video predictors. They are a completed development control for excessive predicted motion. No additional neural-network weights were trained, and this is not claimed as a novel architecture or a state-of-the-art model.

## Intended use

Offline research on feature forecasting from recorded robot video. Input: three observed DINO feature vectors, two past action blocks, and the requested future action blocks. Output: predicted DINO features contracted toward the final observation by a fitted scalar alpha. This package does not synthesize RGB video, execute robot commands, or establish physical safety or counterfactual accuracy.

Each model retains its original validation-selected base checkpoint. Calibration uses 830 eligible training episodes and 7,721 windows from the audited original 851-episode training population. The scalar is fitted by constrained least squares over ten query steps, with equal weighting of episodes; no validation or test targets enter that fit. The same procedure is applied to Framewise, constant-dynamics, factorized ShiftWM, and action-free models across three seeds.

## Validation development results

All 12 wrappers and both horizons completed. ShiftWM's mean h5 feature error decreases from 0.158398 to 0.157704; h10 decreases from 0.219507 to 0.215874. Against **also-calibrated Framewise**, the relative reduction is **0.779% at h5** and **0.154% at h10**. The exploratory session/seed paired interval excludes zero for h5 and includes zero for h10. These are validation findings, not fresh test results. The earlier inspected test set was not read or re-evaluated.

| ShiftWM seed | Fitted alpha |
|---:|---:|
| 0 | 0.9076853228 |
| 1 | 0.8758775455 |
| 2 | 0.9866014900 |

All methods, seeds, scalar values, comparisons, uncertainties, and validation population counts are in `reports/real_droid_residual_calibration_results.{md,json}`. Do not use the stronger 64-episode pilot improvement as the completed campaign result.

## Loading and relocation

From the repository root, add `scripts/real_video_development` to Python's module search path and call:

```python
from calibrate_residual import load_calibrated_package

model = load_calibrated_package(
    "configs/real_video_development/calibrations/droid_factorized_s0.json",
    device="cpu",
    base_checkpoint="/path/to/real_droid_v1/models/droid_factorized_s0",
)
prediction = model.predict(support_features, support_actions, future_actions)
```

Omit `base_checkpoint` when the original release is installed at the relative location recorded by the JSON. Moving a base checkpoint is supported through the override; changing its bytes is rejected. The package also verifies wrapper source and the scalar's training-moment identity. Loading is offline and does not need training data. Twelve campaign reloads passed exact CPU parity on four validation windows; the public JSONs also underwent exact parity after relocation to the existing portable base release.

## Provenance and limitations

`configs/real_video_development/residual_calibration_v1.json` freezes the protocol, scientific source dependencies, selected base checkpoint hashes, and all 851 train/143 validation payload hashes before execution. Those bytes were verified before and after the campaign. Inference used FP32 with autocast and CUDA TF32 disabled; fitting moments used FP64. The scalar configuration is small and reusable, but depends on the separately supplied original base-model weights and frozen DINO feature encoder.

Calibration is a standard least-squares contraction around persistence. It does not repair the observed late overfitting or establish method-level novelty. Confirmatory testing requires a newly registered, untouched recording-session-disjoint population, with method selection and inclusion criteria fixed beforehand. Upstream DINO, LeWM, and DROID attribution/license requirements remain as recorded in the original release; this wrapper does not change their licenses.
