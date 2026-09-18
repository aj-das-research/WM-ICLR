# Matched qualitative prediction diagnostic

This is an additional post-hoc development analysis of the original checkpoints, without training or rerunning CEM. The exact-replay job `200037` completed on one GPU. Both checkpoints predict the same observed support and the same next executed action block separately on each saved behavior trace. The target uses the actual next canonical observation in a tensor-verified common frozen feature space.

All 13 outcome-discordant development tasks are retained: four cases where ShiftWM succeeds and Framewise fails, and nine opposite cases. Both realized controller traces are measured, yielding 70 paired transitions (140 model predictions). Eight requested anchors are excluded because a full next five-native-action block does not exist before that trace stops. No terminal padding or invented future is used. Three observed frames and two executed action blocks infer context; anchors are native times 10, 15 and 20. Inputs use the original floating-point appearance transform; archived quantized RGB is used only to verify replay.

All 343 source hashes and all 23 original scientific source/configuration hashes match. Exported feature vectors reproduce all 140 FP32 prediction MSEs. The diagnostic checks exact tensor equality of both checkpoints' frozen reference and base encoders/projectors, action/pixel normalization, and backbone configuration. Selected original epochs are PushT 4/4 and Reacher 30/29 for ShiftWM/Framewise.

| Environment | Selected outcome category | Paired transitions | Ours lower MSE | Ours mean MSE | Framewise mean MSE |
|---|---|---:|---:|---:|---:|
| pusht | ours_only | 5 | 0 | 0.066160 | 0.049173 |
| pusht | baseline_only | 6 | 4 | 0.183053 | 0.197327 |
| reacher | ours_only | 15 | 6 | 0.006422 | 0.005934 |
| reacher | baseline_only | 44 | 18 | 0.006419 | 0.006126 |

These are descriptive transition counts and means within an outcome-selected development sample, with correlated observations; they are not a representative benchmark success rate or significance test. The input to the two predictors is matched within each row. Different realized behavior traces are not identical counterfactuals.

For the highlighted PushT success (seed 2031024), ShiftWM has higher one-block MSE in all five valid matched comparisons. For the highlighted Reacher success (2031004), it has lower MSE in three of five comparisons: both transitions on its own trace and the first transition on Framewise's trace. Support feature calibration remains worse for ShiftWM in both highlighted cases. These mixed observations rule out a uniformly favorable component explanation; lower local prediction error is not established as the reason for the planning win. The analysis does not test CEM candidate rankings, remove context, identify physical factors, or provide decoded image predictions.

Machine-readable predictions: `results/qualitative_diagnostics/prediction_diagnostics.json`. Independent numerical/source audit: `reports/evidence/qualitative_prediction_validation.json`. Exact simulator replay and per-step physical criteria are in the separate replay diagnostic.
