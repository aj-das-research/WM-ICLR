# DROID metric completion: operational recovery v2

The v1 study registered 34 full evaluations. Twenty-two completed; twelve
mixing-family runs stopped at an exact CUDA first-batch checkpoint-reload check
before full scoring. The finalizer refused aggregation. Failures exhibited
prediction differences of a few FP32 ulps (logged maxima around 3e-6), not a
reported old-MSE parity failure. All v1 sources, logs and successful outputs are
retained unchanged and cryptographically bound by this new registration.

This operational revision reruns all 34 registered rows. It reuses no v1 metric
row. Models, parameters, package loaders, windows, normalization, original batch
sizes, FP32 CUDA inference, disabled TF32/autocast, metric formulas, aggregation,
bootstrap and the original per-window MSE tolerance (rtol2e-5, atol2e-6) remain
unchanged. The original primary model/metric/split and secondary status remain.

Only the serialization diagnostic changes: verify every parameter/buffer exactly
between the active GPU model and two official-loader CPU instances, then require
exact CPU prediction equality on the first two registered input windows, using
one CPU thread. CPU outputs are never substituted into the scored CUDA forecasts.
This is a serialization/reproducibility check, not a changed scoring backend.

Additionally, on the full first registered batch, run the same CUDA model again
and an independently loaded CUDA model. Record exactness, number of differing
elements, maximum absolute prediction differences and maximum per-window MSE
differences. These bounded repeatability diagnostics have no equality threshold;
they do not replace the unchanged old-ledger MSE gate on every scored window.
Nonfinite diagnostic outputs fail rather than being hidden. No precise library
root cause is asserted solely from observing small CUDA differences.

After all 34 reruns pass, use the same complete-only finalizer and paired
session×seed bootstrap as v1. Publish the recovery and retain its adverse outcomes.
The new output namespace is `reports/metrics_completion_v1/droid/recovery_v2/`.
The root agent controls all GPU submissions; no additional jobs are auto-created.
