# Residual calibration: controlled development follow-up

Status: registered for a complete matched development campaign before execution. This is a simple diagnostic control, not a novelty claim or a replacement for the completed paper results.

## Evidence and hypothesis

Validation-only diagnosis found that factorized models predict larger displacement and are less accurate over ten blocks than matched Framewise models despite a small advantage over five blocks. Low-observed-motion validation windows are harmed relative to persistence; high-motion windows benefit. Full 30-epoch training overfits. Strong action substitutions change predictions and increase error, so weak temporal reversal is not evidence that the model ignores commands. On a deterministic 64-episode training subset, fitting a single residual scale of 0.80933 improved seed-0 validation error from 0.220366 to 0.212483 at h10, and from 0.156231 to 0.154773 at h5. These are development results, not test gains.

The controlled hypothesis is that some learned motion is overestimated. Apply one scalar to an existing predicted trajectory about the last observed feature vector, without changing the base model or its recursive computation: calibrated prediction = final observed support + alpha × (base prediction − final observed support).

## Locked development procedure

- Preserve all original 12 best checkpoints and their selection. There is no gradient training and no choice of late checkpoint.
- Apply the same calibration procedure separately to all four methods and three seeds; do not give ours an extra fitting opportunity.
- Fit one scalar alpha per checkpoint on **all eligible original training episodes**, camera 1, support length three, horizon ten, stride five. Average squared displacement and displacement/target inner products over dimensions and ten horizons, then over windows within each episode, then equally across eligible training episodes. Alpha is clip(mean alignment / mean displacement energy, 0, 1), with zero if displacement energy is zero. No validation targets are used for fitting the scalar.
- Evaluate all fitted models and their unmodified counterparts on validation only at h5 and h10 using the original metric and equal-episode aggregation. Report all seeds/methods and both horizons. Also retain persistence and action reversal diagnostics.
- Keep data, encoder, architecture, normalization, base checkpoint, and all inference settings identical. Only the final predicted residual is contracted. No spatial resolution change, loss change, or additional model fitting is mixed into this experiment.
- Write a reusable calibration JSON pinned to the base checkpoint, script, wrapper, protocol, and training payload hashes. This is a calibrated wrapper around an existing trained model, not a newly trained neural checkpoint.
- Before execution register the actual verified bytes of all train/validation cache payloads, selected model packages, original scientific dependencies, wrapper, fitting script, and this protocol. Verify sources/checkpoints before and after each model and all payloads before and after the campaign. Do not read any test payload. Require all 12 matched models for a completion report.
- Use FP32 inference with autocast and CUDA TF32 disabled, and FP64 sufficient statistics for scalar fitting. Reconstruct each calibration JSON offline on CPU with the explicit base-checkpoint path override used for relocation; require exact prediction parity on four fixed validation windows.

## Promotion and new confirmatory data

Use this as a development control. A useful result must reduce validation h10 error without materially worsening h5, and its effect must be checked across all three seeds and the same calibrated Framewise baseline. Do not infer method novelty from a positive result. The scalar can fail because training and unseen-session errors differ; report that outcome.

Before any confirmatory evaluation, create a separate frozen new-data protocol. Deterministically select additional official DROID shards and exclude every episode from any of the original 433 recording sessions using metadata before decoding images or calculating errors. Prove no identical episodes or session keys overlap, pin source generations and hashes, declare all exclusions, fix a primary camera and horizon and equal compute baselines, and select/freeze methods using training/validation only. Archive the new data manifest and final code/configuration hashes before opening its evaluation payloads. No confirmatory result may be claimed on the already revealed original test set.

This calibration is an initial control for excess residual magnitude. It does not establish a novel test-time adaptation algorithm, solve long-horizon physical prediction, or guarantee improved future results.
