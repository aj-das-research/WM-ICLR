# Manuscript and experiment handoff

Updated 2026-09-19 (Asia/Dubai). Research draft; no conference submission. GitHub, the project page, and Overleaf have been published; the latest calibration appendix source is ready for the next synchronized build.

## Completed work

| Study | Completed work | Interpretation |
|---|---|---|
| Original PushT/Reacher | 30 original training runs and three-seed forecast/planning comparisons | Held-out forecast improvements, mixed planning; no established planning superiority |
| Dynamics-residual revision | Two 30-epoch development runs | No improvement over donor across both environments |
| Rollout/context revision | Eight 30-epoch development arms and planning evaluations | Registered promotion criterion not met |
| Drone/tissue simulator extensions | 36 original and six observation-gain runs, all 30 epochs, forecasting and planning | Positive and negative development outcomes retained; final tests remain unrun |
| Real DROID | 12 full 30-epoch runs and 48 evaluations | Small short-horizon gains; long-horizon regressions; separate from physical control |
| Real DROID development diagnosis | All 12 best models, three late checkpoints, train/validation motion and command-sensitivity probes | Real command dependence exists; late overfitting and excessive predicted motion remain |
| Matched residual calibration | 12 train-fitted scalar wrappers, h5/h10 validation comparisons, 24 exact offline/relocated reload checks | Modest validation improvement; no new test result or neural training |

DROID ingestion downloaded and audited 1,126 actual robot episodes (22,017,792,821 raw download bytes), with 851 train / 143 validation / 132 test episodes separated by recording session. The model uses recorded commands and frozen DINO features; no synthetic image corruption is applied. Data are a prespecified subset, not the full published DROID benchmark.

## Real-video findings

| Population | Final-horizon MSE: ours | Framewise | Relative error reduction | Paired difference CI |
|---|---:|---:|---:|---|
| Camera 1, 5 blocks | 0.138198 | 0.138479 | +0.203% | [-0.001038, +0.000413] |
| Camera 1, 10 blocks | 0.193525 | 0.189895 | -1.912% | [-0.000387, +0.007493] |
| Camera 2, 5 blocks | 0.154190 | 0.154337 | +0.095% | [-0.000830, +0.000644] |
| Camera 2, 10 blocks | 0.222748 | 0.220020 | -1.239% | [-0.000741, +0.006195] |

All four final-horizon intervals versus Framewise include zero. Primary five-block error is 2.94% lower than persistence, but only 0.20% lower than Framewise. Early one-block differences have unadjusted intervals below zero; they are small and do not establish a broad advantage. All selected checkpoints come from validation, not test. Late overfitting and low sensitivity to reversed command order require explicit discussion.

## Separate validation development findings

The completed diagnostic reads training and validation recordings only. Substituting future commands from another recording session increases ours' validation h10 error by 5.17%, while the action-free control does not change. Reversal alone therefore understates action sensitivity. These perturbations are not physically executed counterfactuals.

All 12 frozen best models then received the same least-squares residual calibration, fitted on all 830 horizon-eligible training episodes and 7,721 windows. The remaining short training recordings remain in the audited manifest. All 851 train and 143 validation payloads, source files, protocol and checkpoints were verified before and after execution. No original test payload was opened.

| Validation comparison | h5 | h10 |
|---|---:|---:|
| Ours, before calibration | 0.158398 | 0.219507 |
| Ours, calibrated | 0.157704 | 0.215874 |
| Framewise, also calibrated | 0.158942 | 0.216206 |
| Relative reduction versus calibrated Framewise | +0.779% | +0.154% |
| Paired validation MSE-difference interval | [-0.002173, -0.000470] | [-0.001395, +0.001022] |

These are exploratory validation intervals, unadjusted for multiple comparisons; h10 is inconclusive. The original held-out table above remains authoritative for the completed test. Calibration is a standard control, not a newly established methodological contribution. Full results: [calibration report](real_droid_residual_calibration_results.md), [machine-readable results](real_droid_residual_calibration_results.json), and [reusable-wrapper model card](model_cards/real_droid_residual_calibration.md). Twelve compact configurations are in `configs/real_video_development/calibrations`; both original-location and portable-release relocation checks pass exactly on CPU. The paper source now includes a clearly labeled development paragraph, formula and generated table; rendered-page review is pending the root build.

## Local artifacts and verification

- [Results](real_droid_results.md) and [machine-readable ledger](real_droid_results.json): all 12 runs and 48 evaluations validated.
- [Reusable release](../artifacts/releases/real_droid_v1/README.md): 12 validation-selected latent predictors, one shared encoder, 330,577,554 inventoried bytes, and 207 hashed files. All 12 isolated offline CPU prediction checks passed with exact parity. Original run folders retain optimizer/RNG states.
- [Simulator extension report](completed_extension_results.md): all 42 runs audited; the official AdaJEPA reproduction is reported separately because its inputs/checkpoints/budget differ.
- [Current paper](../paper/world_model_draft.pdf), real-data tables and source hashes are generated locally. The real-video comparison is integrated as Figure19 on page43; real-data tables18/19 appear on page44. Compiled pages were visually inspected.
- [Actual DROID footage](../data/real_video/droid_selected/processed/preview/recorded_three_camera_episode.mp4): playback assembled from recorded frames; display rate is not calibrated physical time.

## Important decisions and remaining work

- Keep original test outcomes and post-hoc development studies separate; preserve failures, uncertainty and method naming with `(ours)`.
- Frozen real-data training, feature, evaluation, configuration and protocol sources were not changed during the campaign.
- Real videos establish forecasting evidence on recorded observations. They do not establish physical closed-loop robot success, patient benefit, identified dynamics factors or state of the art.
- The next scientific priority is stronger generalization and action-dependent prediction. Use training/validation diagnostics to develop a revision and reserve fresh held-out sessions for any confirmatory claim; do not tune to this completed test.
- Fresh recording-session-disjoint holdout preparation is pending. No new test data, results or confirmatory improvement are claimed for the calibration wrappers.
- Simulator extension final tests, matched independent published-method comparisons, and additional real domains remain unfinished. Open-H is only an audited physical-phantom sample; Cholec80/SWoMo/navigation sources were researched, not trained in this study.
- [GitHub repository](https://github.com/aj-das-research/WM-ICLR) publication is verified at commit `f5596711695374fbc1b9fc162ca16b2add06a2e1`; the [live project page](https://aj-das-research.github.io/WM-ICLR/) and its assets were verified against local hashes. The live demo plays actual recorded footage and displays audited results; it is not online model inference. [Overleaf](https://www.overleaf.com/project/6aadbb24b37acd9be4eed157) synchronization is complete for the previously compiled paper. Subsequent local changes, including this calibration appendix, await the next verified synchronization. Credentials remain in private stores outside the repository. Neural checkpoints remain local unless a separate release upload succeeds. The user handles conference submission. Latest verified figure-skill revision is `436ea49b7c677210ed67cf1c44624db6ff6a3068`.

## Historical snapshot

The [previous manuscript status](history/manuscript_status_before_real_droid_completion.md) is preserved verbatim. Its in-progress job counts and page references are historical, not current.
