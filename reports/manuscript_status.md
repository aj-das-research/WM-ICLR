# Manuscript and experiment handoff

Updated 2026-09-18T22:31:05.253091+00:00. Research draft; no conference submission. GitHub and Overleaf publication are now explicitly authorized.

## Completed work

| Study | Completed work | Interpretation |
|---|---|---|
| Original PushT/Reacher | 30 original training runs and three-seed forecast/planning comparisons | Held-out forecast improvements, mixed planning; no established planning superiority |
| Dynamics-residual revision | Two 30-epoch development runs | No improvement over donor across both environments |
| Rollout/context revision | Eight 30-epoch development arms and planning evaluations | Registered promotion criterion not met |
| Drone/tissue simulator extensions | 36 original and six observation-gain runs, all 30 epochs, forecasting and planning | Positive and negative development outcomes retained; final tests remain unrun |
| Real DROID | 12 full 30-epoch runs and 48 evaluations | Small short-horizon gains; long-horizon regressions; separate from physical control |

DROID ingestion downloaded and audited 1,126 actual robot episodes (22,017,792,821 raw download bytes), with 851 train / 143 validation / 132 test episodes separated by recording session. The model uses recorded commands and frozen DINO features; no synthetic image corruption is applied. Data are a prespecified subset, not the full published DROID benchmark.

## Real-video findings

| Population | Final-horizon MSE: ours | Framewise | Relative error reduction | Paired difference CI |
|---|---:|---:|---:|---|
| Camera 1, 5 blocks | 0.138198 | 0.138479 | +0.203% | [-0.001038, +0.000413] |
| Camera 1, 10 blocks | 0.193525 | 0.189895 | -1.912% | [-0.000387, +0.007493] |
| Camera 2, 5 blocks | 0.154190 | 0.154337 | +0.095% | [-0.000830, +0.000644] |
| Camera 2, 10 blocks | 0.222748 | 0.220020 | -1.239% | [-0.000741, +0.006195] |

All four final-horizon intervals versus Framewise include zero. Primary five-block error is 2.94% lower than persistence, but only 0.20% lower than Framewise. Early one-block differences have unadjusted intervals below zero; they are small and do not establish a broad advantage. All selected checkpoints come from validation, not test. Late overfitting and low sensitivity to reversed command order require explicit discussion.

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
- Simulator extension final tests, matched independent published-method comparisons, and additional real domains remain unfinished. Open-H is only an audited physical-phantom sample; Cholec80/SWoMo/navigation sources were researched, not trained in this study.
- The user now authorizes code/project-page publication at `aj-das-research/WM-ICLR` and paper synchronization to their Overleaf project. Overleaf sync is complete; GitHub/site publication is being finalized. Checkpoints remain local unless a separate release upload succeeds. The user handles conference submission. Latest verified figure-skill revision is `436ea49b7c677210ed67cf1c44624db6ff6a3068`.

## Historical snapshot

The [previous manuscript status](history/manuscript_status_before_real_droid_completion.md) is preserved verbatim. Its in-progress job counts and page references are historical, not current.
