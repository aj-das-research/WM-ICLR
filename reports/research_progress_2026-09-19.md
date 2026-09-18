# Research progress — 19 September 2026

Current direction: resolve measured overfitting and long-horizon errors, then test whether a spatial predictor with causal adaptation improves matched forecasting. No large state-of-the-art advantage has been established. The original test and fresh-session confirmation remain immutable; all new development uses original train/validation only.

| Workstream | Completed / current work | Next evidence required |
|---|---|---|
| Original real-video study | 12 full training runs, 48 evaluations, 12 released predictors and 12 calibrated wrappers | Original small gains and long-horizon regressions retained |
| Fresh-session confirmation | 96 evaluations, 65 episodes / 52 new sessions; primary calibrated h5 gain0.742%, paired interval excludes0 | No further tuning on these revealed sessions |
| Technical qualitative figure | Figure20 integrated and reviewed; real frames, measured error curves and all65 episode gains | 46 improve /19 regress; no RGB prediction claim |
| Regularization/capacity control | 36 registered30-epoch runs; slower learning, stronger decay, smaller model | Complete all36 /72 evaluations before aggregate tables or model publication |
| Matched h10-training control | 12 full runs, same four original modes /3seeds; implemented and32 engineering checks passed | Registered run completion, matched-window comparison and offline reload |
| Causal reliability diagnostic | Uses13 observed frames to measure earlier forecast errors; donors fixed and fitting train-only | Compare against both donors, constant blends and shuffled/short-backtest controls; report failures |
| Spatial architecture revision | 4×4 tokens, observation-anchored transport plus bounded innovation; five matched arms /3seeds planned | Causal/grid/normalization/parity checks, runtime budget, full training and matched ablations |
| Additional real benchmark | Public IWS manipulation data download submitted as CPU job200194; 2.766GB pinned acquisition | Split/action/time audit; RLA-WM reproduction requires authorized DINOv3-L access and compatibility repair |

## Public artifacts and ongoing synchronization

- [Code and reports](https://github.com/aj-das-research/WM-ICLR)
- [Overleaf manuscript](https://www.overleaf.com/project/6aadbb24b37acd9be4eed157)
- [Interactive project page](https://aj-das-research.github.io/WM-ICLR/)
- [Verified model release](https://github.com/aj-das-research/WM-ICLR/releases/tag/real-droid-v1)

GitHub and Overleaf remote edits have passed an actual round-trip merge/publication test. The five-minute user timer waits for source edits to settle, then rebuilds the paper/site, checks the standalone Overleaf bundle, scans the public snapshot and verifies remote commits. Conflicts or changes to frozen scientific files stop imports. The live inference service uses actual released predictors on fixed attributed examples; its Cloudflare tunnel is temporary.

## Scientific decisions

The model already trains recursively for five steps and demonstrably uses recorded commands. The h10 control changes the supervised rollout horizon while preserving original checkpoint-selection weighting. Spatial transport must use common channel coordinates rather than mixing independently normalized patch positions. All spatial arms receive the same higher-resolution features; common2×2-coordinate scores are supplementary cross-resolution diagnostics. The reliability study's longer observed prefix defines a separate matched information budget. None of these changes is promoted on the basis of isolated favorable examples.

Related diagnosis: [architecture evidence](real_video_development/architecture_diagnosis.md), [causal adaptation review](real_video_development/algorithm_review.md), [external real benchmark audit](real_video_development/rla_wm_iws_audit.md). Exact live run counts are recorded separately from this durable workstream summary.
