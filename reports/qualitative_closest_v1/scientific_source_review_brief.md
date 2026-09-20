# Independent qualitative replay review brief

Status: protocol guidance and existing-interface review complete; new replay source and frozen selection audit still require review before execution. This document does not clear a replay or claim new results.

## Comparator and case selection

Select the comparator once from a complete, source-bound study/task table using its declared primary metric and horizon. Distinguish the overall runner-up from the strongest non-ours baseline: an ablation can be the former without being an independent baseline. Do not select a different comparator for each case, seed, horizon or patch. An unfinished raw-v2 external campaign cannot supply a candidate mean or comparator ranking.

Freeze the eligible methods, finalized report hashes, task population, primary endpoint, tie-break and deterministic case rule before replay. A largest/median/smallest episode-gain panel is a gain-conditioned illustration, not a representative sample or population estimate. Retain the unfavorable case and disclose when the fixed first eligible window has the opposite sign from its episode-average selection score. Preserve all three training seeds and their individual errors; do not select the best checkpoint seed or average predictions before computing errors.

## Matched evidence and maps

Both methods must use identical episode/session, window start, observed frames, native command rows, target frame, preprocessing, frozen normalization and selected checkpoints. Use the same full field of view or one explicitly recorded common crop. Keep the same linear error-map scale across methods and displayed cases; no per-method normalization, percentile clipping or undisclosed log transform. Recompute per-patch squared error in the registered standardized feature coordinates, average channels, then seeds. Check that the mean over all 16 patches reproduces the scalar feature MSE. Preserve native frame indices and distinguish DROID's five-command blocks from IWS native command rows and offsets 1–59.

The existing models expose `transport`, `gate` and `innovation` through `return_details=True`. Their exact operator is

`M = (1 - g) I + g T`, followed by `prediction = M @ observed_anchor + innovation`.

These are learned source-mixing weights in semantic-feature coordinates. They are not image saliency, optical flow, measured physical correspondence, or proof that mixing caused a performance gain. The separate `spatial_attention` encoder currently runs with `need_weights=False`; a transfer matrix must not be relabeled as that module's attention. Any new attention hook needs its layer/head/query definition, weight/output parity check and separate interpretation.

Show a fixed destination-patch row of T or M on the source grid, or explicitly label diagonal maps as own-location weights. A diagonal map does not show the full transfer distribution. Compute each seed's M before averaging: `mean(M)` generally differs from `(1-mean(g)) I + mean(g) mean(T)`. The three-seed mean of per-seed error maps also differs from the error of the mean forecast. A gate is a learned mixture coefficient, not confidence. Strong diagonal retention partly follows by construction and is not an accuracy result. Preserve unbounded-innovation values without applying a bounded display interpretation to the no-tanh ablation.

## Input access and reproducibility

Replay must be inference-only with the exact finalized selected weights and evaluation backend, including the reviewed row-wise GRU for reserved IWS recovery-v2. Check raw command/index identities before model calls; targets are scorer-only. Validate ordinary prediction against detail extraction and reconstruct every saved forecast from its actual per-seed tensors. Bind source code, registrations, finalization/review receipts, checkpoint/config hashes, source inputs and derived arrays. Save a completion manifest only after all cases, methods and seeds pass. No training, checkpoint reselection, changed tolerance or partial table ingestion is part of this review.

The earlier DROID implementation provides a useful reference: `scripts/real_video_spatial_qualitative/{selection,replay}.py` preserves largest/median/smallest episode gains, the first eligible window, all three seeds, a shared linear MSE scale and fixed target patch (1,1). Its old AR comparator and development scope must not silently carry into a new closest-comparator protocol.

## Dataset distribution

The [DROID paper, Section III](https://arxiv.org/html/2403.12945v2) explicitly releases the full dataset under CC BY 4.0. Retain attribution to the DROID contributors, dataset/source link, license notice and the description of extraction/cropping/annotations. The existing `paper/figure_sources/droid_forecast_setup/LICENSE-DROID.txt` and attribution metadata are reusable references.

The [RLA-WM dataset card](https://huggingface.co/datasets/xyzhang368/RLA-WM) supplies no dataset license field or redistribution grant in the text checked on 2026-09-20. The separate [model card](https://huggingface.co/xyzhang368/RLA-WM) labels its weights CC BY 4.0; that does not establish a license for genuine dataset RGB, command or feature archives. Existing local rights metadata agrees (`reports/real_video_iws/release/public_source_metadata.json`). Keep reusable genuine IWS input/target archives local and outside public source packs/releases. Attributed paper excerpts do not by themselves license redistribution of the underlying trajectories. This finding does not prohibit publishing the project's trained weights or source, which have separate provenance and licensing.

Only metadata, source files, existing documentation and public primary-source pages were inspected for this brief. No new replay, model inference, raw dataset/cache payload read or partial accuracy aggregation was performed.
