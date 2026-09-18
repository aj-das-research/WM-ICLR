# Established-code reuse and correctness plan

Status: source inspection and planning, no algorithm implementation. Eleven implementation repositories and one release placeholder are pinned in `references/checkouts.json`. A cloned repository is not a reproduced result.

## Layering

1. Preserve upstream checkouts and commands. Run official baselines first in isolated environments. Keep compatibility fixes in explicit patch files, separate from scientific changes.
2. Add a thin task adapter exposing prediction, proposed update, snapshot/restore, and paired unlabeled score. The controller owns timing and deployment; the host retains preprocessing, tokenizer, prompts, model loading and adaptation internals.
3. Keep labels in an evaluator process/module outside the adaptation interface. Record active-state predictions before probe-dependent decisions, and never replace them retroactively.
4. Reuse expensive frozen encoder features across compatible cache methods, keyed by model and preprocessing hashes. Gradient methods require fresh forwards. A dense feature cache must preserve alignment and image IDs.
5. Use separate single-GPU jobs for datasets/seeds/hosts. A chronological stateful stream runs on one worker. Job checkpoint includes both states, optimizer if applicable, RNG, stream cursor, config and checkpoint hashes.

## Preferred implementation foundations

| Component | Existing source | Reuse strategy |
|---|---|---|
| Classification cache host | TDA, MIT | Thin wrapper around official cache; smallest primary integration |
| Current classification comparisons | PTA, D2O, LTTA, BITTA | Run upstream configurations; inspect licenses before copying code |
| Medical data and corruptions | Histopath-C, MIT; TTAMedVLMs, MIT | Retain Quilt loader, transforms and corruption definitions; distinguish LATTE transductive protocol |
| Segmentation host | MLMP, MIT | Reuse supported CLIP dense prediction and adaptation; snapshot trainable state and align augmented outputs |
| Evaluation cross-check | tta-vlm | Inspect harmonized protocols; do not assume its code can be relicensed |
| Statistical baseline | StatA, AGPL-3.0 | Preserve license obligations; standalone upstream runs simplify provenance |
| Selective-adaptation comparator | CAS repository | Currently contains no implementation. Clearly label any paper-based reimplementation and record ambiguities. |

## Necessary correctness checks

- Shuffle evaluator labels: adaptation states and predictions must be unchanged.
- Candidate-generation IDs and probe IDs must be disjoint within a round; prevent repeated-image identity leakage.
- Freeze active/candidate state during paired probing; score order must not mutate either state.
- Snapshot/restore includes buffers, caches, prompts and optimizer state where applicable; output equivalence before/after restore.
- Resuming a checkpoint must reproduce uninterrupted predictions and gate decisions.
- Zero accepted updates reproduces the frozen/current active host; unconditional acceptance with zero delay reproduces the declared host protocol where mathematically equivalent.
- Segmentation inverse transforms restore correspondence; mask invalid pixels; aggregate per image.
- Metric accumulation agrees with a small manually checked example; no calibration-bin or class-mapping mismatch.

These tests protect scientific validity rather than merely mirror functions. Only after official smoke reproduction and these checks do we run the main grid. Keep dependency locks per upstream until compatibility is demonstrated; avoid upgrading every research repository into one environment.

## Release

New controller, manifests, scripts, configs and tests go in our repository with explicit upstream attribution. Base weights and datasets remain downloads unless redistribution is allowed. Build anonymous supplementary materials separately from public GitHub, project page and Hugging Face Space. All builds run on the cluster; public destinations and authentication remain to be verified. A CPU replay demo must identify itself as replay; a live demo must report its actual inference limits.
