# No-tanh local inference release

All nine selected IWS no-tanh component-ablation predictors are exported to
`artifacts/releases/iws_unbounded_single_observation_local_v1` (74,780,289 bytes).
Each task (PushT, Bimanual Box, Bimanual Rope) includes seeds 0/1/2; every selected
checkpoint is epoch 30 after a complete 30-epoch run. Manifest SHA256:
`906912e301d6abd653f6eb941b23121f18295562cb1ddd4235e2dd6c021b1dec`.

All nine relocated CPU FP32 predictions match the original selected packages
exactly at every one of 59 offsets (`[1,59,6144]`, two CPU threads). The isolated
process made zero network attempts and zero original-workspace reads, with only
installed Python environment libraries exempt from the workspace guard. Inputs
were fixed internal-training episode `000011`, frame 0 and command rows 0–59 for
each task. These are target-free portability fixtures, not an accuracy test.

The complete 9+27 development finalization, all 598 bound source/receipt files,
checkpoint identities, all four normalization buffers and the original 27-model
bundle inventory passed before/after checks. The original bundle is unchanged.
No reserved data, optimizer/RNG state, RGB encoder or decoder is included.

The new builder and runtime are in `scripts/real_video_iws_unbounded_release/`.
Its README gives the build, offline reload and API commands. Twenty-six boundary
and completion-gate tests passed, including partial/duplicate campaigns, stale
sources, wrong package kinds, unlisted files and custom-path escapes. The
unchanged frozen exporter validates full training completion and copies selected
model/config bytes exactly.

Receipts:

- `local_inference_export.json`: all selected epochs, source identities and bundle size.
- `relocated_cpu_parity.json`: nine exact comparisons and offline/import guards.
- `readiness_checks.json`: final file bindings, tests and immutable-source checks.
- `independent_source_review.json`: independent reviewer scope and findings, when present.

The first successful export is preserved under `initial_before_path_guard/` and
`artifacts/releases/iws_unbounded_initial_before_path_guard`. Independent review
then tightened custom output path containment and clarified that commands are
inputs; a clean final export repeated the exact parity proof.

This is a **local** feature-input bundle, separate from the existing 27-model
bundle. No checkpoints were published, no GPU or training job ran, and no
reserved payload was opened. Distribution of real dataset-derived parity
fixtures is not authorized by this local export.
