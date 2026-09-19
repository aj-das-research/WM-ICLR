# Gated spatial development model prerelease

This publisher is separate from the simulator and original real-video releases.
It does not train, tune, add benchmark predictions, or change any registered
spatial scientific source. Publication is already authorized by the user; the
CPU job must depend on successful completion of spatial finalizer Slurm 200230.

- Scientific registration: `configs/real_video_spatial/v1/registration.json`,
  SHA256 `ae99340f7a3761ece066ee7376f4bd119b61a92503948c142196c40a1a76e337`.
- Exact population: autoregressive, anchored additive, transport, context off,
  and action free; seeds 0, 1, 2; all 15 runs complete 30 epochs. The selected
  epoch may precede epoch 30 and must be the first minimum of the unchanged
  window-weighted, FP32, all-ten-query validation criterion.
- Destination: `artifacts/publishing/spatial_v1`; archive
  `shiftwm-spatial-v1.tar.gz`; GitHub repository `aj-das-research/WM-ICLR`, tag
  `spatial-world-models-v1`, research prerelease. Exact bundled source hashes
  are authoritative even if the latest main commit has additional changes.
- Scheduling: one CPU-only job, 4 CPUs, 16 GiB, 2 hours, `afterok:200230`;
  no GPU requested and no new training or validation model-selection work.

## Fail-closed export gates

Before using finalization results, verify the exact scientific registration,
all registered configurations and scientific dependencies, and the separately
frozen exporter/runtime/tests/Slurm/protocol/third-party license identities.
The exporter freezes a copy of its own executable and executes that copy.
The shared publication scanner is a frozen dependency; later changes require
a separately reviewed registration rather than an unnoticed relaxation.

Recheck the cache and resource gate, native cache identity and manifest,
training-only shared-channel statistics, original normalization/cache inputs,
and pinned DINO source/provenance. The original timing gate requires at most
80 projected total GPU-hours and at most 20 hours per model. This projection
does not claim actual elapsed-time or contention guarantees.

For every run, call the unchanged completed-training validator. This checks
all 30 journal epochs, selected and last packages, optimizer/RNG resume-state
identity, precise selection arithmetic, configuration, and training/source
hashes. Optimizer/RNG state hashes are retained as local source evidence;
their payloads are not placed in the public bundle. Verify every finalized
inference package against its selected source checkpoint and old parity hash.

Reconstruct every per-window/per-episode metric through the frozen ledger
validator using the selected checkpoint. Preserve all 15 evaluation ledgers,
all 5 aggregate modes, and all 16 paired comparisons: transport versus each
of 4 controls, native 4×4 and exact-original 2×2 coordinates, horizons 5 and 10.
Recompute aggregate arithmetic and the paired session-and-seed bootstrap
using the frozen tested helper, 10,000 draws, seed 173. This is an arithmetic
recheck, not an independent implementation of the bootstrap. Negative and
inconclusive results do not block publication; missing or inconsistent
results do. These are exploratory original-validation results without
multiple-comparison adjustment, not new confirmatory test results.

## Portable inference and evidence

Each model package includes only its selected model tensors and config,
manifest, model card, and an engineering parity reference. The shared pinned
DINOv2-small encoder is stored once. Source, license notices, configurations,
training journals, complete ledgers, and hash provenance accompany the models.
Inputs are raw channel-major 384×4×4 DINO features, three observed frames,
two past command blocks and the causal future command sequence. Each command
block contains five chronological 7D recorded commands. Models return latent
features; they do not generate RGB video or a robot policy. Attention mixing
is not validated optical flow or physical correspondence.

Use the deterministic first eligible original-validation window only as an
API fixture. Serialize its three observed features and recorded commands;
do not serialize future observation targets or raw video. Recompute source
CPU predictions with the selected frozen models for exact array parity.
Physically copy the complete staged bundle to a new temporary root and run
a separate Python isolated process. Require all 15 relocated predictions to
match their source arrays exactly, with the correct selected epochs.

Offline verification uses HF/Transformers offline flags, local-only encoder
loading, and a Python socket connect/connect_ex/create_connection guard.
This is not OS network-namespace isolation. The synthetic zero-RGB encoder
check proves offline loading, finite outputs and shape. A unit test verifies
resize-then-normalize and channel-major pooling contracts with controlled
patch outputs. Neither establishes numerical CPU-FP32 equivalence to the
original GPU-BF16 feature cache.

Immediately before sealing, recheck all frozen dependencies and the separate
ledger of completion, selected/last checkpoint, journal, cache and evaluation
hashes. Reject secrets, symlinks, unsafe paths, hidden credential files,
optimizer states and incomplete inventories. Generate a deterministic archive
and stream-verify every member, byte hash and exact file inventory. The
project's new predictor weights/code retain MIT; DINO remains Apache 2.0,
LeWM remains MIT, and DROID-derived fixture/provenance retains CC-BY 4.0.
No full dataset or source video is redistributed.

## Publication and retry behavior

Create a draft prerelease only after all local gates pass. Verify each uploaded
asset's GitHub SHA256 digest and byte length before changing the draft to a
public prerelease. An existing public prerelease is immutable: missing,
different or extra assets cause failure. Download all public assets without
credentials and verify their hashes again. Save a publication receipt with
exact source registration, finalizer, archive and public-download identities.
Failed or partial execution leaves no claim that publication completed.

The private credential store is read only by the publisher at execution time.
Its values must never enter logs, source, archives, reports or command-line
arguments. Only its filesystem path is passed to the job.
