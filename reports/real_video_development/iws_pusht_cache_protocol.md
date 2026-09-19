# IWS PushT native-row feature-cache preparation v1

This prepares data, not a training campaign or benchmark performance claim. Only
PushT is enabled. No predictor training, model selection, or official evaluation
is authorized by this protocol. GPU submission remains a separate root scheduling
action after independent implementation review.

## Population and exposure

The frozen `configs/real_video_iws/split_v1.json` supplies 480 internal training
and 120 internal development trajectories, all drawn from upstream training IDs
000010–000609. Official-validation trajectories 000000–000009 remain inaccessible
to this cache pipeline. The loader checks identity and internal split before
resolving/opening payloads. No official-validation video is decoded, no official-
validation command value is read, and no mask stream is used. Upstream split,
metadata-shape and handle identity documents can be read without accessing their
underlying held-out payloads.

The previously inspected first upstream-training trajectory 000010 happens to be
**internal development** under the deterministic split. Its frame0 preview is
already documented in `reports/evidence/iws_training_visual_inspection.json`.
It remains in that partition; this study does not claim all development imagery
was previously unseen. That image shows actual tabletop manipulation with two
robot grippers and a pink T-shaped object. The first accompanying mask was all
255; no semantic/object annotation is inferred from its filename. The first
internal-training ID is 000011, a distinct recording. No outcome, success
attribute or command/image content influences the split.

## Recorded-row contract

Every decoded native RGB row and every recorded `target_qpos` row is retained.
PushT commands have width4; state `qpos` has width14 but is not an input here.
These four coordinates retain their upstream name and order; their physical
meaning and units remain unverified. For each trajectory with audited length N:
features have shape N×6144, commands N×4, and separate frame/command-row index
vectors both equal `[0,...,N-1]`. No final command is dropped, no adjacent-frame
N−1 convention is borrowed from DROID, and no temporal subsampling is applied.

The exact sequential FFMPEG decoder must produce the audited metadata row count,
not merely a plausible container header. Missing/extra decoded frames, wrong
640×480 dimensions, changed bytes or nonfinite commands fail the entire affected
package; no clipping, repetition, repair or trajectory exclusion is allowed.
BGR is explicitly converted to RGB. Input-video bytes, metadata bytes, decoded
RGB bytes and command values have separate SHA256 identities. No masks, success
attributes, physical timestamps or inferred seconds enter the cache.

No model windows are defined. The documented upstream horizon60 ambiguity
(`s`→`s+59` with60 command rows and four15-row chunks) remains reserved for a
future explicitly registered training/evaluation protocol; this cache does not
silently repair it or claim an official RLA-WM reproduction.

## Frozen encoder and preprocessing

Use local `facebook/dinov2-small` revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`, verified against all recorded file
SHA256/size identities and loaded offline with safetensors. The numeric recipe
is copied from the immutable reviewed spatial extractor, whose source hash is
also bound: uint8 RGB→float32/255; bilinear antialiased resize to224×224 with
`align_corners=False`; ImageNet mean `[.485,.456,.406]` and std
`[.229,.224,.225]`; discard CLS; reshape256 final tokens×384 channels into
384×16×16; FP32 adaptive mean-pool to4×4 and flatten channel-major into6144.

CUDA encoding uses BF16 autocast with FP32 pooling/storage and requires an
allocated BF16-capable GPU. CPU uses FP32 and a separate default output path.
Both disable TF32. Batch size32 and exact Torch/Transformers/NumPy/HDF5/OpenCV
versions, OpenCV build identity, CUDA runtime/device when used, preprocessing,
source/config/input/split/encoder hashes are part of immutable extraction
identity. CPU and GPU caches cannot be mixed on resume. No encoder fallback,
weight download, mask substitution or changed preprocessing is allowed.

## Training-only normalization

Fit each of384 feature-channel means/sample standard deviations over every
native internal-training frame and all16 patch positions. Repeat the resulting
channel parameters over the16 cells to retain channel-major coordinates.
Separately fit the four command coordinates over all N training command rows.
Use float64 parallel-Welford moments, ddof1, and standard-deviation floor1e-5.
Record complete training episode identities, package hashes and sample counts.
Development inputs cause rejection in the statistics function; they cannot
silently contribute. No normalizer or model is fitted on upstream validation.

## Registration, transactions and completion

After source/tests/protocol independent review, `prepare_cache.py register`
checks and hashes all600 allowed videos/metadata without decoding video or
reading HDF5 numerical values. It freezes the explicit input ledger and cache
registration, including source, configuration, data and encoder dependencies.
Changing any frozen dependency requires a new explicit version; the preparer
will not overwrite different registration documents.

A writer lock permits one producer per cache output. Each episode is constructed
in a private temporary directory, flushed, round-trip checked, and atomically
renamed with both arrays and receipt present. A stopped process can leave a
hidden pending directory; it is never consumed as an episode. Existing committed
packages are always shape/index/finiteness/hash checked and are never silently
replaced. Wrong identity, missing receipt or corrupted payload causes rejection.
A normal interrupted build resumes only missing episodes.

After every expected package passes validation, an episode index and train-only
statistics are written. The final `manifest.json` with status complete is written
**last**, after statistics checks and a final registration/source dependency
check. A cache without that marker cannot be opened by the complete-cache loader.
No partially prepared cache is considered benchmark evidence or a trained model.

## Checks and launchers

Tests exercise rejected held-out access before payload opens, native N/N row
alignment, decoder color/count guards, shuffled indices, train-only statistics,
large development sentinels, stable variance, identity mismatch, partial resume,
corruption and completion-last failure behavior. Any real-data preflight uses
only allowed upstream-training payloads and is reported separately from these
engineering fixtures. A CPU inspection of all200 native frames of internal-training000011, plus encoder-recipe parity on its first/last frames, is performed before full-cache registration. This compatibility check is not a predictor-performance result and is distinct from the existing000010 preview. Registration therefore says before full-cache extraction, not before any encoder invocation.

After review, CPU preparation registration can run locally. Full extraction may
use `scripts/real_video_iws/cache_gpu.slurm` (ws-ia partition, one GPU, eight CPUs, 24GiB host RAM) or the separate
CPU launcher/output. The agent preparing this task does not submit either job.
The builder checks allocator-visible CUDA free memory and requires at least4GiB before loading the encoder; an idle scheduler entry is not treated as usable memory. No automatic partition fallback occurs. Root may explicitly override Slurm partition/node after a fresh successful allocation probe; the identical CUDA-memory preflight still runs and scientific identity is unchanged.
Root must schedule the GPU builder alongside the two component-training lanes
within the existing three-GPU maximum. Reuse/resume never changes the data split.
