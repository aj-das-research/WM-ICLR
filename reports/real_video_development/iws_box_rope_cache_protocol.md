# IWS Box and Rope native feature-cache preparation v1

This data-preparation extension preserves every frozen PushT source and cache byte.
It defines no predictor, training objective, temporal window, official evaluation,
physical time or command units. New task adapters accept only `bimanual_box`
(14 recorded command coordinates) and `bimanual_rope` (8 coordinates), with explicit
validated task/width parameters; there are no process-global task monkeypatches.

Both tasks use the unchanged `configs/real_video_iws/split_v1.json`: 481 internal
training and 121 internal development trajectories, from 602 upstream-training
trajectories per task. All ten official-validation trajectories per task remain
reserved; their payloads are rejected before HDF5/video opens. Frozen metadata
identities/shapes may be read, including reserved IDs, without their payloads.
Box contains 120,274 native rows and Rope 120,312. Box lengths are 199 or 200;
Rope includes 199, 200 and one 201-row record. All native RGB frames and all N
`target_qpos` rows are retained with independent index vectors 0 through N−1.
The task names and widths are bound in configurations, input records, registration,
static/runtime identity, episode receipts, statistics and completion manifest.

The existing reviewed `load_commands`, sequential FFMPEG RGB decoder, float64
RunningMoments, atomic JSON/writer lock and identity functions are imported directly.
Task-dependent inventory, receipt/schema validation and transaction orchestration
are explicitly adapted in `src/shiftwm/real_video_iws_tasks/`. Metadata/video bytes
and all original dependencies are hashed. The frozen DINOv2-small extractor is
reused directly: revision ed25f3a31f01632728cabb09d1542f84ab7b0056,224×224 bilinear
antialiased RGB/ImageNet preprocessing, 256×384 spatial tokens, FP32 adaptive
16×16→4×4 pooling and channel-major6144 storage. GPU encoding is BF16 with TF32
disabled; CPU encoding is a separately identified FP32 cache. No fallback occurs.

Training-only shared-channel statistics count every internal-training frame×16
patch locations and all internal-training command rows. Development and official
validation never fit statistics. Float64 Welford sample standard deviation uses
ddof1 and floor1e-5. Immutable episode directories are atomically renamed only
after round-trip checks. Resume reuses verified packages and rejects changed
source, task, width, encoder, inputs, split or runtime. The final complete manifest
is written after all packages/statistics and registration guards pass. Validation
on CPU checks the full current static registration of a GPU-produced cache.

Before registration, `inspect_training.py` validates all authorized command values,
then fully decodes the first internal-training identity at each distinct training
native length. The first training identity's first/last frames are encoded on CPU
and compared exactly with an independently spelled-out pinned DINO spatial recipe.
This is declared input-compatibility exposure, not method selection. No masks are
used. Preserved upstream [s,s+59]/60-command temporal ambiguity remains unresolved;
these native-row caches introduce no alignment repair or official reproduction claim.

Each task receives a separate immutable configuration, input manifest and
registration before its full extraction. Both record the same source versions and
encoder but disjoint task identities/output roots. Tests cover both widths,
held-out access before open, missing/extra frames, RGB conversion, all native
command rows, train-only statistics, task/width swaps, corruption, exact static
identity checks, exclusive writes, interrupted resume and completion-last behavior.

Operational launch: at most two simultaneous GPUs total for these cache jobs,
one each on available ws-ia nodes ws-l1-006/ws-l5-004;8CPU,24GiB,6h maximum.
The allocated device must support BF16 and report at least4GiB free CUDA memory
immediately before model load. The scheduler allocation alone is insufficient.
The launcher builds the full cache then validates every committed package.
No model training or official benchmark evaluation is authorized by this protocol.
