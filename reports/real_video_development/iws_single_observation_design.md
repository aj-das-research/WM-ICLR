# Single-observation IWS compact forecasting: implementable design review

**Recommendation:** a distinct DINOv2-S, native-command, full-horizon compact study with three matched learned arms (autoregressive, anchored additive, bounded spatial mixing), plus persistence. Use only the frozen internal training/development split to train and choose checkpoints. Reserve all600 official validation handles until the complete model/evaluation specification and checkpoint identities are locked. This is a proposed implementation contract, not a registration, completed experiment, or RLA-WM reproduction. This review read source and existing audit metadata only; it decoded no image, command array, or model output and launched no job.

## 1. Evidence and temporal boundary

Pinned upstream commit: `6f19048758699bf9a152eaed5ac6dbf1caa07c18`. The dataset's `frame_index_to_trajectory_data` constructs indices `range(s,s+(H-1)*stride+1,stride)`; with `num_frames=2`, `_compute_sampled_frame_indices` selects `[0,H-1]`. IWS uses stored-index stride1. Its trainer's `directly_use_target_qpos=True` branch passes all H rows unchanged, unlike the alternative branch that removes the last row. Its released predictor takes one observed image, uses disjoint15-row command chunks, and compares the last forecast to image `s+59` for released H60 handles.

| Handle horizon H | Actual observed frame | Target frame | Supplied command rows | Stored-index interval count |
|---:|---|---|---|---:|
|15|s|s+14|s…s+14 (15 rows)|14|
|30|s|s+29|s…s+29 (30 rows)|29|
|45|s|s+44|s…s+44 (45 rows)|44|
|60|s|s+59|s…s+59 (60 rows)|59|

The released H60 predictor performs four15-row chunks. Since its H15 training endpoint spans14 stored intervals, chunk composition has an unresolved boundary convention. **Do not silently repair that evaluator.** The compact model below is a separate direct stored-index contract: H60 means59 future feature predictions and60 command-conditioning rows, with no ambiguous15-row chunk mapping. It preserves the official initial frame, final endpoint and complete command sequence, while deliberately using a different architecture and full-H60 training recipe. Its intermediate offsets14/29/44 must not be equated to the official predictor's intermediate chunks without independent alignment evidence.

Command acquisition lead/lag, physical units and capture-time conversion remain unverified. A30fps video-container header does not establish physical duration. Use “stored offset59 / horizon60,” not “60 transitions,” seconds or physical action frequency. Causality below is prefix causality in the supplied stored-row convention, not a claim about the robot controller's physical timebase.

## 2. Exact model interface and equal information

New prediction API: `predict(initial_features[B,6144], native_commands[B,H,A]) -> predicted_features[B,H-1,6144]`, H>=2. There is no observed-history tensor, no past-action tensor and no future-image argument. Output index k-1 is the prediction of stored frame s+k, k=1…H-1. Query RGB/features enter a separate loss/scoring function only.

Reuse the reviewed shared 4×4 DINO patch encoder/projection, spatial attention, pinned LeWM three-slot temporal transformer, zero-initialized residual head and applicable transport/gate heads through a new namespace. Retain hidden96/depth4/heads6 and the shared-channel coordinate layout. Instantiate all arms with matched seed/RNG order and explicitly record total/active counts. Do not repurpose existing DROID packages or claim compatibility with their three-observation API.

Let z0 be the one actually observed feature grid; q_k be native command row s+k standardized by this task's internal-training statistics. A single unidirectional GRU produces h_k = GRU(q_0…q_k). At prediction offset k, feed h_k as the conditioning vector to each of the three temporal slots and every spatial patch. Computing the GRU over the whole supplied sequence is safe only because it is unidirectional; h_k cannot access rows after k. The first future prediction uses two rows, q0 and q1, consistent with upstream H2. Row q0 is real supplied conditioning, not an invented pre-start action. All60 rows are consumed by the last H60 forecast.

All arms begin with the architectural register `[z0,z0,z0]`. Those are **three repeated tokens of one image**, not three observed frames or two observed transitions. The autoregressive register shifts its own prediction after each offset: `[z0,z0,z1_hat]`, then `[z0,z1_hat,z2_hat]`, etc. Anchored arms keep `[z0,z0,z0]` at every offset. This common conditional backbone makes the intended difference explicit without fabricated history or zero-valued pseudo-actions:

| Arm | Visual register | Prediction at offset k |
|---|---|---|
|Persistence (no training)|one observed image|z0|
|Autoregressive|initial repeats, then own predicted features|z_hat_(k-1) + residual_k|
|Anchored additive|initial image repeated|z0 + residual_k|
|Bounded spatial mixing (ours)|initial image repeated|(1-g_k)z0 + g_k(T_k z0) + tanh(residual_k)|

Mixing uses the existing identity bias4, initial gate logit−3 and bound1 in normalized coordinates. Preserve and disclose its additional active query/key/gate parameters and different initial prediction from pure persistence. Repeated temporal-slot position embeddings are architecture, not timestamps. AR uses its own predictions with full backpropagation; no teacher forcing, hidden ground-truth refresh or horizon-dependent detachment.

**Disable and freeze `TransitionContext` and its FiLM adapter for every arm.** There is no observed transition from which to infer motion or perform support-based adaptation. Computing “transition context” from repeated z0 would manufacture evidence. The GRU is command conditioning, not observed test-time adaptation. This is a forecasting-transfer application of the compact decoder mechanism; it does not by itself demonstrate online adaptation. Extra observed frames or a later action-free/context study require separately registered, equal-information controls.

## 3. Task-specific commands and immutable data boundary

Freeze existing split SHA `17be56426dbee136ec883c45813b092a0747d707f25ee6efddb12b8a351d29c9` from `configs/real_video_iws/split_v1.json`.

| Task | Native command width A | Internal train trajectories | Internal dev | Reserved official validation |
|---|---:|---:|---:|---:|
|PushT|4|480|120|10|
|Box (`bimanual_box`)|14|481|121|10|
|Rope (`bimanual_rope`)|8|481|121|10|

Train separate per-task models and statistics; share architecture and recipe, not weights or fabricated common action semantics. Do not truncate to4, pad8/14 with undocumented action meanings, substitute measured qpos, or infer physical joint labels from the generic `target_qpos` name. Within a task all arms have identical native action width and information. Across-task parameter counts differ slightly and must be reported. No universal cross-embodiment checkpoint claim follows from these task-specific adapters.

The split was salted-identity-only; one previously viewed PushT training-split frame is now in internal development and remains disclosed. No trajectory is moved. Independence is trajectory-level only; session/scene/operator separation is unestablished.

Implement an IWS-specific ingestion/cache layer, not the DROID loader: IWS has N stored images **and N command rows**, whereas DROID's sampled loader has N images/N-1 action blocks and a different temporal stride. Preserve every native stored image and command row at stride1. Training-window start stride5 is only a sampling of starts, not temporal downsampling. Check exact video frame counts against metadata, shape, finite arrays, no clipping and no unsafe paths. Filter identities by internal train/dev **before** opening video or HDF5 payloads; use a runtime open guard to reject reserved-validation files. Keep reserved feature extraction in a separate later command/namespace.

Use full RGB `camera_0`, resize224×224 bilinear antialias, ImageNet normalization, the exact local frozen DINOv2-small encoder and channel-major4×4 FP32 pooling/storage. Record any BF16 extraction separately from FP32 scoring. Fit feature means/stds shared per channel across all internal-training frames and16 patches; fit each native command coordinate over training rows once, ddof1 and std floor1e-5. Do not reweight normalization by overlapping windows or include dev. This compact full-RGB track is distinct from upstream masked-RGB DINOv3 processing. The initially inspected white mask is not proof that all masks are white, and provided masks are not established object annotations. No future mask/crop enters a forecast.

## 4. Concrete full-training/selection proposal

Core fixed grid: **3 tasks × 3 learned arms × 3 seeds (0,1,2) =27 full30-epoch runs**; persistence for every task/window. Begin implementation and resource measurement with PushT, then execute the same frozen recipe on box and rope; do not drop later tasks based on PushT outcomes. Keep bounded spatial mixing as the declared candidate, rather than switching to a different winning architecture per final metric. The DROID component study remains a separate revealed-control follow-up. Adding its other two decoder cells to IWS would require an explicit expanded grid registered before reserved validation, not silent selective reporting.

Each eligible training/dev window contains frames s…s+59 and command rows s…s+59. Predict offsets1…59 from frame s alone. Match the upstream unused-tail feasibility rule `s+60<N`; deterministic start set `range(0,N-60,5)` for both internal train and dev. Retain every trajectory's eligibility audit, even if too short; do not fabricate queries. Train loss is the uniform mean standardized squared feature error over all59 offsets and6,144 coordinates, with equal windows. No curriculum or rollout-horizon mismatch: every run trains the entire H60 forecast from its first epoch.

Proposed recipe: AdamW lr1e-4, min_lr1e-6, weight_decay0.01, cosine30epochs, grad_clip1; effective batch64, microbatch16 with four-way accumulation; BF16 training, FP32 validation with TF32 disabled; no augmentation/dropout changes beyond the retained trunk. Accumulate weighted microbatch sums divided by the actual optimizer-batch sample count, including a short final batch, and clip once before each update. All arms share the same indexed windows, batch order, optimizer-update count and full59-offset loss. No gradient truncation for AR. Benchmark full shapes before registering actual training; if this recipe cannot fit, review and freeze one common new recipe for all arms before any scientific run, rather than adapt one arm after observing results.

Select each seed's checkpoint after all30epochs by **internal-dev equal-trajectory mean standardized MSE at stored offset59 (H60)**: mean windows within trajectory, then equal trajectories. Earliest strict minimum wins; intermediate H15/H30/H45 and mean1…59 are diagnostics, not alternative selectors. This differs from DROID's window-weighted all10 selection and therefore needs a new selection string/package kind, e.g. `shiftwm_iws_single_observation_v1` and `internal_dev_equal_trajectory_endpoint_H60_standardized_mse`. Architecture is fixed before IWS outcomes; the only required selection is checkpoint epoch using this declared rule. If a broader version-selection exercise is desired, specify its complete candidate grid and one macro dev criterion before training, retaining every arm; this review does not authorize that expansion.

Reuse atomic generations, safe tensor packages, complete optimizer/scheduler/RNG/dataloader state and strict full30 completion validation from the existing trainer through a private wrapper. Implement the new accumulation/one-observation epoch function and freeze its identity. Compare executed scientific recipe against registered config, not merely against its own metadata. Publish every completed selected checkpoint and every per-seed outcome, not just the winning seed.

## 5. Locked endpoint evaluation and uncertainty

Only after all intended tasks/arms/checkpoints, metrics, preprocessing and any decoder are frozen: unlock the original200 H60 handles per task exactly once. Preserve all600 handle IDs and endpoints without outcome filtering. Derive secondary H15/H30/H45 **prefix evaluations at the same starts** and label them derived endpoints, not additional official handle sets. Each shorter prediction receives only its corresponding15/30/45 command rows. Official H60 uses all60. Inference cannot access target features/RGB; a separate evaluator obtains reference frames and measures errors.

For every task, seed, arm, handle and H in{15,30,45,60}, retain native standardized feature MSE (primary), standardized feature MAE, raw DINOv2 feature L1 and flattened feature cosine distance (secondary). Specify epsilon1e-8 and clip cosine to[-1,1] before computing1−cos; zero-vector cases receive the registered formula, not exclusions. Keep persistence and relative MSE reduction `(baseline−method)/baseline`; undefined zero-baseline ratios are marked undefined. DINOv2 raw L1 is **not** numerically comparable to upstream DINOv3 `dino_l1`.

Report both equal-handle means (matching upstream wrapper aggregation) and equal-trajectory means (our primary clustered estimand), with every denominator named. Primary headline: bounded mixing versus anchored additive at H60, with separate task values and an equal-task macro of task-specific relative MSE reductions; recompute the macro ratio statistic within each bootstrap draw. AR and persistence comparisons, other endpoints and feature metrics remain secondary. Use10,000 paired training-seed × trajectory cluster bootstrap draws, fixed seed173; resample three matched seeds and10 validation trajectories within each task, retaining every selected trajectory's handles. Across-task macro draws resample trajectories independently by task but retain the same selected seed identities. Report unadjusted exploratory95% intervals and all comparisons; do not imply200 independent trajectories, new session-independent test data, or a corrected familywise significance guarantee. Ten trajectory clusters per task make uncertainty an explicit limitation.

No score-based qualitative selection during development on the reserved set. If later needed, register fixed best/median/worst case selection with all-case context before accessing those outcomes, and label observed input, withheld RGB, predicted feature map and decoder output separately.

## 6. Optional RGB extension and separate upstream reproduction

A common RGB decoder is an optional separate registered experiment, not a prerequisite for the feature study. Train one decoder per task from **internal-training ground-truth DINOv2 features to their same-frame RGB**, select it only by fixed internal-dev reconstruction loss, then freeze it identically for AR/anchor/mixing/persistence. Do not train a different decoder per method or use target RGB as decoder conditioning. Suggested small deterministic upsampling CNN:4×4×384→7×7×256→14×14×128→28×28×64→56×56×32→112×112×16→224×224×3, bilinear upsampling and3×3 convolutions, final sigmoid; L1 reconstruction objective. Its complete epoch/batch/resource budget must be registered separately before the final set is opened.

Score generated RGB against the exact resized full-RGB references at all four endpoints with fixed LPIPS-VGG and SSIM implementations; pixel MSE/PSNR can be separately declared secondary. Upstream currently returns `dino_l1`, LPIPS and SSIM, not PSNR. Also show decoding of the true target features as a **reference reconstruction**, with the same metrics; this exposes the4×4 information bottleneck but is not a proven mathematical performance ceiling. Feature forecasts can decode poorly, and reconstruction bias can partly cancel forecast errors. Do not present true target RGB as a generated result or claim RGB superiority from feature MSE alone. Our full-RGB224 pipeline cannot be pooled into the upstream masked-RGB512 score table without a separately matched evaluation contract.

The gated DINOv3-L RLA-WM reproduction stays distinct: resolve its trainer import/checkpoint compatibility, obtain authorized encoder weights, disable silent DINOv2 fallback, retain its original image preprocessing, action widths,15-row chunking, stochastic flow sampling and official metrics, and pin decoder/encoder/predictor identities. The compact model's all59-offset supervised H60 training uses a different budget and supervision than upstream H2–15 endpoint training; report that difference. Do not call this compact study a reproduced RLA-WM score or a SOTA comparison. Physical robot success, causal intervention quality and online adaptation require separate evidence.

## 7. Required tests and execution gates

- Hand-constructed indexing tests for H2/15/30/45/60, first/last feasible starts, exact command-row inclusion, N-image/N-command layout, one-observation-only predictor API, and no silent index clipping.
- Nonzero learned-like output, context-independent and LeWM AdaLN fixtures: later-command perturbations/gradients cannot affect earlier offsets; all future target-image gradients into predictions are zero; permitted native-command gradients are nonzero. Prefix truncation tolerates documented FP32 kernel roundoff, not arbitrary drift.
- Initial-token repetition matches across arms; prediction-fed AR never receives teacher targets; transport is row-stochastic and its bounded normalized innovation obeys its bound; expose total/active capacities and initial-prediction differences.
- Internal-statistics-only provenance; reserved path open guard before payload access; all native widths and exact video/metadata lengths; metadata or decode failures stop the study with an audit rather than silently dropping cases.
- Accumulation equivalence including incomplete last batches; all30epochs with exact uninterrupted-vs-resumed continuation; no cross-package loading; physically relocated offline inference parity using the actual new source.
- Complete window/trajectory/seed population checks, independent reconstruction of means and paired intervals, executed-recipe/config equality, immutable finalization/release manifests, all negative and inconclusive results retained.

GPU extraction and full-shape resource measurements need scheduler allocation. Keep at most the already authorized two ws-ia GPUs/16trainingCPUs; do not conflict with the six running DROID component jobs. There is no measured IWS runtime yet. Derive a27-run time/memory projection from allocated full-shape measurements before committing the training grid; document any resource-driven scope change before outcome access. This report submits no jobs and changes no frozen implementation.

## Source identities for this review

The following files were read without modifying them. Hashes bind this design to the inspected local source, not to unverified latest upstream behavior.

| Local source | SHA256 |
|---|---|
| `reports/real_video_development/next_benchmark_and_versions.md` | `dabdb0d7a5847fbe56a97bfef39045af4d92acba009fa8f646804175c3f07a5a` |
| `reports/real_video_development/iws_temporal_semantics_review.md` | `167d4078222830ea289e94505d038810ff2944c602ac50678e3c7cb4b6747b04` |
| `reports/real_video_development/rla_wm_iws_audit.md` | `94f977a6d0171e52e0d37b5b04c6f4a7dfb358107cced80077c4ef08191cc860` |
| `configs/real_video_iws/split_v1.json` | `17be56426dbee136ec883c45813b092a0747d707f25ee6efddb12b8a351d29c9` |
| `reports/evidence/iws_split_independent_review.json` | `56fa704d8688db1dfbbe8ac846ec3f2a60e56cb0de925be593ef92d11ed8ed4c` |
| `external/rla-wm/src/datasets/trajectory_dataset.py` | `fb1e48699b17452df698a664839cdb7ba883ee7736e672681683c82ad5bfbe4a` |
| `external/rla-wm/src/trainers/rla_wm_trainer.py` | `faa57d9834e41b614e2b9b07540ab9b9bbc76b25483529c88b36b7a47625ff19` |
| `external/rla-wm/src/models/rla_wm.py` | `c0746323e865eb87b8674750f790893a25ccd9447a15888497f915eb95220268` |
| `external/rla-wm/eval/predictors/rla_wm_predictor_iws.py` | `54aa0de48ba8430580d5dd41979d8f8dc0996cb5ce930044aa9cbb1b07ae649a` |
| `external/rla-wm/eval/eval_wrapper.py` | `6fa0f14b9850dc0a4f7bc46e99f23b983084dd5ac49739d0451b9b0528caedfd` |
| `external/rla-wm/src/utils/loss_utils.py` | `0f086575b57102aeaa25d1f209c2beb913126089f08e8736bf8158895fd4d41f` |
| `external/rla-wm/configs/rla_wm/iws_pusht.yaml` | `7f032626ea0575c85018f8f08a17ed4792eb16bfb01017ad31db7f5f50877b9f` |
| `external/rla-wm/configs/rla_wm/iws_box.yaml` | `0ee4b046950cd33a18bdf46f390cb73913c525979b234ef5cffb9120fe46b7a7` |
| `src/shiftwm/real_video_spatial/model.py` | `054d0aa1c479c1dc722b674bb59dc35015c69ebacec0f0602b84ded40651edf2` |
| `src/shiftwm/real_video_spatial/features.py` | `89e76683f28c003c7171ae17f485ea9fe35bd43adc9df854fad5f9beaf648d05` |
| `src/shiftwm/vendor/lewm/module.py` | `0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620` |
| `scripts/real_video_spatial/train.py` | `78e6345bdaaf4e07bdd7a681938b21e6d75e9967c595f8d889bffcea6d02b378` |
