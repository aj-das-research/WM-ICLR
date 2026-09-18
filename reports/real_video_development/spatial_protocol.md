# Spatial observation-anchored world-model development protocol, v1

This protocol is frozen before extraction/training. It is a development experiment on the original DROID train/validation recordings, not fresh-test evidence. No original or fresh test payloads may be decoded, selected against, or evaluated. All original experiments remain immutable.

## Data and representation

- Source: existing audited DROID processed manifest and genuine exterior-camera-one recordings. Select original `train` and `val` entries before opening any payload. Verify source NPZ SHA256 before decoding. Retain exact session IDs, frame indices, and recorded 35-dimensional grouped commands.
- Encoder: existing pinned, frozen DINOv2-small; same 224-square bilinear-antialiased preprocessing, ImageNet normalization, batch32, and bfloat16 encoder as the original cache. Pool the 16×16 final patch grid to 4×4 in float32; store channel-major 384×4×4 arrays in `data/features/droid_spatial_v1`.
- No test camera, labels, target-derived strata, or intervention metadata enters the model.
- Independently pool new raw features to 2×2 and compare with existing original train/validation cache; require absolute tolerance2e-5 and relative tolerance1e-5, identical actions and frame indices. Store maximum absolute discrepancy per episode.
- Fit per-channel mean/std jointly over all spatial positions in all original training frames (ddof1, floor1e-5); repeat those384 values across16positions for the flattened API. Action statistics use training actions only. This shared coordinate system is required for cross-patch transport.
- Expected feature payload ~1.391GB float32 for56,595frames; raw images and encoder remain local and are not duplicated into ordinary Git.

## Models and matched controls

All arms use4×4features, shared384→96projection, exact spatial position embeddings, one shared96-dimensional spatial attention/MLP block, pinned LeWM ARPredictor over each patch's three-frame temporal history (depth4,heads6,headwidth16), and a unidirectional96-dimensional action-prefix GRU. The action prefix for horizon h includes recorded commands only through h. A single transition-context GRU is computed from observed support and two past action blocks and held fixed; no future observation refresh occurs. This spatial architecture candidate does not reproduce the original separate observation/dynamics factorization and must not be labeled an unchanged original ShiftWM model.

| Mode | Decoder and intervention |
|---|---|
| autoregressive | Shared output head predicts additive patch residuals recursively from the last three observed/predicted grids. |
| anchored_additive | Same output head predicts directly from the three observed grids plus the causal action prefix, added to the last actual grid. |
| transport | Attention transports the last actual grid; sigmoid patch gate mixes identity/transport; innovation is `1.0*tanh(residual)` in shared normalized coordinates. |
| context_off | Transport arm with observed support context/FiLM disabled. |
| action_free | Transport arm with every action path disabled, including dynamics context and prefix states. |

Transport heads use an identity-logit bias4.0 and gate-logit bias−3.0. The innovation and additive output heads initialize at zero. The sigmoid gate remains differentiable at initialization; transport starts near persistence, whereas additive arms start exactly at persistence. Report this difference and all total/active parameter counts. Inactive heads are frozen and excluded from optimizers; this is shared-trunk, approximately matched capacity, not exact active-parameter equality. Context-off/action-free intentionally remove parameters and inputs. Transport is semantic feature mixing, not a validated physical image warp.

Output bound: in shared normalized coordinates, each component's magnitude is at most the largest anchor magnitude plus1.0. This bounded-output fact is not an accuracy, safety, or Lipschitz theorem. A positive result does not distinguish the transport and bounded-innovation contributions without a further registered ablation.

## Full training, selection, and evaluation

Exactly5arms×3seeds(0,1,2)=15runs, each30full epochs. AdamW learning rate1e-4, weight decay0.01, cosine decay to1e-6, batch128, clipnorm1.0, bfloat16 training, float32 validation without TF32. Three observed frames and ten query frames; training stride2, validation stride5. No short-run validation metric selects the architecture or a hyperparameter.

Checkpoint selection: window-weighted mean MSE across **all ten** query steps in the same shared-channel coordinates for all arms, matching the original horizon-ten control's aggregation convention. Every epoch is retained in the metric journal; lowest validation epoch is selected, with first minimum on ties. Also record equal-episode validation as a diagnostic; it does not select checkpoints. Final evaluation averages windows inside episodes, then equally weights episodes and seeds. Report h1–h10 native4×4MSE, persistence, and deterministic2×2pooled raw forecasts scored against the EXACT original2×2cached targets, aligned by episode/window, measured under the original frozen2×2normalization. Original targets are not recomputed from4×4features because tiny pooling roundoff can be amplified by the normalization floor. Native errors across different resolutions are not directly comparable.

All positive and negative arms/seeds must appear in the final aggregate. Predefined transport comparisons against autoregression, anchored additive, context-off, and action-free report h5/h10 paired difference intervals in both coordinate metrics, using10,000 joint recording-session/training-seed bootstrap draws(seed173). Intervals are exploratory validation evidence with no multiplicity adjustment. The initial study estimates the package-level architecture effect, not novelty or SOTA. Existing train-only scalar calibration remains a separate baseline; no calibration gain is attributed to this architecture campaign.

## Resource and execution gate

Three GPUs maximum; after the existing horizon-ten campaign's finalizer. One allocated GPU first extracts the complete train/validation cache and measures full-batch128 forward/backward throughput for every arm (two warmup plus five timed training batches and three float32 forward batches, training examples only). This measurement does not replace full training or select by error. Record GPU, memory, parameter counts, and projected runtime. Fail closed if an individual30-epoch estimate exceeds20hours or if the total15-run training estimate exceeds80GPU-hours (approximately26.7ideal wall hours across three identical GPUs). No batch-size/model fallback is allowed without a new registration.

Then run fifteen jobs in three seed chains, five arms sequentially per chain; use two ws-ia GPU allocations and one gpu-partition allocation at any time. Jobs verify frozen source/config/protocol/registration dependencies at startup and training epochs. Each model allocation is7h50m. At the first completed epoch after6hours, unfinished full training is checkpointed and its own Slurm job is requeued; epoch/RNG/optimizer/scheduler continuation is exact. Five restart attempts are the maximum, after which an explicit failure requires review. A CPU finalizer runs only after all three complete seed chains succeed. It validates every30-epoch journal/best-selection/package hash, all15validation ledgers, every expected window/episode/horizon and their arithmetic, and independently relocated CPU checkpoint parity. Optimizer/RNG state remains local for resumption; inference exports contain weights/configuration/manifests and exact source/encoder provenance.

Registration captures code, original data/cache metadata and normalization, encoder provenance, all15configs, protocol, and scheduler scripts before submission. New cache hashes are bound into each training identity after audited extraction and before training; no test data is introduced at that stage.

## Reuse and release contract

Reuse the existing original real-video atomic checkpoint/epoch-resume implementation through an isolated module import; do not edit it. Reuse frozen DINO/LeWM modules and existing SHA/audit utilities. New code lives only in `src/shiftwm/real_video_spatial` and `scripts/real_video_spatial`. Causal prefix/query-independence, action-free invariance, exact grid ordering, shared-normalization enforcement, nondead gate gradients, bounded output, split rejection, and relocated offline reload must pass before scheduling.

Model packages forecast latent DINO features. They are not RGB generators, trained robot policies, executed counterfactuals, or clinical systems. Any published model card must include validation-selected epoch, all completed30epochs, exact coordinate/preprocessing contract, licenses, and original-validation development scope.
