# Registered spatial component follow-up (v1)

This is a development follow-up designed **after** the original five-arm spatial validation results were revealed. It is not a new held-out confirmation. No original frozen scientific source, registration, cache, model, evaluation, or fresh-test file is modified. No fresh/test image, feature payload, prediction, or outcome enters this follow-up. The original train and validation recording sessions retain their existing split. The two missing factorial cells, all six new runs, and all six existing control identities must be registered before any new training begins.

## Question and four fixed cells

The existing spatial package improves some native-coordinate validation endpoints, but changes both anchor mixing and innovation bounding. It does not establish a context benefit or separate the decoder components. Complete the following 2×2 design, without tuning any other quantity:

| Mixing package | Unbounded innovation | Bounded innovation |
|---|---|---|
| Absent | existing `anchored_additive`: anchor + residual | new `bounded_additive`: anchor + tanh(residual) |
| Present | new `unbounded_transport`: (1−g)anchor + g(T anchor) + residual | existing `transport`: (1−g)anchor + g(T anchor) + tanh(residual) |

Here anchor is the last observed 4×4 DINO feature grid, in training-fitted shared-channel standardized coordinates. Residual is the shared decoder's predicted innovation. T is a row-stochastic attention matrix across the 16 observed patches; g is a learned sigmoid gate. The bound is exactly 1.0 per coordinate, matching the existing transport arm. The mixing package includes the gate, identity bias 4.0, initial gate logit −3.0, and its learned query/key parameters. It is not isolated transport alone, does not establish physical correspondence, and is not an exact active-capacity match: nonmixing arms have 1,007,776 active parameters; mixing arms have 1,026,305 (+1.839%), with 1,026,305 instantiated in every model.

For each seed 0, 1, 2, each new arm constructs its corresponding old control with identical RNG draws, tensors, and active parameter masks before changing only the innovation transform. Bounded additive matches original anchored-additive initialization and trainable masks. Unbounded transport matches original transport initialization and trainable masks. Residual output heads are zero initially. Thus initial predictions match within each bounding comparison; between mixing levels, learned gate/identity initialization and active capacity remain disclosed differences. No existing checkpoint is used as an initialization or fine-tuned. All six new models are trained from their matched random initialization.

## Data, causality, training, selection

Reuse `data/features/droid_spatial_v1`, its frozen DINOv2-small provenance and train-only shared-channel mean/std (384 channels × 4×4 = 6,144 features), and the exact original2×2 target cache. Training and validation use only camera `exterior_image_1_left`. Native actions, five-original-frame sampling stride, feature extraction, normalization, frame/action alignment, and support horizon remain unchanged.

Every training/validation window has 3 observed support frames, 2 past actions and 10 query frames with 10 future recorded commands. A unidirectional GRU exposes only command prefixes. Support context is computed once from observed frames and past actions; query images are never predictor inputs. The shared spatial and pinned LeWM temporal trunk are unchanged: hidden96, depth4, 6 heads, context32/128. Predict latent features only, not RGB videos or physical robot actions.

All **2 new arms × 3 seeds × 30 full epochs**: AdamW lr1e−4, minimum lr1e−6, weight decay0.01, cosine schedule30, batch128, gradient clip1, BF16 CUDA training, training window stride2. Validation is FP32 with CUDA TF32 disabled, stride5, on all h10-eligible windows. Checkpoint selection is the **window-weighted MSE averaged across all ten query steps**, exactly as the existing controls, with earliest strict minimum chosen after all30epochs. Equal-episode validation diagnostics never select checkpoints. No hyperparameter search, seed dropping, early stopping, test-based selection, outcome-dependent filtering, or favorable-horizon resampling is permitted.

Training retains the existing atomic best/last checkpoint, optimizer, scheduler, Python/NumPy/PyTorch/CUDA RNG and dataloader-generator continuation semantics. Distinct package kind `shiftwm_real_video_spatial_components_v1` and schema `observed_anchor_mixing_x_innovation_bound_v1` prevent loading as the original spatial family. Original source files are imported into private module objects, not edited or globally rebound. The full dependency/configuration identity is checked at registration, run start, every epoch through the inherited training identity, run completion and finalization. Failures retain logs/checkpoints and cannot become a completed result; the finalizer requires the entire grid.

## Fixed comparisons, aggregation and uncertainty

The primary endpoint family is native standardized feature MSE at query endpoint h10. Secondary reported endpoints are native h5 and exact original2×2 h5/h10. Endpoint means are not mean1–5 or mean1–10. All ten horizon errors and persistence errors are retained per window and episode. Original2×2 scores use the exact old cached targets/anchors and old train-fitted normalization, not recomputed pooled targets. Different coordinate metrics cannot be compared numerically across resolutions.

Expected evaluation population is the existing h10-eligible original validation population: 1,631 windows from 141 episodes, 59 recording sessions, identically aligned for all four arms and all three seeds. Short episodes remain in cache metadata but cannot supply a fabricated h10 query. Within each episode average every eligible stride5 window; then give episodes equal weight and seeds equal weight. Strict ledger validation reconstructs all episode and global values from the full serialized per-window errors, checks exact session/episode/window populations and selected-package hashes, and retains failures instead of substituting values.

Predeclare these **four first-minus-second contrasts**, each for two metrics × h5/h10 = 16 reported effects:

1. Bounding without mixing: bounded_additive − anchored_additive.
2. Bounding with mixing: transport − unbounded_transport.
3. Mixing without bounding: unbounded_transport − anchored_additive.
4. Mixing with bounding: transport − bounded_additive.

Also predeclare **four difference-of-differences interactions**:

`(transport − unbounded_transport) − (bounded_additive − anchored_additive)`

at each native/original2×2 h5/h10 endpoint. Negative interaction means bounding reduces error more in the mixing package; it is a signed absolute MSE interaction, not a relative percentage gain. No component will be relabeled the sole cause based only on a favorable within-arm number. All 20 contrasts, all outcomes, complete seed-level values, selected epochs, and parameter counts will be reported even when negative or inconclusive.

Use 10,000 paired recording-session × training-seed bootstrap draws, seed173, matching the prior spatial analysis convention. For each draw, sample 3 seed identities with replacement and 59 recording sessions with replacement; concatenate all episodes in each selected session, average paired differences across episodes and selected seeds. The same selected units enter every arm of each contrast/interaction. Intervals are percentile2.5/97.5, exploratory and unadjusted for the 20 comparisons. The four-way interaction implementation is separately tested, and pairwise comparisons reuse the frozen tested helper. Existing controls were revealed before design; intervals do not turn this into independent confirmation.

## Completion, portable checkpoints and scheduler plan

Freeze a new registration with all new source/config/protocol/test hashes, original registry SHA `ae99340f7a3761ece066ee7376f4bd119b61a92503948c142196c40a1a76e337`, cache/statistics/DINO provenance, original completed finalizer, and all six control checkpoint/configuration/summary/journal/evaluation identities. Register before any new run directory exists. Six new inference-only packages are exported only after all six complete30epochs and all twelve result ledgers validate. Each export must pass exact CPU prediction parity from a physically relocated directory using copied source/weights, Python `-I`, local vendored upstream code, HF/Transformers offline flags and a Python socket guard. This is not an OS network namespace. The evidence uses the first eligible original validation feature window for deterministic reload parity; no qualitative performance-based selection is involved. No optimizer, raw video or generated prediction imagery is exported. Public release requires a separate publication gate.

Review code/tests/protocol before registration and launch. Schedule on **ws-ia only**, at most two simultaneous GPUs and eight CPUs per GPU (16 CPUs total). Known working nodes: ws-l1-002 and ws-l5-004. The gpu partition is excluded because previous jobs found occupied physical GPUs despite nominal allocation. Draft two serial lanes: bounded_additive seeds0→1→2 on the first node, unbounded_transport seeds0→1→2 on the second. Each full run retains batch128 and the complete30epoch recipe; no runtime-based batch reduction. CPU-only finalization waits `afterok` on both last jobs and requests no GPU. Existing matching-control timings provide a planning proxy (not a measured new-arm speed claim); fail registration if the six-run proxy exceeds six GPU-hours. Each allocation is7h50, with safe epoch-boundary checkpoint/requeue after6hours and at most five restarts. Scheduler metadata records allocated visibility and allocated-device free/total CUDA memory without overriding visibility. Require at least6GiB free on that visible device before training; if another process occupies it, fail before training and record an operational scheduler recovery without changing the scientific recipe. Actual job IDs/resources are recorded in a separate operational amendment, leaving this registered protocol unchanged.
