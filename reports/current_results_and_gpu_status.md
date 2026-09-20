# Current results and GPU status

**Current GPU work: raw-feature DINO-WM follow-up — 3 running, 2 queued, six full 100-epoch runs.** The completed 30-epoch comparison below remains separate.

Checked **2026-09-20T15:31:29.063426+00:00** from the live scheduler and checkpoint summaries.

**IWS jobs: 0 running, 0 queued. There are 27 original models plus nine component ablations.**
V1 training jobs: 0 running, 0 queued.
All 27 models and the complete development comparison have passed the scientific finalizer.

All 36 reserved evaluations and their independent complete-result review passed. The reserved population has 600 handles from 30 trajectories; its metrics remain separate from development.

**External DINO-WM training: 0 jobs running,
0 queued; 6/6 full
training summaries and 6/6 evaluation markers present.**
The six runs use two separately declared objectives, three seeds each, and
30 full epochs on the matched DROID train/development population. Counts are
progress observations; the complete scientific finalizer validates all six
external predictors and fifteen internal controls before publishing comparisons.

The account permits two ws-ia jobs plus up to one GPU on the GPU partition.
Evaluation recovery uses CPU-only allocations to keep numerical reduction
behavior consistent across every comparison; scheduler job names distinguish
these from training. Full H60 training uses all 59 future offsets for 30 epochs, with effective batch 64 and
three seeds for each task/method. The frozen recipe and all reviewed scientific
sources are checked at launch and at every epoch.

The first task is real IWS PushT: one actual image plus recorded native commands
predict future DINOv2 visual features. Box and Rope use the same method/recipe
with their native command widths. These are feature forecasts, not RGB videos
or measured physical robot success. The reserved evaluation has a distinct
registration, population, execution revision and reporting gate.

## Training progress (checkpointed epochs, not benchmark scores)

| Task | Autoregressive | Additive anchor | ShiftWM (ours) |
|---|---|---|---|
| pusht | s0: 30/30, s1: 30/30, s2: 30/30 | s0: 30/30, s1: 30/30, s2: 30/30 | s0: 30/30, s1: 30/30, s2: 30/30 |
| bimanual_box | s0: 30/30, s1: 30/30, s2: 30/30 | s0: 30/30, s1: 30/30, s2: 30/30 | s0: 30/30, s1: 30/30, s2: 30/30 |
| bimanual_rope | s0: 30/30, s1: 30/30, s2: 30/30 | s0: 30/30, s1: 30/30, s2: 30/30 | s0: 30/30, s1: 30/30, s2: 30/30 |

27/27 full-training summaries and
27/27 development evaluation receipts are present.
The finalizer independently checks every model, primitive window error,
trajectory, seed and source hash before numerical paper ingestion. Intermediate
training losses are not substituted for completed comparisons.

## Completed results retained in the paper

| Study | Verified completion | Finding and scope |
|---|---|---|
| Current spatial ShiftWM on DROID | 21 models ×30 epochs;32 method contrasts +4 interaction effects | 3.01% h5 /5.30% h10 relative standardized-MSE reduction vs matched AR;136/141 episodes improve at h10; development evidence |
| Spatial components | Six additional follow-up models included above | Mixing has the clearest native h10 benefit; incremental bounding/context effects are inconclusive |
| Historical DROID/context, capacity and h10 studies | Completed in their separate finalizers | Different architectures/protocols; positive and negative findings retained |
| Historical simulation and domain extensions | Completed separate studies | Mixed forecast/planning findings; not final spatial-model cross-domain evidence |
| Real IWS PushT/Box/Rope | 1,804 recordings /360,473 frames cached; 27/27 full training summaries | All 27 original development evaluations validated; nine component ablations and the reserved study are tracked separately below |
| Adapted official DINO-WM comparison | 6/6 full training summaries; complete comparison receipt present: True | Two distinct objectives; matched DROID development forecasts; no partial accuracy or SOTA claim |

Training profiles and scheduler limits are in `reports/real_video_iws/`.
The new external jobs request two-hour allocations with exact epoch-boundary
continuation if needed. The original IWS training projection is historical
capacity planning, not the remaining evaluation time.
Original GPU receipts and operational recovery provenance are retained under
`reports/real_video_iws/recovery/`; the frozen evaluator and its tolerances
are unchanged.

The paper now states one current proposed decoder, the distinct DROID/IWS
interfaces, the primary feature-error metric, matched internal controls and
pending external comparisons. See `reports/experiment_alignment_2026-09-19.md`.
Source-bound completed DROID evidence remains in
`reports/real_video_spatial/finalization.json` and
`reports/real_video_spatial_components/finalization.json`.

## Controlled innovation-bound ablation

Nine separately registered full 30-epoch runs retain the fixed-source mixer and remove only the innovation bound. These are exploratory development follow-ups; their scores do not replace completed v1 results.

| Task | Seed | Checkpointed epochs |
|---|---:|---:|
| pusht | 0 | 30/30 |
| bimanual_box | 0 | 30/30 |
| bimanual_rope | 0 | 30/30 |
| pusht | 1 | 30/30 |
| bimanual_box | 1 | 30/30 |
| bimanual_rope | 1 | 30/30 |
| pusht | 2 | 30/30 |
| bimanual_box | 2 | 30/30 |
| bimanual_rope | 2 | 30/30 |

Source, profile, registration, completion and all outcome records live in `reports/real_video_iws_unbounded/`.

## External DINO-WM full-training progress

| Objective | Seed | Checkpointed epochs | Evaluation marker |
|---|---:|---:|---|
| official_one_step_shifted | 0 | 30/30 | True |
| matched_recursive_h10 | 0 | 30/30 | True |
| official_one_step_shifted | 1 | 30/30 | True |
| matched_recursive_h10 | 1 | 30/30 | True |
| official_one_step_shifted | 2 | 30/30 | True |
| matched_recursive_h10 | 2 | 30/30 | True |

Source and complete-campaign protocol: `scripts/external_dinowm_train_v1/README.md`. Intermediate losses are not benchmark results.

## Raw-coordinate DINO-WM follow-up

| Objective | Seed | Checkpointed epochs |
|---|---:|---:|
| official_raw_one_step | 0 | 100/100 |
| official_raw_recursive_h10 | 0 | 29/100 |
| official_raw_one_step | 1 | 85/100 |
| official_raw_recursive_h10 | 1 | 0/100 |
| official_raw_one_step | 2 | 100/100 |
| official_raw_recursive_h10 | 2 | 13/100 |

Raw visual inputs/outputs, batch 32, constant learning rate 5e-4 and FP32; automatic epoch-boundary continuation preserves the full budget. No partial accuracy is reported.
