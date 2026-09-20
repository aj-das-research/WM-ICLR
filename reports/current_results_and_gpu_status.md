# Current results and GPU status

Checked **2026-09-20T09:30:05.095408+00:00** from the live scheduler and checkpoint summaries.

**IWS jobs: 0 running, 0 queued. The v1 study has 27 registered runs.**
V1 training jobs: 0 running, 0 queued.
All 27 models and the complete development comparison have passed the scientific finalizer.

The account permits two ws-ia jobs plus up to one GPU on the GPU partition.
Evaluation recovery uses CPU-only allocations to keep numerical reduction
behavior consistent across every comparison; scheduler job names distinguish
these from training. Full H60 training uses all 59 future offsets for 30 epochs, with effective batch 64 and
three seeds for each task/method. The frozen recipe and all reviewed scientific
sources are checked at launch and at every epoch.

The first task is real IWS PushT: one actual image plus recorded native commands
predict future DINOv2 visual features. Box and Rope use the same method/recipe
with their native command widths. These are feature forecasts, not RGB videos
or measured physical robot success. Official validation remains reserved.

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
| Real IWS PushT/Box/Rope | 1,804 recordings /360,473 frames cached; 27/27 full training summaries | All 27 development evaluations validated; complete signed comparisons are in the paper; official validation remains reserved |
| External SOTA comparison | Reproduction work remains incomplete | No current evidence of SOTA superiority |

Training profiles and scheduler limits are in `reports/real_video_iws/`.
The GPU partition has an eight-hour per-job limit. The original training
projection is historical capacity planning, not the remaining evaluation time.
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
