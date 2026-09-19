# Current results and GPU status

Checked **2026-09-19T17:35:43.833024+00:00** from the live scheduler and checkpoint summaries.

**IWS full training: 3 jobs running, 6 queued; 27 registered runs.**
The account permits two ws-ia GPU jobs plus one GPU-partition GPU. Full H60
training uses all59 future offsets for30epochs, with effective batch64 and
three seeds for each task/method. The frozen recipe and all reviewed scientific
sources are checked at launch and at every epoch.

The first task is real IWS PushT: one actual image plus recorded native commands
predict future DINOv2 visual features. Box and Rope use the same method/recipe
with their native command widths. These are feature forecasts, not RGB videos
or measured physical robot success. Official validation remains reserved.

## Training progress (checkpointed epochs, not benchmark scores)

| Task | Autoregressive | Additive anchor | ShiftWM (ours) |
|---|---|---|---|
| pusht | s0: 30/30, s1: 30/30, s2: 0/30 | s0: 30/30, s1: 30/30, s2: 5/30 | s0: 30/30, s1: 30/30, s2: 0/30 |
| bimanual_box | s0: 30/30, s1: 30/30, s2: 30/30 | s0: 30/30, s1: 30/30, s2: 0/30 | s0: 30/30, s1: 30/30, s2: 26/30 |
| bimanual_rope | s0: 30/30, s1: 30/30, s2: 23/30 | s0: 30/30, s1: 30/30, s2: 0/30 | s0: 30/30, s1: 0/30, s2: 0/30 |

18/27 full-training summaries and
14/27 development evaluation receipts are present.
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
| Real IWS PushT/Box/Rope | 1,804 recordings /360,473 frames cached; training progress above | Predictor comparisons pending validated completion |
| External SOTA comparison | Reproduction work remains incomplete | No current evidence of SOTA superiority |

The measured micro64 workload is approximately20.4 GPU-hours for27 runs including
per-epoch validation, or6.8hours at three continuously available GPUs before
loading, checkpointing and final evaluation. This is a resource projection,
not a guaranteed finish time. Full-shape profiles and the scheduler limits are
in `reports/real_video_iws/`. The GPU partition has an eight-hour per-job limit;
its array uses7h50m allocations and preserves epoch checkpoints.

The paper now states one current proposed decoder, the distinct DROID/IWS
interfaces, the primary feature-error metric, matched internal controls and
pending external comparisons. See `reports/experiment_alignment_2026-09-19.md`.
Source-bound completed DROID evidence remains in
`reports/real_video_spatial/finalization.json` and
`reports/real_video_spatial_components/finalization.json`.
