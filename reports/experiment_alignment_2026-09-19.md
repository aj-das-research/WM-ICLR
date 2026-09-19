# Experiment alignment and automatic result reporting — 19 September 2026

The manuscript proposes one current method: **ShiftWM, the bounded observation-anchored spatial decoder**. Its completed main evidence is the DROID development study. Historical context-model studies and the new single-observation IWS transfer interface have different representations, training recipes and selection rules; their scores cannot be pooled into a single cross-domain or state-of-the-art claim.

## Task, information and comparison matrix

| Study | Available predictor input → output | Primary reported score and aggregation | Population / selection | Comparisons and current evidence |
|---|---|---|---|---|
| Current spatial DROID | Three observed images; two past and ten supplied query-command blocks → ten future 4×4 DINOv2 feature grids | Native training-standardized feature MSE, step10; equal windows within episode, then equal episodes and seeds; paired session × seed intervals | 851 training /143 development recordings; h10 eligible141 episodes,59 sessions,1,631 windows. Checkpoint: window-weighted all-ten development loss, earliest minimum | Matched AR, learned additive anchor, action-free and context/mixing/bounding controls. 21 full 30-epoch models;36 paired-bootstrap contrasts =32 method comparisons +4 interactions. Completed development evidence, not an untouched final test |
| IWS PushT | One observed image +60 native4D command rows →59 future feature grids | Planned H60 standardized MSE at stored offset59; equal windows within trajectory, then equal trajectories and matched seeds | 480 internal-train /120 internal-development trajectories; reserved10 official trajectories supply200 H60 handles. New endpoint selector; all 59 offsets trained | AR, additive anchor and bounded spatial mixing,3 seeds each; persistence untrained.600 recordings /119,887 native frames cached. **Predictor results pending** |
| IWS Box | One observed image +60 native14D command rows →59 future feature grids | Same interface-specific metric; additive anchor is primary comparator | 481 internal-train /121 internal-development; reserved10 trajectories /200 handles | Same matched grid.602 recordings /120,274 native frames cached. **Predictor results pending** |
| IWS Rope | One observed image +60 native8D command rows →59 future feature grids | Same metric; equal-task macro averages task-specific relative gains, not raw errors across tasks | 481 internal-train /121 internal-development; reserved10 trajectories /200 handles | Same matched grid.602 recordings /120,312 native frames cached. **Predictor results pending** |
| Historical DROID context model | Three-image support and actions → original2×2 frozen features; different two-context mechanism | Original registered h5 endpoint and other horizons; separate original/fresh populations and calibration controls | Original12 runs /48 evaluations; fresh96 evaluations, primary65 episodes /52 sessions; capacity36 runs /72 evaluations; h10-training12 runs /60 evaluations | Earlier Framewise/persistence/context comparisons retained, including regressions. These are not ablations or fresh confirmation of the current spatial decoder |
| Historical simulation/domain study | Simulator observations, registered actions and paired-context supervision → earlier model's features; CEM used only by its planning protocol | Forecast error and separately labeled planning outcomes | Original PushT/Reacher and completed domain extensions; protocol-specific populations | Positive and negative findings remain in the appendix. No claim that the current spatial model was evaluated for physical control or inherited these outcomes |
| External SOTA reproduction | Requires matched data, encoder, preprocessing, commands, loss, decoding and inference budget | To be fixed before results | Not complete | No external SOTA-superiority claim. The full-RGB DINOv2 H60 IWS recipe is not a reproduction of upstream masked-DINOv3/RLA-WM training or evaluation |

Relative MSE reduction is `100 × (comparator error − method error) / comparator error`; it is not a percentage-point increase in task success. A zero comparator error makes this ratio undefined. “Additive anchor” is the learned `Z0 + R` control, not persistence. The main DROID gains are 3.01% at h5 and 5.30% at h10 versus matched AR; the h10 reduction against additive anchoring is 3.55%.136 of141 episode-level means improve and 5 regress. Incremental bounding/support-context effects and all interaction intervals remain inconclusive; the unbounded arm has a slightly better h5 point estimate. These limitations remain visible in the paper.

## IWS information and split contract

IWS has N recorded images and N command rows. H60 means observation at stored index s, target at s+59, and commands s…s+59. It does not establish physical seconds or60 physical transitions. The model's three temporal slots repeat one observed grid; they are not three observations. Transition context and FiLM are absent. Native commands enter a unidirectional prefix GRU; a prediction at offset k accesses rows0…k only. Future targets enter loss/scoring, never prediction.

All 1,804 internal-train/development recordings and 360,473 native frames have complete caches. Normalization fits internal-training native frames/commands only. The prior preview of PushT trajectory000010 remains disclosed in the frozen split audit; it belongs to internal development and was not moved. Official validation images/commands remain reserved. Ten official trajectories per task are not200 independent trajectories or verified independent sessions.

The proposed grid is 27 learned runs, 30 epochs each, plus persistence. Architecture, recipe, exact metrics and checkpoint identities must be locked before reserved payload access. The internal-development and official finalizers are separate; this implementation cannot populate official results.

## Completion and publication gates

`scripts/real_video_iws/finalize.py --if-ready` is called after per-run evaluation. Before all 27 receipts exist it returns pending and leaves paper results unchanged. An exclusive process lock serializes concurrent finalizers. It validates:

1. The registered task × method × seed grid and unchanged scientific sources/configuration.
2. Each full 30-epoch training journal, earliest selected checkpoint, atomic selected/last package identities and optimizer/RNG package integrity through the audited trainer.
3. The internal-development population reconstructed from authorized cache metadata, without opening raw or reserved payloads.
4. The exact sibling NPZ window-ledger SHA, integer window indices, all 59 offsets and every metric/persistence field.
5. Independent raw-window-to-trajectory aggregation, endpoint agreement with selected-checkpoint validation, and identical persistence across matched arms/seeds.
6. Unchanged evidence at completion. The final receipt is atomic and immutable; retries revalidate bound bytes rather than replace it.

The passed receipt is `reports/real_video_iws/development_finalization.json` with schema `shiftwm_iws_development_finalization_v1`. It binds all 27 evaluation paths/checkpoints and source files. `paper/scripts/refresh_experiment_alignment.py` rechecks the complete contract before computing equal-trajectory/equal-seed means and paired seed × trajectory intervals. Bootstrap draws share seed identities across tasks; trajectory draws are independent within each task; the macro relative-gain statistic is recomputed inside each draw. All signs, tasks and methods are retained. Official receipts and training-log scores are never used as a fallback.

Generated files are under `paper/generated/experiment_alignment/`. The main discussion reads its current transfer status, and the appendix reads the interface matrix. Pending cells contain words, not invented numbers. Completed development numbers will be added only after the gate passes. Public checkpoint counts currently distinguish 117 verified released predictors from 6 further locally verified component predictors; these span multiple historical and current studies, not 123 versions of the proposed method.

## Implementation and compute status

The older `current_results_and_gpu_status.md` is a timestamped research-status snapshot whose “figure development” activity description is now stale. Work has moved to IWS model/trainer/evaluator/finalizer implementation and full-shape resource review. Train-only full-H60 resource profiles exist for all three arms, including a full batch64 profile; they are resource measurements, not predictor results. At this report's read-only scheduler check no user-owned running/pending jobs were listed; root controls registration and subsequent submission.

The verified account limit is two ws-ia jobs with a 24-CPU account cap, plus one GPU-partition GPU with a 16-CPU cap. Three simultaneous GPUs require an actually free allocated third device. Idle-node totals do not authorize 12 concurrent training jobs. Runtime estimates must come from the actual full-shape profiles and include all 27 runs, validation and evaluation; this report gives no unsupported completion deadline.

## Validation and scope

The finalizer/reporting tests use temporary synthetic receipts solely to test integrity and arithmetic. No synthetic values enter the manuscript. The pending appendix compiled with the official ICLR2027 style and its actual PDF was inspected. Current completed scientific records, accepted Figures 1/2/6/7, reserved data and model-training source files were not modified by this editorial/reporting task. Exact source and proof hashes are recorded in `reports/evidence/experiment_alignment_review.json`.
