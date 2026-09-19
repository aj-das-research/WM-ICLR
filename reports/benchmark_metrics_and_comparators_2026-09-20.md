# ShiftWM benchmark, metric and comparison audit

Audit snapshot: 20 September 2026, 03:15 Dubai time. This report distinguishes measured results, recommended additions, internal controls, external methods and unfinished evaluations. It does not change a scientific registration, select a model, or promote preliminary results into manuscript tables.

## Scope and counting

There are eight evaluated task settings: four simulated control tasks (PushT, Reacher, planar drone, SOFA tissue manipulation), plus four recorded-video forecasting settings (DROID, IWS PushT, IWS Box, IWS Rope). The four recorded settings span two dataset families. Open-H is an acquired input example, with no model-performance evaluation.

The simulator results belong to the earlier context-based ShiftWM. The current spatial-mixing ShiftWM has been evaluated on DROID and IWS development data. Simulator scores are not evidence that the current spatial model already controls those environments. Camera views, split revisions, random seeds, checkpoints and repeated evaluations are not additional methods or benchmarks.

All configuration counts below include ours. Internal ablations do not count as independently published algorithms.

## Evaluated simulator studies

| Task | Predictor comparisons | Completed models | Evaluation status |
|---|---|---:|---|
| PushT | Six predictors: Frozen LeWM; Framewise calibration; Shared context; Unpaired contexts; historical factorized ShiftWM (ours); Unaligned predictor. The main comparison displays five, with Unaligned as an appendix diagnostic. | 15 original trained models: five trainable configurations × three seeds; released Frozen LeWM has no additional training | Original forecast and planning grid complete; 576 in-range task conditions/checkpoint, including the 64-task held-out-composition subset, plus 192 extrapolation tasks/checkpoint |
| Reacher | Same six predictors and main/appendix distinction | 15 original trained models | Same completed grid structure as PushT |
| Planar drone | Three method families: Framewise calibration, Constant dynamics, historical factorized ShiftWM; each on a LeWM transformer and an in-house GRU = six configurations. A wider-observation-gain revision adds two transformer configurations: Constant dynamics and ShiftWM. | 18 original + six revision models = 24 | Development forecasting and planning complete. Planning has eight physical tasks repeated across three training seeds. Final model test evaluation remains unrun. |
| SOFA tissue manipulation | Same three method families × two backbones = six configurations | 18 | Development forecasting and planning complete, eight physical planning tasks × three training seeds. Final model test evaluation remains unrun. |

Random action and privileged recorded-command replay references are additional policy controls, not trained world-model algorithms. The GRU is an internal architecture control, not a Dreamer reproduction.

Additional PushT/Reacher development studies completed two frozen-Framewise dynamics-residual runs and eight runs crossing teacher-forced/recursive training with inferred/constant context over two environments, at seed zero. These are revisions/ablations, not ten new methods. The audited historical simulator campaigns therefore contain 82 trained runs: 30 original core + 42 extensions/revisions + 10 core development revisions. Duplicate checkpoint packages and analytic interventions are excluded.

### Simulator metrics

| Metric | Meaning and reporting contract |
|---|---|
| Closed-loop success ↑ | Primary outcome: count and percentage under the exact environment evaluator and native-action budget. Keep all-task and policy-eligible success separate when support actions already solve an episode. Report per shift and backbone. |
| PushT physical error ↓ | Position error in native pixels and symmetry-aware block-angle error in radians. Current joint criterion requires combined pusher/block position error <20 pixels and angle error <π/9. This is the local goal-reaching evaluator, not an interchangeable coverage score. |
| Reacher physical error ↓ | Each unwrapped joint error must be <0.05 radians. Wrapped joint-distance plots are diagnostics, not the success criterion. |
| Drone physical error/failures ↓ | XY error <0.04 m, speed <0.06 m/s, altitude error <0.05 m, and no crash/workspace escape are jointly required. Report distance, speed and failure categories separately. |
| Tissue physical error ↓ | Native XZ tissue-point distance ≤2 mm. Report timeout and measured simulator failures separately; this is not a clinical endpoint. |
| Forecast MSE@1/@3/@5 ↓ | Error in the same frozen 192-dimensional reference representation; five prediction blocks correspond to 25 native commands. Feature accuracy is a secondary diagnostic, not control success. |
| Interaction cost ↓ | Native commands to success, with explicit handling of failures/censoring. Core total budget is 50 commands including 10 support commands; extension budget is 200 including 10 support. |
| Compute cost ↓ | Planning latency and peak memory on matched, uncontended hardware. Existing runtime logs do not establish a complete fair efficiency comparison. |

Current simulator results are mixed. The 12 original learned-comparator planning contrasts have six positive mean differences, five negative and one tie; none has a paired confidence interval strictly above zero. Drone/tissue rankings also change across backbones. Neither favorable forecast error nor a selected qualitative success establishes general planning superiority.

Sources: [simulation protocol](../paper/sections/appendix/simulation_protocols.tex), [extension protocol](../paper/sections/domain_extensions.tex), [completed extension results](completed_extension_results.md), [original completion ledger](evidence/progress_2026-09-19.json), [dynamics revision](../paper/generated/dynamics_revision.json), [rollout revision](../paper/generated/rollout_revision.json).

## Evaluated recorded-video studies

| Task/study | Exact predictors or variants | Completion and scope |
|---|---|---|
| Current DROID spatial | Eight predictors: Autoregressive; Persistence; ShiftWM bounded mixing (ours); No mixing/bounded anchor; No bounding; No mixing or bounding/additive anchor; No support context; No actions | Seven learned arms × three seeds = 21 completed 30-epoch models. Development: 141 eligible episodes, 59 sessions, 1,631 windows. |
| IWS PushT | Four completed predictors: Persistence; Autoregressive; Additive anchor; ShiftWM bounded mixing (ours). Unbounded mixing is one ongoing additional ablation. | Nine completed original models, plus one completed follow-up seed; two follow-up seeds unfinished. Development: 120 trajectories, 3,360 windows. |
| IWS Box | Same four completed predictors plus one ongoing ablation | Nine original models plus one follow-up seed; two unfinished. Development: 121 trajectories, 3,388 windows. |
| IWS Rope | Same four completed predictors plus one ongoing ablation | Nine original models plus one follow-up seed; two unfinished. Development: 121 trajectories, 3,388 windows. |
| Open-H endoscopy sample | Zero trained/evaluated methods | One physical stomach-phantom episode downloaded: 257 frames and command rows. Acquisition, decoding and timestamp audit only. Not a clinical evaluation or a model-result benchmark. |

IWS has 27 completed original models and nine planned unbounded-ablation models. Three follow-up seed-zero models are completed and independently verified; three seed-one jobs are running and three seed-two jobs queued at this snapshot. No full-nine finalization exists. The official reserved IWS validation payloads remain unread. Full-campaign comparisons are withheld from the manuscript until all runs and the registered finalizer pass.

### Historical DROID studies: same dataset, not extra benchmarks

| Study | Methods/configurations | Completed work |
|---|---|---|
| Original DROID | Six predictors: Framewise; Constant dynamics; historical ShiftWM real-video variant; Action-free; Persistence; Constant feature velocity | Four learned methods × three seeds = 12 models; 48 learned evaluations across cameras/horizons. Original test: 132 episodes at h5 and 130 eligible at h10. |
| Residual calibration | Same four learned methods, original and calibrated = eight variants | 12 training-fitted scalars; no new neural training. Development validation only. |
| Fresh-session DROID | Eight original/calibrated learned variants plus two unchanged deterministic baselines = ten rows | 96 learned evaluations reusing original checkpoints; 65 fresh episodes/52 sessions at h5, 64 eligible episodes at h10. |
| Optimization/capacity controls | Four historical methods × Slow learning rate, Decay schedule, Compact capacity = 12 learned configurations | 36 completed models; development only. Saved persistence/constant-velocity predictions are deterministic references. |
| h10-training control | Four historical method families, each h5-trained and h10-trained = eight learned training variants in paired comparison | 12 new h10-trained models, compared with original h5-trained checkpoints; development only. |
| Longer-prefix support reliability | Eleven arms: Framewise calibrated; ShiftWM calibrated; Equal blend; Global-prior blend; Train-query constant blend; Unshrunk support gate; Shrunk support gate; One-step support gate; Shuffled support gate; Persistence; Constant velocity | Completed development screen, no new neural weights. Uses 13 observed frames and 132 validation episodes. Candidate failed the promotion rule. Some arms yield identical predictions. |

### Recorded-video metrics

| Study | Measured metrics | Horizons and interpretation |
|---|---|---|
| Current DROID spatial | Training-standardized native 4×4 feature MSE ↓; separately standardized pooled 2×2 MSE ↓; paired relative error reduction % ↑ | h5/h10 endpoints and curves. One query block contains five native commands; h5/h10 represent 25/50 future native transitions. Native and pooled errors use distinct coordinates/scales and cannot be directly ranked against each other. |
| Historical DROID original/fresh | Standardized feature MSE ↓; raw feature MSE ↓; cosine error ↓; endpoint and all-query means | Original h1/h3/h5 and h10 extrapolation; fresh primary camera-1 h5. Aggregate windows within episodes, with session-aware uncertainty where available. |
| DROID calibration/generalization/horizon controls | Primarily standardized MSE and relative error reductions; generalization/horizon saved evaluations also retain raw MSE/cosine | h10-training primary diagnostic is mean error over all ten future queries, not only endpoint. |
| DROID reliability | Standardized endpoint MSE and gain against global-prior blend | h5/h10 with a longer observed prefix; not directly the same information budget as the main three-frame model. |
| All three IWS tasks | Standardized feature MSE ↓; standardized MAE ↓; raw DINOv2 feature L1 ↓; flattened raw-feature cosine distance ↓ | All offsets 1–59; H15/H30/H45/H60 denote stored offsets 14/29/44/59, not verified seconds. Equal-window averages within trajectories, equal-trajectory and equal-seed aggregation. |

Standardized errors and cosine distance are dimensionless; raw L1 uses feature units and raw MSE squared feature units. Models must share the frozen encoder, preprocessing, normalization and target coordinates for direct feature-error comparison. Logged-action forecasting does not measure robot-control success.

Sources: [main evaluation scope](../paper/sections/main_evaluation.tex), [DROID spatial finalization](real_video_spatial/finalization.json), [DROID component finalization](real_video_spatial_components/finalization.json), [IWS results](real_video_iws), [unbounded IWS evaluations](real_video_iws_unbounded/evaluations), [Open-H audit](real_openh_video_audit.md), historical `real_droid_*_results.json` reports and active manuscript sections.

## New findings: preliminary single-seed follow-up

The three unbounded-mixing seed-zero checkpoints each completed all 30 epochs and selected epoch 30. Independent review reconstructed all 59 offsets × four metrics, verified checkpoint/recipe identities and the 101 registered scientific dependencies, and matched the exact evaluation populations to same-task original seed-zero comparators.

| IWS task | Unbounded MSE at H60 | Bounded MSE, same seed | Autoregressive MSE, same seed | Relative reduction vs bounded | Relative reduction vs autoregression |
|---|---:|---:|---:|---:|---:|
| PushT | 0.241584 | 0.261378 | 0.260042 | +7.57% | +7.10% |
| Box | 0.263645 | 0.291082 | 0.279300 | +9.43% | +5.61% |
| Rope | 0.197473 | 0.223617 | 0.211717 | +11.69% | +6.73% |

These are one-seed internal-development findings, without a completed multi-seed uncertainty estimate. They are relative feature-error reductions, not percentage-point control-success gains. They do not establish general SOTA performance. The bounded original three-seed campaign remains the completed manuscript result; autoregression has lower endpoint MSE than bounded mixing on all three tasks. No post-result changes were made to registrations, evaluation access or model-selection rules.

## Published methods actually exercised

1. **LeWM:** released backbone integrated; Frozen LeWM is an external baseline in the historical matched simulator grid. Separate upstream-protocol evaluations completed PushT 46/50 and Reacher 39/50. Those author-protocol scores have different goals/distributions and must not replace matched adaptation-grid scores. See [upstream reproduction audit](upstream_planning_reproduction.md) and [official implementation](https://github.com/lucas-maes/le-wm).
2. **AdaJEPA:** official code/checkpoint reproduction completed four 50-task PushT stages: clean frozen 34/50, clean adaptive 46/50, blur frozen 29/50, blur adaptive 39/50. These are two modes in two conditions, not four algorithms. Extra proprioceptive inputs and different planning budgets make these unmatched to ShiftWM's simulator grid. See [completed results](completed_extension_results.md) and [official implementation](https://github.com/agentic-learning-ai-lab/adajepa).

There are **zero completed matched independent external-method comparisons for the current real-video spatial model**. Internal autoregression, additive anchors, persistence and mechanism ablations should be labeled accordingly. Use of LeWM predictor code or DINO features alone does not reproduce those published methods. Cited related work is not an evaluated baseline.

## Recommended comparison completion plan

These are proposed additions, not jobs launched or completed by this audit.

| Setting | Priority external additions | Conditions for a defensible comparison |
|---|---|---|
| PushT/Reacher | LeWM, DINO-WM without proprioception, PLDM; matched AdaJEPA where compatible | Same images, goals, action interface, training data, support access and CEM/native-action budget. Released models are listed in the [official LeWM baseline suite](https://github.com/lucas-maes/le-wm). |
| Drone | Established PPO and PID controls | Same task and input/action interface. State-input controllers must be marked privileged. [Official drone framework](https://github.com/learnsyslab/gym-pybullet-drones) hover examples do not automatically reproduce our planar-goal task. |
| Tissue manipulation | LapGym PPO baseline family | Same exact task, visual/state inputs, action scaling, termination and budget; our adapted result is not an official leaderboard score. [LapGym benchmark paper](https://www.jmlr.org/papers/v24/23-0207.html). |
| DROID forecasting | LeWM/DINO-WM-style external predictors trained under our declared forecast contract | DROID's [official evaluation](https://droid-dataset.github.io/) concerns policy success/robustness; it does not supply an interchangeable feature-forecast leaderboard. |
| IWS PushT/Box/Rope | RLA-WM | Verify actual pretrained backbone and access; match handles, action indexing, representation, preprocessing and disclose training budget. Its masked DINOv3 features cannot be directly ranked against our full-image DINOv2-small features. [Official predictor/metrics](https://github.com/mlzxy/rla-wm/blob/main/eval/predictors/rla_wm_predictor_iws.py), [checkpoint inventory](https://github.com/mlzxy/rla-wm/blob/main/docs/data-and-checkpoints.md). |

For every task, retain the task-specific primary score, long-horizon or interaction-budget behavior, per-shift outcomes, paired uncertainty over independent episodes/sessions and training seeds, and a fair efficiency panel (encoder-inclusive and predictor-only latency, CEM replanning cost where relevant, peak GPU memory, parameter count and training GPU-hours).

RGB PSNR/SSIM/LPIPS and video FVD require actual predicted pixels. Current ShiftWM predicts features; these are not presently valid primary scores. RLA-WM reports DINO-token L1 and decoder-based LPIPS/SSIM. A separate image-decoder study would need its own fair protocol and target-feature reconstruction reference. Do not invent RGB scores from latent errors.

Success differences should be reported in **percentage points**. Feature gains should be reported as **relative error reduction**, 100 × (reference error − method error) / reference error, with the reference named. Negative results and all prespecified conditions remain visible.
