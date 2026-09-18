# Extension mechanism diagnosis and next controlled revision

Diagnostic snapshot: 2026-09-18T 21:22:43.678951+00:00; planning snapshot: 2026-09-18T 21:25:45.882208+00:00. **The present evidence points more strongly to weak goal-distance geometry and limited use of the additional dynamics branch than to total action blindness or constant inferred contexts.** These are bounded development diagnostics, not a causal explanation or a final-test result.

## What the completed study actually established

All eight earlier rollout-revision arms completed. On PushT, inferred context scores1/31 under either objective, tying its frozen donor; constant context scores0/31. On Reacher, teacher-forced inferred/constant score 11/30 and 15/30; recursive inferred/constant both score 13/30, against donor 11/30. Neither objective passes the registered requirement to exceed both controls in both environments. The apparent Reacher gain after recursive training is shared by a context-constant control. Do not describe it as proof of episode identification.

Code permits several explanations: the predictor itself already receives temporal observations and actions, so an explicit dynamics vector can be redundant; appearance-consistency regularization admits a constant dynamics code; and training minimizes average latent forecast/alignment errors rather than ranking candidate actions by realized goal progress. These are plausible mechanisms. The completed comparisons alone cannot identify which caused the observed ties and regressions.

## New drone probe: observed facts

All six completed seed-zero drone checkpoints were validated using full 30-epoch histories and strict package loaders. The probe reads **development** cached features and separately stored physical audits only: all 48 trajectories, at fixed window starts0 and 16. The following measurements use the warm/gain 0.75 subset (32windows from 16trajectories). Context shuffling is a fixed roll 17 across all 96 support contexts; observation context, observed history and actions remain unchanged. No training, final-test model outcomes, or new simulator actions were used.

| Predictor / method | Selected recursive val MSE | Probe MSE@5 | Zero-action MSE@5 | Dynamics relative SD | Context-shuffle error change |
|---|---:|---:|---:|---:|---:|
| transformer / framewise | 0.0051986 | 0.0054892 | 0.0071471 | n/a | +0.000% |
| transformer / factorized | 0.0052479 | 0.0055594 | 0.0070730 | 0.218 | -0.037% |
| transformer / constant_dynamics | 0.0052509 | 0.0055503 | 0.0070553 | 0.000 | +0.000% |
| gru / framewise | 0.0061967 | 0.0063178 | 0.0067986 | n/a | +0.000% |
| gru / factorized | 0.0062062 | 0.0062519 | 0.0071794 | 0.285 | +3.694% |
| gru / constant_dynamics | 0.0062234 | 0.0063050 | 0.0068195 | 0.000 | +0.000% |

1. **Inferred contexts are not numerically constant.** Relative dynamics-code SD is0.218 for the transformer and 0.285 for the GRU. Nevertheless, shuffling the transformer dynamics context changes forecast error by−0.037%, whereas the GRU worsens 3.69%. Context variability therefore does not guarantee that its episode information materially affects prediction. The shuffle is exploratory and not an identification test; only the fixed-history code path is intervened on.
2. **Models respond to actions.** Across the six models, recorded-versus-zero predicted action-effect MSE is 31–39% of the observed canonical five-step displacement MSE. This is a scale comparison, not a calibrated action-effect estimate: the zero-action physical rollout was not executed. Recorded actions give lower probe forecast error than zero actions for every model. The existing complete GRU factorized development forecast independently reports MSE@5=0.0064574 versus0.0080605 under zero future actions (19.9% lower), across144 windows/16 trajectories. Its forecast quality does not establish planning quality.

### Goal geometry and a concrete capacity bound

Use all 16 warm/gain 0.75 development endpoint goals, producing 120 location pairs. Physical XY distance is diagnostic metadata; it never enters model inference. Each corrected-goal comparison fixes one support context across all 16 goals, then averages over 16 contexts, preventing context variation from masquerading as location sensitivity.

| Goal representation | Mean pairwise latent MSE | Spearman vs squared XY distance |
|---|---:|---:|
| Canonical frozen encoder | 0.0102054 | 0.614 |
| Warm frozen encoder | 0.0017757 | 0.490 |
| transformer / framewise corrected | 0.0017392 | 0.494 |
| transformer / factorized corrected | 0.0016554 | 0.491 |
| transformer / constant_dynamics corrected | 0.0016518 | 0.490 |
| gru / framewise corrected | 0.0017387 | 0.494 |
| gru / factorized corrected | 0.0016345 | 0.492 |
| gru / constant_dynamics corrected | 0.0016305 | 0.493 |

Warm appearance compresses average pairwise latent distance by 82.6%. Learned corrections leave approximately 16–17% of canonical separation. This is compression, not complete loss of distinguishability; canonical distance itself has only moderate association with physical XY distance and ignores the speed/altitude components of success.

The original observation FiLM has `G(u,c)=u*(1+0.1*tanh(scale(c)))+shift(c)`. At fixed context, translation cancels between two goals and every scale lies in[0.9,1.1]. Consequently, `0.81*||u−v||² <= ||G(u,c)−G(v,c)||² <=1.21*||u−v||²`. No optimization can exceed this fixed-context bound. For the measured warm goals, even the upper bound is only 21.1% of their canonical mean separation. This establishes a representational limitation of that correction family on these cached features; it does not establish that this limitation caused a particular failed trajectory. Framewise calibration has no such algebraic bound but also fails to restore separation in this snapshot.

## Actual selected actions and physical progress

Saved completed development records and trace hashes were checked against their live identities. Correlations below align predicted **next-block** goal MSE with physical distance after exactly five executed native commands. Partial terminal blocks are excluded. The 25-command terminal prediction is deliberately not compared with a five-command observation. No support success occurred in these records.

| GRU method / seed | Successes | Five-command decisions | ρ(cost, next XY distance) | Median within-task ρ | Fraction reducing XY distance |
|---|---:|---:|---:|---:|---:|
| constant_dynamics_s0 | 1/8 | 279 | 0.016 | 0.048 | 37.6% |
| constant_dynamics_s1 | 4/8 | 193 | -0.028 | 0.070 | 42.0% |
| factorized_s0 | 1/8 | 266 | 0.118 | -0.053 | 33.5% |
| framewise_s0 | 1/8 | 280 | 0.077 | 0.163 | 38.6% |
| framewise_s1 | 3/8 | 221 | 0.152 | 0.145 | 38.9% |

**These are not candidate-ranking accuracy measurements.** Only the chosen candidate is executed, contexts/goals change over time, decisions are serially dependent, and planning stops early on success/failure. A positive correlation would indicate descriptive cost calibration; these weak correlations support investigating the learned metric but cannot prove that a different candidate would have succeeded. The three-seed comparison is incomplete at this snapshot; individual favorable seeds must not be promoted.

## One immediate revision, with explicit limits

**Test an observation-only gain-range revision before adding another loss.** Replace only the observation FiLM gain by `exp(log(4)*tanh(0.1*scale/log(4)))`, with unchanged translation, context networks, parameter shapes and zero initialization. This permits gains [0.25,4] and squared-distance expansion up to 16×, which removes the demonstrated 1.21× ceiling. Keep the dynamics action FiLM unchanged. Register six transformer runs: factorized and constant-dynamics × seeds 0/1/2, full 30 epochs, same recursive selection, data, optimizer and planning budget; compare to the already trained corresponding original arms. Preserve all outcomes.

This is a capacity diagnostic, **not an algorithmic-novelty claim**. The new gain has derivative 0.1 at zero, matching the original adapter. Identity, parameter count, translation and first-order scale sensitivity are preserved; higher-order curvature and attainable range still differ. This is a bounded capacity/geometry ablation, not complete equivalence of optimization away from initialization. Require improvements in development goal separability and paired closed-loop success over both corresponding old arms and the constant-dynamics control; do not declare the revision successful from forecast MSE alone.

If metric repair remains insufficient, the next action-ranking hypothesis needs **genuinely branched training data**: replay one recorded observed prefix identically, execute multiple short candidate command sequences from that same prefix, and supervise their relative achieved goal progress with identical access and losses for Framewise, inferred-context and constant-dynamics controls. Do not implement or merge that additional objective into the present gain-range comparison; it would confound the immediate test.

The existing drone matched-seed/gain data are not already such causal pairs. Its collector recomputes `clip(4*(target−currentXY)+noise)` from simulator state. All 32 checked nominal-versus-shifted command-sequence pairs differ; only 21–28 of 200 individual commands are identical, largely including the final settling zeros. Shared seed/noise/target schedule is not shared actions. Surgery uses a different collection protocol and needs a separate command-identity check.

## Reproducibility and scope

- Probe code: `reports/evidence/extension_mechanism_probe.py`; measurements, definitions and source hashes: `reports/evidence/extensions_mechanism_probe.json`.
- Planning alignment code: `reports/evidence/extension_planning_probe.py`; selected-action rows and checked trace hashes: `reports/evidence/extensions_planning_probe.json`.
- Completed prior study: `reports/expansion_status_2026-09-19.md`.
- No weights, frozen scientific source, registered configs, manuscript claims, or final-test model outputs were changed for this diagnosis. The proposed revision requires its own namespace and immutable protocol before any training.
