# Results diagnosis — September 18, 2026

The results are mixed. The strongest completed result is lower held-out
forecast error; a planning advantage is not established. At this audit,
53/64 main planning evaluations and all eight random/replay controls are
complete. Framewise and Unpaired comparisons still lack the prescribed
three-seed evidence. Their missing cells are not negative measurements.

## What is positive, negative, or unresolved?

| Comparison | PushT | Reacher | Interpretation |
|---|---:|---:|---|
| Held-out forecast error vs Framewise calibration | 3.68% lower | 26.78% lower | Positive means across three seeds |
| Held-out forecast error vs Unpaired contexts | 2.30% lower | 9.21% lower | Positive; closest aligned held-out control |
| Extrapolation forecast error vs Framewise | 4.20% lower | 4.15% higher | Mixed |
| Held-out planning vs Shared context | +0.52 pp, CI [−1.04, +2.08] | +4.17 pp, CI [−4.17, +12.50] | Both inconclusive |
| Extrapolation planning vs Shared context | −0.87 pp, CI [−2.95, +1.22] | +2.78 pp, CI [−1.91, +7.64] | Both inconclusive |
| Held-out planning: ours / random control | 2.60% / 7.81% | 40.63% / 4.69% | PushT is below random as a point estimate |
| Development success: ours / Framewise | 2/32 / 2/32 | 8/32 / 13/32 | No development advantage; one training seed |

Forecast reductions are ratios of three-seed mean MSEs, not significance
estimates. Planning confidence intervals are conditional paired initial-state
cluster intervals over the three observed trained models. The random policy
has one fixed evaluation RNG; its comparison is descriptive, not a matched
three-training-seed significance test. Replay of privileged recorded future
actions succeeds on every evaluated task: these goals are reachable within
the replay protocol. That does not establish correctness of every learned
planning operation.

## Most strongly supported failure: goal calibration

On Reacher development, ShiftWM's recorded-action prediction has canonical-goal
MSE **0.01174**, but its calibrated goal has MSE **0.79675**. Framewise goal MSE
is **0.27120**. The planner scores distance to the calibrated goal, so a good
prediction model can still receive an inaccurate target. The checked Reacher
live goal renders match the archived frames; the observed discrepancy is not
explained by that renderer mismatch.

The controlled, post-hoc goal-only swap keeps the ShiftWM predictor, history
calibration, contexts, and planner fixed. Reacher eligible successes change
from **6/30 to 12/30**; the paired difference is +20 pp with exploratory
95% interval [0, +40]. PushT changes from **1/31 to 0/31**. The hybrid does not
establish superiority to the complete Framewise controller (12/30 versus
11/30 on Reacher; paired interval [−20, +26.67] pp). This is evidence that the
goal path matters on those development examples, not a general repair.

## Concrete design limitations whose causal effects are not yet isolated

1. **Calibration capacity.** Our observation adapter uses one history-derived
   diagonal feature scaling and shift, shared across history and goal. Its
   multiplicative scale is bounded to 0.9–1.1. Framewise calibration is an
   image-dependent nonlinear MLP. A shared affine map may not undo the
   state-dependent effect of RGB changes through a nonlinear frozen encoder.
   The measured goal errors are consistent with this explanation; a matched
   capacity intervention is still needed to isolate it.
2. **Training does not match recursive planning.** Training uses observed
   histories for one-step targets 4–7, whereas deployment starts by predicting
   frame 3 and recursively feeds predictions into later steps. The first
   support/query boundary is absent from the prediction loss. This mismatch
   exists in the code; its contribution to failure has not been measured.
3. **The planner searches different actions from the recorded trajectories.**
   Lower average forecast MSE does not guarantee correct ranking of optimized
   candidates. PushT also has a recorded training-action spread about 2.4 times
   the released normalization spread. Candidate-ranking errors and this
   distribution difference need controlled development tests before attribution.
4. **Useful context inference has not been demonstrated.** Pairwise consistency
   encourages matching contexts but does not ensure they encode the intended
   factors. Small consistency losses or reduced context variance can also
   reflect rescaling. Constant/shuffled-context interventions must inspect
   effective adapter outputs and planning, not only raw context distances.
5. **Extrapolation remains weak.** Frozen LeWM has lower Reacher extrapolation
   error than the adapted aligned models. This shows the adaptation does not
   preserve every generalization regime; it does not by itself prove a specific
   forgetting or optimization mechanism.

All 30 runs completed 30 epochs with finite logged train/validation metrics.
Validation prediction error improved for all six ShiftWM runs. The selected
PushT checkpoints are early (epochs 3–4); simply extending training is not
justified by these curves. These checks rule out missing completion records
and logged NaN/Inf losses; they do not prove that the entire system is bug-free.

A separate source and deterministic CPU check ruled out an apparent action-range
bug: the imported CEM solver does not clip proposals or elite means to the
configured Box. It searches unbounded normalized Gaussian actions and clips
only after conversion to native physical bounds. Changing the Box alone left
the fixture output bitwise unchanged, and optimized normalized actions exceeded
one. The initial PushT proposal distribution is narrower than native-uniform
random actions, which remains a search-distribution hypothesis rather than a
hard restriction or established failure cause. Reproduction and source hashes
are in `evidence/cem_action_range_audit_2026-09-18.json`.

## Next decisions

- Finish the fixed three-seed Framewise and Unpaired planning comparisons,
  keeping the current protocol and all unfavorable results.
- Run the already implemented, queued development revision: keep the entire
  completed Framewise model and goal calibration fixed, then train a small
  zero-initialized dynamics residual. This isolates adding the residual from
  changing the visual path. It is not trained or evaluated at this snapshot.
- If that helps, compare it with a matched trained constant-context residual
  before attributing a gain to inferred dynamics information.
- Separately test the missing boundary and recursive training objective with
  matched controls on development data. The queued revision deliberately
  preserves the old objective, so it cannot answer this question.
- Prioritize a credible PushT control improvement over further broad efficacy
  claims. Do not select settings using final-test outcomes. Any development-led
  revision must retain the original negative results and selection history.

Sources and exact hashes: `evidence/results_diagnosis_2026-09-18.json`.
The validated main and paired ledgers used for this snapshot are archived in
`evidence/results_diagnosis_2026-09-18/`. Additional controls are reported
separately from learned-model comparisons in the manuscript.
