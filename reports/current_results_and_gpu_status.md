# Current results and GPU status — 19 September 2026

This status is based on the live scheduler and source-bound completed ledgers.
It supersedes historical “running” descriptions in overnight handoffs.

**No running or pending Slurm jobs are listed under abhijit.das at this check.**
The active work is figure development, numerical/visual review and manuscript
publication. The scheduler advertises idle GPU nodes, but those labels do not
measure device memory availability. Accounting history is unavailable; completed
runs below are established by their experiment finalizers.

| Study | Completion | What the results support |
|---|---|---|
| Current spatial ShiftWM on DROID development | 21 models ×30 epochs; 36 paired-bootstrap contrasts (32 method comparisons +4 interaction effects) | 3.01% h5 and 5.30% h10 relative error reduction vs matched autoregression; 136/141 episodes improve at h10 |
| Spatial mixing/bounding/action/context ablations | Complete, including six new follow-up models | Mixing reduces native h10 error with and without bounding; incremental bounding and support-context benefits are inconclusive |
| Original DROID historical context model | 12 training runs /48 evaluations | Small gains and long-horizon regressions retained; a different architecture from the current spatial model |
| Fresh-session confirmation of historical model | 96 evaluations /65 episodes /52 new sessions | Primary calibrated h5 gain0.742%, paired interval excludes zero; not confirmation of the final spatial model |
| Capacity/regularization development | 36 runs /72 evaluations | Four favorable and two unfavorable comparisons against matched Framewise |
| Matched h10-training development | 12 runs /60 evaluations /20 paired contrasts | 0.405% h10 endpoint improvement over equally trained Framewise; small five-step-prefix regression retained |
| PushT/Reacher simulation and domain extensions | Completed historical studies;36 extension runs/forecasts/planning evaluations | Forecast gains and planning/regression outcomes are mixed; these do not establish the final spatial model's cross-domain strength |
| Real IWS PushT/Box/Rope | Caches complete:1,804 recordings /360,473 native frames | No predictor training or evaluation yet |
| External state-of-the-art reproductions | Incomplete | No justified claim of SOTA superiority or universal5–10-point gains |

There are117 verified public predictors across the historical/current releases;
15 belong to the original spatial campaign. Six additional component predictors
are trained and verified locally. This is not117 versions of the proposed method.

## Explanation supported by the current ablations

Fixed observed features give each horizon direct access to recorded evidence.
Action-conditioned mixing improves native ten-step error by roughly3.4–3.5%
relative to the corresponding no-mixing controls. Bounding has a range guarantee
in standardized feature coordinates, but its extra native accuracy benefit is
inconclusive; the unbounded arm is slightly better at h5. Removing support context
barely changes error. Visual gate/source-weight diagnostics describe computation;
they do not establish why an individual episode succeeds or fails causally.

## Remaining empirical priorities

1. Implement, review and freeze the IWS single-observation training/evaluation
   contract, then run matched autoregressive/additive-anchor/spatial-mixing arms
   with three seeds on all three tasks (27 proposed learned runs plus persistence).
   Reserve official validation until the protocol and checkpoints are fixed.
2. Complete compatible external baseline reproductions with explicitly matched
   data, feature representation, supervision and inference budgets. The current
   compact IWS design must not be labeled an RLA-WM reproduction.
3. Freeze the final spatial choice and evaluate on a genuinely untouched
   population. Previously revealed original/fresh tests cannot be reused for
   architecture selection or described as untouched final confirmation.
4. Add source-linked qualitative cases and release the remaining component
   checkpoints once their final bundles are verified.

Figures6/7 are being redesigned concurrently. Figure polish changes no result.
The manuscript is a developed draft with incomplete empirical breadth, not an
established multi-benchmark SOTA paper. Exact hashes and scheduler output are in
`evidence/research_status_live_2026-09-19.json`.
