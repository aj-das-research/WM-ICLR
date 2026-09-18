# Drone fixed-candidate action-ranking diagnosis

This is a four-task development diagnostic, not a closed-loop benchmark. All six seed-zero models scored the same 32 candidates before any counterfactual branch was simulated. Every branch independently reset and replayed the exact 10-command support. No test tasks were used.

| Model | Prediction–physical rank correlation | Prediction–realized latent rank correlation | Realized latent–physical rank correlation | Mean regret (mm) | Safe winners |
|---|---:|---:|---:|---:|---:|
| Transformer / Framewise | 0.612 | 0.437 | 0.674 | 45.6 | 4/4 |
| Transformer / Constant dynamics | 0.597 | 0.422 | 0.675 | 40.9 | 4/4 |
| Transformer / ShiftWM (ours) | 0.594 | 0.422 | 0.675 | 40.9 | 4/4 |
| GRU / Framewise | 0.587 | 0.362 | 0.674 | 136.6 | 4/4 |
| GRU / Constant dynamics | 0.579 | 0.359 | 0.675 | 136.6 | 4/4 |
| GRU / ShiftWM (ours) | 0.508 | 0.278 | 0.675 | 136.6 | 4/4 |

Spearman correlations are computed within task over safe full-horizon candidates and then averaged. Regret is the selected candidate's physical distance minus the best safe candidate's distance; it is undefined if the predicted winner is unsafe. The full JSON retains all candidates, exclusions, random-reference means and per-task values.

Frozen canonical feature-to-goal geometry (same encoder for every model), correlation with physical distance per task: 0.803, 0.808, 0.891, 0.754.

Interpretation: weak prediction–realized-feature ranking suggests action/dynamics forecasting is not ordering futures correctly. Weak realized-feature–physical ranking suggests the visual goal metric is not aligned with physical progress. Both can occur; this small controlled branch study cannot establish a unique cause.

Artifacts: `reports/evidence/drone_action_ranking`; elapsed 118.8s. Candidate/source/checkpoint hashes and raw commanded sequences are retained. No model, dataset, frozen evaluator or simulator source was changed.

## Observations from these four tasks

The transformer variants select much better candidates than the GRU variants in this fixed library. Mean realized terminal distance is 61.7 mm for transformer ShiftWM and constant dynamics, 66.4 mm for transformer Framewise, and 157.5 mm for all GRU variants. The uniform mean across the candidate library is 177.4 mm; this is a library-average reference, not the separately evaluated random closed-loop controller.

ShiftWM and constant dynamics choose the **same candidate in all four transformer tasks**. All three GRU methods likewise choose the same candidate in each task. Thus the small transformer advantage over Framewise does **not** establish an episode-dependent dynamics-context benefit.

Canonical frozen features preserve useful goal-distance ordering (mean Spearman approximately 0.814). The adapted goal's ordering is weaker (approximately 0.675), while predicted versus realized latent-cost ordering is only 0.422 for transformer ShiftWM and 0.278 for GRU ShiftWM. These observations motivate improving goal correction and action-conditioned prediction/ranking, rather than treating the representation as entirely blind to position. They do not isolate one unique cause of closed-loop failure.

Two of 128 branches hit a workspace boundary before the full horizon. All branches remain in the JSON, all selected model winners are safe, and only equal-horizon safe branches enter the primary rank/regret calculations. Exact support replay passed for every branch. Independent numerical recomputation of all 24 argmins, correlations and regrets passed; see `drone_action_ranking_validation.json`.

The XY-distance ranking is a partial-objective diagnostic. Native success also requires speed below 0.06 m/s, altitude error below 0.05 m, and no crash or workspace escape. A candidate can improve position while failing to settle; the distance correlations and regrets therefore do not establish success-policy optimality. Per-step speed, altitude, and success flags are retained in the raw diagnostic evidence.
