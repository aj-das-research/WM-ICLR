# Figure 2 planning-flow repair

Date: 2026-09-18. Canonical geometry: `world_method.tex`.

## Scope and reading task

Repair the unexplained fan between LeWM predictor P and CEM. Preserve the existing 5.5-inch-wide A/B/C composition, visual emphasis on the proposed contexts and paired supervision, observed images, and training-only boundary. The conventional planning machinery stays gray, with calibrated goal features blue. A small height increase is permitted to obtain clear routing.

Given calibrated observed history and action candidates, the predictor produces candidate latent futures; terminal latent error against the calibrated goal scores each candidate; CEM refits its action distribution to low-cost elites, resamples, and ultimately executes the first block of its final mean plan.

## Node and edge contract

| Visible object | Meaning and connection |
| --- | --- |
| Candidate action strip | Sampled candidate action sequences, input to action embedding/conditioning. |
| A_d composed with E_a | Candidate action embedding conditioned on inferred dynamics context c_d. |
| LeWM P | Reused predictor; receives calibrated support features and conditioned candidate actions. No online weight updates. |
| Three latent rows | Schematic candidate predictions. Reading follows P toward CEM. Repeated small feature tokens denote latent states, not generated pixel frames or observed future states. Outlined final token is terminal predicted latent. |
| Goal MSE block | Compare each terminal prediction with the same calibrated goal z_g. Exact cost is mean squared error across terminal feature coordinates. |
| J_i edge | One scalar cost per candidate, forwarded to CEM. Symbols are schematic; no scores or gains are fabricated. |
| CEM refit | Rank costs, retain low-cost elites, refit mean and population standard deviation, and resample action plans. |
| Mean plan → first block | CEM returns the final updated mean, not a particular best candidate. Execute its first block and obtain real next observations. |
| Feedback loop | Candidate resampling only, not model optimization or parameter adaptation. Any unavoidable crossing must have an explicit continuous overpass and no junction dot. |

## Evidence and provenance

`src/shiftwm/evaluate.py` terminal feature cost and the imported CEM solver establish the contract; root independently verified elite refitting, returned final mean, and first-block execution. Existing deployment and training image assets remain unchanged and retain their existing manifests. All new feature tokens and candidate identities are explanatory vector schematics, not measured results. No external visual asset or generated image is needed to explain latent prediction and matching.

## Review requirements

Compile to `paper/build/planning_flow_redesign/` only, leaving canonical exports to root integration. Inspect actual whole-figure pixels at 5.5 inches, enlarged planning strip, arrowheads and crossing detail, and grayscale. Root independently reviews the diagram and final manuscript insertion. Any diagram/caption mismatch or false implication of predicted image frames, best-candidate execution, or measured candidate ranking is a blocking defect.
