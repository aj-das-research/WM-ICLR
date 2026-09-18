# Goal-cost explanation: mathematical review

Reviewed independently by the paper agent on 18 September 2026, before insertion
in `paper/main.tex`. This is an elementary derivation, not a novelty claim.

For normalized Euclidean norm and a finite nonempty candidate set, uniform
terminal-prediction and goal errors combine by the triangle inequality into
epsilon = e_f + e_g. Expanding the squared residual and applying Cauchy--Schwarz
gives cost error at most B = 2 R epsilon + epsilon squared. Applying this bound
on both sides of an eta-approximate estimated-cost minimizer gives canonical
latent regret at most 2 B + eta. The manuscript includes the proof.

The reviewer confirmed the scalar exact-prediction counterexample: predictions
0 and 1, true goal0, estimated goal1 reverse the two candidate costs. This is
clearly identified as an illustrative mathematical example, not an experiment.

Required limits retained in the manuscript: the returned sequence belongs to
the specified candidate set; the errors are uniform norm bounds, not empirical
mean MSEs; the deployed CEM optimizer has no certified eta; latent-cost regret
does not guarantee physical task success. Neither the derivation nor the example
proves causal attribution for the observed planning failures.
