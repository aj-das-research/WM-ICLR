# Completed spatial component study: independent audit

**Passed training, numeric, package and relocated CPU inference checks.**
All six new models and six registered revealed controls completed 30 epochs.
The new bounded-additive checkpoints select epochs **21/25/17**; unbounded
transport selects **10/13/11**, for seeds 0/1/2 respectively. Each selection is
the first minimum of the registered window-weighted ten-query validation metric.

The audit reopened all 12 evaluation ledgers and checked all **19,572 windows**,
141 episodes and 59 recording sessions per model. Per-window values reproduce
the stored episode means, model means and four-mode aggregates. A separate
multiplicity-count bootstrap computation reproduces all **200 horizon/metric
interval cells**, including the 16 reported edge contrasts and four interactions;
maximum interval-endpoint difference is 1.57e-17.

Among the 16 edge contrasts, 13 relative point gains are positive and three are
negative; six intervals favor the first model and ten include zero. All four
interaction intervals include zero. Mixing improves native endpoint errors
with and without bounding. Native-coordinate gains from bounding remain
inconclusive; two secondary coarse-coordinate bounding-without-mixing intervals
are favorable. The evidence does not establish exact additivity or equivalence.

The six existing inference-only exports match their validation-selected source
weights. I also freshly reexecuted physical relocation for **all six** in isolated
Python processes using copied local sources, CPU inference and application-level
network guards. Each forecast matches exactly: maximum absolute error **0.0**.
These are local package checks, not a claim that six new release assets are public.

The new manuscript discussion is consistent with these results. Mixing includes
the learned gate, identity bias and approximately 1.84% more active parameters;
there is no pure-mixing causal attribution. This is a registered follow-up after
revealed controls on original validation, with unadjusted intervals, not a fresh
independent test or state-of-the-art claim. Updated figure/table pixels will be
reviewed separately once rendered.
