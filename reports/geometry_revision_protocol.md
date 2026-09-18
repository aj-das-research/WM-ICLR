# Registered observation-geometry capacity ablation

Registered 2026-09-19 after the development-only diagnosis in
`reports/extensions_mechanism_diagnosis.md`, before any new revision training.
This is a hypothesis-driven development revision, not an independently confirmed
algorithmic contribution or a state-of-the-art claim.

Change only the observation FiLM gain from `1+0.1*tanh(s)` to
`exp(log(4)*tanh(0.1*s/log(4)))`. Both are identity with derivative0.1 at zero;
translation, parameter shapes, initial parameter values, dynamics/action FiLM,
context networks, recursive prediction loss and consistency losses are unchanged.
Attainable gain range changes from[0.9,1.1] to[0.25,4]. Higher-order curvature also
changes, so optimization trajectories need not be identical. No new geometry,
ranking or simulator-state loss is added in this ablation.

Six drone transformer runs: inferred factorized context and constant dynamics
context, each for training seeds0,1,2. Match the original registered full30epoch
configuration exactly, including data/validation/cache, initialization, optimizer
and batch order. New output namespace `runs/geometry_revision`, explicit model
formatversion2 and strict offline loader prevent silently loading old adapter
semantics. The old runs are retained as controls. The active context parameter
counts are identical within each old/new comparison.

Use the same eight prespecified eligible warm/gain0.75 development goals, common
ten-command support, total200native commands, and CEM horizon5/candidates128/
iterations5/elites16/plannerseed101. Record every native action, success/failure,
distance and source identity. Do not inspect final-test model outcomes. Selection
is still minimum complete FP32 recursive validation MSE, not development success.

Report all six results, the corresponding narrow-gain controls, and uncertainty
over paired training/task seeds. Check whether the revised inferred method beats
both its old inferred version and the revised constant-dynamics control. Improved
goal separation or forecasting alone is insufficient for promotion. The observed
goal-distance compression motivates this test; the test may fail.

The original diagnostic reports16goals/120pairs in fixed contexts. Recompute that
same measurement for the new checkpoints before interpreting the mechanism.
Any later branched-data action-ranking objective requires a separate registered
study and must not be merged into this one-change comparison.
