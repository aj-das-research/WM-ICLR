# Capacity-matched DROID control

Post-hoc original-development study; not held-out confirmation. Retain the
completed three-seed bounded-additive baseline and bounded spatial mixer.
Add a patch-local gated two-layer MLP before the existing residual projection:
h + sigmoid(g(h)) B GELU(Ah). A/B are bias-free96x96; g is affine96-to1.
Exactly18,529 active parameters are added, equal to the mixer's two projections
and gate. The new head never mixes patch positions. The shared upstream spatial
encoder remains unchanged. All old active tensors initialize identically to the
bounded-additive arm at each seed; the added head consumes subsequent RNG draws.
The correction stays tanh-bounded. No dropout or regularization is added.
Three seeds0/1/2,30full epochs, all other original DROID training, normalization,
selection, horizons and development scoring settings are retained. No test
payload is read. This controls parameter count, not every capacity/optimization
property. Report either outcome against BOTH original bounded-additive and
bounded-mixing comparators; no model is promoted based on this experiment.
