# Technical inference schematic review

Reviewed at the intended 5.5 × 2.4 inch size on 2026-09-19 (Dubai).
Exports: `paper/generated/qualitative/technical_contract.pdf`, `.svg`, `.png`.
Editable source: `paper/scripts/render_technical_contract.py`.
Source and output hashes: `paper/generated/qualitative/technical_contract.json`.

Actual pixel inspection used `paper/build/technical_contract_review/paper_width.png`
(550 × 240), plus the 300-ppi top, context, and prediction-route crops recorded
in `inspection_report.json`. The final proof has no extracted out-of-page words.
That automated result supplements, rather than replaces, the visual inspection.

Repairs made after inspection: corrected the escaped theta label; shortened
thumbnail captions to prevent collision; separated the ShiftWM label from its
context block; replaced the opaque observation-adapter label with FiLM_o; added
the corrected-transition [z, Δz] cue; moved the history label off its vertical
branch; lowered top math labels to eliminate clipping. The final paper-width
and enlarged proofs show no overlapping labels, broken connectors, detached
arrowheads, or clipped text. All 20 named connector paths remain continuous.

Semantic trace checked against model.py: Framewise retains temporal and action
conditioning and adds residual per-image calibration. ShiftWM infers observation
context from raw support mean/variance, corrects history before dynamics-context
inference, and supplies corrected z and Δz plus normalized past executed actions
to the GRU. The goal bypasses both context networks; candidate actions do not
enter either context network. E_a(a) includes past and candidate future actions
in predictor windows. Both residual FiLM operators are explicit. Predictor
weights are independently trained; the common region denotes architecture and
planning procedure only. All weights are fixed online, and contexts are fixed
within CEM and refreshed after real execution.

The thumbnails are the original available shifted last-support and goal images
from Reacher development episode 2031004. No synthetic prediction, generated
scene, canonical target, or retouched observation is shown. The feature bars are
symbolic identities. The figure has no quantitative result or causal claim.

The `--if-needed` identity check was exercised: intact sources/exports reuse the
figure; a changed source identity or modified PNG export rejects reuse. This is
a rendering check, not a scientific experiment. Full manuscript placement is
left to the parent agent's integrated PDF review.

Related read-only review: render_technical_qualitative.py recomputes displayed
prediction MSE from saved prediction/target vectors, checks physical success
against both strict per-environment criteria, plots actual native-step traces,
and transparently selects the first lexicographic positive case per environment.
Its summary includes all four positive cases and explicitly avoids independent
benchmark-win or significance interpretations. No substantive math/selection
defect was found in that source review; its final visual proof remains separate.
