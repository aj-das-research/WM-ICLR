# Integrated editorial and evidence review

Reviewed September 18, 2026. The final reviewed snapshot is a **22-page local
research draft**: main content on pages 1–8, references on page 8, and appendices
on pages 9–22. This is not a submission-readiness or efficacy certification.

Snapshot PDF SHA256:
`7a78c252db8d55c43179077cd31dd38bd7a9570db91e80dfdea1833acdcb3c11`.
The preserved local snapshot is
`build/pairing_control_review/final/reviewed_draft.pdf`; the live manuscript
can be rebuilt as experiments complete. Exact review and restart checks are
recorded in `evidence/editorial_revision_review.json`.

## Figure 2 and quantitative presentation

Figure 2 is now on page 4. Its A/B/C composition foregrounds shared observation
calibration, dynamics inference after correction, and the two different
training-only consistency penalties. Real training thumbnails make the pairing
relations visible. The reused predictor and CEM solver are secondary. The figure
does not claim reconstructed images, identified physical factors, equality of
learned contexts, or empirical success. Its 5.5-inch-wide standalone export is
4.41 inches tall. Root and independent agents inspected paper-width pixels,
enlarged goal/action and pairing crops, and the integrated caption. The figure
agent also inspected grayscale and image provenance. Continuous connectors,
target ports, training/deployment separation, and text clearances passed review.

Table 2 remains the primary planning comparison. It identifies **ShiftWM
(ours)**, highlights its row, and adds differences against the fixed Shared
context control with conditional paired intervals. A separate compact block
reports forecast reductions against Framewise calibration and Unpaired
contexts. The forecasting and paired-planning plots annotate signed changes
without covering their points or uncertainty ranges. Negative results remain
visible; all four completed Shared-context planning intervals include zero.

Table 9 adds the completed random-action and privileged-replay controls. Its
counts and task/support checks use the original evaluator and common record
validators. Random actions have a single fixed RNG rule; replay uses privileged
future actions. The main limitations explicitly disclose PushT's lower success
point estimates than the random control. The final caption says that policies
share a **50-step budget**, rather than implying that early-terminating episodes
execute exactly 50 steps.

## Integrated checks and repairs

- Inspected pages 4, 6, 7, 8, 9, 18, 19, and 20 in
  `build/pairing_control_review/`, and the corrected final page 19 at higher
  resolution. Earlier passes also checked title, training and diagnostic plots.
- Repaired the earlier title split, the forecast figure's avoidable page break,
  and appendix floats crossing topic boundaries. The larger final Figure 2
  and its caption fit together. No fonts or margins were reduced to make room.
- The final LaTeX build has no overflow, undefined-reference, or LaTeX warnings.
- All 20 previously completed primary cells retain their numerical values and
  seed SDs. All 20 forecast points retain their original means, SDs, seeds, and
  source identities; gain annotations add derived ratios, not new measurements.
- New gain, paired-plot, and control-reporting guards passed. The full live
  source-validation/render/build pipeline completed successfully. Control
  reporting validated eight completed runs and eight common-support audits.
- Model, evaluator, training configurations, and the six source/config hashes
  pinned for queued dynamics-revision jobs remain unchanged.

Independent review found no unresolved material diagram or table overlap.
Forecast ranges remain seed SDs; paired planning ranges remain conditional
task-cluster intervals. The retained negative results and missing comparisons
are scientific limitations, not formatting defects.

The result diagnosis is in `../reports/results_diagnosis_2026-09-18.md`, including
goal-calibration evidence, the training/recursive-evaluation mismatch, and
unproven explanations. A separate deterministic CPU audit ruled out a hard
normalized-action cap; it did not establish a cause of poor planning.
