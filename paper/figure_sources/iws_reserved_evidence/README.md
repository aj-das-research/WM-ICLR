# Reserved IWS presentation source pack

This pack contains the completed, independently reviewed 36-run reserved study. Numerical files are created only after the complete finalizer passes full local revalidation and the independent result review binds that exact finalization and registration. An incomplete study emits no numerical figure or paper table.

The sole accepted scientific source is `reports/real_video_iws_reserved_recovery_v2/finalization.json`. This is an explicitly post-access numerical recovery: all 36 predictors must be rerun with the same row-wise command-GRU backend after an input-only prefix audit. Weights, normalization, examples, metrics, tolerances and comparators remain unchanged. Original v1 partial receipts are preserved as incident evidence and cannot be pooled into this pack.

Live preparation:

```bash
.venv/bin/python paper/scripts/render_iws_reserved_evidence.py --if-ready
```

Portable replay after completion:

```bash
python paper/scripts/render_iws_reserved_evidence.py --from-pack \
  --pack-dir paper/figure_sources/iws_reserved_evidence \
  --output-dir /tmp/iws-reserved-public-replay
```

The 40 public runtime inputs are the renderer, manifest, exact finalization JSON, independent result-review JSON and 36 exact evaluation JSON receipts. The manifest binds the 38 data/review JSON files. The review must pass and bind both the completed finalization and registration; an absent review keeps presentation pending and a stale or failed review raises an error. All per-trajectory full-H60 metric means, separately invoked prefix endpoints, population denominators, original equal-handle means and source hashes are retained. Primitive per-handle NPZ ledgers remain local; the frozen scientific finalizer independently reconstructs them before pack creation. Their original hashes remain in the finalization and manifest, without becoming public binary dependencies.

Public replay independently recomputes all paired confidence intervals from per-trajectory means using the same seed 173 shared-seed/within-task-trajectory draws, and verifies equal-trajectory means. Reassociating floating-point additions when deriving handle-weighted means from trajectory means uses a strict 2e-14 relative / 2e-15 absolute numerical tolerance; original finalized values are preserved verbatim in plotted marks and tables.

The main forest plot retains all 15 task/contrast standardized-MSE points and CIs. Its four learned-reference rows share one scale across the three tasks; the persistence comparison has an explicitly separate 0–100% strip. This preserves every interval while keeping the smaller learned-reference effects readable. The main endpoint table contains 15 MSE means; the appendix score table retains all 60 metric means. The four compact contrast tables retain all 80 signed gain intervals, including the equal-task macro. All 59-offset curves and per-seed endpoints remain in the source pack. Undefined ratios remain undefined and negative findings are never filtered. Existing development figures are unchanged.

The final 5.5 × 2.6 inch forest uses labels of at least 8 pt, and the tables use 9 pt text. Actual PDF color, grayscale and official-template table proofs were reviewed. Independent presentation review is recorded in `reports/evidence/iws_reserved_presentation_independent_review_2026-09-20.json`; integrated manuscript review is separate. Isolated public-only replay reproduced all 11 exports byte-for-byte, matched PNG and PDF raster pixels, and preserved bytes and modification times on retry. It made no network or original-workspace data access attempts. This verifies presentation reproduction; it does not replace scientific reconstruction from the private primitive ledgers.
