# Reserved IWS presentation source pack

Numerical files are created only after the complete36-run reserved finalizer passes full local revalidation. Until then this directory contains the design brief and reporting tests only; no numerical figure or paper table is emitted.

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

The public runtime inputs are the renderer, manifest, exact finalization JSON and36 exact evaluation JSON receipts. All per-trajectory full-H60 metric means, separately invoked prefix endpoints, population denominators, original equal-handle means and source hashes are retained. Primitive per-handle NPZ ledgers remain local; the frozen scientific finalizer independently reconstructs them before pack creation. Their original hashes remain in the finalization and manifest, without becoming public binary dependencies.

Public replay independently recomputes all paired confidence intervals from per-trajectory means using the same seed173 shared-seed/within-task-trajectory draws, and verifies equal-trajectory means. Reassociating floating-point additions when deriving handle-weighted means from trajectory means uses a strict 2e-14 relative / 2e-15 absolute numerical tolerance; original finalized values are preserved verbatim in plotted marks and tables.

The main forest plot retains all15 task/contrast standardized-MSE points and CIs. The score table retains60 metric means. The four compact contrast tables retain all80 signed gain intervals, including the equal-task macro. All59-offset curves and per-seed endpoints remain in the source pack. Undefined ratios remain undefined and negative findings are never filtered. Existing development figures are unchanged.

Generated outputs are completion-bound and idempotent. Actual color/grayscale/PDF and manuscript reviews are required after real data exist; source tests or a successful export do not pre-certify future pixels.
