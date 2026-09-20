# Development geometry follow-up, 20 September 2026

This is a diagnostic of existing selected checkpoints, not training or a new
benchmark. It compares all six original and six wide-gain transformer packages:
factorized and constant-dynamics, seeds 0/1/2. Selection remains the minimum
validation recursive-MSE package from each completed 30-epoch run.

The fixed population is identical to the original mechanism probe: 48 development
trajectories, support windows 0/16, and the 16 gain-ID-1 goals. Each of those 16
first-window observation contexts is held fixed while mapping every goal. The
120 pairwise latent MSEs are compared with squared physical XY distances. Simulator
coordinates are scoring-only; they never enter context inference. Translation
cancels within each fixed context. The unchanged original `geometry` helper is
extracted with AST, avoiding the original script's top-level model loads/writes.
Canonical and narrow seed-0 results must reproduce the original recorded probe.

The paired report includes distances, descriptive Spearman correlations,
calibration MSE and attained adapter gain ranges. Pairs, contexts and shared goals
are dependent; no confidence interval, causal attribution or final-test promotion
is inferred. A larger latent-distance magnitude alone is not better geometry.
All 12 packages are retained, including adverse results. There is no simulator
execution, fitting, retuning, RGB processing or held-out payload access.

The second report audits all 18 existing tissue-development planning records
(144 method/seed/task instances, eight shared goals). It retains every failure,
invalid action and early stop. Only complete five-command decisions are used for
next-step alignment; this is chosen-action description, not candidate ranking.

Run from the project root:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -B -m pytest -q scripts/extensions_diagnostics_20260920/test_diagnose.py
.venv/bin/python -I scripts/extensions_diagnostics_20260920/diagnose.py prepare
# Independent reviewer writes reports/extensions_diagnostics_20260920/source_review.json,
# with status=passed and the exact registration_sha256, before submission.
.venv/bin/python -I scripts/extensions_diagnostics_20260920/diagnose.py check
sbatch scripts/extensions_diagnostics_20260920/run.slurm
```

The ws-ia request is two CPUs, 8 GiB, 15 minutes, no GPU; its ia-std partition limit
enforces the existing two-job cap. Registration and completed output are
write-once. A failed run produces no completion marker. `results.json` and
`completion.json` live only in `reports/extensions_diagnostics_20260920/`.
Sources, exact immutable package generations, development payloads and saved
tissue traces are hashed before and after execution. Original files are unchanged.
