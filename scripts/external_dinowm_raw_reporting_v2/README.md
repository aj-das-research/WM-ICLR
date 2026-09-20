# Raw-coordinate external DINO-WM follow-up reporting

This separate, complete-only reporter compares six adapted DINO-WM models (two objectives × three seeds, all **100 epochs**) against the same fifteen fixed internal models (five arms × three seeds, all **30 epochs**). It preserves the earlier standardized-coordinate 30-epoch external study and all of its reports.

The external objectives are `official_raw_one_step` and `official_raw_recursive_h10`. Their visual inputs, predictions, recursion and optimized loss use raw pooled DINO coordinates; the selected checkpoint still minimizes the common window-weighted, all-ten-step standardized development MSE. The report uses the unchanged native 4×4 and original 2×2 standardized scoring, including both persistence controls. Original cached targets are not replaced with newly pooled targets. This is a posthoc development follow-up changing representation and recipe together, not a single-factor intervention, held-out test or official benchmark/SOTA reproduction.

## Complete gate and reuse

The source pins and privately imports the unchanged v1 reporting helper. It reuses its exact population/persistence checks, equal-window-within-episode then equal-episode/equal-seed means, and paired recording-session × seed percentile intervals (10,000 draws, seed 20260919). The frozen spatial ledger arithmetic is unchanged. Only a private external ledger instance changes one exact AST expression from a selected-epoch upper limit of 30 to 100. Internal ledgers retain the original upper limit of 30; neither selected epochs nor metric values are falsified.

Before reading any new external evaluation ledger, all six full external packages and all fifteen fixed comparator packages must pass their authoritative completion and checkpoint-selection validators. Raw package kind, coordinate tag, standardized selector/precision, config, registration, checkpoint, summary, evaluation and execution-receipt identities are checked. Every primitive window, episode mean, source identity and shared persistence value is then validated before cross-model aggregation. Every signed result, negative gain and interval crossing zero is retained. Undefined zero-denominator percentages remain undefined. No training, inference, cache payload or reserved/test access occurs here.

A detached immutable registration binds these four source files, the pinned reused helper, the synthetic fixture, both scientific registrations and their complete source closures, all fifteen existing comparator ledgers and the original finalized comparator report. An independent `source_review.json` must bind the new registration before finalization. No raw-v2 evaluation values were read to author or freeze this reporter.

## Reproducible commands

From the project root:

```sh
.venv/bin/python -B -m pytest -q scripts/external_dinowm_raw_reporting_v2/test_finalize.py
.venv/bin/python -B scripts/external_dinowm_raw_reporting_v2/finalize.py --freeze
# Independent reviewer creates reports/external_dinowm_raw_reporting_v2/source_review.json.
.venv/bin/python -B scripts/external_dinowm_raw_reporting_v2/finalize.py --if-ready
# Once both full training arrays finish successfully, fail closed on missing evidence:
sbatch --dependency=afterok:202064:202067 scripts/external_dinowm_raw_reporting_v2/run.slurm
```

The Slurm wrapper requests CPU only: 2 threads, 8 GiB, 30 minutes. Submission is owned by the campaign author/parent and occurs only after the reporting source review passes. `--if-ready` checks metadata and existence only while incomplete and writes no aggregate output. A normal invocation fails if incomplete. Successful outputs are `reports/external_dinowm_raw_reporting_v2/{finalization.json,results.md,completion.json}`; repeat invocations verify exact frozen source/output bindings. The reporting script neither edits the manuscript nor publishes results.

Synthetic tests cover adverse/null results, all metrics/horizons, unequal window counts, full-grid and early-read gates, stale approvals, wrong package/coordinate/precision, exact epoch limits, primitive corruption and source/receipt tampering. Tests write only temporary fixtures; they do not access real raw-v2 results.
