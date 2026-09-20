# Complete-only adapted DINO-WM reporting

This new reporter reads completed checkpoint/evaluation evidence. It does not run
training or inference, change selection, or open cached feature/video payloads.
The external baseline is **posthoc development adapted official DINO-WM**, not a
reproduction of an official benchmark score or a state-of-the-art claim.

Six new runs are mandatory: `official_one_step_shifted` and
`matched_recursive_h10`, each seeds 0/1/2 and thirty complete epochs. Both use the
same recursively evaluated H10 development selector. The source-authoritative
trainer verifies every epoch and the selected/last packages before this reporter
reads any new evaluation values. All fifteen existing spatial controls also pass
their unchanged full-training validator. No aggregate is emitted from a partial
roster, incomplete package, altered source, or malformed/partial evaluation.

The unchanged spatial ledger validator reconstructs the complete camera-one DROID
validation population from both manifests, then recomputes every ten-step metric
from windows to episodes. This reporter extends **only a private imported module's
accepted mode names**. It first validates the external wrapper's source, upstream
metric-source hash, registration, configuration, normalization, selected training
identity, objective, precision and selection metadata; the unchanged validator
then checks the full common result. Original registered files remain untouched.

Report all four metrics (native4×4 standardized MSE, original2×2 standardized MSE,
and their respective persistence errors), all ten horizons, seven method means,
three seed summaries, and source-bound primitive window/episode ledger paths.
Original2×2 targets remain the exact original cache values, not re-pooled targets.
The two coordinate systems are not interchangeable error units. Paired comparisons
use the unchanged `paired_intervals`: matched training seeds and recording sessions,
10,000 draws, fixed seed20260919, unadjusted95% intervals, all ten horizons.
All five internal modes are compared with each of the two external objectives.
The two primary descriptive contrasts use ShiftWM transport as the first method;
negative first-minus-second favors ShiftWM. Adverse values, zero-crossing intervals
and undefined relative gains at zero comparator error remain explicit.

```bash
.venv/bin/python -B -m pytest -q scripts/external_dinowm_reporting_v1/test_finalize.py
# Once the independent six-run training registration exists, before new results:
.venv/bin/python -B scripts/external_dinowm_reporting_v1/finalize.py --freeze
# Reviewer writes reports/external_dinowm_reporting_v1/source_review.json with
# status=passed and registration_sha256 for this reporter's registration.
.venv/bin/python -B scripts/external_dinowm_reporting_v1/finalize.py --if-ready
```

`--if-ready` writes nothing while the six-run evaluation/completion set is absent.
A present malformed roster or a fully present but invalid campaign fails closed.
No pending values or synthetic fixtures enter active manuscript outputs. Successful
results go only to `reports/external_dinowm_reporting_v1/{finalization.json,results.md,
completion.json}`. The parent presentation pipeline may consume these after review.
Synthetic tests stay in pytest temporary directories; the reporting implementation
and policy are frozen before reading any new external result ledger.
