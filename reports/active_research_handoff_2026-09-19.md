# Active research handoff — 19 September 2026

## Current status at 06:42 UTC

The 15-model spatial campaign, finalizer, paper tables and public checkpoint release are complete. There are currently no GPU jobs. Six new full component runs are being implemented and independently reviewed before launch. The qualitative agent is reviewing genuine examples and preparing a plain-language task explainer; root is updating the paper, site and focused IWS study plan.

Native h10 gains: 5.30% vs autoregression, 3.55% vs anchoring, 4.50% vs action-free. Context does not add a clear benefit. Across 16 comparisons, 14 favorable point estimates, 9 favorable intervals, 7 inconclusive intervals and 2 unfavorable point estimates. Total public predictors: 117 across five releases. Independent audit: `reports/evidence/spatial_completed_independent_audit.md`.

The scheduler graph and counts below describe the earlier overnight execution, not live jobs. Preserve the original/fresh tests and completed scientific registrations. User now prioritizes a clearer real-video task presentation, focused real manipulation benchmarks, and fair evaluation of several architectural versions. Gains must be measured; no 5–10-point promise or test-based selection.

## Earlier overnight handoff (historical)

The current priority is a complete, matched evaluation of the observation-anchored
spatial architecture. Original and fresh held-out DROID studies remain frozen.
All ongoing development uses the original train/validation population.

## Completed in this continuation

- All 12 horizon-ten models completed 30 epochs, all 60 evaluations and all 20
  paired contrasts. Every selected checkpoint is epoch 1. Ours versus equally
  trained Framewise gains 0.252% on mean steps1–10 and 0.405% at step10; versus
  the original ours on identical windows, gains are 1.058% and 2.772%. The
  matched five-step-prefix mean retains a −0.116% point regression with an
  interval crossing zero. These are exploratory validation comparisons.
- Independent review checked all ledgers/means/intervals and actual Tables28–33.
  Figure22 is integrated onpage56 of the58-page draft and actualpixels reviewed.
  Reporting and figure source evidence are retained; no test claims changed.
- The simulator-development-v1 release contains 42 predictors and the
  horizon10-development-v1 release contains 12. All passed relocated exact
  inference parity and public asset checksum checks. With original12 and
  generalization36, **102 neural predictors** are public. Calibration wrappers
  are not additional trained networks.
- GitHub/Overleaf/Pages were synchronized at00:26:47UTC, including102-model
  availability and the completed h10 appendix. Main commit
  `74ca68051cdde8af9b2eab576e667eec8a29c9c1`, Overleaf
  `e4993d2f31d98c2276c863496f792f76b96bd97a`, Pages
  `09efa3301c748bef7b7391ea835fa3b83ad017a1`. Later figure/report edits need
  the next sync; publication does not imply unpublished spatial results exist.

## Live scheduler graph

Two ws-ia GPU chains are healthy:

- seed0:200215→200216→200217→200218→200219.
- seed1:200220→200221→200222→200223→200224.
- seed2 laneA:200266→200268→200270, on ws-l1-002 after200219.
- seed2 laneB:200267→200269, on ws-l5-004 after200224.

The first three arms for seeds0/1 are complete (six total); Context-off is currently running. A verified operational amendment distributes the remaining seed2 arms over two GPU lanes when both earlier chains finish; no new jobs or scientific settings were added.
Two seed2 attempts on the gpu partition failed before the first epoch because
allocated cards carried about31GiB of other memory use. Do not repeatedly
submit there based on Slurm's idle label, change batch128, override allocated
CUDA visibility, or kill external processes. GPU04 metadata-only inventory
showed less than1.2GiB free on every card. The ws fallback preserves all frozen
scientific files, configurations and exporter dependencies.

Finalizer **200271** waits for200219,200224,200269,200270. On successful completion:

- **200236** generates three preregistered qualitative candidate cases and18
  matched model/seed replays, with genuine frames and measured latent errors.
- **200254** performs gated relocation/hash/ledger checks and publishes the
  complete fifteen-model spatial prerelease if every check passes.
- **200278** independently recomputes every aggregate and all sixteen paired
  contrasts, creates five complete manuscript tables, and builds the paper.
  Reporting tests:13 passed; independent implementation review passed.

All three dependent jobs request zero GPUs. Paper/source synchronization uses
the existing five-minute user timer. New actual qualitative figures still
require pixel/semantic review; candidate generation does not approve inclusion.

## Frozen identities and next actions

Spatial scientific registry:
`ae99340f7a3761ece066ee7376f4bd119b61a92503948c142196c40a1a76e337`.
Do not edit registered scientific files or `prepare_public_snapshot.py`, which
is frozen in the pending spatial release. New reporting source is also pinned
by `reports/spatial_paper_submission.json` and checked at execution. Operational
job-ID amendments are separate provenance records.

Next inspect all15 completed results together, including anchoring-only,
context-off and action-free comparisons. Native4x4 and original2x2 errors have
different normalization and cannot be compared as if interchangeable. The
candidate contains one support context; it is not unchanged original ShiftWM.
Mixing and bounded innovation are not individually isolated by the current
five-arm study. No large SOTA or universal applicability claim is established.
Review actual generated tables/qualitative cases, update the paper narrative
and release counts only after completion. Any later confirmatory evaluation
must use an independently frozen, genuinely untouched population.
