# Pending results: execution plan and conditional ETA

Audit snapshot: 20 September 2026, 10:58 UTC / 14:58 GST. This planning audit read
source, frozen identities, handle metadata and saved timing receipts only. It
opened no reserved video, HDF5 command values or feature payload, submitted no
job and changed no scientific source or registration.

**Later update, 15:16 GST:** the separately registered resource study finished at 15:09 GST. Its complete summary is `reports/real_video_iws_resources_v1/job_201732/summary.json`. The reserved accuracy implementation below has not started.

**Recommended next accuracy study:** a new, separately registered, feature-only
IWS reserved evaluation. Training is already complete: 27 original predictors
and nine no-tanh ablations, with complete CPU development finalizers and local
offline inference exports. More training is not a prerequisite.

| Pending item | Concrete next action | Completion condition / ETA basis |
|---|---|---|
| IWS original reserved study | Implement and freeze reserved loader/cache/evaluator/finalizer; score all 27 original checkpoints plus persistence | About 4–8 engineering/review hours, then 1–2 hours for allocated cache/evaluation/final audit, plus queue time; estimates, not submitted jobs |
| IWS no-tanh reserved follow-up | Before opening any reserved payload, add all nine selected ablations to the same new freeze as a declared secondary family | Adds nine inference passes, no training; included in the compute allowance above |
| Predictor cost measurements | Completed job 201732: all 36 predictors and three task-specific persistence cases, CPU and CUDA | 39/39 cases, 78 device rows and 2,340 timed calls passed; source-bound resource table is integrated as manuscript Table 12. No accuracy measurement or reserved access |
| Drone and simulated tissue final tests | Keep gated; resolve the existing development promotion criterion and freeze the complete final roster before any test run | No defensible completion date while promotion is unmet; end-to-end final-test runtime is unmeasured |
| Original simulated PushT/Reacher final grid | Already complete | No missing final experiment to launch |
| External RLA-WM comparison | Resolve exact DINOv3 backbone access and define a common representation/metric contract | No reliable ETA from current evidence; downloading an RLA encoder does not supply the missing image backbone |

## Smallest valid IWS reserved protocol

**Population and indexing.** Preserve the exact original 200 H60 handles per task
in `configs/real_video_iws/split_v1.json`: 600 unique trajectory/start pairs from
30 reserved trajectories. Metadata lists 5,996 native frames in total. Handle
counts are unequal across trajectories (Box 16–25; PushT/Rope 15–26), so report
both equal-handle and equal-trajectory means; do not treat 200 handles as 200
independent trajectories. Do not substitute stride-five development windows.

For each handle starting at `s`, observe only frame `s`; supply command rows
`s…s+59`; forecast stored offsets 1–59. H60 target is `s+59`, output-array index
58. Derived H15/H30/H45 endpoints are `s+14/s+29/s+44`, output indices
13/28/43, evaluated with exactly 15/30/45 command rows at the same start. They
are not additional official handle sets. Preserve every original handle and
its order; all currently listed handles satisfy `s+60<N`. These are stored-row
indices, not verified physical seconds or sixty transitions.

**Methods and claims.** Retain the original primary comparison: bounded spatial
mixing versus anchored additive at H60 standardized feature MSE. Original AR
and persistence remain secondary. The no-tanh variant was chosen for study
after development outcomes: include all three seeds for all three tasks as a
newly frozen secondary component comparison versus bounded mixing, not as a
retroactive replacement for the original primary method. The original design
did not preregister this follow-up. No selection, tuning, seed removal or model
switching is allowed after reserved outcomes. A separate explicit decision is
needed before access if a different primary claim is desired.

**Representation and metrics.** Reuse the exact full-RGB DINOv2-small revision,
224×224 preprocessing, CUDA BF16 extraction with TF32 off, FP32 4×4 pooling,
channel-major 6,144-dimensional features, and each task's existing training-only
statistics. Do not fit new statistics on reserved data. Predictor/scoring runs
use CPU FP32 with eight threads, fixed batch size 64, no autocast, and all three
independently checked command prefixes. Retain all 59 offsets for the four
existing metrics: standardized MSE, standardized MAE, raw DINOv2 L1 and flattened
cosine distance. All four are lower-is-better. The exact cosine epsilon/clipping
formula and per-window persistence must match the existing metric helper.

Report equal-trajectory task means and equal-task macro relative reductions;
retain equal-handle secondary means. Use the existing 10,000-draw paired
seed×trajectory bootstrap, seed 173, resampling ten trajectory clusters per
task, with shared seed draws across tasks. Keep all signs and undefined
zero-denominator ratios. Primary H60 intervals are distinct from exploratory
secondary endpoints/metrics. This study supplies feature forecasting, not RGB
LPIPS/SSIM, robot control, or an external SOTA comparison.

## What must be implemented before access

Suggested **new** namespace: `scripts/real_video_iws_reserved_v1/`, a new reserved
registration/config, separate feature-cache root and separate report/finalizer.
These are proposed paths, not existing runnable commands.

1. Bind the two completed development finalizers, all 36 selected checkpoint
   hashes/epochs, both package kinds, immutable training statistics, exact
   handle JSONs, encoder/provenance, preprocessing and source/test hashes.
   Record the existing session's authorization for end-to-end experiments in
   the new access manifest; no further user permission is required merely to
   implement this protocol. Payload access still waits for its independently
   reviewed scientific freeze. Existing download and
   metadata provenance can be bound now; record/verify individual raw-video
   hashes during authorized extraction, without pretending they were inspected
   in this audit.
2. Build a new exact-handle inventory/cache adapter. Authorize the 30 frozen
   reserved IDs before any payload open; preserve native row alignment and all
   original handles. Test missing/duplicate/wrong-ID handles, H2/15/30/45/60
   indexing, command widths 4/14/8, target isolation, unsafe paths and incomplete
   population rejection using synthetic or authorized training fixtures first.
3. Reuse unchanged model loaders and `feature_errors` from the original evaluator;
   use the distinct no-tanh loader for its nine packages. Keep target reading in
   the scorer, outside `predict`. Write immutable per-handle primitive metric
   ledgers, complete trajectory aggregates and backend/prefix receipts.
4. Add a reserved-only finalizer with the complete frozen roster, exact same
   persistence across methods/seeds, independent ledger reconstruction and paired
   uncertainty. The reserved endpoint must **not** be required to equal the
   development checkpoint-selection score. Preserve the original development
   completion check as provenance, not as the reserved score.
5. Independently review source/tests and freeze before cache extraction. After
   all outputs validate, provide a new portable reporting pack and inspect the
   actual table/plot. No partial run populates a final-study table.

The existing `scripts/real_video_iws/evaluate.py` and `finalize.py` cannot be
redirected unchanged: they hard-code `internal_development`, stride-five windows,
the development population and selector-score equality. The Box/Rope and PushT
`InternalInventory` classes reject reserved payloads intentionally. Do not patch
those guards or reinterpret their development receipts. Existing local bundle
commands support offline inference, not the missing reserved evaluation:

```bash
.venv/bin/python -I artifacts/releases/iws_single_observation_local_v1/runtime.py --output /tmp/iws-v1-parity.json
.venv/bin/python -I artifacts/releases/iws_unbounded_single_observation_local_v1/runtime.py --output /tmp/iws-no-tanh-parity.json
```

## Wall-clock planning, with uncertainty explicit

Saved CPU evaluation timing provides a useful scale: nine AR full-development
runs took 124.67–138.14 seconds each; nine additive runs 110.79–172.70 seconds;
nine bounded-mixing runs 156.00–177.90 seconds. Each scored 3,360–3,388 windows,
all 59 offsets and separate prefix calls. Scaling only the inference workload
to 200 handles suggests approximately 7–11 seconds/run. This omits startup,
package validation, I/O, new population checks and scheduling, so it is not an
end-to-end promise. Budget **15–35 minutes for 36 sequential CPU evaluations**
including those overheads, then approximately **20–40 minutes** for finalization,
independent checking and reporting. No-tanh CPU package parity is verified, but
its future reserved wall time is still an estimate.

For 5,996 new frames, budget **5–20 minutes** for GPU cache work as an explicitly
unmeasured planning allowance, to be replaced by an authorized training-only
extraction pilot or actual allocated-job throughput. The saved cache completion
receipts do not provide reliable end-to-end timing. Request conservative headroom:
one GPU / eight CPUs / 24 GiB / one hour for extraction, then zero GPUs / eight
CPUs / 24 GiB / one hour for CPU inference. Sequential inference suffices; more
concurrency is optional, subject to actual account limits and the profiling job.

Engineering/review is the dominant prerequisite: estimate **4–8 focused hours**
for the new adapter, fail-closed tests, finalizer and independent freeze. Overall
planning window is **T0 + 5–10 hours plus actual queue delays**, where T0 is the
start of implementation, not this report's timestamp. If work starts at 15:00
GST today and tests pass on the first review, that corresponds approximately to
20:00 GST today–01:00 GST tomorrow, before queue delays. No job has been submitted
by this audit; no fixed completion time is committed.

## Why the simulated final rows remain gated

`reports/domain_extension_protocol.md` requires better closed-loop success than
both within-family controls with consistent paired evidence across seeds and
domains before promotion. The 36 original drone/tissue models and six wide-gain
follow-ups are complete on development data, but the recorded signs are mixed
and no original paired interval has a strictly positive lower bound. The
wide-gain drone follow-up uses the same development tasks and does not establish
promotion. Checkpoint files are frozen; a promoted final-test candidate/complete
roster is not frozen. A training-only branched drone dataset also exists, but its
collection is not a newly trained or promoted model.

The existing command supports a test split, but command availability is not the
promotion gate:

```text
scripts/extensions/evaluate_campaign.py --domain {drone,surgery}
  --architecture {transformer,gru} --seed N --split test --kind both
  --episodes-per-gain N --output-root NEWPATH --include-controls
```

Do not run its default eight tasks per gain and label that the complete final
set. First freeze the full eligible roster, candidate/control models, exclusions,
success criteria and all seeds. Existing per-decision `solve_seconds` cannot
establish end-to-end final-campaign runtime; no honest final-test ETA exists
until this scientific gate and a bounded resource profile are resolved. Current
rows should say **not run; development promotion unmet**, rather than imply jobs
are queued. PushT/Reacher's original simulated final grid is already complete.

Source anchors: `configs/real_video_iws/split_v1.json`;
`reports/real_video_development/iws_single_observation_design.md`;
`reports/real_video_iws/{next_evaluation_gate_audit,external_baseline_next_step}.md`;
`scripts/real_video_iws/{evaluate,finalize}.py`;
`reports/real_video_iws/recovery/{seed0_cpu_evaluation,seed12_cpu_evaluation}.json`;
`reports/real_video_iws/recovery/common_cpu_v1/runs/*/{started,invocation}.json`;
`reports/real_video_iws_unbounded/development_finalization.json` (SHA256
`1f0fa2bffcbfa957852bda78bbdb377784b77b4b98e17e8ca67bd07c304853d1`);
`reports/domain_extension_protocol.md`; `reports/completed_extension_results.json`;
`scripts/extensions/evaluate_campaign.py`;
`reports/drone_branched_training_completion.md`.
