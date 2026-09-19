# Next evaluation gate audit

Snapshot: 2026-09-19T21:41:20.610563+00:00. Read-only source/metadata/package audit. No reserved IWS video/command payload, DROID test image/feature payload, model inference, job submission or registered-file edit occurred in this audit.

**Recommendation:** finish the authorized common-CPU internal-development repair/finalization, then release all27 selected IWS predictors. The cleanest next new evaluation is the already specified IWS feature-only reserved-handle study, after a separate evaluator/access registration. No current command is authorized by its existing registration to open that reserved set.

## What is ready now

- All27 registered runs report completed30epochs; all27 development receipts exist and pass their individual gates with zero official-validation payload reads. The9 AR receipts came from the exact frozen CPU evaluator. The18 additive/mixing GPU receipts are now scheduled by root for an archived, exact CPU consistency rerun; this audit does not call the present mixed-backend snapshot the final common-CPU comparison. The complete development finalizer was absent when checked; sibling owns it.
- All54 `best`/`last` package manifests exist. Every model/config digest was independently checked, every selected model digest matches its evaluation receipt, and every package has the distinct `shiftwm_iws_single_observation_v1` kind. Training-state files exist; their contents were not re-audited here. These are resumable local training packages, not a published standalone release.
- There is no `artifacts/releases/*iws*` release. Existing `train.py export --run RUN --output NEW_DEST` calls `export_inference_package`: full30 validation, selected-best weights/config copy, safe package manifest and atomic new directory, excluding optimizer/RNG. It exports **best only**; last stays local for resumption. No new training is needed.
- Smallest useful immediate job: finish18 CPU replacement evaluations under the sibling's operational archive/recovery registration, then CPU export/reload parity for all27 selected packages. A portable release additionally needs the minimal copied model/vendor source, dependency/model cards and a relocated `python -I`, offline/socket-guard prediction check on a fixed authorized training input. The existing export alone does not supply that proof or a bundled RGB encoder. Feature-input API is one6144D observation plus native command rows; these checkpoints do not generate RGB.

## IWS reserved evaluation: specification exists, access implementation does not

The frozen split metadata identifies200 original H60 handles and10 reserved trajectories **per task**:600handles/30trajectories total. Reserved metadata reports5,996 stored image/command rows (PushT2,000; Box1,996; Rope2,000); this audit did not open their payloads.

`training_registration_v1.json` explicitly covers internal train/development only. `campaign.py` says it cannot unlock official validation; both IWS inventory classes reject reserved IDs, and `evaluate.py` constructs only `internal_development`. Do not patch those guards or feed official handles through development window sampling.

Before any reserved decode/feature extraction, freeze a **new namespace** that binds the final27 checkpoint hashes/selected epochs, training statistics, exact600handle identities, original preprocessing and encoder revision, full evaluator/cache source/tests, inference backend/batch settings, and all reporting/selection rules. Declare **feature-only, no RGB decoder**; the optional decoder is not required for this study. Test alignment, leakage rejection, fail-closed population checks and causal-prefix checks using training fixtures first. Root's project authorization can then permit the separately recorded access gate.

Keep the existing scientific plan unchanged: one observed initial frame; H60 consumes60rows and predicts offsets1–59; exact H15/30/45 prefixes use only15/30/45rows at the same starts. Primary is bounded mixing versus anchored additive at H60 standardized feature MSE, equal-trajectory averaging and equal-task macro relative reduction. Keep AR/persistence and all four feature metrics as secondary, equal-handle means too, paired seed×trajectory10,000draw bootstrap(seed173), all signs and the frozen best/median/worst rule. Ten trajectory clusters/task are not200 independent samples. This is not an RLA-WM/SOTA reproduction or real-robot control evaluation.

Prefer **CPU FP32 inference for all27** in this new freeze until CUDA prefix consistency is separately resolved. All9 completed CPU AR evaluations passed with zero prefix discrepancy. CPU success does not validate CUDA; do not relax the existing tolerance or silently change frozen model math. Cache extraction may retain the existing BF16 frozen-DINO recipe on an allocated GPU.

## DROID spatial: not an untouched-test switch

The spatial registration, loader and cache explicitly permit original train/validation only; the evaluator hardcodes `val`. There is no frozen spatial test evaluator/cache/access gate. `data/features/droid_spatial_v1` has no test cache. Changing a config split is neither sufficient nor protocol-preserving.

Original DROID test outcomes were already published for the historical models. The65-episode/52-session fresh-v1 set was also already unlocked and evaluated for historical calibrated models (`real_droid_fresh_evaluation_results.json`: completed; verification passed). Neither population is now globally blind. Current spatial checkpoints could be evaluated there only through a separately frozen **secondary transfer** protocol with this reuse disclosed. A genuinely fresh spatial confirmation needs a new acquisition excluding all previously exposed sessions/episodes, followed by its own model/evaluation freeze. Do not reuse the old historical fresh-set freeze as spatial authorization.

## Resource plan (requests, not measurements of a new campaign)

Controller checked this audit: queue empty; account `students` as in prior allocations. `ws-ia`/`ia-std`: MaxJobsPU2 and24CPUs/110000MiB total. `gpu`/`gpu-1`: at most1GPU and16CPUs/92160MiB. Thus at most **2 workstation GPU jobs +1 gpu-partition job**, never27 simultaneous jobs. Physical free GPU memory remains unknown until allocation; retain visibility and inspect allocated-device memory, never override CUDA visibility.

| Stage | Conservative allocation | Timing basis / limit |
|---|---|---|
| Common-CPU dev repair or27-package release/parity | 0GPU,8CPU,16GiB; CPU-only allocation as prior `gpu`/normal recovery | Prior full AR dev evaluations took125–138s/run for~3,360windows. Request1h for repair/release; finalizer budget owned by sibling. |
| Future official feature cache, after gate | 1GPU,8CPU,24GiB,1h; three tasks sequential | Only5,996 metadata-listed frames; decode/encoder throughput not newly profiled. Use existing batch/precision and fail on any mismatch. |
| Future official inference, after gate | 0GPU,8CPU,16GiB,1h; all27 sequential |200handles/run is much smaller than internal dev, but independent prefixes and I/O add cost. The1h request is headroom, not a measured speed claim. |
| Optional GPU backend diagnostic, training inputs only | 1GPU,8CPU,24GiB,30min | Existing batch64 FP32 forward peak~0.42GB and0.0016–0.0023s/window on RTX5000Ada; this establishes capacity, **not** prefix correctness. |

The official forecast campaign need not occupy all three GPUs. Preserve all tasks/arms/seeds in one registered campaign; no task/arm selection after outcomes. Exact official execution command is intentionally not supplied because that evaluator/access registration does not yet exist.

## Source bindings

- `configs/real_video_iws/training_v1.json` — `6c8b34b9422891a23ccf64df853aeee82eff63cd80842f6244f4409ed51332ac`
- `configs/real_video_iws/training_registration_v1.json` — `edbff0ee007201be302adc70a607270759a89f598f76361464a0088df4e25995`
- `configs/real_video_iws/split_v1.json` — `17be56426dbee136ec883c45813b092a0747d707f25ee6efddb12b8a351d29c9`
- `reports/real_video_development/iws_single_observation_design.md` — `19ce4edfde1418a611b7c17eb974010af2cb2448e976329ca138a2fd7f98202d`
- `scripts/real_video_iws/evaluate.py` — `ceeef9b7d79b8e9fa6255a55b23acfaee265b564ff4df51e339f9cb675689714`
- `scripts/real_video_iws/finalize.py` — `9a59b547158782c775e95de059b713bde3010765449764eec7adb518b2af728e`
- `scripts/real_video_iws/train.py` — `2e84d01d1ff085325a93925fe3385f30f5ee0a965823c14b23800572fbd24d74`
- `src/shiftwm/real_video_iws/training.py` — `a3129da1db0f5e6f280c3e700236750df414c89465bd8e2c69d25fe906237e92`
- `src/shiftwm/real_video_iws/data.py` — `cf173ede6f02dabde8ca7a00415e78edc21d61d43e7a3c656ccdccdafcc8ab02`
- `src/shiftwm/real_video_iws_tasks/data.py` — `945602b6683be7c88e2132c3381b71b452b6e58a55eab935b601afb3ab00ccf5`
- `reports/real_video_iws/recovery/seed12_cpu_evaluation.json` — `5fc70a9d39ba5afd83f74af4d56bb0c884700eb8c464a5dcfca8dd369236f14a`
- `reports/real_video_iws/full_shape_profile_final.json` — `33a02884b33cd43470d2ab09d61bd3a060b08e968bfe2c24b0b42ea19a8d4a4a`
- `reports/real_video_development/spatial_protocol.md` — `27a8a4948a1da2af9a1bdccb88414d6c484dfdd2e089d6d900ed266296e2c8a1`
- `src/shiftwm/real_video_spatial/data.py` — `5a502e9acea1ba0da4d5953198185a29a79854df7f7232faf8e0b212646103d4`
- `scripts/real_video_spatial/evaluate.py` — `dad9bd2df511c87fc2dfc7dbbd48d413327885a111cfc8211c8d7169571cdfbb`
- `reports/real_droid_fresh_evaluation_protocol.md` — `f657238486c9fb6da3add683e28915ea9a59a659bb34debd652ea9747b47171d`
- `reports/real_droid_fresh_evaluation_results.json` — `d1d39484ffc35bf0f4a5a4d8f9cc5058e5f0f4ff54964bc486103567a7a255b5`
