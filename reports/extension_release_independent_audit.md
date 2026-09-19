# Independent audit of the simulator checkpoint release

The existing `artifacts/releases/extensions_v1` contains **36 completed, validation-selected world-model checkpoints**, covering two simulation domains × two predictor families × three methods × three seeds. No incomplete run was found in the independent metadata/selection audit. This review does not certify a new deduplicated export before its own parity checks.

The read-only audit is recorded in `reports/extension_release_independent_audit.json` (SHA256 `11815084ce1b8cc0c68bdf0e50f7e3fdadd6e468a7e143094b313d3a532ac3a2`). It binds reviewed source/documents and all 36 model identities. No old package, frozen source, training record, or publication destination was modified. No new inference or training was performed during this review.

## Publication requirements identified

1. **The old offline proof covers 12 representative checkpoints, not all 36.** `offline_verification.json` selects seed 0 for each domain/family/method combination. It proves strict loading and actual cached/RGB API compatibility for those cases. A new deduplicated exporter should verify every one of the 36 reassembled state dictionaries against its source model and exact cached-input prediction parity, plus representative RGB parity. Do not relabel the old 12-case proof as an all-model or new-format proof.
2. **The current archive deliberately leaves its new weight/example distribution notice unspecified.** `LICENSES.md` states that a public model/dataset release needs its own explicit notice. The publication archive should state the license of the project's trained additions and generated example snippets, separately preserve the pinned MIT base-model/source attribution, and avoid inferring generated-data terms from the simulator's code license.
3. **The linked model-license/provenance manifest is missing from the source archive.** Bundled `source/THIRD_PARTY_NOTICES.md` points to `references/world_artifact_sources.json`, which is not in `extensions_v1/source`. Include that file, or a compact self-contained equivalent with the exact pretrained model revision, source URL, license declaration, and weight hash. The information exists in the main repository and can be carried over without changing scientific artifacts.
4. **Public-facing status text needs a factual refresh.** The archive currently says local-only/unpublished and that performance is “pending or mixed.” `reports/completed_extension_results.md` now records completed development forecasting and planning. New release documentation should link that evidence, retain the negative comparisons, and update publication status only when publication has actually succeeded. The old immutable archive can retain its historical wording.

These are requirements for the forthcoming publication artifact. No evidence here supports replacing the completed checkpoints or changing their selected epochs.

## Completed-versus-partial checks

Independently checked all 36 exported packages against the registered Cartesian product and their copied records: exactly 30 chronological epochs; finite nonnegative recursive validation losses; the selected epoch is the **first exact minimum** of the full validation journal; selected epochs lie between 25 and 30; the selected model has no incomplete-batch progress; the model's embedded configuration matches its external configuration; executed and registered scientific configurations agree; and recomputed training identities match the summary, checkpoint metadata, and release record. Bundled relevant source hashes match each checkpoint's training provenance.

The original exporter calls `shiftwm.extensions.train.validate_completed` before copying. That authoritative gate additionally validates both original best and last generations, optimizer/RNG pairing, epoch-30 final state, validation-only selection, and strict CPU model construction. The exported optimizer state belongs to the selected best epoch, not necessarily epoch 30. This is accurately disclosed by the original cards and must remain explicit if optimizer state is retained.

An inference-only successor may omit optimizer/RNG files, but it needs a new package manifest/loader contract and must retain source identity, completion evidence, and selected-epoch provenance. Removing `training_state.pt` while preserving a manifest that requires its hash would break the old loader. Preserve the original archive byte-for-byte while constructing the separate publication format.

Evidence paths: `src/shiftwm/extensions/train.py::validate_completed`, `src/shiftwm/extensions/checkpoint.py::read_package/load_package`, `scripts/extensions/export_packages.py::export_one`, the release `manifest.json`, and each checkpoint's `metrics.jsonl`, `training_summary.json`, `config.json`, and `release_record.json`.

## License and provenance inventory

| Component | Evidence/declared license | Release implication |
|---|---|---|
| Project source | Root/bundled `LICENSE`: MIT, ShiftWM contributors | Preserve project notice. |
| LeWorldModel implementation | MIT, Lucas Maes; source revision `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` | Preserve `vendor/lewm/LICENSE` and `NOTICE.json`; bundled loader verifies source hashes. |
| Frozen visual encoder and transformer initialization | `quentinll/lewm-pusht` revision `22b330c28c27ead4bfd1888615af1340e3fe9052`; pinned model-card/API declaration MIT | Original base weight SHA256 `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`; retain attribution and license evidence. |
| Drone simulator source | `gym-pybullet-drones` revision `7ebad1ecabd28a7000add2d05f888aa2e837c2cc`; MIT, Jacopo Panerati | Notice already copied under `licenses/`; simulator binary/runtime is separate from inference. |
| Tissue simulator source | `sofa_env`/LapGym revision `85bf7e05dd088b824794dda0046679df13b13e6e`; MIT, Paul Scheikl | Notice already copied under `licenses/`; retain simulated-task attribution. |
| SOFA runtime | Version 24.06; bundled notice is LGPL v2.1 text | Runtime binary is not included in the checkpoint archive; its notice does not become a license declaration for new model weights. |
| Extension trajectories/examples | Locally generated simulation, with pinned collection manifests | No real-flight or patient dataset is distributed. Full trajectories are absent; the two small example snippets need an explicit publication notice. |

License descriptions above report the existing source/card declarations, not a blanket statement that one license covers every runtime, asset, output, and model. The pinned base license evidence is in `references/world_artifact_sources.json` and `THIRD_PARTY_NOTICES.md`.

## Portability and inference contract

The current constructor rebuilds the architecture from bundled configuration and fills a complete state dictionary; it does not require downloading encoder weights or following the historical training path. Vendored upstream source hashes are checked. The established interface is three observed frames or `[B,3,192]` frozen features, two observed `[B,2,10]` action blocks, and `[B,K,10]` candidate command blocks. Each action block contains five chronological 2D controls. Outputs are `[B,K,192]` feature forecasts, **not RGB frames or robot-policy actions**. RGB is float BTCHW in `[0,1]`; the implementation normalizes internally and resizes to the pinned 224×224 encoder input with bilinear antialiasing.

A deduplicated format must restore every omitted shared tensor with exact dtype, shape, and content; verify that the frozen/reference encoder and projection copies are actually identical before sharing them; reject unknown/missing state keys; bind each delta to its common base hash; and compare all 36 reconstructed models to the source. Keeping the existing model code for inference reduces format drift.

The supplied environment is a tested Python 3.11/Linux dependency set with CUDA-build PyTorch that also supports CPU execution. That is not evidence for every Python version, accelerator, or operating system. Dependency installation can require internet; subsequent tested inference is offline. The original RGB-versus-cached encoder proof uses a documented tolerance (maximum observed differences approximately 0.00099 for drone and 0.00082 for surgery), whereas same-input repeated CPU forecasts are exact. Do not turn the cross-device tolerance check into a claim of bitwise cache equivalence.

## Scientific scope to retain in model cards

Drone is fixed-external-camera planar simulated CF2X control with an upstream PID controller. Surgery is simulated deformable-tissue positioning in LapGym, not clinical video, autonomous surgery, or a deployed medical system. The GRU is an in-house recurrent architecture control, not Dreamer; its initialization and capacity differ from the transformer family. These are feature-world-model packages; a planning system also requires its separately documented CEM/environment components.

Completed development results are mixed: the original drone-GRU and surgery-transformer planning comparisons favor their baselines; other gains are small, and no original paired interval has a strictly positive lower bound. The report concerns eight common development tasks per domain and three model seeds, not 24 independent tasks. Do not promote these checkpoints as best-performing policies, real-robot transfer models, or a state-of-the-art comparison. The separate six geometry-revision checkpoints require their own version-2 loader and are not part of this 36-model original release.
