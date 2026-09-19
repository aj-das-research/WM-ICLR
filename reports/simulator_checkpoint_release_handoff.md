# Simulator checkpoint release handoff

Published and fully verified at 2026-09-19T00:17:17.666031+00:00.

Release: https://github.com/aj-das-research/WM-ICLR/releases/tag/simulator-development-v1
GitHub tag source commit: `7d6208de1ce8b31dc1c458817dcda378663f8757`.

| Component | Count / status |
|---|---|
| Original simulation predictors | 36: two domains × two predictor families × three methods × three seeds |
| Observation-gain revision predictors | 6: drone transformer × two methods × three seeds |
| Training completion | All 42 runs completed all 30 epochs |
| Selected epoch counts | 25:3; 27:14; 28:3; 29:11; 30:11 |
| Independent relocated CPU parity | 42/42: exact full tensors, cached forecasts, RGB forecasts and RGB encodings |
| Maximum source/export difference | 0.0 |
| Common tensor bank | 421 exact shared tensors; 50,535,679 bytes |
| Public asset verification | All six server digests and full anonymous downloads passed |
| Tests | 15 format, relocation, tamper and population checks passed |
| Secret scan | No findings |

Archive: `1,321,091,128` bytes; SHA256 `528724d917442e6d8e5f02607c65de8aa6ac2665f16e3017701fd73f5a4dce5b`.
Bundle inventory: 1029 payload files plus manifest; 1,521,782,363 uncompressed payload bytes.

## Reuse

Download and extract the release archive. Install its pinned `requirements-inference.txt` dependencies and use `shiftwm.simulator_release.load_package(release_root, model_id, device)`. The IDs live in `models.json`, with `original/` and `geometry/` namespaces. The loader reconstructs the full original state from the shared bank and each model delta, then uses the appropriate original or version-2 constructor. Historical training paths are metadata only.

The package includes a frozen LeWM visual encoder, all predictor tensors, normalization, code, model cards, full source/license provenance, completion journals, every development comparison including regressions, and two genuine validation support snippets. Project-trained additions and generated snippets have an explicit MIT distribution notice. Upstream licenses/attribution are preserved.

These are inference packages, with optimizer/RNG state omitted. The old `artifacts/releases/extensions_v1` and all 42 original training directories remain unchanged and retain resumable states. This is lossless tensor sharing, not quantization or retraining.

Inputs: three observed 192D frozen features, two past 10D action blocks, and K future candidate 10D blocks. Each block groups five chronological 2D commands. Outputs are latent features, not generated RGB or policy actions. RGB histories use the bundled encoder.

## Evidence and limitations

All completed-result evidence (1,491 existing artifact hashes) was reverified before export. The original 36 and six geometry models passed their established full-30-epoch validators and exactly match the selected checkpoints recorded in `reports/completed_extension_results.json`. A separate CPU process using relocated bundled source and a Python socket guard verified every reconstructed tensor fingerprint and same-input cached/RGB forecast exactly. This is not a claim of OS network-namespace isolation. The two validation snippets support engineering portability only, not added benchmark evidence. CPU encoding versus historical GPU caches uses the disclosed rtol0.005/atol0.002 tolerance; source-versus-export CPU outputs are exact.

The domains are simulated planar drone control and simulated LapGym/SOFA deformable tissue positioning. Results are mixed: drone GRU and surgery transformer comparisons favor baselines, and no original paired interval has a strictly positive lower bound. Geometry revision outcomes are follow-up development comparisons. No SOTA, real-flight, patient-data, clinical, physical-control, or independent-test superiority is established. The full historical report also describes a separate AdaJEPA reproduction; AdaJEPA weights are not included in this release.

## Durable files

- `reports/simulator_checkpoint_publication_status.json`: authoritative public receipt and all six asset checksums/URLs.
- `artifacts/publishing/simulator-release/publication_receipt.json`: identical local receipt.
- `artifacts/publishing/simulator-release/shiftwm-simulator-development-v1/manifest.json`: every payload identity and all 42 model records.
- `artifacts/publishing/simulator-release/shiftwm-simulator-development-v1/offline_verification.json`: all 42 independent parity cases.
- `logs/simulator_release_export.log` and `logs/simulator_release_publish.log`: successful CPU-only execution logs.
- `reports/extension_release_independent_audit.md`: earlier independent 36-model audit and publication requirements, now addressed.

No main-branch, site, Overleaf or existing-release mutation was performed by this artifact task. Root can now advertise the verified release and update the public model count to 90 neural predictors (48 real-DROID plus 42 simulator models); calibration wrappers are not additional trained neural models.
