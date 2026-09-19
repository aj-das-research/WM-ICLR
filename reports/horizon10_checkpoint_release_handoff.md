# Horizon-ten model release handoff

Published and fully verified: 2026-09-19T00:25:08.326776+00:00.

Release: https://github.com/aj-das-research/WM-ICLR/releases/tag/horizon10-development-v1
Tag source commit: `63a94ee2f7b784a9ba71ae935daaa4464567eff1`.

| Item | Verified status |
|---|---|
| Models | All 12: four methods × three seeds |
| Full training | All 30 epochs completed for every run |
| Selected weights | Epoch 1 for all 12, using unchanged window-weighted all-ten validation selection |
| Saved evaluations | All 60: 24 standard and 36 matched diagnostics |
| Comparisons | All 20, including negative and inconclusive outcomes |
| Exact source bytes | Every model.pt/config.json matches its selected source generation |
| Independent physical relocation | All 12 exact CPU forecasts, maximum difference 0.0 |
| Public assets | Six server SHA256/size checks and full anonymous downloads passed |
| Secret scan | No findings |
| New tests | 10 parity-contract/selection/population failure cases passed |

Archive bytes: 246,426,955; SHA256 `e4e6e03bdd6cc0d31420b48cb41677b3c162ec83e5e34ade4c4f674215479c5f`.
Archive contains 235 payload files plus its manifest; all 60 complete episode ledgers accompany the models.

## Inference contract

Use the bundled `horizon10_train.load_package`, not the original h5 loader. The distinct package kind is `shiftwm_real_video_droid_horizon10_v1`. Inputs are three raw 1536D DINO feature observations, two past 35D command blocks, and candidate future command blocks; output is one 1536D frozen-feature forecast per future block. Each command block groups five recorded 7D Cartesian/gripper commands. The shared pinned DINO encoder, exact source, normalization, licenses, cards and training journals are included.

The parity fixture contains only the three genuine observed validation feature vectors and command sequence; future image targets are excluded. Fresh physical-copy relocation uses Python isolated mode, a separate source tree, CPU execution and socket guards. Same-input predictions must match exactly. This is engineering portability, not RGB-cache equivalence, OS network-namespace isolation, extra benchmark evidence, or new model selection.

No optimizer/RNG state, raw videos, full datasets, installed environment or credentials are in the archive. The original training runs remain untouched and retain resume state. Project predictor code/weights are MIT; DINO Apache2.0, LeWM MIT and DROID-derived fixture/provenance CC-BY4.0 retain their complete notices.

## Scientific gates and interpretation

The exporter reuses the stable paper postprocessor collect()/aggregate() read-only. It binds exact registration/source/configuration/data/model identities, verifies all 30-epoch journals and selection, reopens all 60 saved evaluation files, verifies exact episode/session/window populations, recomputes 2,304 saved summary cells, and independently recomputes all 20 crossed seed/session intervals with 10,000 draws. It does not rewrite or rerun any registered scientific result. Separate corroboration is in `reports/horizon10_release_independent_audit.md` and `reports/evidence/horizon10_actual_paper_review.json`.

All means are development evidence from original validation recordings. Training target and matching checkpoint-selection horizons change together; weighting and architecture remain fixed. Standard h5 evaluation has 1,772 windows; h10 and matched h5-prefix populations have 1,631 windows, each covering 141 episodes and 59 recording sessions. These populations must not be conflated.

Of the 20 preregistered contrasts, 19 point estimates favor the first named method, 14 intervals exclude zero in that direction, and six include zero. The matched ShiftWM h5-prefix mean has a -0.116% point change with an interval including zero. Against equally h10-trained Framewise, ShiftWM endpoint h10 and all-ten mean changes are +0.405% and +0.252%; these are small validation effects. No SOTA, new held-out/fresh-test improvement, robot-control or clinical result is established.

## Durable artifacts

- `reports/horizon10_checkpoint_publication_status.json`: authoritative verified receipt and six public asset URLs/digests.
- `artifacts/publishing/horizon10-release/publication_receipt.json`: identical local receipt.
- `artifacts/publishing/horizon10-release/shiftwm-horizon10-v1/manifest.json`: full files/source/model inventory.
- `artifacts/publishing/horizon10-release/shiftwm-horizon10-v1/verification/offline_packages.json`: all 12 exact relocation cases.
- `scripts/publishing/publish_horizon10_release.py`: publisher, reusing unchanged source/export/audit components.
- `logs/horizon10_release_export.log` and `logs/horizon10_release_publish.log`: successful CPU-only execution logs.

No frozen scientific/spatial/scanner files, main/source checkouts, site, paper or Overleaf content was modified. Root can now advertise 102 publicly released neural predictors: 12 original real-DROID + 36 generalization-development + 42 simulator-development + 12 horizon-ten-development. Calibration wrappers are not additional neural models; still-running spatial models are not counted.
