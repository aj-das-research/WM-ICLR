# Generalization checkpoint release: independently verified

The `generalization-v1` research prerelease is public and ready to announce. Export job200187 completed successfully with exit0:0 in1minute56seconds on4CPUs and **zero GPUs**. It completed at2026-09-18 23:56:04UTC (2026-09-19 03:56:04Dubai).

- Release: https://github.com/aj-das-research/WM-ICLR/releases/tag/generalization-v1
- Archive: https://github.com/aj-das-research/WM-ICLR/releases/download/generalization-v1/shiftwm-generalization-v1.tar.gz
- Size:443,302,514bytes.
- SHA256:`8ee06a9ea5ea4b10d6c485a173ef2a6921fda350e57656295d3224f212e747e2`.
- Release target commit:`d9d8165cbb26625f82713f83211e102b37c99f0e`.

## What is available

All36registered generalization predictors:3development arms×4methods×3seeds. Every training run completed30epochs; the distributed weights are the validation-selected checkpoints, with11selected at epoch1,11at epoch2, and14at epoch3. The archive includes the shared frozen DINOv2 encoder, exact reusable source, configurations/registrations, all72original-validation evaluations, full positive/negative aggregate reports, model cards, upstream licenses/attribution, and verification records. Optimizer/RNG state is excluded from inference packages and remains in the original training directories.

There are six public assets: the archive, `README.md`, `MODEL_CARD.md`, `manifest.json`, `offline_packages.json`, and `SHA256SUMS`. The release is explicitly marked a research prerelease, not a draft.

Together with the earlier `real-droid-v1` release, there are48public trained predictor checkpoints. The12earlier calibration wrappers reuse original predictors; they are not12additional neural models. The horizon-ten and spatial architecture campaigns are separate and are not included in these counts.

## Verification performed

The exporter verified all six GitHub server digests and anonymous complete-download hashes. A separate post-publication check repeated **all six anonymous downloads**, including the entire443MBarchive, and independently matched every SHA256 and byte count against the publication receipt and GitHub release API.

The local archive was then matched to that public archive digest and independently audited:455regular files, including454manifest-listed payloads and the manifest itself; every member's size/hash verified. Exactly36predictor weight files and one shared encoder are present. There are no unsafe paths, links, or optimizer-state files. The original audited export has zero secret findings; no raw video datasets or environments are distributed.

All36packages have passing isolated, relocated offline CPU forecast parity, with maximum absolute difference exactly0 on the registered validation-support fixture. This is a packaging/inference integrity check, not an additional benchmark result.

Evidence:

- `reports/generalization_checkpoint_publication_status.json`: authoritative successful publication receipt.
- `reports/generalization_checkpoint_independent_verification.json`: independent public downloads, manifest/archive audit, and card review.
- `artifacts/publishing/generalization-release/publication_receipt.json`: immutable exporter receipt.
- `artifacts/publishing/generalization-release/offline_packages.json`: all36CPU parity records.
- `logs/droid-generalization-release-200187.log`: successful asset verification/completion.

The older `artifacts/publishing/generalization-release/scheduled_status.json` is the historical pre-execution scheduling receipt. Its `uploaded:false` field is not the current release status; use the publication receipt above.

## Claims and pending publication work

The cards accurately label original-validation development, matched comparisons within each arm, and latent-feature outputs. They include four positive and two negative point-estimate horizon cells versus matched Framewise; they do not equate a positive point estimate with statistical significance. Selection epochs and completed training budgets are explicit. No stale or unsupported claim was found during card review.

These are optimization/capacity controls, not a new algorithm, fresh-test improvement, RGB generator, robot-control result, clinical system, or SOTA demonstration. Public paper/README/project-page updates may now link this verified release using those limits. The root agent owns those updates and sync; no duplicate publication workflow was started here.

No frozen scientific or export source file was changed. Subsequent manuscript/figure/PDF updates do not alter this completed, hash-bound release snapshot. No operational repair was necessary.
