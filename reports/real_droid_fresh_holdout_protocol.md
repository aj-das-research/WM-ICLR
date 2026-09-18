# Fresh real-DROID candidate holdout: acquisition and metadata protocol

Registered UTC: 2026-09-18T22:51:34.940739+00:00

Status: acquisition and identity audit only. **Not eligible for evaluation.**
The original DROID study's test results have already been revealed. This separate
candidate is reserved for a future confirmatory experiment after development
choices are frozen using original training/validation data only.

## Deterministic selection before download

From the complete official public GCS `robotics/droid/1.0.0/` listing, exclude all
24 previously downloaded full-release shards. Rank the remaining shards by the
hexadecimal SHA256 of `shiftwm-real-fresh-v1-20260919:` concatenated with the exact GCS object name and
take the first 12. No resampling or topping up based on retained counts is allowed.
The full listing and selected inventory, including immutable GCS generations,
publisher MD5 and byte lengths, are stored locally before any new shard download.
Expected source episodes before exclusions: **556**. Download bytes,
including release metadata and CC-BY-4.0 license: **10875301444**.

The official dataset description is https://droid-dataset.github.io/ and the
download specification is https://github.com/droid-dataset/droid/blob/main/docs/the-droid-dataset.md.
Selection does not imply the whole published DROID benchmark was evaluated.

## Identity-only audit and exclusions

Verify every downloaded object's generation-bound publisher MD5, size and local
SHA256; validate TFRecord CRCs. Read only the two episode metadata strings from
each TF Example. Derive episode IDs identically to the original study and derive
the same collection-site/calendar-day session key. Do not decode JPEGs, inspect
images, parse rewards/actions, generate features, or run any model or metric.
The status path component is checked only for schema validity and never filters
episodes. Raw path strings are not included in the candidate manifest.

Exclude the union of (a) any of all **433** original session IDs, irrespective of
their original split; (b) any of all **1,126** original episode IDs; (c) any exact
serialized-example hash already present in the original manifest. Report each
reason's count, intersections and union exactly. Duplicate candidate IDs or
ambiguous/mismatched session metadata fail the audit; they are not silently
removed. There are no success, motion, length, appearance or outcome exclusions.

The retained set is session-disjoint under this site/date definition. It is **not
established as scene-disjoint, object-disjoint or unseen physical environments**.
There is no clinical or physical robot control evaluation claim.

## Separation and future eligibility

All new bytes and metadata stay under `data/real_video/droid_fresh_v1`, outside
the original data, feature and training paths. The candidate manifest has
`eligible_for_evaluation: false` and is intentionally not a trainable data manifest.
No data is redistributed or added to GitHub. Preserve source license/attribution.

Before any held-out image decoding, a **separate** evaluation freeze must pin the
selected methods and all method/baseline checkpoints, calibration files, code,
primary camera, horizon, metrics, inference budgets, all secondary analyses and
statistical comparisons. Freeze that using original training/validation only,
then hash and archive the final candidate manifest with the evaluation freeze.
The current preparation script provides no operation to bypass this gate.
If too few sessions remain, report that fact without inspecting performance or
silently enlarging this registered candidate. Any later acquisition is a new
separately registered version.

## Registered files

- Selected inventory SHA256: `f28ac034531019a78f3c90e33c4813e1ee33d1176af6e14e708533caca1936ab`
- Full official listing SHA256: `11657274b751626b6a2b14e2949b0cdb9c2ada34ba0036f34b744651b2a58dbc`
- Original manifest SHA256: `0f89fa0f3ddfcc27f01e4926c1b1e5e1a039aac1111c4c13be9e454c57f34280`
- Original inventory SHA256: `ad6f3edb982c4c048da2756caf58a963e5b8b5826ea73944b7c1a5ca78684aeb`
- Exclusion identities SHA256: `436258ca32bd23ae1c07fb623538b95d43c435a9408975c9f1be95aeb4c8b52d`
- Preparation script SHA256: `b77b894adc0db599424879b97d66b75b676ce7975fbe34e5211e269dc538cecd`
- Reused immutable downloader SHA256: `35617e550eb5c6dbfff7b273ab4b22f29494b40a3df51eabc73c1079ce593304`

Selected shard object names, in registered rank order:

- `robotics/droid/1.0.0/r2d2_faceblur-train.tfrecord-01582-of-02048`
- `robotics/droid/1.0.0/r2d2_faceblur-train.tfrecord-01271-of-02048`
- `robotics/droid/1.0.0/r2d2_faceblur-train.tfrecord-01449-of-02048`
- `robotics/droid/1.0.0/r2d2_faceblur-train.tfrecord-01186-of-02048`
- `robotics/droid/1.0.0/r2d2_faceblur-train.tfrecord-00531-of-02048`
- `robotics/droid/1.0.0/r2d2_faceblur-train.tfrecord-01563-of-02048`
- `robotics/droid/1.0.0/r2d2_faceblur-train.tfrecord-01355-of-02048`
- `robotics/droid/1.0.0/r2d2_faceblur-train.tfrecord-01588-of-02048`
- `robotics/droid/1.0.0/r2d2_faceblur-train.tfrecord-01864-of-02048`
- `robotics/droid/1.0.0/r2d2_faceblur-train.tfrecord-00001-of-02048`
- `robotics/droid/1.0.0/r2d2_faceblur-train.tfrecord-01867-of-02048`
- `robotics/droid/1.0.0/r2d2_faceblur-train.tfrecord-02013-of-02048`
