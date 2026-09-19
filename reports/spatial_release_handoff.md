# Spatial checkpoint release handoff

The release pipeline is implemented, independently reviewed, tested, frozen,
and scheduled. Publication is **pending**, not complete.

- CPU job **200254** waits for successful spatial finalizer **200230**.
  Verified resources: 4 CPUs, 16 GiB, 2 hours, no GPU requested.
- Exact scientific registry: `ae99340f7a3761ece066ee7376f4bd119b61a92503948c142196c40a1a76e337`.
- Export registration: `artifacts/publishing/spatial_v1/export_registration.json`,
  SHA256 `61e575cb56c122e030c86d6a67bfac03b9d75965af4ab6681aee2894b17e795a`.
- Frozen executable: `artifacts/publishing/spatial_v1/frozen_exporter.py`,
  SHA256 `d6933278124af7b441ab1ccd141faa603503c71c0e72f1adcb91dbc2a99cf601`.
- Frozen shared scanner: `scripts/publishing/prepare_public_snapshot.py`,
  SHA256 `8cec9654376a191d023eb6ca6bd03698e4736104ce5d30ee074191f855ceedcd`.
  Editing this dependency before execution intentionally fails the release;
  coordinate a separate reviewed registration if a change becomes necessary.

The job requires all 15 registered models to complete 30 epochs and checks
full training journals, selected/last packages, source/cache/normalization
identities, all 15 validation ledgers and all 16 paired comparisons. It repeats
exact CPU predictor parity from a physically relocated copy of the actual
new bundle, verifies the shared pinned DINO encoder, scans secrets/paths,
checks every archive byte and publishes an immutable GitHub prerelease only
after these gates. All positive, negative and inconclusive outcomes remain.
No raw videos or optimizer payloads are exported. The root model card and
individual cards are generated from the completed finalizer, not guessed now.

Publication target: `aj-das-research/WM-ICLR`, tag `spatial-world-models-v1`.
Assets and local proof will be in `artifacts/publishing/spatial_v1`;
final public-download hashes go to
`reports/spatial_checkpoint_publication_status.json`. Monitor
`logs/spatial-model-release-200254.log` after the dependency completes.

Validation: 100 combined scientific/release tests passed in 10.77 seconds;
seven additional release checks were then added, and the final 58-test release
suite passed in 2.43 seconds. Two independent agents found no blocking
contract/code issue. The exact reviewed sources, test evidence and scheduler
record are in `reports/spatial_release_preflight.json`. These checks certify
the release mechanism; they do not certify unfinished scientific results.

Full gate and interpretation contract: `reports/spatial_release_protocol.md`.
Existing registered scientific files and the shared public-snapshot script
were not changed by this task. No main paper, root README or site edits.
