# Figure 2 A/B/C review

Date: 2026-09-18. **Review of the paired-supervision composition before the planning-flow clarification.** The subsequent user-reported ambiguity between the predictor and CEM is addressed in `planning_flow_redesign_review.md`. The preceding inference-only records are historical and archived in `paper/build/method_pairing_redesign/prior_editorial_{brief,review}.md`.

## Observed improvement

The old figure devoted much of its area to conventional CEM/predictor mechanics. The new design gives A shared calibration, B calibration-before-dynamics inference, and C paired supervision their own visual identities. Mean/variance pooling differs visibly from sequential GRU inference. C uses real preserved training inputs to show two different relations rather than a symmetric disentanglement cartoon: appearance changes at fixed trajectory/actions for dynamics consistency; equal appearance across distinct batch entries for observation consistency. The C header explicitly identifies the losses absent from Unpaired contexts. Conventional planning remains readable but secondary.

## Concrete defects found and repaired

| First mock issue | Repair | Final inspection |
|---|---|---|
| Goal route crossed the executed-action route | Routed the goal in a separate upper/outer gutter | Goal-action crop has no junction/crossing between those inputs. |
| Dynamics title crossed the action connector | Removed redundant exterior title; B heading and transition symbols carry its role | The executed-action edge now terminates directly on the GRU input. |
| Candidate label approached the outer border | Shortened label and moved it inside a clear gutter | Paper-width and detail views show separation. |
| Training appearance-pair vignette omitted shared actions | Added one explicit `a_S` object feeding both A→B instances | Paired crop shows continuous separate inputs and no claim that images alone determine dynamics. |
| Training context maps appeared to accept RGB directly | Labeled each input mapping with the same frozen `E_0` | Context inference is understood in feature space. |
| Compact future branches could imply only one reaches CEM | Added a collection bracket across all three schematic branches | Calibration/planning crop shows the common score input. |

## Source-grounded semantic check

Reviewed `model.py` context inference, residual adapters, losses, rollout, and goal embedding; `data.py` appearance transforms and pairing; and the manuscript's exact losses. Raw observed support alone produces observation context. Corrected support and executed actions produce dynamics context. Goal correction is shared with support; the goal reaches scoring only. Each paired view independently obtains its observation context/correction before dynamics-context comparison. The observation penalty matches only context vectors for equal appearance IDs at unequal batch indices. No context-swap supervision, contrastive negatives, physical labels, guaranteed separation, or success claim is introduced.

Canonical targets and pairing IDs are training-only. The two frozen-encoder locks are correct; gray reused LeWM/action machinery is trained offline. All weights and inferred contexts are fixed during each CEM search, with contexts refreshed from new real history on replanning. The loss links denote optimization penalties, not achieved equality. Exact query indexing/canonical losses remain outside this diagram.

## Actual visual checks

- Inspected the previous exported image and the first new mock at 550px and 1650px.
- Inspected final `paper/build/method_pairing_redesign/final/paper_width.png` and the corresponding detail image; final proof size is 550×442 and 1650×1324.
- Opened enlarged paired, inference, goal-action, and calibration/planning regions (coordinates/hashes in `inspection_report.json`).
- Opened `paper/build/method_pairing_redesign/proof/grayscale.png`: module grouping, relation labels, dashed training boundary, matched identities, and solid-versus-penalty links remain distinguishable.
- No material text/connector overlap, missing endpoint, wrong junction, clipped image, or off-canvas object was found in the inspected final views. The first feedback elbow and all paired-action/penalty endpoints remain visible.
- LaTeX compilation has no warnings. Poppler found no out-of-bounds extracted words. These checks do not establish semantic correctness or absence of all geometric defects; the actual pixel/semantic reviews are recorded separately here.

## Independent review

Root inspected first mock and final paper-width/detail plus goal-action/paired crops. The approved composition foregrounds the proposed components and makes the distinction from Unpaired visible; root found no material overlap in the corrected final views and approved exports.

`/root/paper_organization` separately inspected paper width and the final changed crops. The reviewer confirmed shared executed actions, per-view context construction, actual GRU conditioning, and distinct training penalties; no missing tips, overflow, or connector crossing was found. The writer owns the A/B/C caption and final manuscript placement. Standalone approval is not a claim that the final integrated page has already been inspected.

## Export and evidence checks

The standalone PDF displays at 5.5×4.4116in. Labels remain approximately 8.5pt, headings 9.5pt; math subscripts are conventionally smaller. Four deployment and three unique training rasters retain their native 224×224 pixels, approximately 478–522ppi at their final sizes. Asset hashes match all three source ledgers. SVG has 10 embedded image nodes and no external image references. The deliverable is hybrid: exact vector geometry and source-editable TikZ labels plus recorded rasters. SVG text is outlined, not claimed to be live text.

The paired example is an explanatory selection of actual inputs, not a sampled optimization trace or empirical result. The new composition adds no performance marks and does not certify novelty.

## Reviewed standalone artifact SHA256

- `paper/figures/world_method.tex`: `80ce5829b59ade4309cd70d4f91d3e6cee00eba42a604cae8ff4403e981c1dd9`
- `paper/figures/method_standalone.tex`: `f55e81125098332378a86d79dc8abfb7dba741c6e7b2ad3ec6f6bde60aa86d0e`
- `paper/figures/method.pdf`: `c8f386ac042b3c3b8a9df76af58498bbfbeab8ac7736bd5da7de767f64842fb7`
- `paper/figures/method.svg`: `b3bd43cefa43de7c5c2673c10619e70bbc4a7f87af5800a7fb9bd48fdee73111`
- `paper/figures/method.png`: `732f9bb6b6976376b24048cb5bed8d272225f746822c6577331ed841a9539899`
- `src/shiftwm/model.py`: `2f783caf8eed699a8d024f689aa956efd126a2c8b6c67aea23e4ff5dfb8536f8`
- `src/shiftwm/data.py`: `28bda1ce5ed755fb9f1fe9c1975c46ba983763062af82121a838b0876b64567c`
- `paper/figures/assets/method_assets.json`: `6df08d9d615783f07932355b1782e873c1e292afd04709e8c363ff128fd0efec`
- `paper/figures/split_assets/manifest.json`: `f9334bb84144e2265806a2d42b38e4490a7dae8b43d3cd3f65f911ec8f935b34`
- `paper/figures/assets/method_training_assets.json`: `ff9e54a2c720ae3e14600c253f4591a5432205eca54799cfc8b0f6b15afbb77c`

These hashes identify the source and standalone artifacts reviewed here. Later integrated builds can regenerate PDF metadata. The final integrated manuscript and source hashes are pinned separately by `paper/evidence/manuscript_sources.json`; this record does not silently adopt another build's hash.
