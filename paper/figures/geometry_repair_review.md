# Connector, asset, and layout review — 18 September 2026

This is a historical intermediate review. Further user-provided screenshots exposed arrowhead/shaft defects missed here. The regenerated layout and the current checks are documented in `arrow_regeneration_review.md`.

This is the second review of the sparse visual redesign. A more focused zoom inspection found defects missed by the initial broad legibility check. Those defects were repaired in editable source, not covered over in exported images.

| Observed defect | Repair | Verification |
| --- | --- | --- |
| The feedback label masked part of the action/observation wire | Offset the label above a single continuous path | Full-width proof and bottom-loop crop |
| Blue history and goal routes crossed `z_S` and `z_g` | Move routes into a right-side gutter; add an explicit history fork | Context/goal crops; labels clear of shafts |
| CEM output pointed toward an unexecuted future block | Connect its output to the highlighted first block's south port; feedback exits that block's west port | Bottom-loop crop traces one continuous execution route |
| Terminal collection stubs were separated from candidate dots | Use exact shared center coordinates | Enlarged terminal crop |
| The corrected bus overpainted the selected terminal dot | Draw the collection wires before the terminal markers | Final crop shows complete circular markers and attached wires |
| Several encoder/context endpoints used approximate coordinates | Named encoder/predictor/context anchors and declared score/action ports | Enlarged endpoint crops |
| Candidate input lacked an explicit sequence-group boundary | Add a headless group bracket with a single outgoing path | Candidate-to-adapter route is recoverable |
| The teaser PDF silently downsampled the illustration to 100ppi | Export embedded raster layers at 300dpi in PDF/SVG as well as PNG | `pdfimages -list`: 668×446 pixels at 300×300ppi, replacing 223×149 at 100ppi |

The generated concept illustration and downloaded official Material lock PNG both have true alpha; no background-removal operation was needed. The lock image is unchanged official artwork, used only to identify frozen encoders. Its source, Apache-2.0 license, original checksum, pixel dimensions, display width and transparency check are retained in `assets/material_lock.json` and the adjacent license. The generated illustration's prompt and original checksum remain in `assets/concept_asset.json`. The model diagram's task frames are real, unmodified simulator inputs. No logo, icon, or generated asset is used as performance evidence.

## Applied skill workflow

The installed skill now routes to `asset-sourcing.md` and `geometry-and-inspection.md`. These cover web/image search, actual original downloads, official logos and model illustrations, generated assets, background removal when helpful, before/after alpha-edge review, named ports, route continuity, alignment, spacing, overflow, crossings, and z-order. Image generation remains separate from exact scientific wiring and numerical evidence.

The reusable `scripts/inspect_figure.py` produces 100ppi-equivalent paper-width proofs, 300ppi detail views, and reproducible normalized crops. It records source/output hashes, dimensions, scaling, and conservative extracted-text boundary checks. It explicitly does not claim to detect all overlaps, arrow defects, scientific errors, or visual quality. The script was exercised on the real method/setup/teaser PDFs; the root agent opened the actual paper proofs and context, terminal, feedback, factor-header and illustration-edge crops. Their JSON reports are under `paper/build/precision_review/`; a durable summary is in `paper/evidence/figure_geometry_checks.json`.

An independent reviewer confirmed the major route repairs and identified the terminal-dot overpainting, which was then fixed and checked in the final crop. Setup and teaser geometry also passed independent paper-width review; grayscale checks preserve their meaningful distinctions. No final label/shaft crossing, broken information path, ambiguous execution destination, or cropped label was observed in the inspected figures. This is a record of completed visual inspection, not a promise that automated checks guarantee perfection.

The integrated manuscript is rebuilt from these sources. PDF/SVG retain precise vector diagrams and text around their raster assets, with 300dpi PNG previews. Editable source, saved assets, source/license records, and the portable skill patch are all local. GPU experiments continue separately.
