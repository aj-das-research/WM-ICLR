# Landscape evidence review — 20 September 2026

Status: standalone PDF evidence and geometry review passed. Integrated manuscript pagination and float placement are reviewed separately by root.

## Reviewed outputs

| Figure | Size at manuscript width | SHA-256 |
|---|---|---|
| `paper/generated/editorial/spatial_composite_landscape_v1.pdf` | 5.5 × 2.25 in | `bc9df24a6db13bc733c973397ee97792144da1598be9187e23070b233a9f6cfc` |
| `paper/generated/editorial/forecast_landscape_v1.pdf` | 5.5 × 2.00 in | `2a7e3f2ba3568b75aa883883fc36cc8b4db018cba98fdf6226aca0ecb9348a63` |

Renderer: `paper/scripts/render_landscape_evidence_v1.py`, SHA-256 `40225f9b5b5d5d78cd8092b85064f693032aa387c283968d4a855250265b9559`.
Source manifest: `manifest.json`, SHA-256 `c1db6ea21e563cf40e6010532ed0e0e083b19f7af2d2ecb06af0c075b25e8a6c`.

## Scientific checks

- Main spatial composite retains all 50 exact original curve means and all 12 native endpoint paired bootstrap intervals; full 36-contrast ledger is retained in the pack. Persistence uses its own native persistence metric. All interval endpoints are sign-reversed and rescaled by ×1000 exactly from the source method-minus-comparator intervals.
- Historical forecast retains all 20 means, all 16 three-seed sample SDs, and four frozen points with no invented SD. Its 52 raw result JSON hashes were checked; seed aggregation and signed changes were recomputed. Historical context model and current spatial model are explicitly separated.
- Small negative h5 no-bounding comparison, zero-crossing ablations and +4.15% Reacher extrapolation regression remain visible. Green changes indicate favorable point estimates only. Neither figure implies state-of-the-art performance or causal component necessity.

## Actual PDF pixel review

Reviewed PDF-rendered 792px-wide proofs at a 5.5-inch physical slot, 240dpi enlarged proofs and grayscale versions under `proofs/`. All text is at least 8pt at 5.5in. No clipped labels, text collisions, missing markers or clipped error bars were found. Axis units, independent log axes, curve identities and shared labels are legible. Distinct markers/line styles retain method identity in grayscale. Geometry audit finds no text overlap or clipping.

The optional standalone spatial alternate is not intended to be added alongside the composite: use the composite in place of the prior main spatial figure and remove the old standalone appendix ablation. Use the historical forecast in place of the previous tall 2×2 forecast figure.

## Data-backed prompts applied

- Can every visible mark be mapped to one source result and the intended metric/horizon? Yes.
- Are the intervals correctly identified as paired CI versus between-seed SD? Yes; captions distinguish them.
- Are unfavorable results retained at the same visual scale? Yes.
- Are the fonts readable after real manuscript-width placement rather than enlarged screenshot inspection alone? Yes, with final integrated-page review left to root.

No models, training, evaluations, payloads, or frozen scientific source files were changed.
