# Matched Reacher outcomes

Reproduce from repository root with `.venv/bin/python paper/scripts/render_reacher_outcomes_compact.py` (Python, NumPy, Pillow, Matplotlib). Runtime reads only this public source pack; no models, raw data, network or experiment execution. Deterministic outputs are written under `paper/generated/editorial/reacher_outcomes_compact.*`. PDF/SVG labels, rules and borders are vectors; full recorded RGB observations remain raster evidence. Native draw.io is not needed for this image comparison.

The figure merges the four cases formerly spread over two Reacher plates. Each is the original first lexicographic development episode in its outcome stratum against Framewise, at training seed 0. These are the historical context model, not the current spatial decoder. Outcome strata are not frequencies. The selected layout deduplicates only goals and support observations proven pixel-identical across all three methods. Method endpoints are their actual stopping times, not common-time snapshots.

`cases.json` retains all twelve records, including Shared context, exact action blocks/timelines, source JSON/NPZ identities, original selection and pixel hashes. The twenty PNGs are complete 224×224 source arrays, without crops, interpolation, recoloring or generated imagery. `manifest.json` binds every renderer input. The companion `caption.tex` is proposed manuscript prose; the root paper controls placement and cross-references.

## Original complete archive

- [All 64 tasks and Shared context with full recorded timelines](../../../artifacts/qualitative/index.html)
- [Original evidence inventory](../../generated/qualitative/evidence.json)
- [Original renderer](../../scripts/render_qualitative.py)
- [Original Reacher contrasts](../../generated/qualitative/reacher_contrasts.pdf)
- [Original Reacher controls](../../generated/qualitative/reacher_controls.pdf)

No old archive or renderer is replaced. Shared context fails on all four selected Reacher cases, including the case in which both displayed methods succeed. Joint-angle L2 is diagnostic; the simulator success event requires **each unwrapped joint error <0.05 radians**. Ten support calls count within the 50-native-call allowance; exact final calls are 23/50,50/41,50/50,33/36 for ShiftWM/Framewise.

## Image attribution

Images are recorded observations from the completed local Reacher simulator evaluations. Infrastructure: GalilAI-group/stable-worldmodel at revision `4821c8e6a3f0f83b7e6a80da3a757e026ea9026b` (MIT); dm-control/MuJoCo and their assets retain upstream notices. See [THIRD_PARTY_NOTICES.md](../../../THIRD_PARTY_NOTICES.md). Source evaluation and video SHA256 hashes are preserved in `cases.json`; this pack does not substitute a model-repository license for simulator assets.
