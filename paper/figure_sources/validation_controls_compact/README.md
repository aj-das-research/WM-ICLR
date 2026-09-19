# Historical DROID validation controls

Reproduce with `python paper/scripts/render_validation_controls_compact.py`.
Runtime inputs are this directory's data.json, caption.tex, and manifest.json only.
No raw data, checkpoints, cluster dependencies, or experiment execution are required.

The complete original six-row optimization ledger (all four means/seed SDs),
all twenty horizon contrasts, exact fifteen displayed effects, and 215 source
hash bindings are retained in data.json. Original source paths are provenance,
not runtime inputs. They refer to original validation of the historical context
model and do not describe the newer spatial method.

Signed MSE difference = first model minus second; display scale multiplies by
1000. Open marks denote paired 95% intervals crossing zero. No new uncertainty
estimates, experiment outcomes or pooled task means are introduced by rendering.
The extraction audit recomputed the 26 existing paired confidence intervals.

SVG/PDF contain editable vector quantitative marks and text. Original renderer
code and exact data, rather than manual editing, are authoritative.
