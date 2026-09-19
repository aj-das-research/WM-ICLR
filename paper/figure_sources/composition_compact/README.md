# Compact historical composition protocol

Run `python paper/scripts/render_composition_compact.py` from the repository root.
The standalone renderer reads only the hash-bound protocol and full PNG assets.
It needs Matplotlib, NumPy, Pillow and Liberation Sans; no checkpoints, simulator,
network or private files are required.

The design preserves the previously reviewed scientific content and exact image
pixels, tightening image sizes and vertical spacing without shrinking type.
The canvas is 5.5 × 1.57143 inches, exactly 21:6, with at least 8-point labels.
Seven training/ordinary-validation pairs, the development and held-out pair,
three extrapolation pairs and independent all-nine test coverage are unchanged.
The physical values and same-state RGB interventions are unchanged.

PDF/SVG have editable vector typography and split marks around full raster
frames; the native `.drawio` is generated from the same geometry. Native-app
review is separate from XML validity. The earlier19:6 source remains archived
in `composition_landscape/`; the old tall TikZ source is preserved as well.
