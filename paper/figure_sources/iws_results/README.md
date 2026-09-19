# Replay the completed IWS figure from public inputs

`reproduce.py` reconstructs the IWS forecast plot from its finalized public JSON.
It does not load model checkpoints, feature caches, raw videos, or reserved data,
and it does not rerun or certify the scientific evaluation. No placeholder or
synthetic numerical plot is generated before the real complete-campaign record
exists.

Keep these public files, preserving the repository layout or providing the
explicit CLI paths:

- This `reproduce.py` and `brief.md`.
- `paper/scripts/render_iws_results.py`, pinned to SHA256
  `c762da602f874336891628bb13ed55815c3f01145f684806ef2e3609f4655671`.
- `paper/generated/experiment_alignment/forecast_transfer.json` and its four
  authoritative siblings: `.pdf`, `.svg`, `.png`, and `_figure.tex`.

Use Python with the exact NumPy/Matplotlib versions recorded in the JSON's
`bound_payload.runtime`, Pillow, the hash-matching Liberation Sans font, and
Poppler's `pdftoppm`. No package installation or network access occurs in the
replay. The original font path need not exist on another machine; its bytes
must match. Renderer imports use the supplied pinned public source, and only
its pure numerical-payload and plotting functions are called. Its private
scientific evidence loader is never called.

From a repository-shaped public snapshot:

```sh
python paper/figure_sources/iws_results/reproduce.py \
  --output /tmp/iws-public-plot-replay --if-ready
```

Or supply `--evidence`, `--renderer` and `--brief` paths explicitly. Output must
be a new, separate directory. `--if-ready` returns pending without creating
outputs if finalized plot evidence is absent. Source mismatch, incomplete or
changed public exports, invalid numerical payload, or pixel differences fail
closed. No existing figure is overwritten.

The replay retains all 708 means: three tasks, four methods and 59 future
offsets. It rechecks the signed H60 annotations against the public endpoints,
renders PDF/SVG/PNG and compares decoded PNG pixels plus actual PDF pixels
rasterized at 144 dpi with the authoritative files. Both comparisons must be
exact. PDF dates and SVG internal IDs can differ, so byte identity is not
claimed. A `replay_manifest.json` appears only after every check passes; it
records the source/input/export hashes, numerical payload hash, geometry
checks and exact pixel parity. No successful replay manifest exists at design
time; actual replay must wait for completed results. The first completed
public-only replay is now recorded in `replay_validation.json`: all 708 means
were retained, PNG and actual PDF pixels matched exactly, and the isolated
process was blocked from opening the original scientific workspace. This
receipt binds that particular source and evidence snapshot; it does not apply
automatically to later revisions.

Full scientific reproduction remains separate: it requires the registered
data, selected checkpoints, evaluator and completion gates. This public replay
preserves the validated evidence and its display; it cannot independently
establish the correctness of private evaluation inputs. Actual manuscript-page
and grayscale review also remain separate from automatic pixel equality.
