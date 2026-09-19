# Current real-data scorecards

These presentation tables consolidate completed evidence without changing any
model, evaluation, checkpoint selection, registration, or original renderer.

- `droid_spatial.tex`: all eight current DROID predictors, native and pooled
  standardized MSE at h5/h10. Intended to replace the two appendix absolute-score
  tables, preserving `tab:spatial-native` and `tab:spatial-pooled`. The main native
  table and every paired-inference table remain separate.
- `iws_development.tex`: all four completed predictors per task and all four
  measured feature metrics at H60. Intended to replace the secondary appendix
  table, preserving `tab:iws-secondary`. The main comparison and paired intervals
  remain separate. An additional no-tanh row appears only after the complete
  nine-new-plus-27-original finalizer validates every task/seed.

Black bold identifies the lowest unrounded point mean in each comparable column;
it does not indicate significance. No gain is colored or inferred by these
scorecards. Metrics not measured by these studies are stated in the captions.
Spatial DROID and single-observation IWS use distinct model interfaces, data
populations, and normalizations. They are not one cross-dataset ranking.

`data.json` retains all per-seed, per-offset means for every measured metric and
the original source/checkpoint receipt hashes. `manifest.json` binds its exact
bytes. Private backing paths are provenance records, not public runtime inputs.
Only these two JSON files, the renderer, Python and NumPy are needed publicly:

```bash
python paper/scripts/render_current_real_scorecards.py --from-pack
```

In the experiment workspace, refresh from finalized JSON reports and bound
per-episode metric receipts with:

```bash
python paper/scripts/render_current_real_scorecards.py --if-ready
```

The live refresh never opens model weights, feature caches, raw videos, reserved
payloads, or runs inference. Missing required study completion writes nothing
with `--if-ready`; a present invalid or partial completion fails. Portable mode
validates pack hashes, complete method/seed grids, finite metric arrays, and
recomputed means. Outputs and packs are deterministic and unchanged on retries.
Optional `--output-dir` redirects the generated LaTeX and render receipt.

Outputs are under `paper/generated/benchmark_scorecards/`. Standalone official
template proofs and author checks are under `paper/design/current_real_scorecards/`.
