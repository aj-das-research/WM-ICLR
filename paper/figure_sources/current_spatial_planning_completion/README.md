# Current spatial planning report source

The pack is created only after all 24 full-30-epoch selected predictors and all
2,400 planning trials have passed the new study's finalizer. No partial or
synthetic accuracy values are stored here.

The eventual `data.json` retains all task/method/seed/case numerical diagnostics,
first-success native calls, physical errors, costs and complete registered
summaries/paired intervals. `manifest.json` binds it and the original finalized
source receipts. No raw RGB, action arrays, physical state arrays, feature
caches, model weights or private binary traces are redistributed in this pack.

A public clone can reproduce the tables and vector figure with:

```bash
python paper/scripts/render_current_spatial_planning_completion.py --from-pack \
  --output-dir /tmp/current-spatial-planning-replay
```

Before completion, `--if-ready` writes only an explicit pending paragraph to
`paper/generated/current_spatial_planning_completion/section.tex`. Real complete
figures require manuscript-width and independent numerical/pixel review before
being described as reviewed. This new current-spatial study does not rename or
reuse the historical context-model planning results.
