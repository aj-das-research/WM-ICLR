# Exact spatial figures: reviewed replay pack

This reviewed derived-example pack accompanies the paper. The publication scanner allows its single NPZ at this exact path and SHA256 only. It contains measured figure arrays, not model weights or a dataset mirror. The accompanying manifest records its earlier local preparation status; public inclusion is recorded separately.

From the repository root, with NumPy, PyTorch, Matplotlib, Pillow, pdflatex and pdftoppm available, redraw the actual measured gallery into a separate directory:

```bash
python scripts/real_video_spatial_qualitative/render.py --input paper/figure_sources/spatial_qualitative --output /tmp/spatial-gallery-redraw --portable-replay-sha256 12d805f0692f9cd0068f7eb392cf9be996dd778f1fff00f5c1ee4273cfffac20
python paper/scripts/render_spatial_task.py --input paper/figure_sources/spatial_qualitative --output /tmp/spatial-task-redraw
python paper/scripts/render_spatial_method.py --output /tmp/spatial-method-redraw
```

The first two commands consume this exact reviewed measured pack and do not retrain or reopen original model/source data. Their hashes are checked before plotting. The architecture schematic uses no measured outputs. Vector PDFs/SVGs and PNGs are generated; a reviewed manuscript copy is a separate publication step.

For full scientific replay with the original processed data and verified checkpoint packages available, use the registered scripts/real_video_spatial_qualitative/replay.py, followed by the strict default renderer and review_measurements.py. Preserve the frozen selection registration; do not reselect windows for visual quality. Default strict rendering verifies all original scientific source hashes.

The native 4x4 endpoint population gain is 5.2998%, which is distinct from the coarser original2x2 metric. The median episode gains 5.3582%, but its prespecified first window regresses by 0.3053%. All three selected best/median/worst cases and shared scales are retained.
