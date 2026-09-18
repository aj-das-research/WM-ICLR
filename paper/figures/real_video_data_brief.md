# Real DROID data illustration: figure brief

- **Slot:** ICLR appendix, 5.5 inches wide; standalone subsection `paper/sections/real_video_study.tex`.
- **Mode:** existing-data/ingestion illustration supporting an adapted offline forecasting setup, not a new benchmark and not a qualitative algorithm result.
- **Thesis:** one physical robot recording provides three distinct camera views at matched native frame indices.
- **Evidence:** first schema-audit record of the author's DROID-100 debugging release, `droid-ea00b8c4fd59911984e43058`; source TFRecord shard 00000, source episode index 0. The image example is separate from the selected 24 full-release shards / 1,126 audited study episodes. Physical episode overlap across releases is not asserted either way.
- **Selection:** choose stored positions0,16,33 by `round(linspace(0,33,3))`, giving native indices0,80,165. Selection uses only position, never a model result, reward or quality ranking.
- **Representation:** three camera rows, three native-index columns; same crop policy and aspect ratio everywhere. Preserve original180×320 pixels, full frame, publisher blur and colors. Short vector labels outside images. No action overlays because calibrated camera projection is not established.
- **Claim boundary:** displays real acquired footage and data layout only. No predictions, method comparison, action effects, success indicator or precision time annotation. No person is identified.
- **Novelty:** none attributed to the existing dataset or to this image panel. The registered ShiftWM real-video comparison remains pending.
- **Label budget:** fewer than40 words on canvas. Details, study status and attribution in caption.
- **Source contract:** original camera NPZ hashes checked against schema-audit manifest; selected pixel-array and exportedPNG hashes enter ledger. Fig source/export hashes included. CC-BY-4.0 DROID source attributed.

## Composition alternatives

1. **Camera rows × time columns** (selected): aligns different viewpoints vertically and chronological change horizontally; all nine real frames equally readable. Lowest reading burden for comparing camera views.
2. **Time rows × camera columns:** emphasizes each synchronous observation but changes the conventional left-to-right temporal reading path. Rejected for this appendix's temporal forecasting context.
3. **One exterior hero frame with two inset camera strips:** improves one scene's size but makes the other camera views much smaller and obscures exact temporal correspondence. Rejected because camera transfer is part of the registered study.

The current five-frame source contact sheet was inspected directly. The figure adapts that truthful filmstrip grammar into three time samples rather than inventing a conceptual architecture around the data.

## Production contract

Canonical layout is deterministic Matplotlib code. PDF and SVG retain editable vector labels/borders with embedded observed raster frames; this is a hybrid figure, not an all-vector depiction. PNG is preview only. Minimum font8pt at exact5.5inch width. Record label/image intersection checks; inspect paper-width proof and enlarged crop, then isolated compiled appendix. Main paper integration is deferred to root.
