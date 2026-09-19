# Historical technical diagnostics

Run `.venv/bin/python paper/scripts/render_technical_story_compact.py` from repository root. The renderer reads only this public pack and emits deterministic PDF/SVG/PNG/evidence under `paper/generated/editorial/technical_story_compact.*`. No private raw data, checkpoint or model execution is required.

The compact plate replaces repeated visual explanations with a comparison table, one goal and two **same-time native-call20 observations** per task, two exact task-criterion traces, and all ten existing paired one-block prediction comparisons. These are historical context-based models, not the current spatial decoder. Original cases remain PushT2031024 and Reacher2031004, the first positive discordant case in each task. Their own actions and Framewise actions define different conditional diagnostic groups; both prediction models receive identical inputs within a group/transition.

`data.json` retains all39 exact-replay traces (13 discordant tasks ×3 methods), all70 paired prediction records with their saved vectors, all source hashes, and the four-positive-case summary. It preserves unfavorable outcomes: displayed PushT prediction MSE is higher on5/5 pairs; Reacher is lower on3/5. Across all4positive cases prediction improves on6/20 and support calibration on0/20. These descriptive, temporally dependent counts are not independent benchmark wins or causal evidence.

## Images and coordinates

All six original224×224 RGB arrays are preserved without pixel changes. PushT is displayed in full. Both Reacher call20 images use the **original** common square crop `[75,87,169,181]`; a rectangle marks that crop on the full goal overview. Dashed silhouettes are vector, display-only goal outlines using the old renderer's exact mask rule. Binary mask PNGs are annotation sources, not new observations. Actual RGB and annotation layers are distinct. PushT's green reset marker is unscored; its gray block and blue pusher are scored. PushT position errors remain in native512 coordinates, not display pixels.

Curves use every saved native call from10 to each actual stopping point. They never extend a stopped execution. Green bands are separate strict criterion regions; success requires both together. Lower paired points retain canonical fixed-coordinate MSE (log axes), with the same full five-action block for both predictors. No terminal partial block is padded or assigned zero error. Plotting changes do not alter the observations or scientific metrics.

## Archived detail and provenance

- [Detailed historical inference graph](../../generated/qualitative/technical_contract.pdf)
- [Original PushT diagnostic](../../generated/qualitative/technical_pusht.pdf)
- [Original Reacher diagnostic](../../generated/qualitative/technical_reacher.pdf)
- [Complete qualitative gallery](../../../artifacts/qualitative/index.html)
- [Original exact-replay diagnostic inventory](../../../results/qualitative_diagnostics/replay_diagnostics.json)
- [Original prediction inventory](../../../results/qualitative_diagnostics/prediction_diagnostics.json)

The runtime needs none of these archives; identities are retained for scientific audit. `historical_contract.json` maps the compact operation table to the earlier graph and explicitly preserves goal/context restrictions, action FiLM and temporal/search semantics. The visual encoder and its projection are frozen during training; both methods' predictors are trained separately. At deployment all model weights stay fixed, while contexts are re-inferred after observation blocks and held fixed inside each CEM search.

Images are recorded simulator observations from the completed local evaluations. Infrastructure: GalilAI-group/stable-worldmodel pinned4821c8e6a3f0f83b7e6a80da3a757e026ea9026b (MIT); PushT, dm-control/MuJoCo and their assets retain upstream notices in [THIRD_PARTY_NOTICES.md](../../../THIRD_PARTY_NOTICES.md). No generated experiment imagery was used. Precise labels, curves and contours are vector/editable; raster source observations remain raster. Native draw.io is not used for this quantitative plate.
