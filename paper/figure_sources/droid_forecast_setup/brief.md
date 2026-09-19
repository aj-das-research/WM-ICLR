# Figure 7: explain the recorded-video experiment

This is an existing-data, adapted evaluation setup, described in `paper/sections/real_video_study.tex` and frozen by `reports/real_droid_protocol.md`. It is the historical DROID context-model study, not the main spatial decoder architecture.

Reader question: given recorded camera observations and logged robot commands, what is the world model asked to predict, which information can it access, and how is it scored?

## Representation

One unchanged source-study episode supplies a genuine observed history at native frames 0/5/10 and withheld future at35. Blue marks observed inputs; amber marks recorded commands/evaluator targets with explicit region headings; teal marks model-produced features. The same frozen encoder appears twice to explain a single shared feature coordinate system. A schematic 2x2 tile denotes pooled features, not generated RGB or measured feature values. The displayed comparison is h5; all five future features are forecast recursively. Commands contain two past and five future blocks, five native seven-dimensional commands per block. The target photo has no route to the predictor. No arrow leads from the predictor to the physical robot.

Panel A depicts camera roles with an original vector robot/table/camera illustration. Positions are schematic, not measured calibration or camera extrinsics. Real camera1/2 thumbnail observations share the episode and frame10; camera1 is used for training, normalization and primary testing, camera2 is used only in a separate held-out transfer evaluation, and the wrist camera is retained for inspection. One camera is used per evaluation run; no multimodal camera merge exists.

## Alternatives actually rendered

- A: camera/robot acquisition vignette at left; model-visible upper lane; separate evaluator-only lower lane. Selected because all roles and information boundaries have an immediate concrete interpretation, and connectors have short clear corridors.
- B: camera-role rail above a full-width input/model/evaluator composition. Space-efficient but the prototype's output arrow crosses its feature label, and hardware roles lack a common physical scene.
- C: observed/withheld timeline across the top, with forecast and evaluator below. The timeline communicates time well but its reference arrow competes with the evaluation heading and longer routes make information access harder to inspect.

Private first drafts used repeated ingestion photos solely as layout placeholders; final candidates must replace these with verified actual study frames before source promotion. The old DROID-100 inspection plate remains archived and is not relabeled as study evidence.

## Publication style and outputs

5.5-inch column, 396x232pt landscape scene, 8pt minimum text. Actual observations provide sufficient visual context; original editable vector camera/robot/feature illustrations avoid fabricating a photorealistic experiment. One ordered scene controls PDF/SVG/PNG and native draw.io XML; PNG experimental imagery remains raster and editable vector objects remain separate. Validate native file in the official diagrams.net app and independently inspect final print-size pixels, enlarged arrows, grayscale, source identities and actual manuscript pages.

No training, inference, predictions or quantitative experimental outcomes are produced or changed by this figure redesign. Camera image source/license and source/model contracts must remain in the portable pack; numeric result tables remain unchanged.
