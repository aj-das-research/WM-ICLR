# Figure 2: proposed components and paired supervision

Date: 2026-09-18. **Current main-figure brief.** This A/B/C composition supersedes the earlier inference-only serpentine figure. The immediately preceding brief, review, source, and PNG are archived under `paper/build/method_pairing_redesign/`; older named review documents describe historical figures.

## Reader, slot, and visual argument

At 5.5 inches wide and 4.42 inches tall or less, make the proposed ShiftWM machinery visible before the conventional planner: **A** shares an observation calibration between history and goal; **B** infers dynamics after that calibration from corrected transitions and executed actions; **C** adds two deliberately different offline pairing penalties. This describes proposed components, not certified algorithmic novelty, identification, or an observed benefit. The Unpaired contexts control retains the two-context architecture while removing C.

Labels use 8.5 pt and headings 9.5 pt before near-unit scaling. Mathematical subscripts follow conventional smaller typesetting. A and B are blue/orange; C is a separate dashed training-only region. Conventional encoder/predictor/CEM machinery is smaller and gray. Gray does not mean frozen: only the encoder has a lock; reused action encoder/predictor are also trained offline.

Active skill references: installed `paper-figure-creation` art-direction, visual-elements, geometry-and-inspection, and asset-sourcing guidance. The previous image and every final proof were viewed as actual pixels. Existing observed assets make new generated art unnecessary.

## Three compositions considered before drawing

1. **Inference above worked paired-training examples — selected.**

   `observations → A calibration → B dynamics → quiet LeWM/CEM`
   `same trajectory / different appearances → dynamics penalty`
   `same appearance / batch i≠j             → observation penalty`

   Separates deployment access from training supervision, gives the two different relations substantial area, and exposes what Unpaired contexts omits.

2. **Two context-construction paths around quiet LeWM.**

   `raw-support pooling → observation context ┐`
   `corrected transitions/actions → dynamics ├→ LeWM`
   `paired relations as side annotations     ┘`

   Strong estimator comparison, but the side annotations undersell the extra supervision that the reader needs to locate.

3. **Paired-example training bridge above deployment.**

   `appearance-pair ← training bridge → batch-appearance pair`
   `                      ↓`
   `                  deployment loop`

   Strong learning narrative, but vertical arrows risk implying that paired IDs or privileged examples are deployment inputs. The chosen separated panel makes that boundary clearer.

## Scientific and representation contract

- **A:** raw support features only supply their temporal mean and population variance to the learned observation-context network. The same inferred context and learned residual-FiLM map correct both support and available goal features. Goal images/actions never enter observation-context inference. The illustrative four channels retain order and shade identity before/after the same affine operation. Raw row amplitudes are(.18,.44,.30,.53) and those values minus.07; both use scales(1.08,.94,1.06,.96) and shifts(.12,-.05,.13,-.08). These are explanatory values, not measured activations or evidence of recovery.
- **B:** chronological corrected support produces features `z_t` and differences `Δz_t`; normalized executed actions feed the GRU directly. Its inferred dynamics context conditions action embeddings through `A_d ∘ E_a`. Candidate future actions never enter either context estimator. Corrected history separately reaches the reused predictor.
- **Deployment:** corrected goal features reach only scoring. Contexts are inferred from real history at each replan, held fixed during the search, and all online weights are fixed. The first action block executes five native controls. The compact candidate/future glyphs abbreviate five-transition CEM rollout, not actual counts, predicted pixels, measured trajectories, or task success.
- **C, dynamics:** the two appearances show the same physical trajectory/timepoints with the same executed actions and physics. Each view independently executes A then B, including its own observation context and support correction. A dashed two-ended link denotes the MSE penalty on their dynamics contexts; neither branch is stop-gradient. It is not an exact-equality or disentanglement claim.
- **C, observation:** distinct minibatch indices with matching appearance IDs produce the observation-context penalty. The implementation uses all matching off-diagonal pairs. Trajectory IDs and dynamics IDs need not differ; there is no matched-state, cross-physics, independence, or negative-pair requirement. Only the contexts are matched, not states from different trajectories.
- Paired alternatives are restricted to allowed training combinations. Canonical query alignment and teacher-forced one-step prediction losses remain described in the method/caption; they are not shown as support-target arrows. With H=3,T=8, context sees 0–2, alignment targets 3–7, and prediction targets 4–7. No future query observation enters context inference.

## Observed assets and scope

The top deployment panel keeps four unchanged TEST demo images: `demo/samples/pusht_test-s3031000-d2.npz`, indices 0/1/2 and available goal 7. They illustrate model inputs only. Their original hash/selection is `assets/method_assets.json`.

The lower training panel uses real TRAIN representatives, not the demo images:

- `split_assets/pusht_v0.png` and `pusht_v1.png` are frame 0 of `data/world/pusht_relative/episodes/train-s31001-d0.npz` under canonical and warm pointwise RGB transforms. State, trajectory, timepoint, and action history are the same. Both appearances are allowed with dynamics 0. Source/transforms/selection/checksums: `split_assets/manifest.json`.
- The same canonical frame is paired illustratively with `assets/method_train_other.png`, frame 0 of `train-s31002-d0.npz`, canonical appearance. Root extracted this losslessly with `assets/build_method_training_asset.py`. Full provenance is `assets/method_training_assets.json`. Different trajectories are one permitted illustration, not a loss requirement or a recorded claim that this exact pair co-occurred in a sampled minibatch.
- Stacked blank card outlines denote support clips. A displayed frame is a representative observed image, not a claim that context is inferred from one frame or that the repeated backing sheets are separately observed timepoints.
- All observed images are 224×224 with no crop or generated alteration. The training assets' RGB transformation and display quantization are the established dataset visualization. The Google lock asset retains its original transparent source and attribution in `assets/material_lock.json`.

## Caption and integration contract

Name A/B/C and separate their information sources. State that dashed links are offline consistency penalties, explain both pairing rules, and mention that Unpaired disables them. Make the shared actions and per-view A→B computation explicit. Training cards represent support clips; all feature/code/future/control marks are schematic. Retain fixed contexts/weights within search and the first five-native-control block. Distinguish the frozen encoder from reused modules trained offline. Canonical query prediction/alignment remain in the text; avoid claiming they are drawn. No obsolete cost-bar description applies to this version.
