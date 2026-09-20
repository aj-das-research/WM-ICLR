# Temporal qualitative presentation: layout advice

Use the existing reviewed replay, not a new inference run. The scientific question is: **as the forecast horizon grows, how do matched prediction errors and the decoder's use of fixed observed evidence change?** A measured error advantage and a measured internal change are separate observations; their co-occurrence does not identify a cause.

## What the current evidence supports

I read the abstract, main method, `sections/closest_qualitative.tex`, current renderer, and verified derived pack; the renderer's 126 mean/map checks pass. I inspected the existing DROID, middle-ranked IWS and decoder figures at paper width. They explain endpoints clearly, but repeat grids and observed/target pairs while leaving the available temporal information unused.

- DROID has saved error and mechanism curves at query steps 1–10; spatial maps and target RGB exist at query steps 5 and 10. The three support observations are native frames 0, 5, 10 for the selected windows, and target frames are 35 and 60. Label **query step** and **recorded frame** separately.
- IWS has saved error and mechanism curves at stored offsets 1–59, with maps and target RGB at offsets 14, 29, 44, 59. These are outputs of the full 60-command call, not separately re-executed H15/H30/H45 prefix endpoints. Label **stored offset**; do not imply seconds or sixty physical transitions.
- The pack contains actual `T16x16`, effective `M16x16` (= paper's W), gate and correction maps at these saved offsets for both bounded and no-tanh variants. It contains AR error maps, but not an equivalent AR source-mixing operator.
- Existing `mechanism_curves.effective_self_weight`, `raw_self_weight`, `mean_gate` and `correction_rms` summarize patches and seeds. They are not the fixed-query q measurements used by the endpoint decoder figure. Do not label an all-patch curve as q's weight. If fixed-query time points are wanted, derive them from the saved matrices at the available map offsets only.

## Three different compositions

### 1. Synchronized filmstrip and diagnostic rails — recommended

One wide worked example carries a single reading direction:

```text
Observed source(s) | Recorded future targets: offset a    b    c    d
                  | Matched error traces: AR, bounded, no-tanh
                  | Source-use rail                  Correction rail
```

Use a fixed observed thumbnail on the left and a filmstrip of unchanged recorded targets across the top. A thin vertical divider labeled “evaluation only” separates future observations from model input. Below, plot full saved per-offset MSE curves for all three methods with sparse markers exactly aligned to the filmstrip offsets. At the bottom, two small synchronized plots show mean effective own-location weight and mean correction RMS for bounded/no-tanh. These show what changed over time without repeating twelve error maps. One compact map inset at a predetermined offset can illustrate the spatial meaning of a curve; do not add an inset for every sample.

**Why it works:** it connects recognizable scene evolution, the evaluator's criterion and a measured internal computation while preserving their different meanings. A reader sees crossings or delayed regressions that an endpoint gallery conceals. Keep a three-line method legend with redundant dash/marker styles. AR stays visible in the error plot; mark the two internal plots “mixing variants only.”

### 2. Time-by-patch matrix with a filmstrip header

Put recorded targets at the available offsets above three aligned 16-by-time heatmaps, one per method. Each heatmap column is a saved forecast offset and each row is a fixed patch index; show a tiny 4×4 key once. Add two narrow matrix bands for effective own-location weights and correction magnitude below. This reveals whether differences are localized or spread across patches and how they evolve.

**Tradeoff:** quantitative structure is compact, but patch-index rows are less intuitive than real spatial maps. The current pack only supports spatial matrices at two DROID/four IWS times. Do not fill missing patchwise times by interpolation or relabel scalar curves as dense patchwise evidence. This approach is better for a technical appendix than the main explanation.

### 3. One expanded operation beside a temporal outcome plot

Left third: observed grid → measured W at one fixed saved offset → correction, with the exact reconstruction `W Z0 + Δ`. Show the same query patch throughout. Right two-thirds: a recorded filmstrip above the three error curves; a small pair of bounded/no-tanh point sequences marks the saved q weights. This separates “how the decoder is executed” from “how this case evolves.”

**Tradeoff:** it explains the algebra best, but duplicates the architecture and current decoder figure unless it replaces those endpoint details. It needs clear labels to avoid making a one-offset operator look valid at every horizon.

## Recommended integration

Use composition 1 as the new temporal explanation. Start with an **already selected middle-ranked case**, not a newly chosen visually favorable sequence. Rope's middle-ranked case is a useful positive example: no-tanh is lower than AR at each of the four saved map offsets (MSE 0.1974/0.2179/0.1831/0.2103 versus 0.2026/0.2238/0.1983/0.2278). Show the full 59-point curves before describing their entire evolution; four favorable samples alone do not prove a pointwise advantage at every offset.

Use the same composition for the retained Box regression or DROID median reversal as a compact companion, so the mechanism story includes a boundary case. DROID's median window has bounded MSE 0.0849 versus AR 0.0941 at query step 5, but 0.1727 versus 0.1722 at step 10: a real sign reversal the temporal layout can explain directly. These are selected-window observations, not population frequency estimates.

Aim for each band at 5.5 inches wide and roughly 1.8–2.1 inches high, with 8-point or larger labels. A strict 19:6 ratio may require omitting an optional map inset; do not shrink all labels to preserve the ratio. Put no more than two bands in one float. Keep DROID query steps and IWS stored offsets on separate axes, and keep task-specific normalized error scales separate. A shared grammar is useful; sharing incompatible task scales is not.

**Replace redundancy:** retain the complete largest/middle/smallest case inventory and numerical source pack, but replace the current four-row endpoint-only decoder figure with temporal diagnostic rails rather than appending another full endpoint gallery. The existing gain table remains the population-level comparison. No temporal drawing should imply that no-tanh has replaced the bounded primary method.

## Semantic and rendering checks

- Observed inputs and withheld future RGB must have distinct labels and containers. Future photos are recorded targets, never decoded model output or feedback into the predictor.
- Preserve the fixed case, frame indices, method identities, three seeds and original image field of view. Mark the selected-trajectory endpoint gain separately from the illustrated-window outcome.
- Plot per-seed error then average; do not square an averaged forecast. Use the already saved mean curves and mean maps. Any seed spread is variation across three seeds, not a confidence interval.
- Compute each seed's W before averaging; `mean(g T)` generally differs from `mean(g) mean(T)`. Keep raw T, effective W, gates and corrections named separately.
- Weights are learned feature-source mixtures, not object attention, saliency, optical flow or physical correspondence. Correction magnitude is not correction accuracy. A larger correction or lower self-weight alone is neither proof of better forecasting nor causal attribution.
- Use one absolute scale for matched error maps within the same task/case comparison, and [0,1] for source weights. A signed AR-minus-method map may use a symmetric zero-centered scale, with both signs retained. Do not use a different per-offset auto-scale that visually suppresses increasing error.
- Draw at saved coordinates only. Do not invent intermediate RGB or patch maps; full scalar curves are available and can be plotted without new inference. Inspect the final manuscript page and adjacent pages after integration.

## Bound sources

- `paper/figure_sources/qualitative_closest_v1/derived.json`: `da07f473966c180061d1ee6c67e27c94ce6ad4aef33b21af39492889ea45c7df`.
- `reports/qualitative_closest_v1/result_review.json`: `8ef21f746952755965265d31c7ff5f237c39829b41238fbdfaf618ed72ef64fd`.
- Relevant skills: personal `paper-visual-design`, `paper-figure-creation` and its qualitative-comparisons, visual-story and evidence references. No generated imagery is needed for these evidence panels.

This is read-only scientific/layout advice. No manuscript, renderer, model, or experiment was changed. The ongoing raw-feature DINO-WM campaign was excluded; only its queue/completion metadata was checked.

## Tighter implementation of the parent's two-panorama plan

The proposed three-task temporal outcome panorama plus three-task decoder panorama is a good application of composition 1 when cross-task coverage is essential. Use the same already selected middle-ranked IWS case in each row of both figures. Each panorama can fit approximately **5.5 × 3.4–3.7 inches** without reducing the 8-point label floor:

- Reserve about 0.55 inch for observed input/case identity, 2.35 inches for four aligned temporal columns, and 1.45 inches for curves, with the remaining width for labels and gutters. Three roughly 0.95–1.05-inch task bands leave space for shared legends. These are starting dimensions to verify against actual rendered labels, not a reason to compress illegibly.
- **Outcome panorama:** each temporal column pairs one real recorded future target with the signed `AR − no-tanh` patch-error map beneath. The right plot shows both bounded and no-tanh relative MSE reductions against a horizontal AR zero line over all 59 saved offsets. The bounded curve is essential: on PushT it is the actual table runner-up. Keep signed-map MSE units distinct from relative-percent curve units; never share a misleading legend. A task's map scale stays constant across all four times.
- **Decoder panorama:** show the fixed query q once on the observed source, then four `W[q,:]` maps over that same observed 4×4 source grid. Do not repeat the future photographs. The right column contains separate B/U rails for all-patch mean gate and correction RMS, with named units and direct line labels; do not use a dual y-axis. This means raw T is explained by the method equation/caption and can be omitted from the panorama if the measured W story is clearer.
- Label the temporal columns **stored offsets 14, 29, 44, 59**, with “full 60-command forecast” once in the caption. Calling them H15/H30/H45/H60 would risk conflating these stored full-horizon outputs with the separately executed prefix experiments. Add exact native frame indices under photographs where they fit.
- Do not compress the DROID reversal into a corner at the cost of legibility. The existing DROID figure plus a concise h5→h10 contrast in the text or caption supplies the boundary case. If a dedicated inset has at least ~1.3 × 0.65 inch at 8 points, the two observed error comparisons can be shown; otherwise omit it from these IWS panoramas.

Retaining the complete nine-case endpoint gallery is defensible as selection/boundary context, but it should remain in the appendix while the temporal panoramas answer the new question. The decoder panorama replaces the endpoint-only decoder figure. This avoids adding another redundant set of endpoint matrices while preserving unfavorable examples and source provenance.
