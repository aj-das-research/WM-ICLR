# Figure 6 (fig:ablations): compact candidates

Current figure: `figures/ablations.pdf`, 5.5 x 1.76 in, a full-width float plus a 3-line caption (about 2.2 in of page height across the full width).
Everything here comes from `make_fig6_candidates.py` (repo root: `PYTHONPATH=src python reviews/fig6_candidates/make_fig6_candidates.py [A B C]`).
Nothing in `paper/` or git was changed.

Data sources (no hand-typed numbers):
- Ablations: `scripts/v2/make_tables.py::ablation_data()`, DROID validation, seed 0. This is the same source as the current figure and the `\abl*` macros.
- (b)/(c) of B: `make_tables.load()` on `eval_test.npz`, averaged over seeds, using the same loader and root rule as Table 1. The gain is `100(1 - err_ShiftWM/err_Direct)`, the definition behind the text macros.
- CIs: 95% paired bootstrap with 4000 resamples. DROID resamples its recording sessions and the other datasets resample episodes. IWS is the equal-weight mean of its 3 tasks, resampled within each task, as in `iws_ci_pct`.
- Seed-noise band: ±0.3% (`NOISE`, the only hand-set constant). The full model's seed range is 0.2% (std 0.1%), and the text puts the threshold at 0.3%.
- Spot checks against earlier output: the DROID gain over Direct is 6.8% at k=1 and 7.3% at k=10, and the Δ values reproduce the current figure exactly.

Style: `make_figures.py` rcParams and palette (STIX serif). Text is 6.2 pt at print size (headings 7.5 pt) and figures are authored at their final width, so source pt = print pt. Direct keeps its blue diamond. Red means clearly worse (>1%, the current figure's rule), grey means within about 1%, and dark means clearly better. The QA audit (`mf.qa`, the paper-figures layout checker) reports 0 issues for A and B. For C it reports 1 flag, '+contrastive' vs 'no corr.'. That flag comes from the leader line's bounding box; the pixels at 450 dpi show no overlap. Print-size previews (100 ppi) and 450-dpi crops are in `qa/`.

---

## A: half-width ablation list (`fig6A_ablations_half.{pdf,png}`, 2.70 x 1.62 in)

A drop-in replacement meant for `\begin{wrapfigure}{r}{0.49\linewidth}`. It uses half the area of the current figure and needs no text changes.
- Rows are grouped by component (actions, transport, memory, correction), with each group sorted by Δ. The component column of the current figure becomes an italic group tag with a separator, so the four "what we removed" questions read as blocks.
- The Δ lollipops have a grey ±0.3% seed-noise band, labelled in the full-model row. The band is visibly tiny next to the +7% effects, and that is the honest point.
- A small rank-accuracy dot column keeps the "global window ranks actions worse" evidence. Direct is the blue diamond and ShiftWM is the green dashed reference line.
- **Dropped: the absolute MSE / k=K column.** Right now those numbers appear only in Fig. 6: `tables/ablations.tex` exists but is not `\input` anywhere. If A or B is adopted, add `\input{submission_folder/tables/ablations.tex}` to App. C (next to the seed-noise sentence at appendix.tex l.259) so the absolute values stay in the paper.

Suggested caption (3 lines at 0.49\linewidth):
> **Ablations** (DROID validation, seed 0): change of validation error against the full model (positive = worse) and action-ranking accuracy. Grey band: seed noise (±0.3%); absolute errors in Table X.

Text supported: all of §5.4 unchanged (the \abl* macros, "The gain comes from moving content", global 92.7 vs 93.4 ranking). Move the 768-vs-147 candidates clause wherever convenient; A labels the row "global (768 cand.)".

## B: full-width composite (`fig6B_composite.{pdf,png}`, 5.50 x 1.80 in)

Same slot and a similar height (1.80 vs 1.76 in), but it does three jobs. It brings the currently figure-less claim *"The gains hold across horizons and surgical tasks"* (§5.1) into the main text, next to the controlled ablation that explains it.
- **(a)** = A (ablations, DROID validation, seed 0).
- **(b) Across horizons.** ShiftWM's error reduction against Direct (the no-transport ablation) at every forecast step k on DROID (3 seeds), Language-Table, Open-H Hamlyn and IWS (3-task macro, k up to 12), each with a 95% CI band. Every curve stays above 0 at every k. DROID stays at about 7-8% (the test-set counterpart of the +7.3% in (a)). DROID's second camera tracks DROID within 1 point, so it is left out to avoid a duplicate line. The lines are green shades with distinct markers and dashes and direct labels, because each one is a ShiftWM gain; no method colour is reused for a dataset.
- **(c) Across tasks.** The same gain (mean over k) for each of the 7 Open-H tasks and 3 IWS tasks, with 95% CIs. Every CI excludes 0. Colours and markers match (b).
- Design choice: Direct is used as the only comparator in (b)/(c) because it is exactly the "no transport head" row of (a). The whole figure then answers one question: what does transport add, and does that hold across datasets, horizons and tasks? Gains over AR are left in the appendix (AR's long-horizon IWS exceptions would need a second series).

Suggested caption:
> **What transport adds.** (a) Ablations on DROID validation (seed 0): change of error against the full model (positive = worse; grey band: seed noise) and action-ranking accuracy. (b, c) Test error of ShiftWM below Direct, the same model without transport, per forecast step and per surgical/IWS task (95% paired bootstrap CIs).

Text supported or replaced:
- §5.4: unchanged, as for A.
- §5.1 "The gains hold across horizons and surgical tasks ... the gain over Direct is positive at every horizon on every dataset": now shown in the main text by (b)+(c), which can replace the pointer to App. B. The AR/AR-TF half of that paragraph stays with its appendix pointer.
- §5.1 IWS: "lowest average error on each task" vs Direct is visible in (c). Significance vs AR stays in `tab:iws`.
- App. B `fig:horizon`: its bottom panel (`hamlyn_tasks.pdf`, absolute MSE per task) could be dropped or kept. (c) shows relative gains, not absolute MSE.

## C: ablation map (`fig6C_ablation_map.{pdf,png}`, 2.70 x 1.75 in)

The most compact way to show both ablation metrics at once. Each variant is a point at (Δ validation error, action-ranking accuracy), ShiftWM sits at the origin, and the "good" region is top-left.
- The key messages become positions. Removing transport (Direct), w=1 and S=1 fall together in the bottom-right corner (worse on both metrics). Global window sits left (lower error) but below ShiftWM (ranks actions worse), so the trade-off is visible without reading numbers. tanh, contrastive and w=5 cluster at the origin.
- Action-free has no ranking accuracy (there are no actions to swap), so it is stated as text in the corner rather than drawn at a false position. tanh lies under the ShiftWM marker (Δ = −0.06%, rank 93.40 vs 93.43) and is labelled as such.
- Weaknesses: component grouping is implicit (by name only), exact Δ values must be read off the axis, and scatter-plot ablations are less conventional. The §5.4 prose already states every number, so this is acceptable.

Suggested caption:
> **Ablations as a trade-off map** (DROID validation, seed 0). Each variant's change of validation error against ShiftWM (right = worse; grey band: seed noise) and its action-ranking accuracy (up = better). Removing or immobilising transport is worse on both.

Text supported: §5.4 as written. The "global window lowers error slightly but ranks actions less well" sentence becomes a visible trade-off.

---

## Recommendation

**B, if the page budget allows a full-width float at about 1.8 in.** It costs no more space than the current Fig. 6 and removes no evidence except the MSE column (move that to App. C via `tables/ablations.tex`). It turns a pure ablation readout into the paper's main "what transport adds" figure: the controlled ablation plus its generalisation across 4 datasets, 10-12 horizons and 10 tasks. That claim currently has no main-text visual. **If space must be freed, use A** as a 0.49\linewidth wrapfigure: it is the most faithful and readable half-width version (grouped, noise band, rank column). C is the most elegant half-width option but less conventional; consider it if reviewers find A dense.

Open items: C has the one false-positive QA flag described above (verified clean at 450 dpi). None of these has been checked in the compiled paper page; place the chosen one and check the wrap and caption on the real page. The seed-noise band is ±0.3% (the text's threshold), not a per-variant seed interval: the ablations have one seed each.
