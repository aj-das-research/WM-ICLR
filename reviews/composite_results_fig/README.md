# Composite results figure: "Where and why ShiftWM wins"

This composite replaces **Fig. 5** (`fig:winmap-main`), **Fig. 6** (`fig:ablations`) and **panels (b)–(d) of Fig. 4** (`fig:qualitative`) with a single full-width float. Nothing in `paper/` or git was changed.

**Files**
- `composite_main.{pdf,png}`: 5.5 × 3.0 in, 2 rows × 3 panels. This is the recommended version.
- `composite_alt.{pdf,png}`: 5.5 × 1.9 in, 1 row × 4 panels. This is the budget option.
- `make_composite.py`: the generator. From the repo root, run `PYTHONPATH=src .venv/bin/python reviews/composite_results_fig/make_composite.py`.
- `ledger.json`: every plotted number, as the generator computes it.
- `inspect_pdf.py` and `qa/`: PyMuPDF proofs (100 ppi at print size, 300-ppi crops), the smallest font size and a check for text off the page.

**Data.** No numbers are hand-typed; the script reuses the existing loaders:
- (a), (b): `make_fig5_candidates.py`, which reads `results/v2/analysis/winmap` and the `eval_test.npz` files.
- (c): `results/v2/analysis/geometry/summary.json` and `oracle_null/summary_any_seed0.json`.
- (d): `results/v2/analysis/interpret/summary.json`.
- (e), (f): `make_fig6_candidates.py`, which calls `make_tables.ablation_data` and `make_tables.load` and runs the paired bootstrap (4000 resamples).

The only hand-set constant is the ±0.3 % seed-noise band.

## The story, one row each

- **Row 1, "where":** ShiftWM wins almost everywhere (a). The gain is largest where things move (b). That is expected, because most of the future is already visible, just displaced (c).
- **Row 2, "why":** the moving content really does go through the transport (d). Every core component is needed (e). The effect is not a DROID artefact: it holds at every horizon and on every task (f).

| Panel | Title (states the finding) | What it shows | Source / scope |
|---|---|---|---|
| (a) | Lower error on 127 of 130 episodes | Per-episode error reduction against Direct (blue) and AR (orange), ranked. The winning region is shaded green; the 3 losses stay visible ("3 not lower"). Medians are 7 % and 12 %. | DROID test, seed 0, mean over k. Uses the paper's relative metric, so it matches `\wmWinDirect` and `\wmWinAR` exactly. |
| (b) | The gain grows with motion | Absolute error reduction against episode motion. Faint points are episodes; lines are quintile medians with IQR bands. Spearman ρ is 0.37 against Direct and 0.49 against AR. | Same episodes. This panel shows no count, because the absolute aggregation gives 125/130 against Direct (see `fig5_candidates/README`). |
| (c) | The future is the present, moved | Moving-patch error at k=10 as bars: move from another episode 1.31 (+40 %), persistence 0.93, AR 0.62, Direct 0.60, **ShiftWM 0.56**, oracle move 0.46 (−51 %). Method colours match the rest of the paper. | geometry / oracle_null, seed 0, 3,923 windows. |
| (d) | Moving content goes via transport | Two moving-vs-static pairs. Setting the gate to 0 raises error by +57 % on moving patches and +2 % on static ones. Swapping in another episode's actions shifts the transport by 0.54 vs 0.19 patches (2.8×). | interpret, DROID test, k=10, seed 0. |
| (e) | Each core component matters | Increase in validation error for the 5 removals (actions +10.5, move +7.4, transport head / Direct +7.3, memory +4.8, correction +1.7). Below a divider, in grey, are the 4 design alternatives, all within 1.2 %. The grey band is seed noise. | DROID val., seed 0. |
| (f) | Holds at every horizon and task | Error reduction against Direct per forecast step on DROID (3 seeds), Language-Table, Open-H and IWS, with 95 % CI bands. The "tasks" strip on the right (same y-axis) shows 7 Open-H tasks (squares) and 3 IWS tasks (diamonds) with 95 % CIs. Every CI lies above 0. | test, paired bootstrap (DROID resamples sessions). |

**Consistent encodings**
- Blue always means "relative to Direct" and orange "relative to AR" (a, b, e, f). Green means ShiftWM.
- The dark ink/grey pair means moving/static in (d).
- All axes are framed so that higher is better (reduction or rise when removed). No panel says "ShiftWM worse" and no axis reads "below 0 is better".
- The losses in (a) and (b) and the neutral or negative ablations in (e) stay visible, but they are not the headline.

## What was dropped, and why (no redundancy)

- **Fig. 4c (gain per motion decile):** same message as (b). Keep the text sentence ("+5 to +13 % in all ten deciles") and the pointer to App. `app:geometry`.
- **Fig. 4b "high-gate +36 %":** a nuance. It stays in the text; (d) shows only moving/static.
- **Ablation rank-accuracy column and absolute MSE column:** move these to App. C via `\input{tables/ablations.tex}`. That table exists but is not currently input anywhere. The text keeps "92.7 vs. 93.4 %".
- **Gains over AR per horizon and task:** these stay in App. B, as they do now.
- **Knock-out (d) vs. the w=1 ablation (e):** both are kept because they answer different questions. (d) is a test-time intervention on the trained model and shows *where* transport acts (moving, not static). (e) is retraining and shows transport is *necessary*.
- **Fig. 4a (qualitative anatomy):** stays as its own figure. It can shrink to a single-row strip (two windows × 4 tiles, about 5.5 × 1.2 in), since (b)–(d) no longer share its float.

## Suggested caption (main, 4 lines)

> **Where and why \ours{} wins** (held-out DROID, seed 0, unless noted). (a) Per-episode error reduction vs.\ Direct and AR, ranked (mean over $k$). (b) Absolute reduction vs.\ episode motion (lines: quintile medians, bands: IQR). (c) Moving-patch error at $k{=}10$; oracle move: best observed feature in the transport window; other ep.: same oracle with candidates from an unrelated episode. (d) Gate knock-out and action swap, moving vs.\ static patches. (e) Ablations on validation (grey: seed noise $\pm$0.3\%). (f) Reduction vs.\ Direct per step (DROID 3 seeds) and per task (7 Open-H, 3 IWS), 95\% paired-bootstrap CIs.

For the alternative, drop (b) and (d) and the per-task clause.

## Changes to the main text this enables

- **Remove the floats:**
  - The Fig. 5 wrapfigure: `winmap_compact.pdf` plus its caption, about 16 wrapped lines.
  - The Fig. 6 float: `ablations.pdf` at 1.76 in plus a 3-line caption.
  - Fig. 4 panels (b)–(d): Fig. 4 becomes the anatomy strip only.
- **§5.1, "The gains hold across horizons and surgical tasks":** "positive at every horizon on every dataset" now points to (f), which gives the main text a visual for the first time. Keep the AR/AR-TF clause and its pointer to the appendix.
- **§5.3, "Where do the gains come from?":** the per-episode sentence can be cut to "…on \wmWinDirect\% of episodes against both, and the absolute gain grows with motion (Fig. Xa,b)". The decile sentence can drop "(Fig. 4c)".
- **§5.3, "Reliance on transport":** the pointers become (Fig. Xd). The sentence can shorten, because the moving/static contrast is now visible.
- **§5.3, "Is the future a transport?":** the oracle, null and share numbers get a visual in (c). The 51 % / 40 % sentence can be shortened, and the 79/71/65 % shares can be read off (c) as (0.93−x)/(0.93−0.46).
- **§5.4:** keep the \abl* sentences but drop the "\Cref{fig:ablations} changes one component…" framing sentence. Point to (e).

## Approximate main-text space

| | Current | Composite (main) | Composite (alt) |
|---|---|---|---|
| Fig. 4 | 2.45 in + 2-line caption ≈ 2.75 in | anatomy strip ≈ 1.2 in + caption ≈ 1.45 in | ≈ 1.45 in |
| Fig. 5 | 0.42-width wrap, 16 lines ≈ 0.9 in full-width equivalent | — | — |
| Fig. 6 | 1.76 in + 3-line caption ≈ 2.2 in | — | — |
| New composite | — | 3.0 in + 4-line caption ≈ 3.6 in | 1.9 in + 3-line caption ≈ 2.4 in |
| **Total** | **≈ 5.9 in** | **≈ 5.05 in (saves ≈ 0.85 in)** | **≈ 3.85 in (saves ≈ 2 in)** |

The main version adds four visuals that are not in the main text today: the oracle ladder (c), the horizon and task generalisation (f), the ranked episodes on the paper's metric (a), and the gain-vs-motion trend (b). It still frees about 0.85 in, before counting the text shortening listed above (about 4–6 more lines).

## QA

- **Layout audit:** the skill's `layout_quality` audit (`mf.qa`, at 5.5 in) reports 0 issues for both figures.
- **PyMuPDF inspection** (`qa/inspection.json`): the smallest font is 6.2 pt at print size and no words fall off the page. Panel titles are 7.2 pt bold.
- **Visual check:** I viewed the 100-ppi print proofs and the 300-ppi crops myself.
- **Fixes made along the way:**
  - The legend in (a) overlapped the curves.
  - The labels in (c) and (d) were clipped or colliding.
  - The ×10⁻² superscript in (b) printed at 4.3 pt; the label now reads ×0.01.
  - The Open-H/IWS tick labels in (f) overlapped.
  - The IWS line in (f) had too little contrast.
- **Colour:** Okabe-Ito method colours throughout. The dataset lines in (f) differ by lightness, marker and dash as well as having direct labels. In grayscale, (a)'s two curves are separated by the legend order and their fixed vertical order (AR above Direct).

**Open items**
- The figure has not been placed in the compiled paper. Check the float position and the caption length on the real page.
- The task strip in (f) identifies tasks only by marker (square = Open-H, diamond = IWS). Per-task names stay in App. B.
- (f) mixes 3 seeds (DROID) with 1 seed (the others). The caption says so.

---

# Final versions (approved design; round 2)

## `composite_main_v2.{pdf,png}`: approved composite, with (f) labelled (5.5 × 3.0 in)

- **What changed in (f).** The unlabelled "tasks" strip is now a separate labelled panel.
  - It lists the 10 tasks by name, grouped as Open-H (7, squares) and IWS (3, diamonds), and sorted by gain within each group.
  - Each task shows its gain over Direct (mean over k) with a 95 % paired-bootstrap CI. Every CI is above 0.
  - Rotated italic labels mark the two groups. Colours and markers match the Open-H and IWS horizon curves.
- **Room for it.**
  - (d) was narrowed, and its title shortened to "(d) Moving parts use transport".
  - The axis in (e) was narrowed. Its notes are now stacked beside the alternative rows ("alternatives: / within 1.2 % / grey band: / seed noise").
- **Unchanged.** All data, and panels (a)–(c).
- **Build.** `make_composite.py v2`. The generator writes all three versions by default (alt, v2, main); v1 (`composite_main`) is kept.
- **QA.** The layout audit reports 0 issues. The smallest font is 6.2 pt at print size and no words fall off the page. I viewed the 300-ppi crops (`qa/composite_main_v2_*`).

**Final caption (4 lines):**
> **Where and why \ours{} wins** (held-out DROID, seed 0, unless noted). (a) Per-episode error reduction vs.\ Direct and AR, ranked (mean over $k$). (b) Absolute reduction vs.\ episode motion (lines: quintile medians, bands: IQR). (c) Moving-patch error at $k{=}10$; oracle move: best observed feature in the transport window; other ep.: same oracle with candidates from an unrelated episode. (d) Gate knock-out and action swap, moving vs.\ static patches. (e) Ablations on validation (grey: seed noise $\pm$0.3\%). (f) Reduction vs.\ Direct per step (DROID 3 seeds) and per task (7 Open-H, 3 IWS), with 95\% paired-bootstrap CIs.

## `anatomy_strip.{pdf,png}`: slim Fig. 4(a) (5.5 × 1.0 in)

- **Layout.** One row with the two selected windows side by side. Each window has 4 tiles on the same zoomed crop:
  - observed frame $t$ with the transport arrows and a full-frame inset;
  - true $t{+}10$;
  - Direct error;
  - ShiftWM error, on a shared colour scale, with the crop-mean error in the corner.
- **Data.** Same data, selection rule, crop rule, error scale, arrow rule and helpers as panel (a) of `make_interpret.py`. I imported the helpers without running its `main()`, so nothing in `paper/` was written.
  - The corner values reproduce the current figure: 0.30 → 0.20 and 0.55 → 0.45.
  - `anatomy_strip_ledger.json` records the episodes, start frames and values.
- **Height.** It is 1.0 in rather than 1.2 in, because 8 square tiles across 5.5 in set the height.
- **Build.** `PYTHONPATH=src .venv/bin/python reviews/composite_results_fig/make_anatomy_strip.py`.
- **QA.** The layout audit reports 0 issues. The smallest font is 6.2 pt and no words fall off the page. I viewed the crops.

**Final caption (2 lines):**
> **Anatomy of a win** (held-out DROID, $k{=}10$, two high-motion windows chosen for the largest advantage over Direct). Green arrows move observed features (source $\to$ target); on the same zoomed crop, \ours{}'s error (right) is lower than Direct's on the moving arm (corner: crop-mean error).

**Space with v2 + strip.** The strip takes about 1.0 in plus a 2-line caption (≈ 1.25 in). The composite takes about 3.6 in with its caption. Together that is ≈ 4.85 in, against ≈ 5.9 in for the current Figs. 4, 5 and 6, a saving of ≈ 1.05 in before any text cuts.

---

# Round 3: `composite_main_v3.{pdf,png}` (5.5 × 3.0 in), made for the paper's Figure 7

This responds to the user's feedback that the per-task strip in (f) was hard to read and congested. Only row 2 changed; panels (a)–(c) and all data are identical to v2.

**(d): narrower (1.2 in instead of 1.36 in)**
- The title is now "(d) Motion uses transport".
- The mini-plot headers are shorter: "gate off ($g{=}0$): error rises" and "other actions: shift (patches)".
- The row labels are "moving" and "static". The values (+57 % / +2 %, 0.54 (2.8×) / 0.19) are unchanged.

**(e): same content, narrower axis**
- The separator between the two groups now carries a note, in grey italics: "design alternatives (all within 1.2 %)".
- "grey band: / seed noise" sits beside the last two rows.

**(f): now two sub-panels with more width (2.55 in instead of 1.74 in)**
- **Title:** "(f) Every step and every task: gain over Direct".
- **f1 (left): per-step curves.** Gain over Direct at each forecast step, with 95 % CI bands, for DROID (3 seeds), Language-Table, Open-H and IWS. Datasets are labelled directly at the line ends.
- **f2 (right): per-task bar chart.**
  - One horizontal bar per task, sorted within its group, with 95 % paired-bootstrap CI whiskers.
  - Tasks carry their full names (tissue lifting, tissue retraction, suturing 1, peg transfer, knot tying, suturing 2, needle handover; Rope, PushT, Box), under bold "Open-H" and "IWS" headers.
  - Bars are coloured by group. Open-H uses the same medium blue as its curve in f1; IWS uses a light blue that also matches its curve.
  - The x-axis "gain over Direct (%)" starts at 0. Every whisker stays right of 0; the smallest lower bound is 1.6 %, for needle handover.

**Build and checks**
- Build: `make_composite.py v3`. The default run builds alt, v2, v3 and main.
- The layout audit reports 0 issues. The smallest font is 6.2 pt at print size, and no words fall off the page.
- I viewed the 300-ppi crops (`qa/composite_main_v3_*`).

**Caption (4 lines; only (f) reworded):**
> **Where and why \ours{} wins** (held-out DROID, seed 0, unless noted). (a) Per-episode error reduction vs.\ Direct and AR, ranked (mean over $k$). (b) Absolute reduction vs.\ episode motion (lines: quintile medians, bands: IQR). (c) Moving-patch error at $k{=}10$; oracle move: best observed feature in the transport window; other ep.: same oracle with candidates from an unrelated episode. (d) Gate knock-out and action swap, moving vs.\ static patches. (e) Ablations on validation (grey: seed noise $\pm$0.3\%). (f) Gain over Direct per step (left; DROID 3 seeds) and per task (right; mean over $k$), with 95\% paired-bootstrap CIs.
