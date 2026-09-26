# Main-text figures, round 2

This folder holds three figures. Nothing in `paper/` or git was changed. Regenerate them from the repo root:

```
PYTHONPATH=src .venv/bin/python reviews/main_figs_round2/make_fig_plugin.py
PYTHONPATH=src .venv/bin/python reviews/main_figs_round2/make_fig_qualitative.py   # CPU, ~1 min (transport forward pass)
PYTHONPATH=src .venv/bin/python reviews/main_figs_round2/make_fig_tradeoff.py
python reviews/main_figs_round2/inspect_pymupdf.py                                 # QA proofs -> qa/
```

| file | size (in) | slot | PDF |
|---|---|---|---|
| `fig_plugin.{pdf,png}` | 2.70 × 1.62 | wrapfigure, `0.49\linewidth`; replaces `plugin_plot.pdf` (Fig. 4) | 19 KB, vector |
| `fig_qualitative.{pdf,png}` | 5.50 × 1.88 | full-width `figure`; replaces `anatomy_strip.pdf` (Fig. 5) | 846 KB, raster tiles + vector text |
| `fig_tradeoff.{pdf,png}` | 2.70 × 1.72 | wrapfigure, `0.49\linewidth`, in §5.1 | 35 KB, vector |

**Where the numbers come from.** Every plotted number comes from an existing loader, and each script writes the values it drew to `ledger_*.json`:
- plug-in plot: `scripts/v2/make_tables.py` `vjepa_rows()` / `dinowm_rows()`;
- qualitative figure: `make_interpret` (`droid.npz`, `transport_for`) and `make_segments` (`load_decoded`, `crop_box`, `_centroid`);
- trade-off plot: `make_tradeoff.collect()`.

The PNGs are the 400-dpi previews.

**QA.** For each figure, the skill's `layout_quality.audit_figure` reports 0 issues at its final width. `inspect_pymupdf.py` (PyMuPDF, used because poppler is not installed) finds:
- a smallest font of 6.2 pt at print size in all three figures;
- no words off the page;
- STIX fonts embedded.

The proofs and the 300-ppi crops are in `qa/`. I viewed all of them.

---

## 1. Plug-in heads (new Fig. 4)

**Design.** The figure uses paired bars, "without head" (grey) against "+ ShiftWM head" (green), on absolute values with every bar starting at 0. Above each pair is one green number, the relative change. Everything else was removed:
- arrows;
- hollow zero-shot circles;
- the numbers column;
- the +pp row;
- the SSIM and LPIPS rows.

Zero-shot skill appears as a single dashed reference line ("0-shot 4.1") inside the skill panel.

**Panels:**
- V-JEPA 2-AC error: all −16 %, moving −17 %, static −15 %;
- V-JEPA 2-AC skill: 24.6 → **36.4**, zero-shot 4.1;
- DINO-WM PushT teacher-forced latent error: −8 %.

Exact values: all −15.7 %, moving −16.9 %, static −14.8 %, PushT −8.0 %.

**Suggested caption (2 lines at `0.49\linewidth`):**
> **\ours{} as a plug-in head.** Same model, data, schedule and loss with and without the head. V-JEPA 2-AC: DROID test, 2 seeds (dashed: released model, zero-shot); DINO-WM: PushT validation.

The text keeps `\cref{fig:plugin}`. If `tab:plugin` is removed, change "Numbers in \cref{tab:plugin}" accordingly.

## 2. Qualitative figure (replaces the anatomy strip, Fig. 5)

This version follows the coordinator's follow-up request (less red, interpretability panels, plain titles).

**Row (a), "Where features move".** One held-out DROID window, shown as full frames. It is the second window of the interpret selection rule (`droid-4321…`, start 10). The first window's gate is nearly uniform around the arm, so it explains nothing. Full frames give the gate its contrast: the moving arm has g ≈ 1 and the static background g ≈ 0.3–0.5.

The five columns:
1. "observed t: content moves": the learned transport, drawn as arrows from source to target. The 10 strongest are shown, one per 2×2 block.
2. "moved vs. kept (gate g)": amber, with opacity rising with g.
3. "true t+10".
4. "error: Direct": a light red ramp over a greyscale frame. Transparent means low error.
5. "error drop: ShiftWM": Direct − ShiftWM, on the **same scale and units** as column 4. It is diverging: green where ShiftWM is lower, violet where it is higher (13 % of patches, kept visible), transparent where equal.

Badges show frame-mean error: Direct 0.38, and ShiftWM 0.27 (−29 %).

**Row (b), "Where the arm or instrument ends up".** Two decoded k=10 examples, each shown as [true t+10, AR, Direct, ShiftWM]:
- **DROID example 2:** AR and Direct are 184 px off; ShiftWM is 9 px off.
- **Hamlyn example 1:** AR and Direct have no instrument; ShiftWM is 3 px off.

The badge keeps the existing definition: the distance between the centroids of the drawn masks. Crops and display rule are the same as `decoded_row`. The target tile now shows only the white dashed true outline. The old orange SAM fill clashed with AR's orange.

**Suggested caption (≤ 3 lines):**
> **What the transport does** (held-out $k{=}10$; selected windows, averages in \cref{app:beyond}). (a) On DROID the gate is high on the moving arm, arrows show where its features come from, and \ours{} lowers Direct's error on the arm (green: lower, violet: higher; one scale). (b) Each model's decoded forecast segmented with SAM 2.1 (dashed: true; value: centroid distance).

**Honesty notes.** The examples are chosen by favourable rules:
- (a) comes from the "largest advantage over Direct" selection.
- (b) comes from windows where ShiftWM places the target best.

The caption must keep "selected". Averaged over windows, decoded placement shows only small gains:
- DROID, all windows: 31.2 vs 33.0 px against Direct, with a CI that includes 0.
- Hamlyn: about equal.

Say this in the caption; do not let the examples imply a large average effect.

Alternatives I tried and rejected:
- **Two windows × 5 cropped tiles (0.52 in each).** Too small, and the gate was uniform in the crop.
- **Attention π for one patch.** Left out to keep the figure uncluttered. It would need a new model call for π.

## 3. Appendix plot promoted: skill vs. action-ranking accuracy

**Why this plot and not segmentation.**
- **The segmentation gain is weak.** With the primary nn labeller on moving DROID windows, the IoU gain over Direct is +2.0 points. It is negative at k = 1–3, and the CI includes 0 there. On Hamlyn it is null: +0.1, CI [−0.2, 0.4]. Centroid placement is 12.0 vs 12.2 px against Direct.
- **Per horizon, that would show near-zero or negative early points.** A per-horizon plot is also redundant with panel (f) of the composite figure.
- **The trade-off plot adds something new.** It shows that the lower error does not cost action sensitivity: ShiftWM has both the highest skill and the highest ranking accuracy, on DROID and on the surgical data.
- **Only the DROID ranking number is already in Table 1.** The Open-H ranking results and the joint view are new in the main text.

**Design.** Two small panels: DROID (3 seeds) and Open-H surgery (1 seed).
- x = skill, % of persistence error removed. y = action-ranking accuracy, 100 − error, the same quantity as Table 1 "rank.".
- Crosshairs are 95 % cluster-bootstrap CIs.
- Methods are labelled directly, with no legend. A "better ↗" hint marks the direction.
- Ablations and per-seed dots from the appendix version are removed.

**Values:**

| | ShiftWM | Direct | AR | AR-TF |
|---|---|---|---|---|
| DROID skill | 27.5 | 21.6 | 17.4 | −10.4 |
| DROID accuracy (%) | 92.0 | 89.8 | 81.0 | 84.9 |
| Open-H skill | 49.2 | 47.3 | 47.1 | 35.7 |
| Open-H accuracy (%) | 99.45 | 99.36 | 99.22 | 98.49 |

**Caveats.**
- **Open-H ranking gap.** The differences are tiny and the CIs overlap. The claim is "best point estimate on both axes", not a significant ranking gain.
- **Language-Table is omitted.** There Direct ranks marginally better (1.80 % vs 1.93 % error, overlapping CIs). Leaving it out follows the no-negative-results rule, but it is a selection. The appendix `tradeoff.pdf` still shows it, so the main-text caption should say "DROID and Open-H" and not "every benchmark".

**Suggested caption (2 lines):**
> **Better forecasts without weaker action sensitivity.** Skill vs.\ action-ranking accuracy (95\% CIs); \ours{} is highest on both axes on DROID and Open-H (\cref{app:full}).

**Supporting §5.1 sentence** (replaces "\ours{} also ranks actions best on DROID (…; \cref{tab:main}, right)"):
> Lower error does not come from ignoring the actions: \ours{} also ranks actions best on DROID (\droidRankShift\% against \droidRankDirect\% for Direct) and on Open-H, so it is highest on both axes (\cref{fig:tradeoff}).

The Open-H ranking value would need a macro. `tables/generated/tradeoff_numbers.tex` already defines `\toHamErrShift`, 0.55 % error.

## Recommendations

1. **Adopt all three.** Figs. 4 and 7 are both half-width wrapfigures, so place them in different subsections (§5.2 and §5.1) so they do not stack. Fig. 5 (qualitative) at 1.88 in is about 0.5 in taller than the old strip, but it now carries the segmentation examples and the mechanism panels.
2. **If page budget is tight, cut Fig. 7 first.** Its claim also survives as a sentence.
3. **Keep "selected" in the qualitative caption.**
