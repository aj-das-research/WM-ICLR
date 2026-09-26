# Figure 3 (featurespace) redesign candidates — previews only

Slot: the left half of the Figure 3 + Algorithm 1 top float (0.5\linewidth ≈ 2.75 in). Algorithm 1 is ≈ 2.0 in tall.
Every candidate is **2.75 × 1.50 in** (the current figure is 2.66 × 2.29 in), so it goes in at 100% scale.
Nothing in `paper/` was changed.

Files: `A_wide_featurespace.{pdf,png}`, `B_observed_to_forecast.{pdf,png}`, `C_unrolled_band.{pdf,png}`
(`*_print.png` = 150 dpi proof). Source: `make_fig3_candidates.py` (matplotlib, STIX fonts and the palette
from `make_figures.py`). QA output: `qa_audit.json`.

## Candidates and proposed captions (3 lines each at 2.75 in, 10 pt Times)

**A. Wide feature space.** The same concept as the current figure, redrawn wide and short. The band of real
patch features runs across the figure. ① ShiftWM's transport lines (width = π) meet at T, the gate is a dotted
segment from z₀,ᵢ, and r lifts the result to ẑ. ② Direct is an arc from z₀,ᵢ to the star. ③ AR is a dashed
path that drifts off the band.
```latex
\caption{\ding{172}~\ours{} mixes observed features (width $=\pi$), gated, plus $\mathbf{r}$;
\ding{173}~Direct synthesises the displacement; \ding{174}~AR compounds errors.}
```

**B. Observed → forecast strip.** Three aligned lanes that share one output column (ẑ next to the true star).
① ShiftWM shows the window W(i) as a 3×3 grid of chips (z₀,ᵢ in the centre, z₀,ρ(i) dark). Lines of width π
merge into T, T enters the gate g, a "keep" path carries z₀,ᵢ in with weight 1−g, and the result then passes
the "+ r (correct)" step. ② Direct: one arrow labelled "synthesise z₀,ρ(i) − z₀,ᵢ from h". ③ AR: chained
steps with growing error halos that end off the target.
```latex
\caption{\ding{172}~\ours{} keeps $\mathbf{z}_{0,i}$, moves to $\mathbf{T}$ (width $=\pi$) by $g$, adds
$\mathbf{r}$; \ding{173}~Direct synthesises the displacement; \ding{174}~AR compounds errors.}
```

**C. Unrolled band (my own idea).** The band of real features is flattened onto one axis. Transport weights
are drawn as stems hanging below the candidates (a readable histogram, which line widths are not). T is their
weighted mean. An amber bracket marks how far the gate g moves from z₀,ᵢ toward T, and r rises from that point
to ẑ. Direct and AR are drawn as in A. The geometry of Eq. (2) is exact: ẑ − r lies on the segment from z₀,ᵢ to T.
```latex
\caption{Unrolled features. \ding{172}~\ours{}: gated move to $\mathbf{T}$ (bars $=\pi$) plus $\mathbf{r}$;
\ding{173}~Direct synthesises the displacement; \ding{174}~AR compounds errors.}
```

Line counts come from Nimbus Roman 10 pt at 198.7 pt width. Each caption fits in 3 lines with about one word
to spare. Small-caps \ours{} or a different interword stretch could push one word onto a fourth line, so
recheck the line count in the compiled PDF.

## Comparison

| | A wide feature space | B observed→forecast | C unrolled band |
|---|---|---|---|
| Faithful to Props 1–3 / "drawn for one patch in Fig. 3" (Sec. 3.5) | yes (same picture) | partly: the geometry is gone, so the displacement size of Prop. 2 is not visible | yes |
| Transport weights π readable | line width only (hard at a glance) | line width | **bar heights (best)** |
| Keep / move / correct legible | moderate: gate and lowest transport line nearly coincide | **best** | good (bracket + r) |
| Redundancy with Algorithm 1 beside it | low | **high**: it restates line 9 of the algorithm | low |
| Text change needed | none (still "line width $=\pi$") | none | "line width" → "bars" in the caption |
| Visual density at 2.75 in | medium | highest (three lanes) | lowest |

## Recommendation

**Use A.** It keeps the feature-space picture that Sec. 3.5 and Props. 1–3 refer to, needs no change to the
text, and at 1.50 in tall with a 3-line caption the left column is about as tall as Algorithm 1. **C** is a
close second and the better choice if reviewers find the line widths hard to read. **B** reads best on its own
but repeats Algorithm 1, which sits right next to it.

## QA

- Physical size is 2.75 × 1.50 in for all three (checked with PyMuPDF). The fonts are embedded STIX (Type 42).
- The skill's `audit_figure` check (min 6 pt at 2.75 in display width) found no overlap, clipping or small-text
  issues in any candidate.
- Main labels are 6.8 pt and supporting labels 6.2 pt. Math subscripts render at 4.3–4.8 pt, because TeX script
  style is 0.7× the base size. The current TikZ figure has the same convention (\scriptsize with ~5 pt
  subscripts). Getting subscripts to 6 pt would need a base size of about 8.6 pt, which does not fit this height.
- I inspected the pixels at 600 dpi and 110 dpi (print size) and in grayscale. The three routes stay distinct
  without colour: solid arc, dashed or halo path, and the green fan with its diamond.
- Not run: the skill's `inspect_figure.py`, because `pdfinfo`/`pdftoppm` are not installed. PyMuPDF renders
  were used instead. No independent reader test was done.
- All three are schematics, not measured data. Point positions and π values are illustrative, as in the
  current figure.
