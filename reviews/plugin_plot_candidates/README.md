# Plot candidates to replace Table 2 (`tab:plugin`)

Regenerate from the repo root with `PYTHONPATH=src python reviews/plugin_plot_candidates/make_plugin_plot.py`.
The numbers come from `scripts/v2/make_tables.py` (`vjepa_rows()`, `dinowm_rows()`), the same generator as
`tables/generated/{vjepa,dinowm}_rows.tex`, and every table value is kept. Palette, fonts and type scale are
imported from `paper/submission_folder/figures/src/make_figures.py`: ShiftWM green #009E73, LOSS red #B8433A,
BACKBONE grey, STIX serif, and at least 6.2 pt at print size.

What is plotted: the relative change of each raw metric, 100·(head/base − 1), against the matched baseline.
- **V-JEPA 2-AC (DROID test).** The baseline is the fine-tuned model (2 seeds). The + head arm is also 2 seeds.
  Zero-shot appears as a hollow marker, measured against the same baseline (+27 / +10 / +42 %). Skill is a
  percentage, so its change is given in pp as text. Zero-shot skill is 4.1, fine-tuned 24.6 and + head 36.4.
  The difference is computed from unrounded values and comes to +11.9 pp; subtracting the rounded values gives 11.8.
- **DINO-WM (PushT and Wall, validation, teacher-forced).** One seed per arm.
- Colour shows the direction: green = better and red = worse, judged by each metric's own arrow (↓ or ↑).
- **Wall is shown in full.** Latent error is +18 %, SSIM −0.6 % and LPIPS +194 %. The LPIPS arrow is clipped at
  +48 % with a break mark (//), and its value is printed.

| file | size (in) | layout |
|---|---|---|
| `plugin_plot_A_wrap.pdf/.png` | 2.70 × 1.96 | wrapfigure beside text: one shared axis; right column shows base→head and % change (all table numbers) |
| `plugin_plot_B_strip.pdf/.png` | 5.50 × 1.13 | full-width strip: three panels on one shared axis; shows % change + baseline value |

QA: the skill's `layout_quality.audit_figure` reports 0 issues for both figures at their final widths. I viewed the
PNGs and a print-size PDF raster and embedded STIX fonts. `inspect_figure.py` could not run because `pdfinfo` is
not installed.

**Recommendation: A.** The table plus its caption fills about 16 text lines at full width. Variant A is half
width and ~2.3 in tall with its caption, so text can wrap beside it. It keeps every absolute value from the table.
Variant B saves about 6–7 lines and reads more easily, but it prints only baseline values; the head values have to
be worked out from the %.

## Suggested captions

A (`\begin{wrapfigure}{r}{0.49\linewidth}`, `\includegraphics[width=\linewidth]`):

> **\ours{} as a plug-in head.** Change vs. the same model without the head (same data, schedule, loss). V-JEPA 2-AC: DROID test, base = fine-tuned (2 seeds); DINO-WM: validation, 1 seed. Green better, red worse.

B (`figure`, `width=\linewidth`):

> **\ours{} as a plug-in head for published models.** Each arrow is the change from adding the head to the same model trained with the same data, schedule and loss. V-JEPA 2-AC: DROID test, 2 seeds; skill = zero-shot / fine-tuned / + head. DINO-WM: validation, 1 seed. The head hurts on Wall ($S{=}1$).

Note: `\cref{tab:plugin}` in 05_results.tex must be changed to point at the figure label.
