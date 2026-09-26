# Figure 5 (fig:winmap-main): redesign candidates

**Data.** All four candidates use held-out DROID, seed 0, 130 test episodes, averaged over horizons k=1..10.
- `results/v2/analysis/winmap/` holds the paper's per-episode metric, produced by `scripts/v2/winmap.py`:
  - rel = mean over k of (err_ShiftWM − err_base)/err_base.
  - ShiftWM "wins" an episode when rel < 0. That is 127/130 episodes (98 %) against both Direct and AR.
- `results/v2s/droid/dinov2s/<arm>/s0/eval_test.npz` holds the absolute per-episode errors.
- Motion is the persistence error averaged over k, as in `winmap.py`.

**Build.** `.venv/bin/python reviews/fig5_candidates/make_fig5_candidates.py` writes `cand_{A,B,C,D}.{pdf,png}` and
`numbers.json`, which lists every number shown.

**QA.**
- The skill's `layout_quality.audit_figure` reports 0 issues for every candidate.
- `inspect_pymupdf.py` renders proofs at 100 ppi and crops at 300 ppi into `review/`. It finds nothing off the
  page, and the smallest font is 6.2 pt at print size.
- I inspected the proofs and crops myself.

## Important: two ways to aggregate give slightly different counts

| aggregation | Direct | AR |
|---|---|---|
| mean over k of the *relative* difference (paper text, `\wmWinDirect`) | 127/130 | 127/130 |
| difference of the horizon-mean *absolute* errors (A, C, D-b) | 125/130 | 127/130 |

Two Direct episodes flip from win to loss under the absolute aggregation. They are near-ties: ShiftWM's error is
0.2 % and 0.7 % higher (motion ranks 33 and 97). With them included, it is no longer true that "all losses are in the
less-moving half" (rank 97 is in the more-moving half). That claim holds only for the relative metric.

- Any figure that plots absolute errors must print 125/130 against Direct, and the caption must say which
  aggregation it uses.
- Candidate B uses the paper's relative metric throughout, so it is consistent with the text as written.
- The coordinator's example "128/130" does not appear under either metric.

## Candidates (wrapfigure slot: 2.31 × 1.94 in)

### A. Paired scatter (`cand_A`)
**What it shows.**
- Each point is one episode, with baseline error on x and ShiftWM error on y, both on log axes.
- Direct points are blue diamonds and AR points are orange triangles.
- The region below the y=x line is shaded green and labelled "ShiftWM lower error".
- The legend reads "episodes below y=x: vs. Direct 125/130, vs. AR 127/130".

**Pro.** It is the most self-explanatory comparison: every point is a matched pair.

**Con.**
- The gains are 7–12 %, which on log axes spanning 1.6 decades is only about 0.03–0.05 decades. The cloud therefore
  hugs the diagonal, and the advantage looks small.
- It says nothing about motion.
- It must use the absolute aggregation, so the Direct count is 125.

**Caption.** *Per-episode feature error on held-out DROID (seed 0, mean over k): ShiftWM vs. each matched baseline;
points below y=x (shaded) are episodes where ShiftWM has lower error.*

### B. Ranked per-episode error reduction, "waterfall" (`cand_B`)
**What it shows.**
- The per-episode error reduction (%) under the paper's metric, sorted from largest to smallest.
- Two step curves: vs. Direct (blue, median 7 %) and vs. AR (orange, median 12 %).
- The area above zero is shaded green.
- A dashed marker at episode 127 is labelled "127/130 episodes (98 %) ShiftWM lower error".
- The 3 episodes where ShiftWM is not lower stay visible below zero, labelled in muted grey.
- One AR value (70 %) is clipped and labelled as clipped.

**Pro.**
- It matches the text's 98 % exactly.
- It is positive and readable at a glance: the curve stays above zero almost to the end.

**Con.** It does not show the relation to motion. That would move to the text or the appendix.

**Caption.** *Per-episode error reduction of ShiftWM over the matched baselines on 130 held-out DROID episodes
(seed 0, mean over k), ranked; ShiftWM has lower error on 127 (98 %) against each.*

### C. Absolute error reduction vs. motion, with trend (`cand_C`)
**What it shows.**
- Each point is one episode: x is episode motion (log scale), y is the absolute error reduction (baseline error −
  ShiftWM error). Above zero (shaded) means ShiftWM has lower error.
- Lines join the medians of five motion quintiles, one line per baseline.
- The legend reads "ShiftWM lower error on: vs. Direct 125/130 episodes, vs. AR 127/130 episodes".

**Pro.**
- It carries both messages in one panel: nearly every point is above zero, and the gain rises with motion.
- The quintile medians against Direct go from 0.8 to 1.8 (×10⁻²) and against AR from 1.3 to 3.2.
- Spearman ρ between motion and absolute gain is 0.37 (Direct) and 0.49 (AR).

**Con.** It uses the absolute aggregation, so the Direct count is 125/130 (96 %), not the 98 % in the text.

**Caption.** *Per-episode absolute error reduction vs. episode motion (held-out DROID, seed 0, mean over k); above 0:
ShiftWM lower error; lines: medians of motion quintiles.*

### D. Full-width strip, B + C side by side (5.5 × 1.62 in, `cand_D`)
**What it shows.**
- (a) The waterfall from B, titled "ShiftWM has lower error on almost every episode" and labelled 127/130 (98 %).
- (b) The gain-vs-motion panel from C, titled "and its absolute gain grows with motion".
- Panel (b) carries no counts, so the only count in the figure is the paper-metric 127/130.

**Pro.** It gives both messages, each with its own panel headline, and the one count stays consistent with the text.

**Con.** It needs a full-width slot of 1.62 in instead of the wrapfigure.

**Caption.** *Held-out DROID, 130 episodes, seed 0, mean over k. (a) Per-episode error reduction, ranked: ShiftWM has
lower error than each matched baseline on 127 (98 %). (b) The absolute reduction grows with episode motion (lines:
medians of motion quintiles).*

## Recommendation

**If Figure 5 must stay a wrapfigure, use B.** It is the only single-panel option that tells the headline story with
the paper's own metric: 127/130 (98 %), consistent with `\wmWinDirect` and `\wmWinAR`. It is positively framed and
keeps the 3 losses visible without making them the headline. The motion claim is already in the text with
`\cref{app:geometry}`.

**If a full-width 1.6 in strip fits, use D.** It is the clearest way to show both messages, and it keeps the only
count on the paper's metric.

Candidates A and C are better suited to the appendix, unless the text switches to the absolute aggregation, in which
case it must report 125/130 against Direct.
