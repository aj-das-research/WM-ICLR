# Figure 2 (fig:method): redesign candidates

Both candidates are 5.5 in x 2.05 in (full text width, lower than the current 5.86 x 3.18 in figure). Body text is
6.2 to 7.6 pt at print size. Sub- and superscripts are 70 % of their base size (about 4.3 pt, the usual LaTeX ratio).
Colours follow the existing figure: slate = observed features, purple = actions, green = ShiftWM/transport,
amber = gate/keep, blue = Direct baseline. Every colour also has a text label.

All pictures show real data from the same window the current Figure 2 uses: held-out DROID episode
`droid-c70170b3ca00f756c4bb35c8`, checkpoint `results/v2s/droid/dinov2s/shiftwm/s0/best.pt`, step k=10, and the
patch with the highest gate (patch 51, g=0.99). The following all come from `assets/head_data.tex`, which
`arch_assets.py --head` produced: the 7x7 transport weights for the S=3 source frames, their mass (33/25/42 %), and
the 12-D per-patch vector colours. `make_c_assets.py` recomputes the PCA colours for Z_-2 and Z_-1 with exactly the
same projection. It reproduces `head_obs.png` pixel for pixel and gives the same errors (0.44 for copying the last
frame, 0.28 for ShiftWM).

## Files

| File | What |
|---|---|
| `cand_A.pdf/.svg/.png` | Candidate A, "h decides where, Z supplies what" |
| `cand_C.pdf/.svg/.png` | Candidate C, "rays through the observed grids" |
| `make_arch_candidates.py` | Builds both (`.venv/bin/python reviews/arch_candidates/make_arch_candidates.py A C`). Runs the skill's `layout_quality.audit_figure` (text overlap, clipping, min font) |
| `make_c_assets.py` | Real PCA views of Z_-2, Z_-1, Z_0 (`PYTHONPATH=src .venv/bin/python reviews/arch_candidates/make_c_assets.py`) |
| `inspect_pymupdf.py` | Stand-in for the skill's `inspect_figure.py`, because poppler is not installed. Makes print-size proofs at 100 ppi and 300 ppi crops, finds the smallest font and checks for words off the page |
| `review/` | Proofs, crops and `inspection.json` |
| `assets/` | Read-only copies of the real-data rasters from `paper/submission_folder/figures/src/arch_assets/`, plus the new `obs_m2/obs_m1/obs_0.png` |
| `_ref/` | PNG renders of the current architecture, teaser and featurespace figures, for comparison |

QA status: the audit reports 0 issues for both figures and no words leave the page. I looked at the print-size
proofs and the 300 ppi crops. I found no overlaps, the arrowheads touch their targets, and the one wire crossing in C
has a visible break (keep-line over h-line).

---

## Candidate A: "h decides where, Z supplies what" (two streams)

**Layout.** On the left, a grey *Predictor* panel: frames, frozen DINOv2, the grids Z_{-H+1:0}, the Memory encoder
(L_e layers, space/frame/action embeddings) giving M, and the Decoder x L_d (self-attn over patches, cross-attn to M,
AdaLN(c_k)). The decoder is drawn as K stacked copies labelled "all K steps in parallel". Actions go a_<0 to the
encoder and a_0:K-1 through the causal GRU, plus e_k, to give c_k.

On the right, a green *ShiftWM head* panel. It draws Eq. 1 as an attention with three separate inputs:
- **q** from h_{k,i}
- **k** from M
- **values**: the observed grids themselves, carried by a thick slate "content stream" that runs from Z above the
  predictor straight into the head.

The three real 7x7 windows over Z_-2, Z_-1, Z_0 show pi_{k,i}, with the identity cell (beta) outlined in amber.
Below them, Eq. 2 is written out on real 12-D patch vectors: (1-g) Z_{0,i} + g T_{k,i} + r_{k,i} = Zhat_{k,i}.
Then come the forecast stack Zhat_{1:K} and the formulas for g and r, with h feeding both. A blue note says that
g=0 gives the Direct head.

**Rationale.** It separates the two roles cleanly: the network only computes *where* content comes from and *how
much* to move, while the content comes from the observed features. That is the paper's central argument. It also
matches Alg. 1 line for line. The predictor panel is labelled "shared with the baselines; or V-JEPA 2-AC, DINO-WM",
so the drop-in claim can be read off the layout: everything on the right is the head you swap in. It is the most
complete and faithful option.

**Weakness.** It is dense, with about 45 labels, and it looks like an engineering diagram. The idea of *spatial*
motion is carried only by the heat-map windows.

**Suggested caption.**
> **\ours{} predictor and gated-transport head.** Left: a predictor (shared with all baselines, or a published world
> model) turns observed grids and actions into one query $\h_{k,i}$ per patch and step, for all $K$ steps in parallel.
> Right: the head attends over a $w{\times}w$ window in the last $S$ observed grids (weights $\pi$, real values from a
> held-out DROID window) and \emph{moves} observed features, $\mathbf{T}=\sum_j\pi(j)\Z_j$; a gate keeps $\Z_0$ and a
> small correction $\mathbf{r}$ adds new content. With $g{=}0$ the head reduces to the Direct baseline.

---

## Candidate C: "rays through the observed grids" (spatial)

**Layout.** On the left, three input frames pass through one tall *frozen DINOv2* block. It produces a column of
real feature grids Z_-2, Z_-1, Z_0 (PCA colours), with time running downward. On each grid the w x w window around
patch i carries the real transport weights. The 15 strongest sources send **rays** (width proportional to pi) that
converge on the node T_{k,i}, and each frame's mass (33/25/42 %) is printed next to its grid.

In the green head panel, three labelled wires meet at a single sum node:
- *move*: g Sum pi Z
- *keep*: (1-g) Z_{0,i}, drawn from the amber identity patch in Z_0
- *correct*: r_{k,i}, from h

The output wire lands on patch i of the forecast Zhat_10. Behind it, Zhat_1 and Zhat_5 are drawn as "all K steps at
once", next to "approx. true Z_10" and the window's error (copy 0.44 to ShiftWM 0.28). A compact predictor lane runs
along the bottom: Memory encoder, then Decoder x L_d (K copies), then h_{k,i}, with actions going through GRU + e_k
to c_k. h drives the query, g and r. Notes on the right say that g=0 reduces to the Direct head and that no forecast
is fed back.

**Rationale.** It *shows* the core message rather than stating it. The forecast patch is built from lines that
reach back into features that were actually observed, a few patches away and up to S frames back. "Moving, not
regenerating" is visible before any symbol is read, and it uses the same move/keep/correct words as the introduction
and the old visual equation. The predictor is visibly secondary: a thin lane labelled "shared; or V-JEPA 2-AC,
DINO-WM", which supports the drop-in claim. It keeps the real-data grounding of the current Figure 2 but is 35 %
shorter, and it merges the old top and bottom rows into one reading path.

**Simplifications (say them in the caption or text).**
- The keys come from M. C shows only "query" from h, while A draws M to k explicitly.
- The memory encoder's embeddings appear as "+ emb.".
- Only the 15 strongest of the 147 window sources are drawn as rays.

**Suggested caption.**
> **\ours{} forecasts by moving what it has observed.** For each future step $k$ and patch $i$, the predictor (bottom;
> shared with all baselines, or a published world model) gives a query $\h_{k,i}$. The head attends over a
> $w{\times}w$ window in the last $S$ observed grids (rays: the strongest weights $\pi_{k,i}$ on a held-out DROID
> window, $k{=}10$) and outputs $\hat\Z_{k,i}=(1-g)\Z_{0,i}+g\mathbf{T}_{k,i}+\mathbf{r}_{k,i}$; all $K$ steps are
> decoded in parallel and never fed back. With $g{=}0$ it is the Direct baseline $\Z_0+\mathbf{r}$.

---

## Recommendation

**Use Candidate C as Figure 2.** It makes the novelty visible at first glance: the rays from observed features to
the forecast patch. It still carries every element the method section needs (DINOv2, memory encoder, GRU + e_k,
decoder with AdaLN, h, pi, T, g, r, parallel K, the g=0 Direct reduction and the drop-in framing). It is also shorter
than the current figure.

Use A's head panel (windows plus the vector equation) as an appendix figure if a reviewer wants the
q/k/value wiring spelled out. Candidate B, a side-by-side "regression head vs transport head on one backbone", was
considered and dropped because teaser panel (a) already makes the Direct vs ShiftWM contrast.

**Before inserting either candidate:**
- Figure 3 (featurespace) numbers its heads 1/2/3. If C goes in, the caption could point to it: "the spatial view of
  ① in Fig. 3".
- `\ours` in a caption renders as "ShiftWM"; the figure hard-codes the name.
- Fonts are STIX (Times-compatible) to match the paper's `times` package.

---

## Candidate A v2 (selected; polished) — `cand_A_v2.pdf/.svg/.png`

Built by `design_A_v2()` in `make_arch_candidates.py` (`... make_arch_candidates.py A_v2`); the originals `cand_A.*`
are unchanged. Same data, 5.5 x 2.25 in. Changes from A:
- **Lanes:** each connector and label now has its own horizontal lane. From top to bottom: title, italic stream
  label, Z content stream, M->k wire, then "all K steps in parallel". Every gap is at least 0.1 in.
- **Main row:**
  - The M label sits above the encoder->decoder arrow, clear of the junction and the decoder box.
  - The decoder copies are offset up-right, so the h junction sits clear of them.
  - "self-attn: patches" is now "patch self-attn", so it fits the box.
  - The trapezoid is wider, so DINOv2 fits inside it.
- **Bottom strip:**
  - The h bus now ends in an arrow that points into the green panel.
  - The g and r formulas and the "g=0: Direct head" note sit inside the green panel with a margin of at least
    0.06 in.
  - The action row and e_k were raised off the panel edge.
- **Equation:**
  - The four vector chips have an even 0.40 in pitch. The coefficients (1-g) and +g, the + and = signs are centred
    in the gaps between chips.
  - keep/move/new sit under the symbol labels.
  - The T chip is centred under the Z_-1 window.
  - The transport-sum label, now written inline as Σ_j π(j) Z_j, is offset to the right of the down-arrow.
- **Right margins:** the Eq. 1 legend now leaves at least 0.18 in to the right edge, and the Ẑ_{1:K} stack at least
  0.14 in.

QA: `layout_quality.audit_figure` reports 0 issues, and `inspect_pymupdf.py` finds no words off the page. I looked
at the 100 ppi print proof and all three 300 ppi crops (`review/A_v2_*`).

**Suggested caption (A).**
> **\ours{} architecture.** Left: a predictor, shared with every baseline or taken from a published world model, turns
> the observed DINOv2 grids and the actions into a query $\h_{k,i}$ for every patch $i$ and step $k$, for all $K$
> steps in parallel. Right: the \ours{} head attends over a $w{\times}w$ window in the last $S$ observed grids
> (Eq.~\ref{eq:transport}; weights $\pi$ from a held-out DROID window, $k{=}10$), moves the observed features,
> $\mathbf{T}=\sum_j\pi(j)\Z_j$, and outputs $(1-g)\Z_0+g\mathbf{T}+\mathbf{r}$ (Eq.~\ref{eq:output}); with $g{=}0$
> it is the Direct baseline. Vectors: 12 principal components of one patch.
