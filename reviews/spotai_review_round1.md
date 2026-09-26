# Review by Abhijit (AI Assisted), round 1: ShiftWM (ICLR 2027 submission)

Reviewer: Abhijit (AI Assisted). Workflow: paper-review (Step 5) and latex-annotate (Step 6) of the SPOT AI sequence.
Paper reviewed: `paper/iclr2027.tex` and `paper/submission_folder/` at commit `c97f50d`, main text first, then appendix.
Annotated copy: `paper/submission_folder/review_round1/review_annotated.pdf` (draft mode, 26 pages). The Overleaf main file is `submission_folder/review_round1/review_main.tex`. Every comment below has the same id in the PDF margin.

## Summary score

| | Score |
| --- | --- |
| Overall (ICLR 1 to 10) | **6**: marginally above the bar. 5 if the blocking item stays, 8 is reachable with the major fixes |
| Soundness (1 to 4) | 3 |
| Presentation (1 to 4) | 3 |
| Contribution (1 to 4) | 3 |
| Confidence (1 to 5) | 4 |

Comment counts: **1 blocking, 14 major, 27 minor** (42 in total, all anchored in the annotated PDF).

## Review basis

- **Main claim.** An output head that forms each future feature grid as a gated, action-conditioned local transport of observed patch features, plus a small correction, forecasts better than heads that regenerate the grid. It also improves published world models when attached as a plug-in head.
- **Evidence.** Table 1 compares matched heads (same frozen DINOv2-S encoder and predictor) on DROID, DROID cam 2 zero-shot, Open-H, Language-Table and IWS. Table 2 covers the V-JEPA 2-AC and DINO-WM plug-ins. The rest is region, win-map and knockout analyses (S5.3, Figs. 4 and 5) and one-seed ablations (Fig. 6).
- **Assumptions.** Most content persists and moves locally within a 7x7 window. Feature MSE in a frozen encoder space is a meaningful proxy for world-model quality.
- **Scope.** Deterministic forecasting of frozen-encoder features over 3.3 s horizons. No planning gain is claimed or shown.
- **Falsifying result.** The claim fails if a matched regression head with the same capacity equals ShiftWM on moving patches, or if the oracle-move gain also appears with candidates from an unrelated episode.

## Assessment

The idea is simple, well motivated and well controlled against Direct, which isolates the output head. The gains are consistent in sign and small in size: 3 to 8% over the best matched baseline, and 15.7% as a V-JEPA 2-AC plug-in. The paper mostly reports its negative results honestly (Wall, PushT planning, Language-Table leakage). Three things weaken it for an ICLR reviewer.

1. **The newest dataset (IWS) is not integrated.** It appears in Table 1 and in contribution 4 and carries a per-task claim. It has no per-task table, no data paragraph in App. C.1 and no reported interval. The abstract, contribution 3, the conclusion, the reproducibility statement and "all three datasets" in S5.1 still describe three datasets. This is the one blocking item [7b2a], with follow-ups [hdhl, 2fwo, 2np2, lmtx, u6wc, 8ahu, uneu, 9huk].
2. **Several analysis claims hold by construction or lack a control.** Prop. 1 is optimal under a model that assumes the answer [cpzm]. Prop. 3's direct-decoding half is a definition [5r4r]. The oracle move has no null [p8ke]. Direct recovers 71% of the "transport" gain without any transport [v2pm]. Per-episode gains shrink with motion, which sits badly with the moving-content story [8yon].
3. **Selective presentation that a reviewer will notice.** The teaser (c) shows only wins [fagq]. Contribution 2 omits the Wall loss [zq7y]. Table 2 shades the Wall rows as if they were wins [ldfb]. Planning opens the paper but has no main-text evidence [njrv]. Single-seed datasets carry significance daggers [c5b8].

Closest related work: F2M (warp plus regressed blend) and space-time memory networks (attention copy) are close enough that the delta needs one explicit sentence and ideally a learned F2M-style baseline [csmo].

## Anonymity and required statements

- **Anonymity: PASS.** The author block reads "Anonymous authors". PDF metadata has no author field. No names, affiliations, repository URLs or acknowledgements appear in the text or the software appendix (B.4 names only the anonymous `shiftwm` package).
- **Required statements: present.** The ethics statement, reproducibility statement and AI-use statement are all present after the conclusion and before the references. They do not count toward the page limit, and the main text ends on page 9. Open items: the AI-use statement still has an authors TODO comment in the source [j07s], the ethics statement conflicts with S4 on phantoms versus ex-vivo tissue [vfsf], and the reproducibility dataset list omits IWS [u6wc].

## Smallest high-impact revision order

1. Add the IWS per-task table with intervals and an App. C.1 IWS paragraph, and name the IWS selection split [7b2a, 8ahu, uneu].
2. Make the dataset scope consistent in the abstract, contributions, S5.1, the conclusion and the statements [hdhl, 2fwo, 2np2, lmtx, u6wc].
3. Add the oracle-move null. Report the gain-recovered share for the w=1 ablation, or drop the share [p8ke, v2pm].
4. Recast Props. 1 and 3 as motivation or remarks, and remove "MSE-optimal" from the contributions [cpzm, 5r4r].
5. Show the Wall loss in contribution 2, the teaser and Table 2 shading [zq7y, fagq, ldfb].
6. Seeds: add seeds or soften the daggers, and explain the 130 vs 132 episodes [c5b8, 20ni].

## Prioritised comment list

Page numbers refer to `review_annotated.pdf`. Every comment has a concrete fix in its text.


### Blocking (1)

| id | page | location | issue | comment and suggested fix |
| --- | --- | --- | --- | --- |
| 7b2a | 7 | S5.1, IWS paragraph | IWS per-task claim has no table: iws_per_task.tex is all \pend and not included; add it with CIs or cut the claim | no table shows the three IWS tasks. the IWS per-task table is all pending and not input. add it with intervals, or cut this claim. |

### Major (14)

| id | page | location | issue | comment and suggested fix |
| --- | --- | --- | --- | --- |
| hdhl | 1 | Abstract | abstract says 'every dataset' but lists DROID, Language-Table, Open-H only; IWS missing | this list names three datasets, Table 1 has four plus cam 2. name IWS here or say which benchmarks. |
| fagq | 1 | Fig. 1 (teaser), panel c | teaser panel (c) shows only wins; omits IWS, cam 2 and the Wall regression | (c) shows only wins. add IWS and cam 2, and the DINO-WM Wall loss ($+$18.4\% error), or label the panel as selected. |
| njrv | 2 | S1, first sentence | planning motivation without main-text planning evidence (PushT success unchanged, App. D.4) | the paper opens with planning but shows no planning gain, and App. D.4 finds none on PushT. frame it as forecasting or add a planning result. |
| cpzm | 2 | S1, contribution 1 | Prop. 1 optimality holds by the assumed generative model; not a contribution | true by construction. the model assumes the future is moved content. present this as motivation, not a contribution. |
| zq7y | 2 | S1, contribution 2 | contribution 2 cites the PushT gain and omits the Wall loss | PushT is reported but Wall is not, where the head raises error by 18.4\%. state both here. |
| csmo | 3 | S2, motion and feature warping | closest work (F2M, STM) not separated clearly; only RAFT warping is compared | this is close to gate $+$ transport $+$ correction. state the delta to F2M and STM in one sentence and compare to a learned F2M-style head. |
| 5r4r | 5 | S3.5, Prop. 3 | Prop. 3's direct-decoding half is tautological and not a comparison | true by definition, and $\delta_k$ can grow with $k$. compare bounds on the same quantity or call this a remark. |
| uneu | 6 | S4, datasets (IWS) | IWS is scored on 'official validation handles' while selection uses validation; leakage ambiguity | \S4 says selection uses validation only. which IWS split selects the checkpoint, and is it disjoint from these handles? |
| p8ke | 8 | S5.3, is the future a transport | oracle-move 51% has no null control; nearest-of-147 selection reduces error by construction | the nearest of 147 candidates lowers error even for unrelated features. add a null from another episode. |
| v2pm | 8 | S5.3, is the future a transport | 'gain recovered' does not isolate transport (Direct 71% vs ShiftWM 79%) | Direct recovers 71\% with no transport. this share does not isolate transport. |
| 8yon | 8 | S5.3, where do the gains come from | per-episode gain shrinks with motion (rho=0.38), in tension with the moving-content story | App. D.6 says the gain shrinks with motion ($\rho{=}0.38$), and Fig. 5 has its largest gains at low motion. this conflicts with Prop. 2. reconcile. |
| 20ni | 8 | S5.3 and Fig. 1b | 130 vs 132 DROID test episodes (win map/teaser vs Table 3) | 130 test episodes here and in Fig. 1b, 132 in Table 3. say which two are dropped and why. |
| c5b8 | 9 | S5.3, statistical analysis | single training seed on Open-H, Language-Table, IWS; daggers ignore seed variance | one seed on Open-H, LT and IWS, yet daggers. add seeds or soften. |
| 8ahu | 19 | App. C.1 | IWS dataset is undocumented in App. C.1 | IWS has no paragraph in this section. add source, handles, split unit, frame rate and $K{=}12$. |

### Minor (27)

| id | page | location | issue | comment and suggested fix |
| --- | --- | --- | --- | --- |
| 8lm3 | 2 | S1, para 2 | vague filler sentence before the oracle number | `much' is vague and the next sentence has the number. drop this sentence. |
| 8wy7 | 2 | S1, contributions | filler lead-in | filler. (edit: replace with "Our contributions are:") |
| 2fwo | 2 | S1, contribution 3 | contribution 3 omits IWS | IWS is missing here and from the episode counts, but item 4 and Table 1 use it. |
| jzb5 | 2 | S1, contribution 4 | contribution 4 is a number dump | too dense. ten numbers in one item. keep DROID vs Direct and one transfer number, the rest belongs in \S5. |
| ibjr | 3 | S3, first paragraph | redundant paragraph heading | `Method:' repeats the section title. drop the paragraph heading. |
| rgj7 | 4 | S3.3, Eq. 1 | Eq. 1 simplification | simplify: attention over the window plus an identity bias $\beta$. say that in words first. |
| ymi7 | 4 | S3.3, Eq. 2 | Eq. 2: state Direct as the g=0 case | keep. add that Direct is this with $g{=}0$, so the reader sees the one change. |
| m528 | 5 | S3.4, Eq. 3 | unused loss in main text | $\lambda{=}0$ in every result and the ablation shows no gain. move this loss to the appendix. |
| qgxd | 5 | S3.5, Eq. 4 | Eq. 4 simplification | simplify in words: each future patch copies one observed patch, plus zero-mean new content. |
| 9huk | 6 | S4, metrics | K=10 stated, IWS uses K=12 | IWS uses $K{=}12$ (Table 3). say so here. |
| 5uh1 | 6 | S4, baselines | Linear baseline is degenerate (4.8 vs 0.249) | Linear is 20$\times$ worse than persistence in Table 1. explain why, or drop the row. |
| 7tr4 | 6 | Table 1 | Table 1: CIs for cam 2 and IWS not reported | keep as a table. daggers on cam 2 and IWS, but \S5.1 gives intervals only for three datasets. report all six. |
| ihoh | 7 | S5.1 | AR-TF is worse than persistence only from k=5 | per-horizon table: AR-TF beats persistence for $k{\le}4$. (edit: replace with "is worse than persistence from $k{=}5$ on") |
| mdwi | 7 | S5.1 | AR-TF not run on Language-Table | the DINO-WM-style AR-TF is missing here (`--' in Table 1). run it or say why. |
| 2np2 | 7 | S5.1 | 'all three datasets' predates IWS | which three? Table 1 has four datasets plus cam 2. |
| 4byo | 7 | S5.2, V-JEPA 2-AC | V-JEPA overlap affects all rows, not only zero-shot | not only. both fine-tuned arms start from a model that may have seen the test set. call the gain relative. |
| ldfb | 7 | Table 2, DINO-WM rows | Wall rows shaded as ours though worse on every metric | Wall is worse on all three metrics. unshade these rows. |
| 0lnq | 8 | Fig. 5 | win-map caption does not say which seed(s) it shows | which seed is this? Table 1 averages three. say it in the caption. |
| djrj | 8 | S5.3, beyond feature error | flow-warp comparison is not matched (no correction) | warping has no correction term. not matched. |
| qye0 | 8 | S5.3, beyond feature error | segmentation protocol unclear in main text | how do masks come out of feature forecasts? name the decoder. |
| fbga | 8 | S5.3, statistical analysis | Holm family vs daggers unclear | are the Table 1 daggers Holm-adjusted? name the family. |
| lm6v | 9 | Fig. 6 (ablations) | single-seed ablation; small deltas uninterpretable | one seed. gaps under about 1.5\% ($w{=}5$, tanh, contrastive, global) may be noise. report the full model's seed spread. |
| 5pax | 9 | S5.4 | cost claim is a candidate count | this counts candidates, not cost. report time or memory, or say `a fifth of the candidates'. |
| lmtx | 9 | S6, conclusion | conclusion omits LT and IWS | Language-Table and IWS are missing. match the abstract and Table 1. |
| u6wc | 10 | Reproducibility statement | IWS missing from dataset/licence list | IWS is missing from this list. name it and its licence. |
| vfsf | 10 | Ethics statement | phantoms vs ex-vivo tissue inconsistency | \S4 says phantoms only. which is it? |
| j07s | 10 | AI use statement | AI statement carries an unresolved author TODO | the source still has an authors TODO. check against the ICLR 2027 policy and delete the comment. |

## Annotation notes

- The review copy lives in `paper/submission_folder/review_round1/`. It holds copies of `sections/*.tex`, `appendix/appendix.tex`, and `tables/main_forecasting.tex` and `tables/plugin.tex` (for the table boxes), plus `labreview.sty` v3.13 and `review_main.tex`. `paper/review_main.tex` is an identical copy for local builds. One file detects its location, so it compiles both from `paper/` and from its own folder, which is Overleaf's layout when it is set as the main document.
- The clean paper (`iclr2027.tex`, `sections/`, `appendix/`, `tables/`) is byte-identical. Stripping every review command from the copies reproduces the originals exactly. The only exceptions are two review-only layout changes: the table inputs point to the review copies, and the win-map `wrapfigure` is one line taller so the figure box does not collide with its caption.
- labreview patch (review copy only): under tectonic/XeTeX, soul drops the highlight bars behind words when `\sethlcolor` receives an xcolor expression such as `reviewer!20`. The copy stores the tint in a named colour first (`\colorlet`), which restores the highlights. The canonical skill asset is unchanged. Consider upstreaming this fix.
- Draft mode shows 42 notes. Final mode hides them all and applies the two edits [8wy7, ihoh]. Every annotated page was rendered and checked for anchor, arrow target, clipping and overlap.
