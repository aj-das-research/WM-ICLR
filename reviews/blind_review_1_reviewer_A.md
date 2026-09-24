# Blind review: "ShiftWM: World Models that Move What They Saw"

- **Reviewed PDF snapshot:** `/home/Test/abhijit.das/projects/WM-ICLR/reviews/blind_review_1_paper_snapshot.pdf` (32 pages: 10 main pages + references + appendix)
- **Date:** 2026-09-24
- **Reviewer:** Reviewer A (ICLR reviewer persona)
- **Scope:** Blind review. Only the PDF, its extracted text, and a literature search for novelty were used.

---

## Summary

The paper proposes ShiftWM, an output parameterisation for latent world models that forecast frozen foundation-model (DINOv2) patch features. The forecast is not regressed. For each future step k, conditioned on a GRU summary of the action prefix, the model predicts three things for every patch: (i) a local soft transport (a softmax over a 7×7 window in each of the last 3 observed frames, with a learned identity bias) of *measured* features, (ii) a sigmoid gate that mixes "keep Z0" with "move", and (iii) an additive correction (Eq. 2). All horizons are decoded in parallel from observed features, so predictions are never fed back. An optional action-contrastive hinge (Eq. 3) is meant to stop the anchored forecast from ignoring actions.

Three propositions motivate the design (Sec. 3.1). (1) Under a "motion + zero-mean innovation" model, the MSE-optimal forecast is a transport of observed features. (2) An additive head has to output the feature contrast along the motion. (3) Recursive rollouts compound error geometrically.

Empirically, against matched predictors on the same backbone (Direct, rollout-trained AR, a DINO-WM-style teacher-forced AR-TF, persistence, linear), ShiftWM has the lowest held-out feature MSE on DROID (0.181 vs 0.196 Direct, −7.6%) and on the Open-H Hamlyn dVRK surgical tasks (−3.6% vs Direct). As a plug-in head, it improves a fine-tuned V-JEPA 2-AC on DROID (skill 24.6% → 36.4%, 2 seeds). On DINO-WM it improves PushT latent error by 8.0% and worsens Wall by 18.4%. Analyses cover a gate-ablation intervention, motion-decile breakdowns, an "oracle move" check of Prop. 1, and a SAM 2.1-based arm-placement readout.

A large part of the stated evaluation is still marked **pending**:
- 5 of 8 forecasting columns in Table 1
- the entire ablation table (Table 3)
- all closed-loop planning (Table 4 and DINO-WM success)
- decoded-pixel metrics (Table 14), probes (Table 15), efficiency (Table 16), IWS (Table 12)
- the optical-flow comparison (Fig. 14) and several other figures

---

## Strengths

1. **Clear, well-motivated idea with a clean inductive bias.** "Move what you saw, correct the rest" is simple, easy to explain, and suits manipulation and surgical scenes where most content persists. Eq. 2 is minimal and initialised at persistence (W_o = 0, b_g = −3), which is sensible engineering.
2. **Carefully matched baselines.** Direct shares the memory encoder, decoder, width, depth, cross-patch attention and schedule (21.70M vs 21.75M params). The comparison therefore isolates the output head, not capacity. This is better experimental hygiene than most world-model papers.
3. **Honest reporting of a negative result.** The DINO-WM Wall degradation (+18.4% latent error, LPIPS 0.009 → 0.026) is stated in the abstract, and the paper avoids overselling the marginal Hamlyn segmentation gains (App. E.2). Selection rules for qualitative examples are stated explicitly (App. E.3), and random galleries are included (Figs. 18, 19).
4. **Good statistical practice where results exist.** DROID splits are by recording session, with a session-level paired bootstrap and Holm correction (App. C.6). Per-horizon (Table 10) and per-task (Table 11) breakdowns are given. On Hamlyn the ordering ShiftWM < Direct ≤ AR holds on all 7 tasks.
5. **Mechanistic analyses go beyond a leaderboard.** The static/moving decomposition (Table 17) shows every learned baseline is *worse than persistence on static patches* and ShiftWM is not. This is an insightful finding. The oracle-move check (Fig. 12b, 51% of the stay error removable by a single observed candidate; ShiftWM recovers 79% vs 71%/69%) is a neat model-free test of the premise.
6. **The V-JEPA 2-AC plug-in result is striking if it holds.** Adding 0.16M parameters raises skill by ~12 points at every horizon (Fig. 8c), and the gate opens to ~0.65 within 250 steps. Validation error is lower at every step for both seeds.
7. **Interpretability.** The gate and transport field are explicit tensors (Figs. 2, 6, 21), and the failure-case figure is a welcome addition.

---

## Weaknesses (prioritised)

### W1. The submission is incomplete: a large share of the claimed evaluation is "pending" (critical)
- **Table 1 (p. 7):** IWS, Bridge V2, RT-1, Language-Table and DROID cam-2 (zero-shot) are all pending. Only 3 of 8 columns have numbers. Fig. 5 has a pending Bridge panel.
- **Table 3 (p. 8), the entire ablation study, is pending.** Sec. 4.4 has one sentence followed by "pending". Without it, none of these design choices is supported: window size, the number of source frames, local vs global transport, the correction type (scaled vs tanh vs none), action-free vs action-contrastive, and encoder choice (DINOv2-B, DINOv3, pooled grid).
- **Table 4 (p. 9), all closed-loop planning, is pending**, as is the DINO-WM success rate (Sec. 4.3, Table 2 last column). Sec. 4.5 ends with "pending".
- **Appendix:** Tables 12, 14, 15, 16, Fig. 9, Fig. 14 (transport vs RAFT), Fig. 10 right, Fig. 20, parts of Table 13, Table 6 (IWS license, planning split sizes), compute (App. C.4) and the planning protocol text (App. C.5 is an empty heading) are all pending.
- **The text still claims results that are not there.** The abstract and contributions say the method is evaluated "on IWS, BridgeData V2, RT-1, Language-Table and three planning suites" (p. 2). Contribution 5 claims "how an action-contrastive objective keeps anchored forecasts sensitive to actions for planning". Sec. 5 says "Figure 14 compares it with RAFT optical flow". None of this is supported in the current PDF.

As submitted, questions Q2 (planning part), Q3 (ablations and planning) and the multi-dataset part of Q1 are unanswered. A reviewer must score the paper as it stands.

### W2. Seed and statistical inconsistencies undermine the headline tables (major)
- The Table 1 caption says "mean of 3 seeds". The same caption carries "[Interim: seeds completed – Hamlyn: ar 1/3, ar-tf 1/3, direct 1/3, shiftwm 1/3]". So the surgical column, and the claim "best predictor on seven surgical robot tasks", rests on **one seed**, and Table 11 ("mean over horizons and 3 seeds") contradicts this. Table 17 carries the same interim note.
- **DROID numbers differ across tables with no explanation:**

  | Method | Table 1 | Table 17 |
  |---|---|---|
  | ShiftWM | 0.181 | 0.179 |
  | Direct | 0.196 | 0.194 |
  | AR | 0.206 | 0.203 |
  | AR-TF | 0.275 | 0.268 |

  Skill is 27.5% in Table 7 and the abstract but 28.3% in Table 17. AR-TF skill is −10.4% in Table 7 but −7.6% in Table 17. These look like different seed subsets (Table 13 base 0.179 is seed 0). Every table must state which seeds it uses.
- **V-JEPA 2-AC plug-in:** 2 seeds and no CIs in Table 2 or Table 7 (only the zero-shot row has a CI). The strongest result in the paper therefore has the weakest statistics.
- **DINO-WM plug-in:** the number of seeds and any CI are not stated.
- **Segmentation analysis:** one seed (App. E.2, caveat iv), yet the CIs are presented as evidence of method differences. They only capture episode variance.
- **Fig. 13:** the 10×10 decile grid is drawn with "cubic interpolation" as filled contours. This smooths the data and can hide noisy cells. Show the raw grid.

### W3. The effect size is modest and the evaluation regime is small (major)
- The gain over the strongest matched baseline is **7.6% on DROID and 3.6% on Hamlyn**. Direct already takes most of the step from AR.
- DROID is used at **1,126 episodes (24 shards)**, with a 132-episode test set, out of a ~76k-episode dataset. B.3 and D.2 note that all learned predictors reach their best validation error early and then overfit. The benefit of a strong copy prior is expected to be largest in exactly this low-data, overfitting regime. Whether it survives more data, or a larger predictor, is the key open question, and Table 13 ("+2nd camera, EMA, dropout") only partly addresses it.
- There is **no model-size or data scaling curve**.

### W4. The theory is largely tautological and does not support the claims it is used for (major)
- **Proposition 1** follows immediately from linearity of expectation under the assumption E[ε_k | P_k, Z_0, a] = 0. That assumption carries all the weight. In practice DINOv2 patch features are *not* translation-equivariant copies: sub-patch motion (14 px patches), viewpoint and illumination change, deformation (tissue, rope) and context-dependent self-attention features all violate it. The paper needs the correction r precisely because the assumption fails. The statement "the optimal forecast therefore never leaves the convex hull of observed features" (p. 5) is then used as if it described real data.
- **Proposition 2** is about *representational cost* (the norm of the residual), not learnability. The Direct baseline has cross-attention to memory and can in principle implement a copy through attention. The proposition says nothing about why it fails to learn this, which is the empirically relevant question. Calling it "the cost of regenerating motion" overstates it.
- **Proposition 3** is the standard Lipschitz rollout bound. The "direct predictor incurs exactly δ_k" half is vacuous, because nothing bounds δ_k relative to δ. Table 10 shows Direct and ShiftWM error still grows ~2.6× from k=1 to k=10. The abstract's "anchoring every horizon to measured features prevents compounding error" and the intro's "errors cannot compound" are overclaims. Direct shares this property, so it is not a ShiftWM-specific advantage.
- The "row-stochastic" P_k ∈ {0,1} is also later called a "permutation" in Prop. 2. Many-to-one copying (occlusion) is not a permutation. The indicator in Eq. 1 renders as a broken glyph ("β⊮[j=(0,i)]").

### W5. Novelty is overstated; closely related feature-space warping prior work is missing (major)
- The core idea of forecasting *features* by warping observed features with a predicted motion field plus a residual appeared in dense semantic forecasting:
  - Šarić et al., "Warp to the Future: Joint Forecasting of Features and Feature Motion" (CVPR 2020), and its TNNLS follow-up.
  - Luc et al., F2F (ECCV 2018), which forecasts convolutional features.
  - Terwilliger et al. (WACV 2019), recurrent flow-guided semantic forecasting.

  None of these is cited.
- DDP-WM (static/dynamic separation) and anchored navigation world models are cited but not compared against.
- The claim "first latent world model that forecasts by action-conditioned transport of frozen foundation features" may be literally true, but the novelty is narrow: an action-conditioned, attention-based (soft, local) version of feature warping, applied to DINOv2. The related-work section should state this delta honestly and position against the feature-warping line.

### W6. Missing baselines that would isolate *why* transport helps
- **A feature-warping baseline driven by optical flow:** warp Z0 with RAFT flow extrapolated or predicted from actions. Also an **oracle-flow warp**, which bounds what transport can achieve.
- **Direct + identity/skip gate:** Direct with a learned gate between Z0 and its regression, i.e. Eq. 2 without T. Part of the static-patch gain (Table 17) may come from gating alone, not transport. Table 3 has "no transport (= Direct)" but not "gate without transport", and it is pending anyway.
- **Global vs local transport** (pending), which would show whether the locality prior or the copy mechanism matters.
- **The actual published DINO-WM, V-JEPA 2-AC or LeWM predictors on the same DINOv2 feature forecasting task** (not just the "DINO-WM-style" re-implementation AR-TF). AR-TF is worse than persistence (Table 1: 0.275 vs 0.249), which suggests it is a weak strawman. Please verify it against the official DINO-WM training recipe (frame-skip, history length, loss).

### W7. The plug-in results need controls and confound checks
- **V-JEPA 2-AC:** the head reads layer-normalised tokens of the *last S=3 measured frames* (App. B.3). Does the base V-JEPA 2-AC fine-tune receive the same 3-frame context? If the base uses less context, part of the gain is extra information, not the parameterisation.
- Fine-tuning is short (3,000 steps) and both arms overfit (Fig. 8a). The fine-tuned baseline peaks at step ~250 and then degrades, so the best-checkpoint comparison is sensitive to evaluation frequency.
- V-JEPA 2-AC was trained on DROID and "likely include[s] our test trajectories" (App. C.3). This matters for the zero-shot number, and the paper should say it explicitly in the main text.
- **DINO-WM Wall:** the explanation ("DINO-WM conditions on a single history frame") is a hypothesis, not tested. It is a direct counterexample to "works as a plug-in head" and needs analysis, e.g. S=1 ShiftWM on DROID (a pending Table 3 row), or Wall with more history.

### W8. The action-sensitivity problem is acknowledged but unresolved
- The authors correctly note that anchoring to Z0 makes it easy to ignore actions (p. 5). The fix (Eq. 3, λ=1, margin 0.05) is evaluated only in pending cells.
- Fig. 10 (left) suggests ShiftWM without the hinge already has the highest action-ranking accuracy (~0.92). That is actually good news, but it is buried in the appendix, and the "rank acc." metric in Table 3 is never defined in the main text.
- Without planning results there is no evidence that lower feature MSE translates into better control. For a world-model paper at ICLR, this is the key downstream test.

### W9. Metric scope: feature MSE only, deterministic model
- All main results are standardised DINOv2-feature MSE. MSE-optimal deterministic forecasts are conditional means. ShiftWM's softmax mixture is *by design* a conditional mean of observed features (Prop. 1), so it is well suited to MSE and says little about multimodal futures.
- Pixel metrics are pending (Table 14). Probes are pending (Table 15). The segmentation readout shows only marginal gains over Direct (+0.6 px, CI [+0.0, +1.1] on DROID; a tie on Hamlyn).
- The "kept spatial contrast" (Fig. 11b) is a proxy for sharpness, not correctness.
- Stochastic/generative feature forecasters (VFMF, FlowWM) are cited but not compared, and uncertainty is not discussed in the limitations.

### W10. Presentation and professionalism issues
- **Anonymity/hygiene:** internal repository paths leak into the paper. App. E.1 has "results/v2/analysis/geometry/summary.json" and App. E.2 has "re-running scripts/v2/segments_all.sh". "Datasets whose ShiftWM/Direct checkpoints are not trained yet are marked pending" (p. 25) reads like a lab notebook.
- **Broken cross-references:**
  - "Appendix D.2 compares it with the bounded (tanh) correction" (p. 5), but D.2 contains no such comparison.
  - "Earlier variant … reported in Appendix D.2" (p. 20) does not exist.
  - "budgets are reduced … (Appendix C.3)" (p. 7), but C.3 does not describe DINO-WM budgets.
  - "reach their best validation error early (Appendix B.1)" (p. 21) should point to B.3.
  - C.5 is an empty section.
- **Fig. 4** caption overlaps the body text on p. 6 ("PushT, TwoRoom and Reacher with the official counts" collides).
- **Fig. 18:** thumbnails are too small to read, and the gate/transport insets are placed in the "observed" column of the Direct/ShiftWM rows, which is confusing.
- **Fig. 1(b)** uses a cherry-picked window (the largest advantage among the 30% with most motion; App. E.3). This is disclosed only in the appendix. The teaser shows MSE 0.28 vs 0.33, which misrepresents the typical ~7% gap.
- **Wording:** "AR-TF … whose teacher-forced absolute predictions drift below persistence" (p. 7) means *worse than* persistence. "Below" is confusing when lower MSE is better.
- **Section 5 vs Fig. 13:** the gain range over Direct is "+5.1% to +13.2%" in Sec. 5 but "+3% to +14%" in the Fig. 13 legend. Clarify that one is per decile and the other per cell.

### W11. Reproducibility gaps
- Compute (App. C.4) and training GPU-hours (Table 16) are pending.
- Planning dataset sizes are pending (Table 6), and the IWS license is pending.
- "Code … will be released" gives no anonymous link for reviewers.
- The DINO-WM plug-in reports a commit hash (good), but the reduced planning budgets are not specified.

---

## Questions for the authors

1. Which seeds underlie each number in Tables 1, 2, 10, 11, 17 and Fig. 1(c)? Why does ShiftWM's DROID MSE differ between Table 1 (0.181) and Table 17 (0.179)? Is Hamlyn really one seed per method?
2. Please provide Table 3. In particular: (a) "Direct + gate to Z0, no transport"; (b) global vs local window; (c) S=1; (d) no correction. Which component produces the static-patch gain in Table 17?
3. Does the fine-tuned V-JEPA 2-AC baseline see the same 3 observed frames that the ShiftWM head reads? What happens if the head reads only the frames the base model already conditions on?
4. How does ShiftWM compare to warping Z0 with (a) RAFT flow extrapolated from the past, and (b) oracle RAFT flow from the future frame, both at patch resolution?
5. Does the advantage over Direct persist when training on 10× more DROID episodes, or with a 2–4× larger predictor? Your own observation that all models overfit early suggests the gain may be a data-regime effect.
6. Does lower feature MSE translate into planning success on any benchmark? When will Table 4 and the DINO-WM success rates be available? Is action-ranking accuracy (Fig. 10) predictive of success?
7. Why does the head hurt on DINO-WM Wall? Have you tried more history frames or a larger window there? Does the Wall agent move further per step than the 7×7-patch window can reach?
8. How is Prop. 1's zero-mean innovation assumption checked empirically? The oracle-move experiment shows that only ~51% of the stay error is removable by copying. Doesn't that imply the innovation term is large?
9. How does the method relate to Šarić et al. (CVPR 2020) and to F2F? What is new beyond action conditioning and soft attention-based transport?
10. How does ShiftWM handle camera motion (the DROID wrist camera, the RT-1 mobile base)? Base-motion episodes were removed from RT-1. Is this a limitation of the local-window prior?

---

## Scores

- **Soundness: 2 / 4.** The core DROID comparison is well controlled, but the ablations are missing, seed counts are inconsistent (Hamlyn 1/3), there are no CIs for the plug-in results, the theory is tautological, and there is no downstream control evidence.
- **Presentation: 2 / 4.** The figures are polished and the writing is clear, but pervasive "pending" cells, broken cross-references, leaked internal paths, an empty section, and a cherry-picked teaser drag this down.
- **Contribution: 2 / 4.** A useful, well-executed inductive bias with modest gains (3.6–7.6%) and one strong plug-in result. The novelty relative to feature-warping forecasting is narrower than claimed, and the multi-benchmark and planning contributions are not delivered.

**Overall rating: 3 (reject, not good enough)**
**Confidence: 4** (I am confident in my assessment and familiar with the related work, but did not check every appendix derivation in full detail.)

*Justification:* The idea is sound and the DROID and V-JEPA results are promising. However, the paper as submitted is a work in progress. The ablations, planning and five of eight datasets are "pending", and the headline surgical claim rests on a single seed. A complete version with the ablations, planning results and consistent multi-seed statistics could plausibly reach 6. With convincing planning gains and a scaling result, it could reach 8.

---

## (A) What undersells the paper

1. **The static-patch result is the most surprising finding, and it is buried.** Table 17 (p. 26) shows every learned baseline is *worse than copying* on static patches (Direct 0.097, AR 0.105 vs persistence 0.090), while ShiftWM is not (0.088). This is a crisp, general insight about regression world models. It belongs in the main paper as a small table or bar plot next to Table 1, not in Appendix E.
2. **The V-JEPA 2-AC result is the strongest evidence, but it is presented in "skill" percentages that are hard to parse.** "Lowers forecast error by 15.7% (24.6% → 36.4% skill)" mixes two relative measures in one sentence. State it once as MSE 0.396 → 0.334 (−15.7%) with a CI. The gain holding at every horizon (Fig. 8c) and in every validation evaluation (Fig. 8a) is persuasive and should be in the main paper, not Appendix B.3.
3. **Action sensitivity is already a strength, but it is framed as a weakness.** Fig. 10 (left, p. 25) suggests ShiftWM has the *highest* action-ranking accuracy (~0.92) and the lowest error. The main text instead says anchoring "makes it easy to ignore actions" and adds a fix. Lead with the measured result.
4. **The teaser uses a selected window with a 15% gap (0.28 vs 0.33)**, which a sceptical reader will discount as soon as they find App. E.3. A *typical* window, or a histogram of per-window gains (Fig. 12c already shows "better on 64% of moving patches"), would build more trust. Fig. 1(c) is informative but dense, and its "open-loop error only" annotations draw the eye to the missing planning results.
5. **The novelty statement competes with itself.** "What is new" lists three properties. Contributions list five bullets, including "first". Related work lists four differences. Converge on one sentence, e.g. "we make the output of a latent world model a gated, action-conditioned soft transport of measured features", and explicitly concede the pixel-warping and feature-warping precedents. This makes the true delta, action-conditioned and plug-in for foundation-feature world models, more credible.
6. **The propositions are oversold in the abstract.** "We show that the optimal forecast under motion is itself such a transport" invites a tough theory review of what is a motivating observation. Framing them as "a motion model that motivates the design" would reduce the attack surface without losing anything.
7. **The oracle-move analysis is excellent but under-explained in the main text.** "Half of the future is already in the observed grid, displaced" (p. 9) is the best one-line motivation for the whole paper. Move it into the introduction as the empirical premise.
8. **Pending placeholders are everywhere.** Even where the text is honest, a reader sees ~15 "pending" markers and concludes the work is unfinished. Either fill them in or cut the rows, columns and figures, and scope the claims to what is complete (DROID, Hamlyn, V-JEPA, DINO-WM latent error).
9. **Mixed metrics in the headline.** The abstract uses "persistence error removed", "forecast error", "skill" and "latent prediction error". Pick one primary metric (relative MSE vs the best baseline) and one secondary (skill).
10. **The parameter efficiency is only mentioned in passing.** "<0.3% parameters" and "0.16M parameters on a 1.3B model" are strong selling points. Put them in a compact efficiency table once Table 16 is filled.

---

## (B) Top 10 concrete improvements, ranked by expected impact on the score

1. **Complete and report the ablation table (Table 3) with 3 seeds and paired CIs.** Include a new row, "Direct + gate to Z0 (no transport)", to separate gating from transport, plus global vs local window, S=1 vs 3, and correction none/tanh/scaled. Add 3–4 sentences of interpretation in Sec. 4.4. *(Resolves W1 and W6; the single largest soundness gain.)*
2. **Deliver closed-loop planning (Table 4 and DINO-WM success) for all matched predictors**, with CIs over 3 planner seeds × training seeds. Show whether ShiftWM (± action-contrastive) beats Direct and AR-TF in success rate. If it does not, say so and reframe the paper as a forecasting contribution. *(Resolves W8; a world-model paper without control evidence is unlikely to clear the bar.)*
3. **Fix the statistics and seed consistency.** Finish Hamlyn to 3 seeds. Use one seed set for all DROID tables, or label each table's seeds. Add CIs to Table 2 (V-JEPA, ideally 3 seeds) and to DINO-WM. Show the raw 10×10 decile grid instead of the interpolated contour in Fig. 13. *(Resolves W2.)*
4. **Fill in, or remove and de-claim, the pending datasets.** Report at least Bridge V2, RT-1 and DROID cam-2 in Table 1. If Language-Table and IWS are not ready, delete those columns and the corresponding sentences in the abstract, contributions (p. 2) and Sec. 4.1. *(Resolves W1; unsupported claims are heavily penalised.)*
5. **Add feature-warping baselines and cite the prior work.** Add (a) RAFT-flow-extrapolated warping of Z0 at patch resolution and (b) an oracle-flow warp upper bound. Add Šarić et al. 2020/2021, Luc et al. 2018 (F2F) and Terwilliger et al. 2019 to related work, with a precise delta statement. Also fill in Fig. 14 (transport vs RAFT). *(Resolves W5 and W6.)*
6. **Add a data and model scaling experiment on DROID.** Use 25%/50%/100%/400% of the current episodes, and optionally 2× width, for Direct vs ShiftWM. Plot the gap vs data size. *(Resolves W3. If the gap persists, it becomes a strong result; if it shrinks, it is an honest scope statement.)*
7. **Control the V-JEPA 2-AC plug-in for context and checkpointing.** Confirm or equalise the number of observed frames available to both arms. Report the final-checkpoint as well as the best-validation results. Add 1 more seed and a CI. Move Fig. 8 into the main paper. *(Resolves W7 and strengthens the best result.)*
8. **Reframe the theory.** Retitle Sec. 3.1 as a "motivating motion model". Turn "prevents compounding error" into a claim about feedback-free decoding shared with Direct. Correct "permutation" and fix the Eq. 1 glyph. Add an empirical estimate of the innovation term, e.g. the residual after the oracle move (from Fig. 12b), to link Prop. 1 to the data. *(Resolves W4 and removes easy reviewer attacks.)*
9. **Diagnose the DINO-WM Wall failure.** Run ShiftWM with S=1 on DROID, and DINO-WM Wall with more history frames or a larger window. Report which fixes the degradation. Add a short "when transport hurts" paragraph with Fig. 21-style failure examples. *(Turns a counterexample into an understood limitation.)*
10. **Clean up the presentation.**
    - Remove internal paths ("scripts/v2/…", "results/v2/…"), lab-notebook sentences and "Interim" captions.
    - Fix the broken appendix references (D.2, C.3, B.1 → B.3) and the empty C.5.
    - Fix the Fig. 4 caption overlap.
    - Enlarge the Fig. 18/19 panels.
    - Replace the teaser's cherry-picked window with a typical one or a gain histogram.
    - Move the static-patch finding (Table 17) and the oracle-move premise into the main text.
    - Define "Rank acc." in Sec. 4.
    - Fill in the compute and GPU-hours and provide an anonymous code link.

    *(Resolves W10 and W11; cheap and raises the presentation score.)*
