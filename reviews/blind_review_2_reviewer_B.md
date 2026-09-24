# Blind Review 2: Reviewer B (critical AC persona)

- **Snapshot reviewed:** `blind_review_2_paper_snapshot.pdf` (33 pages; extracted text `blind_review_2_paper_snapshot.txt`), "ShiftWM: World Models that Move What They Saw"
- **Date:** 2026-09-25
- **Reviewer:** Reviewer B (critical AC persona)
- **Scope:** Blind review. I used only the snapshot PDF and its text, with pages rendered to PNG to check the figures, plus a literature search. I opened no code, results, git history or earlier reviews.

---

## Part 1: ICLR-format review

### Summary

The paper proposes ShiftWM, an output head for action-conditioned latent world models that work on frozen foundation-model patch features (DINOv2-S/14, 16x16 grid). The head does not regress the next feature grid. For every future step k and every patch i it outputs three things:

- a soft local transport π_{k,i} over a 7x7 window in the last S=3 observed grids, computed as windowed attention with a learned identity bias;
- a sigmoid gate g that blends between keeping Z_0 and using the transported features;
- a zero-initialised additive correction r.

All K=10 horizons are decoded in parallel from measured features, and the action prefix enters through a GRU and AdaLN. The paper also describes an action-contrastive hinge loss (Eq. 3). A short theory section argues three things:

- Prop. 1: under a "motion plus innovation" model, the MSE-optimal forecast is a transport of observed features.
- Prop. 2: an additive head must output the feature contrast along the motion.
- Prop. 3: a Lipschitz bound on recursive error accumulation.

The completed experiments are:

- **Matched-backbone forecasting.** On DROID (3 seeds), ShiftWM beats Direct by 7.6% and AR by 12.2%. On Open-H Hamlyn surgical tasks (1 seed) it beats Direct by 3.6%. On Language-Table (1 seed) it beats Direct by 6.9%.
- **Plug-in to V-JEPA 2-AC.** With fine-tuning, the head lowers forecast error by 15.7% (2 seeds).
- **Plug-in to DINO-WM.** Latent error drops 8.0% on PushT and rises 18.4% on Wall.
- **Mechanism analyses on DROID:** moving/static split, a gate knockout, an "oracle move" bound, a SAM 2.1 segmentation read-out and action steering.

Many parts of the paper are still marked "pending":

- the ablations (Table 3, entirely empty);
- all closed-loop planning results (Table 4, and the DINO-WM success column);
- four of the seven forecasting columns in Table 1 (DROID cam 2, IWS, Bridge, RT-1);
- decoded-pixel metrics, probes, efficiency, and the transport-vs-flow comparison.

### Strengths

1. **Clean, sensible inductive bias with a careful matched-backbone protocol.** Every learned baseline shares the encoder, memory encoder, decoder, optimiser, schedule and parameter count (21.70M vs 21.75M). Only the output head and rollout differ. Direct has the same cross-patch attention, so "spatial communication" is controlled. This is the right way to isolate an output parameterisation.
2. **The DROID result is solid for what it claims.** It uses 3 seeds, held-out recording sessions, and a session-level paired bootstrap with Holm correction. ShiftWM is lowest at every horizon (Table 10) and beats Direct in every motion decile. The gains are moderate (7.6%) but consistent.
3. **The plug-in to V-JEPA 2-AC is the most striking result.** A 0.16M-parameter head on a 1.3B model lowers DROID forecast error by 15.7% under an identical fine-tuning budget. Validation error is lower at every evaluation step for both seeds, and the learned gate opens to around 0.65 (Fig. 9b), so the backbone actively routes most of its forecast through transport. This is the result most likely to interest the latent-WM community.
4. **Mechanism analyses go beyond a leaderboard.**
   - Test-time gate knockout: moving-patch error rises 56.8%, static-patch error only 1.8%.
   - Action swap: the transport field changes 2.8x more on moving patches than on static ones.
   - Model-free oracle-move check: copying the best observed feature removes 51% of the persistence error on moving patches.
   - Fraction of the oracle gain recovered: 79% for ShiftWM vs 71% for Direct and 69% for AR.

   Together these make a coherent case that the gain comes through the transport pathway.
5. **Unusually honest reporting.**
   - The DINO-WM Wall regression is in the abstract.
   - The marginal arm-placement gain over Direct (+0.6 px, CI touching 0) is stated.
   - Selected examples are labelled as selected, with random galleries provided.
   - V-JEPA 2-AC training-data contamination is flagged.
   - Single-seed status is disclosed.
6. **Useful, underused diagnostic.** The action-ranking accuracy in Fig. 10 (ShiftWM ~0.92, Direct ~0.90, AR ~0.81, AR-TF ~0.85) is the metric that matters for planning.

### Weaknesses

**W1. The paper is incomplete as submitted. The contribution list promises far more than the evidence delivers.**
- "Our evaluation suite has 17 benchmarks", while Fig. 4 counts "18 evaluations".
- Of the benchmarks, 4 of 7 forecasting columns in Table 1 are pending, as are all 5 planning suites and the DINO-WM success rates.
- Table 3 (ablations) and Table 4 (planning) are entirely empty in the main text, and Sections 5.4 and 5.5 each end in "pending".
- Q4 ("which components matter, and what happens in closed-loop planning?") is unanswered.
- Contribution 3 ("we set up an evaluation suite of 17 benchmarks") claims a suite that was not run.
- For a paper whose motivation is planning ("forecasting ... yields compact models that support planning"), having **no closed-loop result at all** is the biggest gap. Reviewers will read feature MSE alone as insufficient: the paper itself argues (App. D.7) that low open-loop error can coexist with poor action sensitivity.

**W2. Novelty is incremental relative to prior warp-and-correct predictors, and the paper under-positions against the closest ones.**
- Action-conditioned transformation of observed content is well established:
  - CDNA/DNA (Finn et al., 2016) are action-conditioned motion kernels with a compositing mask, i.e. the gate.
  - SNA (Ebert et al., 2017) adds skip connections to earlier frames, i.e. S>1 sources.
  - Dynamic filter networks give input-dependent local kernels.
  - SDC-Net-style spatially-displaced kernels (not cited) are a close relative.
  - F2M ("Warp to the Future", Šarić et al., 2020) already warps *features* with a forecast motion field and blends them with a directly regressed forecast, which is exactly the "move + correct" structure.
- Soft attention over a local window of past features, used to copy them forward, is also the mechanism of space-time memory and correspondence-propagation methods for video object segmentation and label propagation (e.g. STM, contrastive random walk). None of these is cited.
- The closest latent-WM competitor, DDP-WM (static/dynamic disentanglement), is cited but not compared.
- The new ingredients are real but modest:
  - frozen foundation features instead of pixels or CNN features;
  - conditioning on the action prefix with parallel horizon decoding;
  - the plug-in framing.
- The related-work sentence "Feature-warping forecasters are not action-conditioned ... while robot models warp pixels or flow" draws a narrow line. A reviewer familiar with CDNA/SNA will read ShiftWM as "CDNA/SNA in DINO space with attention-parameterised kernels". The paper needs to own this framing and show what the feature-space version buys that the pixel-space version did not.

**W3. The key control that would isolate "transport" from the "gated identity path" is missing, and the paper says so (App. E.4).**
- Direct is Z_0 + W_o h. It has no gate and no identity bias.
- ShiftWM's advantage could therefore come partly or mostly from a better-conditioned parameterisation near persistence, namely the gate initialised at b=-3 and the identity bias β=4, rather than from moving content.
- The gate knockout at test time (set g=0) is not the same thing. It removes a pathway the model was trained to rely on, so a large error increase is expected for *any* gated architecture.
- The needed baseline is a **gated regression head** trained with the same initialisation: Ẑ = (1-g)Z_0 + g·(Z_0 + W h) + r. Its absence is acknowledged as "future work", which is not acceptable for the paper's central causal claim.
- For the plug-ins, the corresponding control is **(1-g)P + g·Z_0**: a gated blend toward persistence with no transport.
- This control matters most for V-JEPA 2-AC. Fine-tuned V-JEPA 2-AC removes only 24.6% of persistence error, and Fig. 9a shows the fine-tuned baseline overfitting from about step 250 of a 3,000-step budget. A persistence-blend regulariser could plausibly deliver much of the 15.7%.

**W4. Statistical support is uneven, and several headline numbers rest on one seed.**
- Hamlyn, Language-Table and the segmentation study use 1 seed. V-JEPA uses 2 seeds. The seed count for DINO-WM is not given.
- With one seed, the "†" intervals capture episode variance only, as the paper says. The abstract nevertheless reports the Language-Table (6.9%/8.6%) and surgical (3.6%/4.0%) gains alongside the 3-seed DROID gains, with no qualifier.
- The V-JEPA "identical budget" comparison uses the baseline's default fine-tuning recipe, and that baseline overfits early. A learning-rate or regularisation sweep for the baseline arm is needed before 15.7% can be attributed to the head.
- The DINO-WM plug-in has no error bars and no seed count. Its reported gain (0.105 to 0.097) is small in absolute terms. On Wall, LPIPS nearly triples (0.009 to 0.026), which is more than "raises error by 18.4%" suggests.

**W5. Internal inconsistencies that a careful reviewer will catch.**
- **Table 1 vs Table 17.** For the "all patches" DROID test MSE, Table 1 gives ShiftWM 0.181, Direct 0.196, AR 0.206, AR-TF 0.275. Table 17 gives 0.179, 0.194, 0.203, 0.268.
- **Skill values.**
  - ShiftWM: 27.5% in the text and Table 7 vs 28.3% in Table 17.
  - AR-TF: -10.4% in Table 7 vs -7.6% in Table 17.
  - Table 17 looks like seed 0 (compare the Table 13 base row) but is not labelled as such.
- **Seed-count labels.**
  - Table 11 says "mean over horizons and 3 seeds" for Hamlyn, but the text says Hamlyn has 1 seed.
  - The caption of Table 17 (a DROID table) carries a "Hamlyn 1/3 seeds" interim note.
- **Rounding.** Fig. 1c shows 7.5 (DROID) and 15.6 (V-JEPA); the text says 7.6 and 15.7. From the rounded Table 1 values:
  - Language-Table vs Direct is 7.5%, not 6.9%.
  - Surgical vs AR is 3.8%, not 4.0%.

  These may come from unrounded numbers, but readers cannot check that. Report 4 digits or state the convention.
- **Fig. 1c labelling.** The y-axis reads "lower error than best competitor". On static patches the best competitor is persistence (a 2.1% gain), not the 8.8% shown, which is measured against Direct.
- **Possible split confusion.** The Language-Table random test gallery (Fig. 20) lists episode IDs such as `train00662_312`. This is probably the Open-X source split name, re-split by hash, but it must be explained, or readers will suspect leakage.
- **DINO-WM plug-in and "measured features".** App. B.3 says keys and values come from the predictor's input token at the same slot, with S=1. During DINO-WM's autoregressive rollout that input is the model's own prediction, not a measured token. This contradicts the claim that transport "only ever reads measured tokens" (stated for V-JEPA) and weakens the causal story for the DINO-WM numbers. It may also explain the Wall regression.
- **Faithfulness overclaim.** "The released LeWM checkpoint reaches 88% ... confirming that the planner is faithful" overstates the case: 88% is about 3 SD below the reported 96 ± 2.8. The seed argument is reasonable, but "consistent with" is the right wording.

**W6. The theory is correct but weak, and partly mis-sold.**
- **Prop. 1 does not favour transport.**
  - It shows the Bayes-optimal predictor *lies in* the transport family. But the MSE-optimal forecast is the same function, P̄_k Z_0, for *any* sufficiently expressive head, including Direct (Z_0 + W h can represent (P̄-I)Z_0).
  - So Prop. 1 gives no reason to prefer transport. The advantage must come from inductive bias, optimisation or sample efficiency, none of which is formalised.
  - The sentence "keeps it in the convex hull of observed content instead of pulling it toward a generic mean feature" suggests the optimum differs between heads. It does not. Under uncertain motion both optima are equally "blurry".
- **Prop. 1's assumptions do not match the setting.**
  - It assumes binary row-stochastic P, i.e. integer-patch motion on a 16x16 grid of 14-px patches.
  - Real motion is mostly sub-patch, and ViT patch features are not translation-equivariant at sub-patch shifts, so the "copy one source patch" model is at best approximate.
  - The support-in-window assumption (±3 patches over 3.3 s) is not checked against data. The global-window ablation that would check it is pending.
  - The proof conditions on c, while the noise assumption conditions on (P_k, Z_0, a). It needs E[ε | P_k, c] = 0, a minor gap.
- **Prop. 2** is a representational statement, and the text says so. It is not connected to any learnability or sample-complexity quantity. The prediction that "the gain should grow with feature contrast along motion" is testable: correlate per-patch gain with ‖z_{0,π(i)} − z_{0,i}‖. The paper instead bins by *true change*, which is related but not the same.
- **Prop. 3** compares an *upper bound* for recursive rollout with an *exact* quantity for direct prediction. That is not a comparison. Direct's δ_k also grows with k: Direct goes from 0.095 to 0.250 in Table 10. The proposition also does not bear on ShiftWM vs Direct, the paper's main contrast. It is textbook material and could go to the appendix.

**W7. Baselines are matched but not strong, and some look under-tuned.**
- The "DINO-WM-style" AR-TF is worse than persistence on DROID (skill -10.4%). This is used rhetorically ("recursion compounds errors"), but it more likely shows an untuned teacher-forced recipe on 3-Hz DROID than a property of DINO-WM.
- Table 1 has no explicit-flow warping baseline, e.g. predicting a displacement field and bilinearly sampling Z, as in F2M or voxel flow. Such a baseline would show whether soft attention transport matters or whether any warp would do.
- Table 1 also has no comparison to DDP-WM or Delta-JEPA, which target the same static/dynamic issue.
- All models are small (22M) and trained on about 1.1k DROID episodes, about 1.5% of DROID. Inductive-bias gains often shrink with data, so a data-scaling curve is needed before claiming generality.

**W8. The role of the action-contrastive loss is unclear.**
- Eq. 3 is presented as part of the method, and Table 8 lists λ=1 "(planning)".
- Table 3 has a row "+ action-contrastive (λ=1)", which implies the main results do not use it. If so, it should be moved out of the method section. If they do use it, the baselines are not trained with the same objective.
- The stop-gradient in Eq. 3 is on the true-action term. The loss can therefore be lowered by making forecasts under *other* actions worse, not by making true-action forecasts better. This can reward exaggerated action sensitivity. The design choice needs justification.

**W9. Presentation.**
- **Page limit.** The main text ends on page 9 with the Limitations paragraph, so the 9-page limit is met.
- **Space use.** About half of page 9 is spent on two fully empty tables and two "pending" paragraphs. The evaluation wheel (Fig. 4) takes a quarter page to list evaluations that were mostly not run.
- **Selected examples.** Fig. 1b, Fig. 2 (bottom) and Fig. 6a are all selected by "largest ShiftWM advantage". The rule is disclosed, but the first impression is best-case. A median example would build more trust.
- **Fig. 18.** Row labels are misaligned: the gate map appears in the "Direct" row, and the AR row has a blank observed cell. The random decoded rollouts are visually almost indistinguishable across methods, which says something about how the 7.6% MSE gain translates perceptually.
- **Writing.** Clear and direct. The intro makes a compelling argument through three concrete failure modes.

### Questions for the authors

1. What happens with a **gated regression head** (same gate, same identity initialisation, no transport)? And for the plug-ins, with a **gated persistence blend** (1-g)P + g·Z_0? These are the decisive controls.
2. Is L_act (Eq. 3) used in any number in Tables 1, 2 or 17? If yes, are the baselines trained with it too?
3. Why do Table 1 and Table 17 differ (0.181 vs 0.179, and so on)? Which seeds and windows does each use?
4. In the DINO-WM plug-in, does transport read measured tokens or the predictor's own rolled-out tokens? Does this explain the Wall regression?
5. How did you choose the V-JEPA 2-AC fine-tuning recipe (LR, steps)? Given the early overfitting in Fig. 9a, would a tuned baseline (lower LR, fewer steps, more regularisation) close the gap?
6. Does the gain persist with more DROID data, for example 5x or 20x the 24 shards?
7. How does soft-attention transport compare with an explicit displacement-field warp (F2M-style) under the same backbone?
8. How often does the true source fall outside the 7x7 window at k=10? The oracle-move analysis can answer this without training.
9. Why do the Language-Table test episode IDs carry a `train` prefix?
10. Did planning experiments (Table 4) run, and do the open-loop gains translate into success-rate gains? If planning is not finished, would you drop the planning framing from the abstract and intro?

### Scores

- **Soundness:** 2 (fair). The DROID and V-JEPA evidence is careful, but the central causal control is missing, most results are single-seed or pending, and there are inconsistencies between tables.
- **Presentation:** 2 (fair). The writing and figures are strong, but pending tables in the main text, inconsistent numbers and best-case-first examples hurt.
- **Contribution:** 2 (fair). The idea is sensible and useful as a plug-in, but it is incremental relative to CDNA/SNA/F2M. Its value depends on planning results that are not there.

**Rating: 3 (reject, not good enough) for this snapshot.**

This would move to 5–6 with completed ablations and a planning table that shows at least parity-plus, plus the gated-regression and gated-persistence controls. It would move to 6–8 if the V-JEPA plug-in gain survives the persistence-blend control and a tuned baseline, and if transport translates into planning success.

**Confidence:** 4. I am confident in my assessment but have not verified the code.

---

## Part 2: AC meta-view

**Likely decision: Reject (as submitted).** The consensus would likely be 3/3/5, or 3/5/5 with an enthusiastic reviewer. The core idea is liked and DROID + V-JEPA are credible. But no ICLR panel accepts a paper with a fully empty ablation table, a fully empty planning table, and a stated 17-benchmark suite of which about 4 are reported. If everything pending is filled and the key controls are added, this becomes a borderline-accept paper (5–6), with upside if planning success improves.

**Biggest risk: the central causal claim is not isolated.** "Transport, not regeneration, is why ShiftWM wins" is untested against the obvious alternative, a gated identity/persistence path with regression. The paper admits this in App. E.4. A reviewer who trains that control and finds it recovers most of the gain would reduce the contribution to "initialise near persistence with a gate". That would also explain the V-JEPA plug-in gain through regularisation toward persistence. The second-largest risk is that open-loop MSE gains of 3–8% do not translate into planning success. Without Table 4 the paper cannot answer this, and the paper's own motivation is planning.

**Secondary risks:**
- Novelty pushback citing CDNA/SNA/F2M/STM.
- Single-seed results in the abstract.
- The Table 1 vs Table 17 inconsistencies read as sloppiness.
- The DINO-WM Wall regression is left unexplained.

---

## Part 3: What undersells the paper

1. **The V-JEPA 2-AC plug-in is buried as contribution 4.** It is the most interesting result: 0.16M parameters on a 1.3B model, a 15.7% error reduction, and the backbone *chooses* to route about 65% of its forecast through transport (Fig. 9b). "Large latent WMs want to copy" could be the lead finding. Its skill gain is also flat or growing across horizons (+11.5 to +13.1 points), which is not stated in the main text.
2. **Action-ranking accuracy (Fig. 10, left) sits in the appendix.** ShiftWM has both the lowest error and the highest ranking accuracy (~0.92 vs Direct ~0.90, AR ~0.81). This directly answers the "anchored models ignore actions" objection and is the best open-loop proxy for planning available now. It belongs in the main text next to Table 1.
3. **The oracle-move analysis is a genuinely nice model-free result.** Half the future error on moving patches is recoverable by copying an observed feature, and ShiftWM recovers 79% of that. It is framed as "a check of Prop. 1" when it is really a *data finding* about foundation-feature dynamics that stands on its own and motivates the whole line of work.
4. **The 100/100 horizon-by-motion cells and the 10/10 motion deciles** are more convincing than the average 7.6%, and they appear only in a tiny Fig. 1 caption. A sign test over the 7/7 Hamlyn tasks (ShiftWM wins every task in Table 11) would likewise make the small surgical gain more credible.
5. **Interpretability is claimed but not cashed in.** The transport-vs-RAFT comparison (Fig. 14) is pending. If the learned transport correlates with optical flow without any flow supervision, that is a strong and cheap result.
6. **Efficiency is left on the table.** Parallel decoding needs one forward pass for all K steps, while AR needs K. For CEM with 300x30 samples this directly affects planning cost. Table 16 is pending, and it needs no training.
7. **The honest failure reporting is a strength.** The DINO-WM Wall result and the marginal placement gain earn credibility with reviewers, and the paper should say explicitly that it reports both.

---

## Part 4: Top 10 improvements ranked by expected score impact

1. **[NEW EXPERIMENT] Fill the closed-loop planning results.** This means Table 4 (LeWM PushT/TwoRoom/Reacher) and the DINO-WM success rates. Even parity with better efficiency is publishable, and a success gain changes the paper. Without this, the planning motivation in the abstract and intro is unsupported. *Expected impact: very high, the difference between reject and borderline.*
2. **[NEW EXPERIMENT] Add the decisive controls.**
   - (a) A gated regression head with identical gate and identity initialisation but no transport.
   - (b) For the plug-ins, a gated persistence blend (1-g)P + g·Z_0, especially on V-JEPA 2-AC.
   - (c) An explicit displacement-field warp (F2M or voxel-flow style) with the same backbone.

   If (a) and (b) fall clearly short of ShiftWM, the central claim becomes robust. *Expected impact: very high, as it addresses the biggest AC risk.*
3. **[NEW EXPERIMENT] Complete Table 3.** At minimum include: no transport, S=1, window w ∈ {3, 11, global}, action-free, ±L_act, and pure transport (r=0). The S=1 row also tests the stated explanation for the Wall regression. *Expected impact: high, since Q4 is currently unanswered.*
4. **[TEXT] Remove "pending" from the main text and make claims match the evidence.**
   - Drop "17 benchmarks" (Fig. 4 says 18) or report them all.
   - Move Fig. 4 to the appendix.
   - Qualify single-seed numbers in the abstract.
   - Resolve the Table 1 / Table 17 / Table 7 inconsistencies and the Table 11 seed label.
   - Fix the Fig. 1c labels and the rounding.
   - Explain the Language-Table `train` IDs.
   - Change "confirming the planner is faithful" to "consistent with".

   *Expected impact: high on presentation and soundness, at zero compute.*
5. **[NEW EXPERIMENT] Seeds and baseline tuning.**
   - Run 3 seeds for Hamlyn, Language-Table and DINO-WM, and a 3rd seed for V-JEPA.
   - Report seed-level spread next to the episode-bootstrap CIs.
   - Run a small LR/steps sweep for the V-JEPA fine-tuned baseline arm, given its early overfitting in Fig. 9a.

   *Expected impact: high, as it turns abstract claims from anecdotal into robust.*
6. **[TEXT] Reposition the novelty honestly.** Add an explicit comparison, as a table or paragraph, to CDNA/DNA and SNA (action-conditioned kernels + mask + skip to earlier frames), dynamic filters and SDC-Net, F2M (feature warp + regressed blend), STM-style attention memory, and DDP-WM and Delta-JEPA (static/dynamic latent WMs). Then state precisely what is new: frozen foundation features, action-prefix conditioning with parallel horizons, and a plug-in to large pretrained WMs. Also state why feature space makes copying work better than in pixels, since semantic stability under motion is the real argument. *Expected impact: medium-high; it pre-empts the "this is CDNA in DINO space" review.*
7. **[TEXT] Reframe the theory.**
   - State that the Bayes optimum is the same for every sufficiently expressive head, so the argument concerns target complexity and inductive bias.
   - Remove the convex-hull and "generic mean" implication.
   - Acknowledge sub-patch motion and the window-support assumption.
   - Fix the conditioning in Prop. 1.
   - Move Prop. 3 to the appendix, or restate it as Direct vs AR only.
   - Test Prop. 2's prediction directly: correlate per-patch gain with the feature contrast ‖z_{0,π(i)} − z_{0,i}‖, using the oracle source.

   *Expected impact: medium; prevents a theory-savvy reviewer from lowering soundness.*
8. **[ANALYSIS without new training] Finish the analyses that need no world-model training.**
   - Transport vs RAFT flow (EPE and cosine; Fig. 14).
   - Efficiency (Table 16: GFLOPs and latency; parallel vs K-step rollout).
   - Window-coverage statistics from the oracle analysis.
   - Hamlyn per-task sign test.
   - Probes (Table 15) and decoded-pixel metrics (Table 14; the decoder is trained once, independently of the forecasters).
   - Move the action-ranking accuracy into the main text.

   *Expected impact: medium; fills visible holes cheaply and strengthens the interpretability and efficiency claims.*
9. **[NEW EXPERIMENT] Data and model scaling on DROID.** Train all matched predictors on 4–20x more DROID shards, and ideally at DINOv2-B, to show whether the transport gain persists or grows as data grows. This addresses "inductive-bias gains vanish at scale", which is a standard ICLR objection for a world-model paper trained on about 1.1k episodes. *Expected impact: medium.*
10. **[FIGURE] Replace best-case-first visuals with typical ones and fix figure errors.**
    - Use a median-gain DROID window for Fig. 1b and Fig. 2, keeping the best case in the appendix.
    - Add a per-episode gain histogram next to Table 1.
    - Fix the row labels in Fig. 18.
    - Make Fig. 1c's baseline explicit per bar.
    - Show the DINO-WM Wall failure visually next to PushT.

    *Expected impact: medium-low on score, high on reviewer trust.*

---

*Literature consulted beyond the paper's own bibliography:* SDC-Net (spatially-displaced convolution for video prediction), space-time memory networks for VOS (Oh et al., 2019), contrastive random walk (Jabri et al., 2020), and recent flow-assisted latent video prediction (e.g. "Flow and Depth Assisted Video Prediction with Latent Transformer", arXiv:2511.16484), as context for positioning.
