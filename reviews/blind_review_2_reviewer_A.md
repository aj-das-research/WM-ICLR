# Blind review: "ShiftWM: World Models that Move What They Saw"

- **Snapshot reviewed:** `blind_review_2_paper_snapshot.pdf` (33 pages, including appendix)
- **Date:** 2026-09-25
- **Reviewer:** Reviewer A (ICLR reviewer persona)
- **Basis:** the paper only (rendered pages plus extracted text), plus a short web search to check novelty. No code, results or earlier reviews were consulted.

---

## Summary

The paper proposes ShiftWM, an output head for action-conditioned latent world models that operate on frozen foundation-model patch features (DINOv2-S/14, 16×16 grid). The head does not regress the future feature grid. For every horizon k and patch i it predicts:

1. local soft-transport weights π over a w×w window in the last S observed frames, with a learned identity bias;
2. a sigmoid gate g that blends between persistence (Z0) and transported features T;
3. an additive correction r.

All K horizons are decoded in parallel from a GRU action-prefix embedding, and predictions are never fed back. An optional action-contrastive hinge loss aims to stop the anchored forecast from ignoring actions. Section 3.5 gives three short propositions:

- Prop. 1: the MSE-optimal forecast under a "copy-plus-innovation" motion model is a convex transport.
- Prop. 2: an additive head must output the feature contrast along the motion.
- Prop. 3: the standard geometric error bound for recursive rollouts.

**Empirical results.** Against matched 22M-parameter predictors with the same backbone (Direct, rollout-trained AR, teacher-forced AR-TF), ShiftWM lowers held-out feature MSE:

| Dataset | vs. Direct | vs. AR | Seeds |
|---|---|---|---|
| DROID (1,126-episode subset, session split) | 7.6% | 12.2% | 3 |
| Open-H Hamlyn (7 dVRK tasks) | 3.6% | 4.0% | 1 |
| Language-Table | 6.9% | 8.6% | 1 |

As a plug-in head:

- **V-JEPA 2-AC:** 15.7% lower MSE after fine-tuning under the same budget (2 seeds).
- **DINO-WM:** 8.0% lower latent error on PushT, but 18.4% higher on Wall.

The analysis section (DROID) includes:

- a moving/static patch split;
- a test-time gate knockout;
- an oracle "best single observed candidate" analysis (51% error reduction on moving patches);
- action-swap sensitivity;
- a SAM 2.1-based arm-placement/IoU readout.

**What is still pending.** The paper announces a 17-benchmark suite (Figure 4 says 18 evaluations), but many results are marked "pending":

- all ablations (Table 3);
- all closed-loop planning (Table 4, DINO-WM success);
- Bridge, RT-1, IWS and DROID cam-2 zero-shot;
- decoded-pixel metrics, probes, efficiency, and transport vs. RAFT flow.

---

## Strengths

1. **Clear, well-motivated idea.** Separating "keep / move / correct" is intuitive for manipulation video, where most content persists. Putting it in the output parameterisation, instead of adding a new architecture, makes it cheap (+0.05M parameters, <0.3%) and usable with other models. The zero-initialised correction and negative gate bias mean an untrained model starts close to persistence, which is a sensible inductive bias.
2. **Matched-baseline design.** Direct, AR and AR-TF share the encoder, memory encoder, decoder, width, optimiser and schedule. Direct has the same cross-patch attention and the same residual-to-Z0 anchor, so the comparison isolates the output head rather than capacity (p. 6, Sec. 4; App. C.3).
3. **DROID evidence is reasonably solid.** It uses 3 seeds, a split by recording session, paired session-level bootstrap and Holm adjustment (App. C.6). The gain holds at every horizon (Table 10, p. 22), in every motion decile (Fig. 6c, Fig. 13a), and on both moving and static patches (Table 17).
4. **Plug-in result on a strong published model.** On V-JEPA 2-AC (ViT-g, 300M predictor), adding 0.16M parameters raises skill from 24.6% to 36.4% under the same budget. Validation error is lower at every checkpoint for both seeds (Fig. 9a, p. 24). The gate opens from 0.018 to ~0.65, which suggests the backbone does use transport.
5. **Unusually honest reporting in places.** The paper:
   - reports the DINO-WM Wall regression in the abstract;
   - flags the likely train/test contamination for V-JEPA 2-AC;
   - calls the +0.6 px gain over Direct "marginal";
   - labels selected qualitative examples as best-case, with explicit selection rules (App. E.3);
   - includes random galleries (Figs. 18–20) and a failure-case figure (Fig. 21).
6. **Interpretable intermediate quantities.** Transport fields, gate maps and the decomposition "moved term carries 87% of the squared departure" (App. E.1, Fig. 13c) make the model easier to inspect than a black-box regressor.
7. **Useful model-free diagnostic.** The oracle-move analysis (Fig. 12b) is a nice way to measure how much of the future is already in the observed grid, although it needs a null control (see W5).

---

## Weaknesses (in priority order)

### W1. The submission is incomplete, and the headline claims rest on a fraction of the promised evidence (critical)

The pending material covers the core questions of the paper:

- **Table 3 (p. 9), ablations:** fully pending. Section 5.4 is a single sentence ending in "pending".
- **Table 4 (p. 9), closed-loop planning:** fully pending. Section 5.5 ends in "pending".
- **Table 2 (p. 8):** DINO-WM planning success pending, and Sec. 5.2 prints "from pending% to pending%".
- **Table 1 (p. 7):** four of the eight dataset columns pending (DROID cam 2, IWS, Bridge, RT-1). Figure 5 has an empty "bridge" panel inside the main paper.
- **Appendix:** Tables 12, 14, 15 and 16 and Figs. 8, 10 (right) and 14 are pending. The compute statement (C.4) is pending, and Table 6 has pending licence and planning split entries.

Consequences:

- **Q4** ("which components matter, and what happens in closed-loop planning?") has no answer.
- **The central motivation, that these models are "used for planning", is not tested for the proposed model at all.** The paper argues (Sec. 3.4, App. D.7) that anchoring can make forecasts ignore actions and hurt planning. That is exactly the failure a planning table would reveal.
- **The "17-benchmark evaluation suite" is listed as a contribution** (p. 2, bullet 3), yet results exist for about 6 of them.

An ICLR paper has to be judged as submitted, and in its current form it cannot be accepted.

### W2. Selective or inconsistent presentation in the teaser and abstract

**Figure 1(c) (p. 1) is titled "lower error than best competitor" and has these problems:**

- It shows DINO-WM PushT (+8.0) but **omits DINO-WM Wall (−18.4)**, the one negative result. The abstract mentions Wall, but the figure that most readers will remember does not.
- **The "static" bar (8.8%) is measured against Direct, not the best competitor.** On static patches the best competitor is persistence (0.090 vs ShiftWM 0.088, Table 17), which gives a gain of about 2%. The caption says "vs. the best baseline", which is not what is plotted.
- DROID is 7.5 in the figure but 7.6% in the text, and V-JEPA is 15.6 vs 15.7%.
- **"Beats Direct in 100/100 DROID horizon-motion bins"** presents 100 strongly correlated cells as if they were 100 pieces of evidence.

**Figure 1(b) and Figure 2 show the single test window with the largest ShiftWM advantage** (App. E.3). The 0.28 vs 0.33 errors in the teaser are therefore best-case, not typical. The decoder that renders Fig. 1(b) has no quantitative evaluation yet (Table 14 pending).

**Abstract and introduction overclaim the mechanism.** "Regenerating it … blurs moving objects, corrupts static background" is stated as fact. The supporting numbers are small:

- contrast retained: 80% vs 77% for Direct (Fig. 11b);
- static-patch MSE: 0.097 (Direct) vs 0.090 (persistence).

### W3. Internal numerical inconsistencies

**Table 17 (p. 26) "all" column vs Table 1 (p. 7), both labelled DROID test.** These differ by more than rounding and have no stated provenance (single seed? different window set?):

| Quantity | Table 17 | Table 1 / Table 7 |
|---|---|---|
| ShiftWM MSE | 0.179 | 0.181 |
| Direct MSE | 0.194 | 0.196 |
| AR MSE | 0.203 | 0.206 |
| AR-TF MSE | 0.268 | 0.275 |
| ShiftWM skill | 28.3% | 27.5% |
| AR-TF skill | −7.6% | −10.4% |

**Seed counts for Hamlyn contradict each other:**

- Table 11 caption (p. 22): "mean over horizons and 3 seeds".
- Table 1 caption and Sec. 5.3: 1 seed per method.
- Table 17 (DROID-only) carries a stray "Interim: … Hamlyn 1/3" note.

**Table 11 per-task means do not reproduce Table 1's Hamlyn column.** The unweighted task mean for ShiftWM is ≈0.242 vs 0.226 in Table 1, and for persistence ≈0.491 vs 0.444. The weighting (by episode? by window?) is not stated.

**Benchmark counts disagree:** 17 benchmarks (p. 2) vs "18 evaluations" (Fig. 4, p. 6).

**Several relative gains cannot be reproduced exactly from the 3-decimal table entries.** Examples: Language-Table 6.9% vs 7.5% from the table; DROID k=10 7.3% vs 7.6%. These are consistent with rounding, but the paper should report the unrounded values or compute gains from the displayed numbers.

**Table 16 (p. 25):** "Transport adds pending parameters", although 21.75M vs 21.70M is given in the same table.

### W4. Test-set hygiene and statistical claims

**The paper says "Selection uses validation splits only; test splits are scored once" (p. 6). Yet Table 13 (p. 23) compares training recipes on test MSE.** The recipes are 8k vs 16k steps, second camera + EMA + dropout, and correlation features. App. D.2 says the next experiment was chosen *because* the 8k-step test error barely changed. This is test-set model development and should be moved to validation.

**Seeds are too few, and seed variance is never reported:**

- Hamlyn, Language-Table and the segmentation study use 1 seed.
- V-JEPA 2-AC uses 2 seeds.
- DINO-WM seed count is unstated.
- Bootstrap intervals over episodes, averaged over seeds, ignore training-seed variance. The gains that are statistically weakest (Hamlyn 3.6%, Language-Table 6.9%) are exactly the single-seed ones. The † on those columns only means episode noise is excluded, not that the result is robust to retraining.

**Table 1 shows only daggers, never the intervals or Holm-adjusted p-values.** The effect sizes with CIs belong in the main table.

**Number of DROID sessions not reported.** Only 1,126 episodes from 24 shards are used, with 132 test episodes. Resampling sessions is correct, but with few sessions the percentile bootstrap may be unreliable. Report the number of test sessions.

**DINO-WM plug-in:** no seeds, CIs or episode counts. In Table 2 the PushT SSIM "win" of 0.977 vs 0.976 is bolded.

### W5. Several analyses are weaker than presented

**The gate-knockout "causal" test (Sec. 5.3 p. 8; Fig. 6b; contribution bullet p. 2) is a train/test mismatch, not a controlled ablation.**

- Setting g=0 at test time for a model whose correction r was trained to work alongside transport will inflate error almost by construction.
- The +56.8% moving-patch number is quoted in the contributions as "removing transport raises moving-patch error by 56.8%".
- The properly retrained counterpart (Direct) is only 7.8% worse on moving patches (0.484 vs 0.449). The retrained no-transport row of Table 3 is pending.
- Quote the retrained number, or at least describe the knockout honestly.

**The oracle-move analysis ("half of the future is already in the observed grid", p. 8; Fig. 12b) has no null control.** Picking the single candidate out of 147 that is closest to the target will reduce MSE even when there is no motion structure (a selection effect over many noisy high-dimensional candidates). Needed comparisons:

- the same oracle applied to candidates from a *different* episode;
- the same oracle with spatially shuffled windows.

Only the excess over those tells us that "the future is a transport".

**The narrative of an "identity path" on static patches is not supported by the model's own statistics.**

- Fig. 13c (p. 27) shows the mean gate on static patches rising to ≈0.6 by k=10, so static patches are mostly *transported*, not kept.
- The static-patch advantage over persistence is small: 0.088 vs 0.090.
- The paper itself admits (App. E.4, "Gate versus transport") that a gated regression head without transport is needed to separate the gate from transport. **That baseline is the single most important missing control:** without it, one cannot tell whether transport or simply the gate/anchor produces the gains.

**The action-sensitivity evidence is thin:**

- The 2.8× "actions steer moving parts" (Fig. 6d) compares swapped actions from another episode. These are easy negatives.
- The action-ranking accuracy (Fig. 10 left: ShiftWM ≈0.92, Direct ≈0.90, AR ≈0.81) has no CIs.
- Planners discriminate between *nearby* action sequences, so fine-grained sensitivity (perturbations of a few cm) is what matters.
- It is not stated whether the main ShiftWM results use the action-contrastive loss (Table 8 says "λ 1.0 / margin 0.05 (planning)"; Table 3 lists it as a variant).
- There is no action-free ShiftWM result (Table 3 pending). With H=3 history frames, a large part of the transport gain could come from extrapolating visible motion rather than from action conditioning, which matters for a world model.

**The segmentation readout gain over Direct is negligible:**

- DROID: +0.6 px, CI [+0.0, +1.1]; tied at k=10.
- Hamlyn: IoU +0.1, CI [−0.2, +0.4].

The main-text sentence reports "+2.1 px vs AR" in highlighted colour and the Direct comparison in plain text.

### W6. Baseline strength and external comparisons

**AR-TF ("DINO-WM-style") is worse than persistence on DROID** (skill −10.4%) and is used to support the claim "a DINO-WM-style predictor … is worse than persistence" (p. 2). A reimplementation that loses to copy-last-frame looks under-tuned rather than representative of DINO-WM. DINO-WM itself was run only on its own simulated suites.

**All forecasting baselines are the authors' own heads on one backbone.** There is no comparison with:

- published feature forecasters on the same data (DINO-Foresight, DINO-world, VFMF);
- the closest structural competitors:
  - **F2M** (Šarić et al. 2020), which warps features and blends them with a regressed forecast, essentially the unconditioned version of Eq. (2);
  - **DDP-WM** (Yin et al. 2026), which separates static and dynamic latent regions, reports 98% PushT success, and is cited but not compared.

  An action-conditioned F2M-style baseline (explicit flow-field warp + blend) would test whether *soft attention transport* matters or any warp would do.

**V-JEPA 2-AC fine-tuning protocol looks untuned.**

- Fig. 9a shows both arms reach their best validation error at the first evaluation (≈250 steps) and overfit afterwards. Both test numbers are therefore effectively "250-step fine-tunes" at a learning rate that is too high for the baseline.
- A per-arm learning-rate sweep (or lower LR, or early stopping) is needed before a 15.7% gain can be attributed to the head.
- Zero-shot skill of 4.1% for V-JEPA 2-AC suggests a mismatch in action convention or frame rate (V-JEPA 2-AC was trained at 4 fps with 7-D deltas; here 35-D 5-step blocks at 3 Hz). Please explain.

### W7. Data and protocol concerns

**Language-Table (App. C.2, p. 20–21).**

- The "episodes" are hindsight-labelled segments (median 12 native steps) of longer continuous play. They are split by a hash of the segment ID.
- Adjacent segments of the same underlying episode can land in both train and test, which is a leakage risk. The gallery IDs such as `train00662_312` in Fig. 20 are labelled test rollouts but come from the original train split.
- The horizon is 1 s (K=10 at 10 Hz), not 3.3 s as on DROID. The abstract compares gains across datasets without mentioning this.

**DROID scale.** Only ≈1.5% of DROID (1,126 episodes) is used. Table 13 shows 8k vs 16k steps barely matters, which suggests a data-limited regime where a copy-based inductive bias should help most. It is unclear whether the gain survives at scale.

**Surgical data are scene-camera views of phantom training tasks**, not endoscopy. This is acknowledged, but "surgical robot tasks" in the abstract suggests more than this.

**Feature error in standardised DINOv2 space is the only real-robot metric with results.** Pixel metrics, probes and planning are pending. Linear extrapolation at MSE 4.8–11.9 (Table 1) adds nothing and should be removed or explained.

### W8. Theory is light and partly mis-scoped

- **Props. 1–3 are elementary.** They are the conditional-mean property, a one-line algebraic identity, and the textbook recursive error bound (Venkatraman et al. 2015).
- **Prop. 1 assumes features are translation-equivariant** (a moving object's patch equals an observed patch elsewhere). DINOv2 patch features carry positional and contextual information, which limits this. The oracle analysis partly addresses the premise, but the text should say so.
- **Prop. 3 does not separate ShiftWM from Direct**, which are both direct. It explains only the AR comparison, which is not the novel contribution.
- **"Can be approximated to arbitrary precision … given logits the decoder can express"** hides the question of whether a finite-temperature softmax with identity bias β=4 can represent sharp transports. This is not a real guarantee.
- I would present Sec. 3.5 as motivation and move the "propositions" framing to the appendix.

### W9. Presentation and readability

- **Figure 4 (evaluation wheel, p. 6):** unreadable at print size, and most of its wedges are pending.
- **Figure 13a,b (p. 27):** cubic-interpolated contours of a 10×10 grid produce smooth artefacts. Contour labels (e.g. "6, 10") are illegible. A plain heatmap with cell values would be more honest.
- **Figure 18 (p. 31):** row labels ("true/AR/Direct/ShiftWM") appear offset from the rows. The gate map sits in the "Direct" row and the "transport" inset in the ShiftWM row's observed column. Thumbnails are too small to read.
- **Figure 1(b) and Fig. 16/17:** decoded frames are blurry for all methods, so the visual difference is not convincing.
- **Mismatched names:** "Hamlyn" vs "Open-H surgical" vs "Open-H Hamlyn", used interchangeably.
- **Placeholder text in the main body:** grey "pending" tokens in running sentences (pp. 7, 9).
- **The contribution list duplicates the abstract numbers.** One bullet re-quotes the knockout 56.8% (see W5).

---

## Questions for the authors

1. Where does Table 17's "all" column come from, and why does it differ from Table 1 (0.179 vs 0.181; AR-TF 0.268 vs 0.275)? How are Table 1's Hamlyn numbers aggregated from the per-task Table 11, and how many seeds does Table 11 really use?
2. Do the main-table ShiftWM models use the action-contrastive loss (λ>0)? If yes, do Direct and AR also get it? If not, what is the action-ranking accuracy of each method with CIs, and with fine-grained (nearby) negatives?
3. What is the test error of a **gated regression head** (Eq. 2 with T replaced by Z0 + W h, or with a learned gate and no transport)? And of an **action-free** ShiftWM?
4. What does the oracle-move reduction look like with candidates from a different episode, or with shuffled spatial locations (null control)?
5. How many recording sessions are in the DROID test split? What are the seed standard deviations for every Table 1 entry?
6. For V-JEPA 2-AC: which learning rate was used, was it tuned per arm, and what happens at a lower learning rate where the baseline does not overfit after 250 steps? Why is zero-shot skill only 4.1%?
7. Why do you think the DINO-WM Wall result is negative, beyond the S=1 hypothesis? Is transport on a single frame simply an identity-biased copy that fights the backbone's own prediction? Please report seeds and CIs for both DINO-WM environments.
8. Language-Table: are segments from the same underlying play episode kept within one split? What happens with an episode-level split?
9. Why was test MSE used for the recipe study (Table 13), given the statement that tests are scored once?
10. On static patches the mean gate reaches ≈0.6 at k=10 (Fig. 13c). What does transport do there? Is the transport weight concentrated on the identity location, and so effectively a copy?
11. How sensitive are results to window size w, and what happens with large motions or camera ego-motion (RT-1 base motion was removed)?
12. Planning: when will Table 4 be filled in? Does ShiftWM's lower open-loop error lead to higher CEM success, or does anchoring hurt?

---

## Scores

- **Soundness: 2 / 4.** The core DROID comparison is carefully matched, but key controls are missing (gated no-transport head, action-free variant, null oracle, tuned plug-in baselines). There is test-set recipe selection, numbers that do not agree across tables, and single-seed claims on the weaker datasets.
- **Presentation: 2 / 4.** Well written in prose, but pending cells and placeholder text run through the main paper. The teaser is selective (Wall omitted; static bar vs the wrong reference), and several figures are unreadable.
- **Contribution: 2 / 4.** A reasonable, cheap and interpretable head with a consistent DROID gain and an encouraging V-JEPA 2-AC plug-in. However, warp-and-correct in feature space is established (F2M, CDNA/SNA, DDP-WM-style static/dynamic separation), and the planning value, the main purpose of such models, is not shown.

**Rating: 3 (reject, not good enough)**

**Confidence: 4** (I know this literature well; I did not check proofs beyond reading them, and did not see code).

The idea is sensible and parts of the evidence are well done. But the paper as submitted has empty ablation and planning tables, a mixed plug-in record, and several presentation choices that overstate the result. With the pending experiments completed and the controls in W5 added, it could plausibly reach the 5–6 range.

---

## (A) What undersells the paper

1. **Action-ranking accuracy is buried in the appendix.** Fig. 10 (left, p. 24) shows ShiftWM is *both* lowest-error *and* most action-sensitive (≈0.92 vs 0.90 Direct, 0.81 AR). This directly answers the obvious objection that anchoring to Z0 lets the model ignore actions, and it deserves a main-text sentence with CIs.
2. **The V-JEPA 2-AC plug-in is the most convincing result in the paper, and is under-exploited.**
   - The gain grows with horizon (+11.5 to +13.1 skill points, App. D.3).
   - The backbone routes about 65% of its forecast through transport.
   - It is lower at every validation checkpoint.

   This should be the second headline, with Fig. 9 moved to the main paper, rather than being shown alongside the weak DINO-WM result.
3. **The honest decomposition (moved term = 87% of the departure; gate rises from 0.51 to 0.85 on moving patches, Fig. 13c) is good mechanistic evidence**, but it is scattered across appendix text. A compact "mechanism" panel in the main paper would help.
4. **Per-task Hamlyn results (Table 11): ShiftWM is best on all 7 tasks.** That is more persuasive than the single aggregate 3.6%, which reads as "small".
5. **Per-horizon DROID (Table 10): ShiftWM is best at every k, including k=1.** This argues against the view that it wins only by avoiding recursion.
6. **Parameter overhead (<0.3%) and parallel decoding** are practical advantages for CEM planning. With a latency number (Table 16) they would become a concrete efficiency claim.
7. **The selection rules and random galleries are good practice**, but they are hidden. A one-line statement in the Fig. 1 caption ("best-case window; random examples in Fig. 18") would turn a reviewer-alarm into a credibility point.

---

## (B) Top 10 concrete improvements, ranked by expected score impact

1. **[NEW EXPERIMENT] Fill in closed-loop planning (Table 4 and DINO-WM success in Table 2)** with 3 seeds and CIs, including ShiftWM with and without the action-contrastive loss. Without this, the "world model for planning" framing is unsupported. This is the largest single driver of the score.
2. **[NEW EXPERIMENT] Complete the ablations (Table 3), prioritising three controls:**
   - (a) a gated regression head with no transport;
   - (b) action-free ShiftWM;
   - (c) window size / S=1 (which also tests the Wall hypothesis).

   These decide whether transport, rather than gating or history extrapolation, drives the gain.
3. **[TEXT] + [FIGURE] Fix the selective and inconsistent teaser:**
   - add the Wall bar (−18.4) to Fig. 1(c);
   - measure "static" against persistence (the actual best competitor), or relabel the axis;
   - make the figure numbers match the text (7.6, 15.7);
   - drop "100/100 bins";
   - state in the caption that Fig. 1(b) is the best-case window.
4. **[ANALYSIS without new training] Reconcile all numbers:**
   - Table 17 vs Table 1/7;
   - Hamlyn seed count (Table 11 vs Table 1);
   - Table 11 → Table 1 aggregation;
   - 17 vs 18 benchmarks;
   - Table 16 "pending parameters".

   Report unrounded relative gains, add per-entry seed std and the actual paired CIs to Table 1, and report the number of DROID test sessions.
5. **[NEW EXPERIMENT] Run a fair V-JEPA 2-AC fine-tuning protocol:** a per-arm LR sweep (or lower LR) chosen on validation, 3 seeds, and an explanation or fix for the 4.1% zero-shot skill (action/frame-rate convention). If the 15.7% survives, it becomes a strong result.
6. **[ANALYSIS without new training] Add a null control to the oracle-move analysis** (candidates from another episode or shuffled positions). Replace the train/test-mismatched gate knockout (56.8%) in the contributions with the retrained comparison (Direct, 7.8% on moving patches), or clearly label the knockout as non-retrained.
7. **[NEW EXPERIMENT] Add seeds to Hamlyn and Language-Table (3 each), and make the Language-Table split grouped by underlying play episode.** Move the recipe study (Table 13) to validation, or state openly that test was used there.
8. **[NEW EXPERIMENT] Add stronger external or structural baselines on DROID:**
   - an action-conditioned F2M-style explicit flow warp + blend;
   - a tuned DINO-WM predictor that at least beats persistence;
   - DDP-WM on PushT planning.

   This addresses the novelty and strawman concerns.
9. **[ANALYSIS without new training] Report action sensitivity with CIs and hard negatives:**
   - ranking accuracy against *perturbed* versions of the true action (e.g. ±1–5 cm);
   - transport change per unit action change.

   Promote Fig. 10 (left) to the main text.
10. **[FIGURE] + [TEXT] Clean up presentation:**
    - remove every "pending" placeholder from the main body (or drop unfinished benchmarks from the claims, rescoping "17 benchmarks" to what is reported);
    - replace the Fig. 4 wheel with a compact table;
    - redraw Fig. 13a,b as plain heatmaps with values;
    - fix the row labels and enlarge the thumbnails in Figs. 18–20;
    - move Props. 1–3 to "motivation" wording and discuss the translation-equivariance assumption of DINOv2 features;
    - remove the Linear-extrapolation row.

---

*Novelty check (web):* DDP-WM ([arXiv 2602.01780](https://arxiv.org/abs/2602.01780)) already separates sparse dynamic regions from background in DINO-feature world models and reports strong PushT planning. RLA-WM ([arXiv 2605.07079](https://arxiv.org/pdf/2605.07079)) and DINO-world ([arXiv 2507.19468](https://arxiv.org/pdf/2507.19468)) are close latent-feature forecasters. I did not find prior work doing action-conditioned *soft local transport of frozen foundation-model patch features* as an output head, so the specific combination appears new, but its components are established.
