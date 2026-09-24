# Blind review: `blind_review_1_paper_snapshot.pdf`

**Date:** 2026-09-24
**Reviewer:** Reviewer B (critical AC persona)
**Paper:** "ShiftWM: World Models that Move What They Saw" (anonymous ICLR 2027 submission snapshot, 32 pages including appendix)
**Basis:** The PDF only. I rendered all pages and checked figures and tables visually. I also searched the literature for novelty. I looked at no code, results or repository files.

---

## 1. ICLR-format review

### Summary

The paper proposes ShiftWM, an output parameterization for action-conditioned latent world models that operate on frozen DINOv2 patch features. It keeps the standard memory encoder and a query decoder conditioned on a GRU action-prefix state. For each horizon k and patch i it then predicts:
- a local softmax "transport" over a 7x7 window in the last S=3 observed feature grids, with an identity bias;
- a sigmoid gate that blends the transported features with the last observed grid Z0;
- an additive correction r.

Every horizon is decoded in parallel from measured features, with no recursive feedback. An action-contrastive hinge is added for planning.

Three propositions motivate the design:
1. Under a "permutation plus zero-mean innovation" model, the MSE-optimal forecast is a transport.
2. An additive head must output the feature contrast along the motion.
3. Recursive rollouts accumulate error geometrically.

Reported results:
- **DROID (held out by session, DINOv2-S, 3 seeds):** ShiftWM reaches 0.181 feature MSE, against 0.196 for a matched direct residual head (−7.6%) and 0.206 for rollout-trained AR (−12.2%).
- **Open-H Hamlyn dVRK (seven tasks, currently 1 of 3 seeds):** −3.6% against Direct.
- **Plug-in heads:**
  - V-JEPA 2-AC fine-tuned on DROID: skill rises from 24.6% to 36.4% (2 seeds).
  - DINO-WM PushT: latent error −8.0%.
  - DINO-WM Wall: +18.4%, i.e. worse, with LPIPS roughly 3x worse.
- **Analysis:** gate knockout, gains by motion decile, an oracle "best single observed feature" check of Prop. 1, and a SAM 2.1-based arm-placement readout.

A large share of the promised evidence is still marked "pending":
- 5 of the 7 columns of Table 1 (IWS, Bridge, RT-1, Language-Table, DROID cam-2);
- the entire ablation table (Table 3);
- the entire planning table (Table 4) and the DINO-WM planning success rates;
- decoded-pixel metrics (Table 14) and probes (Table 15);
- the transport-vs-RAFT comparison (Fig. 14);
- efficiency (Table 16).

### Strengths

1. **Clear and well-motivated idea, cleanly implemented.** Writing the forecast as keep + move + correct over measured features is intuitive. The zero-init/persistence-start design (b_g=−3, zero-init W_o) is sensible, and the decomposition is exactly additive, so it is inspectable (Fig. 2 bottom, Fig. 13c).
2. **Careful matched-baseline design for the core claim.** Direct, AR and AR-TF share encoder, decoder, width, depth, optimizer, schedule and steps, and differ by 0.05M parameters (21.75M vs 21.70M). Direct has the same cross-attention, so the comparison isolates the output head better than most latent-WM papers do.
3. **Honest statistical reporting where results exist.** The paper uses:
   - session-level paired bootstrap for DROID and Holm correction;
   - explicit selection rules for qualitative examples, with random galleries (Figs. 18–19) and failure cases (Fig. 21);
   - an admission that the Wall plug-in hurts, and that the Direct placement gain is marginal (CI [+0.0, +1.1] px).

   This is more self-critical than typical submissions.
4. **Consistent effect on DROID.** ShiftWM is best at every horizon (Table 10) and has lower error than Direct in all 100 horizon×motion-decile cells. It is also the only learned model below persistence on static patches (Table 17). These results support a real, if modest, advantage.
5. **Strong mechanism analyses.**
   - Gate knockout: +57% error on moving patches vs +2% on static.
   - Action-swap steering: 2.8x.
   - The model-free oracle-move check (51% of stay-error on moving patches is recoverable by copying one observed feature; ShiftWM recovers 79% of that vs 71% for Direct) is a nice data-level argument that "the future is largely a displacement of the observed grid".
6. **The V-JEPA 2-AC plug-in result is striking.** +11.8 skill points at 0.16M parameters, with the gate opening to about 0.65 within 250 steps (Fig. 8). If it holds up under proper controls, it is the most practically interesting finding in the paper.
7. **Uncommon breadth of domains**, including surgical dVRK data, which is rarely used for latent WMs.

### Weaknesses

**W1. The paper is incomplete; much of the claimed evidence is placeholders.** Pending items: the ablations (Table 3, all cells), closed-loop planning (Table 4, all learned rows), DINO-WM success rates, four of six real-robot datasets, decoded pixels, probes, the RAFT comparison and efficiency. In addition, Hamlyn has only 1 of 3 seeds, yet the Table 1 caption says "mean of 3 seeds" and marks † significance.

The abstract and contributions claim evaluation "on IWS, BridgeData V2, RT-1, Language-Table and three planning suites". Section 1 promises a planning cost analysis ("what it costs for control"), and the Mechanism bullet claims the action-contrastive objective "keeps anchored forecasts sensitive to actions for planning". None of this is supported in the current PDF. As submitted, I would have to evaluate a DROID-plus-partial-Hamlyn forecasting paper. For a world-model paper at ICLR, having no planning results is close to disqualifying.

**W2. Novelty is overstated relative to feature-space warping forecasters.** The related-work section frames transport as a pixel-space idea (CDNA/DNA, dynamic filters, voxel flow, SNA). It misses the directly relevant line of work that forecasts by warping *observed deep features* with predicted feature motion:
- Šarić et al., *Warp to the Future: Joint Forecasting of Features and Feature Motion*, CVPR 2020 (F2M: forecasts by warping observed features with regressed feature flow, fused with a direct F2F branch, plus correlation features);
- Šarić et al., *Single Level Feature-to-Feature Forecasting with Deformable Convolutions*, GCPR 2019;
- Šarić et al., *Dense Semantic Forecasting in Video by Joint Regression of Features and Feature Motion*, TNNLS 2021;
- Luc et al., F2F (ECCV 2018);
- Terwilliger et al., *Recurrent flow-guided semantic forecasting* (WACV 2019);
- Zhu et al., *Deep Feature Flow* (CVPR 2017);
- Gao et al., *Disentangling Propagation and Generation for Video Prediction* (ICCV 2019), which also uses a propagate-vs-generate occlusion gate.

A critic will say: "ShiftWM is F2M (warp observed features + regress residual + blend) with soft local attention instead of bilinear flow, applied to DINOv2 features and conditioned on actions." That makes the statements "the idea has not been tested for foundation-feature world models" and "to our knowledge the first" narrow claims that need precise scoping.

Other missing prior work:
- Kulkarni et al.'s *Transporter* (NeurIPS 2019), which literally "transports" features between frames for control; the name collision alone will be raised.
- Flow-as-interface robot work: AVDC (Ko et al., ICLR 2024), Track2Act, ATM, Im2Flow2Act, FLIP, and *Future Optical Flow Prediction Improves Robot Control & Video Generation* (arXiv 2601.10781).
- The multi-step "direct vs. iterated" forecasting and multi-step dynamics model literature (Venkatraman et al. 2015; Asadi et al. 2019; Lambert et al. 2021), which already makes the Prop. 3 argument.

**W3. The main effect is modest, and the attribution to "transport" is not yet isolated.** Against the strongest matched baseline, the gain is 7.6% on DROID and 3.4–3.6% on Hamlyn (0.226 vs 0.234; the text says 3.6%, the table rounds to 3.4%). Segmentation placement gains over Direct are within about 0.6 px on DROID and tie on Hamlyn.

More importantly, two components are confounded: (a) transport, and (b) a gated identity path to Z0, i.e. a better skip connection. The Direct baseline is Z0 + W_o h with no gate. Missing controls that would separate (a) from (b):
- **Gated-Direct:** (1−g)Z0 + g(Z0 + W h), or (1−g)Z0 + g·W h, with the same b_g init.
- **Global rather than local transport** (listed but pending).
- **Transport with values = projected memory M** rather than raw Z.
- **An explicit bilinear-flow warp head** (F2M-style) with the same budget.

The static-patch result, where ShiftWM is the only model below persistence, suggests much of the win is the gate/identity path rather than "motion". A gated regression head could plausibly match it.

**W4. The plug-in claims have control and fairness problems.**
- **V-JEPA 2-AC.**
  - The head gives an autoregressive predictor fresh access to *measured* tokens of the last 3 frames at every horizon, which the base model's rollout does not have. The "copy last observed tokens" control, (1−g)P + g·Z0, is needed.
  - Only 2 seeds and no CI in Table 2 (Table 7 gives a CI only for zero-shot).
  - Both arms overfit after ~250 steps on an 851-episode split (Fig. 8a). The baseline fine-tune (LR, steps, regularization) looks untuned, so the "identical budget" may favor the arm with an easy-to-use persistence shortcut.
  - The authors themselves note that V-JEPA 2-AC's training data "likely include our test trajectories".
  - Zero-shot skill of 4.1% suggests the action convention or preprocessing may not match the released model.
- **DINO-WM.**
  - Seeds are not stated.
  - The Wall result gets worse on every metric (latent +18.4%, LPIPS 0.009→0.026). The explanation offered ("single history frame") is a hypothesis, not a test.
  - Success rates are pending, so the claim "improves DINO-WM on PushT" rests on a 0.008 latent-error difference.
- The abstract and teaser present these results with more weight than the evidence supports.

**W5. The "DINO-WM-style" AR-TF baseline is weak and is used rhetorically.** AR-TF is worse than persistence on DROID (skill −10.4%), and Linear extrapolation is 20x worse than persistence (4.812). Both are straw baselines. Headline comparisons such as "34.3% lower than DINO-WM-style" and the teaser's AR-TF tile (0.47) should not be framed as comparisons to DINO-WM. The real DINO-WM uses its own frameskip, history and proprioception, and is not trained on DROID.

There is also no strong *external* forecasting baseline in DINO feature space trained on the same data at matched capacity: DINO-Foresight, DINO-world, VFMF, FlowWM (flow matching in feature space), DDP-WM, RLA-WM (whose IWS handles are used, yet no RLA-WM numbers appear) or Delta-JEPA. Deterministic MSE comparisons also sidestep stochastic predictors entirely.

**W6. The theory is shallow and only partly relevant to what distinguishes ShiftWM.**
- **Prop. 1** is the conditional-mean identity under an assumed linear permutation model with E[ε|·]=0. "Approximated to arbitrary precision given logits the decoder can express" requires logits → −∞, which is a statement about expressivity, not learnability. The model also implies that the MSE-optimal forecast *under motion uncertainty is a blur* (a mixture P̄Z0). This sits awkwardly with the "transport keeps forecasts sharp" narrative (Fig. 11b).
- **Prop. 2** measures the *norm of the target residual*, not a learning difficulty. A Direct head with cross-attention to M (which contains projected Z0) can implement a copy internally, so the proposition does not show that additive heads are harder to learn, need more capacity, or generalize worse. A sample-complexity or approximation-with-finite-width statement would be needed.
- **Prop. 3** is the standard Lipschitz rollout bound. It applies equally to the Direct baseline, which also predicts all horizons in parallel from measured features. It therefore does not explain ShiftWM vs Direct, the comparison that matters.

The abstract's "anchoring every horizon to measured features prevents compounding error" is presented as a ShiftWM property, but it is shared by the Direct baseline and by the classical direct multi-step forecasting strategy. None of the three propositions is tested quantitatively. For example, the paper does not show that the per-patch gain over Direct scales with ||z0,π(i) − z0,i||² as Prop. 2 would predict (Fig. 10 right is pending).

**W7. Several numbers are inconsistent across tables and figures.**
- Table 1 reports DROID MSE 0.181 / 0.196 / 0.206 / 0.275, while Table 17 reports "all" = 0.179 / 0.194 / 0.203 / 0.268 for the same test set.
- Skill is 27.5% in the abstract and Table 7 but 28.3% in Table 17.
- AR-TF skill is −10.4% in Table 7 but −7.6% in Table 17.
- The teaser gives "static patches −8.8%", but Table 17 gives (0.097−0.088)/0.097 = 9.3% vs Direct.
- Hamlyn is called "3 seeds" while being 1 of 3.

These are probably seed-0 vs 3-seed discrepancies, but they need to be reconciled.

**W8. Presentation and formatting issues that could get the paper desk-rejected.**
- The Conclusion starts on **page 10**, which appears to exceed the 9-page main-text limit.
- The Figure 4 caption visibly overlaps body text ("counts." collides with "PushT, TwoRoom…", p. 6).
- Visible "pending" boxes remain in main-text Fig. 5 (Bridge panel), Table 3, Table 4 and §4.4/§4.5, which end in the bare word "pending".
- The appendix leaks internal paths ("results/v2/analysis/geometry/summary.json", "scripts/v2/segments_all.sh", "not trained yet… re-running"). This is unprofessional, and depending on the repository it could be an anonymity risk.
- Row labels in the gallery figures (Figs. 18–19) are misaligned: the "AR" row shows a blank tile and gate/transport overlays sit in the Direct/ShiftWM rows, so the figures are hard to read.
- The teaser (Fig. 1) is dense: three panels with about 25 small labels, and the decoded tiles depend on an unvalidated decoder whose metrics (Table 14) are pending.
- The teaser and Figs. 2/6/11 show *best-case* windows (largest ShiftWM advantage). This is disclosed, but it is still cherry-picked imagery on page 1.

**W9. Evaluation scope versus claims.**
- Everything is measured in standardized feature MSE of a frozen DINOv2-S at 16×16 resolution.
- The paper does not show that lower feature MSE yields better control. Planning is pending, and the paper itself notes that anchored models can ignore actions, which is why it adds L_act.
- The action-contrastive loss changes the training objective, so any planning comparison needs all baselines trained with L_act as well.
- The surgical data are scene-camera views of phantom tasks, not endoscopy, as the authors acknowledge.
- H=3, K=10 at 3 Hz is a single operating point, and there is no longer-horizon or higher-frame-rate test. Local 7×7 windows at 14-px patches bound the displacement per step; failure under fast motion or camera motion is not characterized.

### Questions for the authors

1. What is the error of a **gated Direct head** ((1−g)Z0 + g·(Z0+W_o h), same gate init) and of a head that uses **attention over projected memory values** instead of raw Z? How much of the 7.6% gain survives?
2. For V-JEPA 2-AC, what happens with a head that gates to **measured last-frame tokens only** (no transport)? Was the baseline fine-tune tuned (LR, steps, early stopping)? Can you report ≥3 seeds with paired CIs, and test episodes provably absent from V-JEPA 2-AC's training set?
3. How does ShiftWM compare to an **F2M-style warp head** (Šarić et al., 2020) or a bilinear-flow warp of DINO features with the same backbone?
4. Why does the plug-in hurt DINO-WM Wall on all metrics? Does S>1 history fix it, or is it a failure of local transport when a single agent moves through a textureless scene?
5. Do planning success rates follow the forecasting ranking? Are Direct and AR also trained with L_act in Table 4?
6. Please reconcile Table 1 vs Table 17 vs Table 7 (0.181 vs 0.179; 27.5% vs 28.3%; −10.4% vs −7.6%).
7. How does the gain over Direct scale with per-patch feature contrast ||z0,π(i)−z0,i||²? This is the quantitative test of Prop. 2.
8. What happens beyond the window (fast motion, camera motion in RT-1/Bridge), and what does the transport field do under ego-motion?
9. How are Hamlyn significance daggers computed with a single seed?

### Scores

- **Soundness: 2 / 4.** The core DROID comparison is careful, but key controls are missing (gated-Direct, copy-only plug-in), plug-in seeds are few, numbers are inconsistent, and the planning and ablation evidence is absent.
- **Presentation: 2 / 4.** The writing and figures are good in places, but pending placeholders appear in the main text, the paper likely exceeds the page limit, a caption overlaps text, and internal paths leak.
- **Contribution: 2 / 4.** A reasonable but incremental reparameterization; prior feature-warping forecasters are not cited, and gains are modest.

**Rating: 3 (reject, not good enough)** in the current snapshot. If the pending results land and support the claims, with planning gains, clean ablations and a copy-only control on the V-JEPA plug-in, I would move to 5–6.

**Confidence: 4 / 5.**

---

## 2. AC meta-view

**Likely decision: reject in the current state.** With the pending tables filled and the missing controls added, this becomes a borderline paper (5/6 average) that could go either way depending on planning results.

The reviewing pool for latent WMs will value the matched-baseline discipline and the V-JEPA plug-in number. However, at least one reviewer will raise the prior feature-warping work (Šarić et al. F2M; Deep Feature Flow; Transporter). Another will object that there are no planning results in a paper whose premise is world models "used for planning". The Wall regression and the 3.6% surgical gain from a single seed will read as mixed evidence.

**Single biggest risk:** the paper is not a complete submission. The ablations, all planning results and most datasets are "pending", the main text appears to run past 9 pages, and internal file paths are visible. This risks a desk reject. If it is reviewed anyway, it will be judged on DROID alone, where the contribution reduces to "a gated local-attention copy head beats an ungated residual head by 7.6%". Without the gated-Direct and copy-only controls, reviewers can plausibly attribute that gain to a better skip connection rather than to transport.

---

## 3. What undersells the paper

- **The oracle-move analysis is buried.** "51% of moving-patch error is recoverable by copying a single observed feature within the window" is a strong, model-free, reusable empirical finding about foundation-feature dynamics. It directly motivates the whole approach and deserves a small main-text figure, ideally across datasets, before the method section.
- **The static-patch result is a sharp message.** "Every learned regressor is worse than copying on static patches; ShiftWM is the only one that is not" is a clean observation about why regression heads underperform, and it applies to DINO-WM, V-JEPA 2-AC and others. It deserves more prominence, and it should be tested on the published models too: do DINO-WM and V-JEPA 2-AC also corrupt static background?
- **V-JEPA 2-AC plug-in dynamics (Fig. 8)** are in the appendix. The gate going from 0.018 to 0.55 in 250 steps, with validation error lower at every step, is compelling evidence that a 1.3B model "wants" to transport. This belongs in the main text instead of the weaker DINO-WM rows.
- **Inspectability is claimed but not exploited.** The transport field could serve as free, action-conditioned motion or flow prediction. Once Fig. 14 is filled, EPE against RAFT or a zero-shot point-tracking evaluation would turn "interpretable tensors" into a measurable secondary capability.
- **Efficiency is left unquantified.** Parallel decoding of all K steps and one encoder pass versus K recursive passes is a real advantage for CEM planning, where thousands of rollouts are needed. Table 16 is pending. Planning wall-clock per success could be a selling point even if success rates tie.
- **Honesty is a strength but is phrased as weakness.** The abstract spends a clause on "raises it by 18.4% on Wall". It would read better as a scoped claim ("helps when history S>1 is available; with a single history frame it can hurt") backed by an S-ablation.
- **The contribution list mixes a method paper with a benchmark paper.** Having 17 benchmarks and 48k episodes (Fig. 4) mostly pending dilutes the story. A tight DROID + surgical + planning paper with complete results would read stronger than a broad but half-filled one.

---

## 4. Top 10 improvements, ranked by expected score impact

1. **Finish or cut every "pending" item in the main text, and fix the page limit.** Either fill Tables 1, 3 and 4, §4.4/§4.5, Fig. 5's Bridge panel and the DINO-WM success rates, or remove the columns and claims (including "we evaluate it on IWS, BridgeData V2, RT-1, Language-Table and three planning suites") that are not backed. Bring the Conclusion onto page 9, fix the Fig. 4 caption overlap, and scrub internal paths and "re-run script" text from the appendix. *(Moves the paper from desk-reject risk to reviewable; largest single impact.)*
2. **Deliver closed-loop planning results (Table 4) with all baselines trained under the same L_act.** Report success ± CI over 3 training seeds × 3 planner seeds, plus planning wall-clock. If ShiftWM + L_act beats Direct + L_act and AR + L_act on PushT, TwoRoom and Reacher, the "world model" framing becomes credible. If it ties, report it honestly and lean on efficiency.
3. **Add the controls that isolate transport from gating.** Needed rows: gated-Direct with an identical gate init; transport with values = projected M; global vs local window; S=1; no correction; and an F2M-style bilinear-flow warp head. Put them in Table 3 with 3 seeds and paired CIs. This answers the "just a better skip connection" critique.
4. **Harden the V-JEPA 2-AC plug-in.**
   - Add a copy-only control: (1−g)P + g·Z0_measured.
   - Use ≥3 seeds with paired session-bootstrap CIs.
   - Tune the baseline fine-tune (LR sweep and early stopping; show both arms at their best).
   - Evaluate on DROID episodes verifiably outside V-JEPA 2-AC's training data, or on a non-DROID dataset such as Bridge.
   - Debug the 4.1% zero-shot skill (action frame, normalization).

   If it survives, this is the headline result.
5. **Rewrite related work and the novelty claim around feature-space warping.** Cite and contrast Šarić et al. (CVPR 2020, GCPR 2019, TNNLS 2021), Luc et al. F2F, Deep Feature Flow, Terwilliger et al., Gao et al. ICCV 2019, Transporter, and 2024–2026 flow-for-robotics work (AVDC, Track2Act, ATM, Im2Flow2Act, FLIP, arXiv 2601.10781). Rescope "first" to the specific combination: action-prefix-conditioned, parallel-horizon, soft local transport of frozen foundation features used as a plug-in head for latent planners.
6. **Complete multi-seed runs and reconcile all numbers.** Hamlyn needs 3 of 3 seeds before † marks are shown. Make Table 1, Table 7, Table 17 and the teaser consistent (same seeds and same windows), or label seed-0 tables explicitly. Report paired CIs next to every headline percentage in the abstract.
7. **Make the theory earn its place or shrink it.**
   - Test Prop. 2 quantitatively: plot the per-patch gain over Direct against the feature contrast ||z0,π(i)−z0,i||² (the pending Fig. 10 right).
   - Reframe Prop. 3 as shared by ShiftWM and Direct, i.e. an argument for direct over recursive forecasting only, and cite the direct/iterated multi-step literature.
   - Address the "optimal forecast is a blur" tension with the sharpness claim, or replace Prop. 1–2 with a statement about sample complexity or approximation error for finite-width additive heads.

   Move the proofs' substance to half a page and use the freed space for results.
8. **Add at least one strong external baseline on the same data and backbone.** Options: a DINO-Foresight- or DINO-world-style predictor, RLA-WM on its own IWS handles, DDP-WM (static/dynamic split, conceptually closest) or a stochastic feature-space model (FlowWM). Also replace or de-emphasize the AR-TF "DINO-WM-style" straw baseline, and remove "34.3% lower than DINO-WM-style" from the headline text.
9. **Resolve the DINO-WM Wall failure with an experiment.** Run DINO-WM with history S≥2, or ShiftWM with S=1 on DROID, to show whether a single history frame explains the regression. Report seeds. This turns a negative result into a scoped, understood limitation.
10. **Tighten the story and the figures.**
    - Simplify the teaser: one mechanism panel plus one bar chart with CIs, and random rather than best-case example windows (or both side by side).
    - Promote the oracle-move and static-patch findings to the main text.
    - Fix the gallery row labels (Figs. 18–19).
    - Fill Fig. 14 (transport vs RAFT EPE) and Table 16 (efficiency).
    - Add a failure-mode analysis for out-of-window motion and camera ego-motion (RT-1/Bridge).

---

### Literature referenced in this review (not cited by the paper)

- Šarić et al., Warp to the Future: Joint Forecasting of Features and Feature Motion, CVPR 2020. https://openaccess.thecvf.com/content_CVPR_2020/html/Saric_Warp_to_the_Future_Joint_Forecasting_of_Features_and_Feature_CVPR_2020_paper.html
- Šarić et al., Dense Semantic Forecasting in Video by Joint Regression of Features and Feature Motion, TNNLS 2021. https://arxiv.org/abs/2101.10777
- Šarić et al., Single Level Feature-to-Feature Forecasting with Deformable Convolutions, GCPR 2019.
- Luc et al., Predicting Future Instance Segmentation by Forecasting Convolutional Features (F2F), ECCV 2018.
- Zhu et al., Deep Feature Flow for Video Recognition, CVPR 2017.
- Terwilliger et al., Recurrent Flow-Guided Semantic Forecasting, WACV 2019.
- Gao et al., Disentangling Propagation and Generation for Video Prediction, ICCV 2019.
- Kulkarni et al., Unsupervised Learning of Object Keypoints for Perception and Control (Transporter), NeurIPS 2019.
- Ko et al., Learning to Act from Actionless Videos through Dense Correspondences (AVDC), ICLR 2024; Bharadhwaj et al., Track2Act, 2024; Wen et al., Any-point Trajectory Modeling (ATM), RSS 2024; Gao et al., FLIP, 2025.
- Future Optical Flow Prediction Improves Robot Control & Video Generation, arXiv 2601.10781 (2026). https://arxiv.org/abs/2601.10781
- Venkatraman et al., Improving Multi-step Prediction of Learned Time Series Models, AAAI 2015; Asadi et al., Combating the Compounding-Error Problem with a Multi-step Model, 2019; Lambert et al., Learning Accurate Long-term Dynamics for Model-based RL, 2021.
