# Jargon and ambiguity audit, ShiftWM ICLR 2027 (phase 1, read-only)

Source: `paper/submission_folder/` as of commit a7c00f2 plus working tree; PDF `paper/iclr2027.pdf` (25 pages,
main text pp. 1-9, references pp. 9-13, appendix pp. 14-25). Page numbers are PDF pages of the current build.
Reader model: a general ML reviewer who knows ViTs, attention, MSE, bootstrap and DiT, but not robotics,
surgery or the latent-world-model literature. Rules applied: paper-writing SKILL.md mechanics 3, 4, 21, 23, 24
(coin once, define at first use with an appositive, one word per thing, scan for undefined noun phrases).

File abbreviations: `abs` = sections/00_abstract.tex, `intro` = 01_intro.tex, `rel` = 02_related.tex,
`meth` = 03_method.tex, `exp` = 04_experiments.tex, `res` = 05_results.tex, `concl` = 06_conclusion.tex,
`stmt` = 07_statements.tex, `app` = appendix/appendix.tex, `tab/x` = tables/x.tex.

Legend for "Defined?": **No** = never defined; **Late** = defined after first use; **Yes** = defined at first use;
**App** = defined only in the appendix.

---

## 1. Main text: method and world-model vocabulary

| # | Term | First occurrence | Defined? | Proposed fix (one clause at first use, or glossary) |
|---|------|------------------|----------|------------------------------------------------------|
| 1 | **soft local transport / transport** | abs:14, p1 (also teaser caption abs:34) | Late: only formally at meth:51-60, p4. An ML reader may read "transport" as optimal transport. | abs:14: "an action-conditioned soft local transport, *an attention-weighted copy of observed patch features from a small neighbourhood of each patch*,". Intro:42 already paraphrases well; keep. |
| 2 | **gate / "gates it against keeping each feature in place"** | abs:14-15, p1 | Late (meth:62, p4). The abstract phrase has two readings (gate the transport vs. gate the keeping). | abs: "...and a per-patch gate *chooses between moving content and keeping the last feature in place*, and a correction adds new content". |
| 3 | **"gated against the identity"** | intro:57, p2 (contribution 1) | No. "Identity" (identity map) is unexplained and clashes with "identity bias" (meth:58). | Replace with "gated against keeping the last observed feature". |
| 4 | **correction** | abs:15, p1 | Yes in spirit ("a correction for new content"); formal meth:63. | OK. Keep "correction" everywhere (appendix fig. captions also use "correct"; fine). |
| 5 | **innovation / "motion-plus-innovation model"** | intro:58, p2; meth:106, p5 | Late/partial: meth:112 says "$\varepsilon_k$ is new content" but the word "innovation" is signal-processing jargon. | Replace "innovation" by "new content" in intro:58 ("under a model in which features move and new content is added") and meth:106 ("motion plus new content"). Keeps one word per thing. |
| 6 | **persistence / copying the last frame / copy / staying put** | intro:23 "copying the last frame", intro:29 "persistence", intro:37 "staying put", teaser caption "copying the last frame", res:81 "copy" | Late: "Persistence copies $Z_0$" at exp:46, p6. Four names for one baseline (violates mechanic 21). | Define once at intro:23: "copying the last frame, *which we call persistence*,"; then use "persistence" at intro:29, intro:37 ("the error of persistence"), res:80-81. Teaser/figure labels may keep "copy" (figure agent). |
| 7 | **window (temporal)** vs **window (transport, spatial)** | Temporal: teaser caption abs:32 "One held-out DROID window", p1; meth:9. Spatial: intro:36 "local window", meth:51 | Temporal sense never defined in main text (app:228 hints "windows (stride 2)"). Two meanings of one word. | At exp:50 (Metrics): "Each test *window* is $H{=}3$ observed and $K{=}10$ future steps of one episode." Always write "transport window" for the spatial sense (intro:36, res:111, abl). Teaser caption: "One held-out DROID clip" or add "(3 observed + 10 future steps)". |
| 8 | **moving / static patches** | teaser caption abs:35 "dashed: moving patches", p1; intro:36, p2 | Late: res:74, p8 ("the 25% with the largest true feature change"). | intro:36: "On moving patches of held-out DROID, *the quarter with the largest true feature change*,". Teaser caption: "(dashed: the 25% of patches that change most)". |
| 9 | **Direct, AR, AR-TF** | teaser caption abs:33 "Direct rewrites every patch", p1; intro:67 (contribution 4), p2 | Late: exp:37-44, p6. Reader meets three undefined method names in contribution 4. | intro:67: "below Direct, AR and AR-TF, *matched predictors that regress all steps at once, roll out recursively, or train with teacher forcing*,". Teaser caption: "Direct (a matched regression head) rewrites every patch". |
| 10 | **teacher forcing / teacher-forced** | intro:28, p2 | No (ML readers from sequence models mostly know it, but the consequence is the point). | intro:28: "trained with teacher forcing, *on true past features instead of its own predictions*,". |
| 11 | **rollout / recursive rollout / rolled out** | abs:10, p1 ("over the rollout") | Partial: abs:7 "feed their own predictions back as inputs" defines it implicitly. | Tie them: abs:7 "...feed their own predictions back as inputs, a *recursive rollout*," then "rollout" is defined. |
| 12 | **feature contrast** | intro:26, p2 | Late: meth:132 (Prop. 2) "the feature contrast along the motion". | intro:26: "a target that grows with the motion and with the feature difference between where content comes from and where it goes". Or drop "feature contrast" from intro. |
| 13 | **"the single best observed feature from a local window"** (oracle) | intro:36, p2 | Late: "oracle move" defined res:110-112, p8. Reader cannot tell the choice uses the future. | intro:36: "the observed feature in a $7{\times}7$ neighbourhood that is closest to the true future, *an oracle that looks at the future*, removes ...". |
| 14 | **output head / plug-in head** | abs:19, p1 | Partial. "Head" normally means a final layer; here it replaces how a predictor forms its output. | abs:19: "because \ours{} is an output head, *a final layer that forms the forecast from the predictor's outputs*, it attaches ...". Optional if space is tight. |
| 15 | **frozen backbone** (abstract) vs **frozen encoder** | abs:21, p1 | Ambiguous: "backbone" can mean encoder+predictor; the predictor is trained. | abs:21: "under a shared frozen encoder and identical predictor" (one word per thing: the paper otherwise says "frozen encoder"). |
| 16 | **measured features** vs **observed features** | abs:16, p1; meth:60 emphasises "measured" | Synonyms (mechanic 21). | Use "observed features" throughout; meth:60 may keep "*observed* features, not latent states". |
| 17 | **patch-feature space of a frozen visual foundation model** | abs:3, p1; intro:7 | Mostly clear to ML readers. | Optional: intro:7 "patch features, *one vector per $14{\times}14$ image patch*,". Low priority. |
| 18 | **horizon / step $k$** | abs:15 "all horizons", p1 | Partial (meth:25 uses $k$). | exp:21: "we forecast $K{=}10$ steps (3.3 s); the *horizon* $k$ is the number of steps ahead". |
| 19 | **skill** | exp:53, p6 (teaser uses "error vs. copying") | Yes (exp:53). Term from weather forecasting; fine once defined. Clash risk with "surgical skill" if that phrase is ever used. | Keep. Do not use "skill" for surgical tasks (see #26). |
| 20 | **action ranking / "ranks actions best" / Rank acc. / action-ranking error** | res:29, p7; tab/ablations "Rank acc." p8 | Yes in-sentence at res:29. Appendix (app:257) reports the complement "action-ranking error" as an error rate; main reports accuracy. Two quantities, one name family. | Keep; in app:257 write "action-ranking error, 100% minus the action-ranking accuracy of \cref{sec:forecasting}". |
| 21 | **transport field** | res:107, p8 | App only (software.tex:33 "expected source offset of every patch"). | res:107: "changes the transport field, *the expected source offset of each patch*, ...". |
| 22 | **"horizon-by-motion cells"** | res:119, p8 | No; ambiguous. | "in all 100 cells of 10 horizons $\times$ 10 motion deciles". |
| 23 | **motion decile / true-motion decile** | res:84, p8; fig. 5c label | Partial ("deciles of true motion"). | res:84: "Across all ten deciles of true feature change". Consistent with #8. |
| 24 | **teacher-forced latent error** (DINO-WM metric), **latent err.** | res:57, p7; tab/plugin p7 | No. Reader cannot tell it is one-step error from true inputs. | res:57: "teacher-forced latent error, *the one-step feature error when the model is fed true past features*,". |
| 25 | **SSIM / LPIPS** in Table 2 (plugin) | tab/plugin p7 | No in main text (decoded-pixel metrics; app:284 context). | Table caption: "SSIM, LPIPS: quality of frames decoded from the features". |

## 2. Main text: robotics and surgery vocabulary

| # | Term | First occurrence | Defined? | Proposed fix |
|---|------|------------------|----------|--------------|
| 26 | **"surgical training tasks" / "dVRK training tasks"** | exp:23, p6; concl:25, p9; tab/datasets | No. **Two readings**: tasks surgeons practise vs. ML training split. High-risk ambiguity. | "surgical practice tasks (knot tying, suturing, peg transfer on practice models)". Glossary: *surgical practice task*. |
| 27 | **dVRK** | exp:23, p6 | No. | "dVRK, *the da Vinci Research Kit surgical robot*,". Glossary. |
| 28 | **Open-H / Open-H Hamlyn / Hamlyn / Open-H surgical / Open-H-Embodiment** | abs:22 "Open-H", p1; exp:23 "Open-H Hamlyn"; app uses "Hamlyn"; stmt:5 "Open-H-Embodiment"; tab/main "Open-H surgical" | Named but never explained as the Hamlyn subset of Open-H-Embodiment. Five names. | exp:23: "\textbf{Open-H Hamlyn}, *the Hamlyn Centre subset of the Open-H-Embodiment surgical-robot collection*,". Then use "Open-H" in main-text prose and tables, "Open-H Hamlyn" only in the Datasets paragraph and glossary; replace bare "Hamlyn" in app (app:243, 264, 449, 462, figure labels). |
| 29 | **phantom / ex-vivo / in-vivo / endoscopy / scene camera** | stmt:8 "phantoms and ex-vivo tissue", p9; concl:25 "scene-camera views ... not in-vivo endoscopy", p9; app:25 "training phantom" | No. | Glossary: *phantom* (artificial practice model of tissue), *ex vivo* (tissue removed from the body), *in vivo endoscopy* (camera inside a living patient), *scene camera* (external camera viewing the robot). Inline in concl:25: "external-camera views of practice tasks, not endoscopic video from patients". |
| 30 | **Franka** | exp:18, p6 | No. | "real Franka *(7-joint arm)* episodes". Glossary. |
| 31 | **35-D Cartesian and gripper action block** | exp:20, p6 | App (app:187: "5 commanded poses and gripper positions of one step"). "Cartesian" and "gripper" unexplained. | exp:20: "a 35-D action block, *the five commanded hand poses (position and orientation) and gripper openings within one step*". Glossary: *end-effector*, *gripper*, *action block*. |
| 32 | **80-D bimanual action blocks** | exp:23, p6 | App (app:193). | "80-D action blocks for the two arms". |
| 33 | **episode** | abs? no; teaser caption "held-out DROID window"; intro:65 "episodes", p2 | No. Robot-learning term (one recorded demonstration from start to end). | exp:18: "real Franka *episodes, each one recorded task demonstration*,". Glossary. |
| 34 | **recording session / session-disjoint** | intro:64, p2 "session-disjoint DROID"; exp:18 | Partial (exp:18 "split by recording session"). "Session" never explained. | intro:64: "DROID split by recording session, so test scenes are unseen,". Glossary: *session* (a block of episodes recorded in one place and sitting). |
| 35 | **zero-shot second camera / exterior camera** | intro:64, p2; exp:20 | Partial (exp:20, res:15 "which no model saw in training"). | intro:64: "a second camera view never seen in training". |
| 36 | **xArm** | exp:26, p6 | No. | "an xArm *robot arm* pushing ...". Glossary. |
| 37 | **Open-X / pinned subset** | exp:26-27, p6 | No. "pinned" is software jargon. | "taken from a fixed-version subset of the Open-X Embodiment collection". |
| 38 | **Language-Table "play session" / segments** | concl:24, p9 | App (app:197-202). Uses "segments" and "episodes" for the same object. | concl:24: "Language-Table test and training clips may come from one long unscripted recording". Glossary: *play data*. |
| 39 | **PushT, Wall** | teaser fig. (c) "DINO-WM PushT", p1; intro:62, p2; exp:29, p6 | No in main text (tab/datasets p17 describes PushT only). | exp:29: "PushT, *a simulated task of pushing a T-shaped block to a target pose*, and Wall, *a 2-D navigation task through a door in a wall*,". Glossary. |
| 40 | **planning success** | concl:22, p9 | App (tab/plan-protocol). | Glossary: *planning success*; pointer already in concl:22. |
| 41 | **model-predictive control (MPC)** | rel:9, p2 | No. | rel:9: "plans with model-predictive control, *re-optimising the action sequence at every step using the model's forecasts*". Glossary (with CEM). |
| 42 | **V-JEPA 2-AC ("AC" = action-conditioned), 1.3B** | abs:6, p1 | Partial (rel:9 "post-trains an action-conditioned predictor"). | rel:9: "V-JEPA 2-AC, *its action-conditioned (AC) variant*,". Low priority. |
| 43 | **DINO-WM, DINO-WM-style** | abs:6, p1; intro:28 "DINO-WM-style predictor" | Late (rel:8, p3). | Acceptable in abstract ("such as"). intro:28: "a predictor trained like DINO-WM, with teacher forcing,". |
| 44 | **"hold on scenes it has not seen"** | intro:12, p2 | Idiom, vague. | "generalise to scenes it has not seen". Also "the gains hold" (res:15, 24): fine. |

## 3. Main text: statistics

| # | Term | First occurrence | Defined? | Proposed fix |
|---|------|------------------|----------|--------------|
| 45 | **paired bootstrap over episodes and sessions** | intro:65, p2 | Yes at res:129 / app:230 (p8, p18). ML readers know bootstrap; "paired" and "session" are fine after #34. | OK. |
| 46 | **Holm-adjust** | res:130, p8 | No. | "and Holm-adjust them *for multiple comparisons*". |
| 47 | **"Episodes are the statistical unit"** | exp:56, p6 | Conflicts with DROID where sessions are resampled (res:130). | "Episodes (sessions for DROID) are the statistical unit". |

## 4. Symbols and notation

| # | Symbol | Where | Issue | Fix |
|---|--------|-------|-------|-----|
| 48 | $\mathbf{P}$ | meth:108 motion matrix $\mathbf{P}_k$; res:39 backbone prediction $\mathbf{P}$; app:150 $\mathbf{P}_k$ backbone prediction | **Clash**: same letter, two meanings, pp. 5 and 7. | Rename backbone prediction to $\hat\Z^{\text{base}}$ or $\mathbf{B}$ (res:39, app:150). |
| 49 | $\pi$ | meth:54 transport weights $\pi_{k,i}(j)$; meth:112 source index $\pi(i)$ | Overload on one page. | Rename source index to $s(i)$ or $\sigma(i)$... ($\sigma$ is sigmoid) so use $s(i)$ in meth:112-132 and app proofs. |
| 50 | $\operatorname{sg}$, $m$ | meth:84, p5 | Undefined (stop-gradient, margin). | "where sg stops the gradient and $m$ is a margin". Also $m$ reused in proof app:105. |
| 51 | $d_k$, $\mathbf{W}_q$, $\mathbf{W}_k$ | meth:55, p4 | $d_k$ undefined; $k$ subscript clashes with step index $k$. | "$d_k$" -> "$d_{\text{key}}$ (key width)"; $\mathbf{W}_k$ -> $\mathbf{W}_{\text{key}}$. |
| 52 | $\sigma$ | meth:63 | Undefined (sigmoid). Common, low priority. | "the sigmoid $\sigma$". |
| 53 | $\mathbf{M}$ vs $M$ | meth:35 memory states; app:106 logit constant $M$ | Minor clash in appendix. | Use $\Lambda$ or $B$ for the constant in the proof. |
| 54 | $S$, $w$ | meth:51 | Defined. OK. | - |
| 55 | AdaLN | Fig. 2 label (p4); text meth:41 says "adaptive layer normalisation" without the abbreviation; app:79 "AdaLN-zero" | Abbreviation not introduced. | meth:41: "adaptive layer normalisation (AdaLN)". |

## 5. Sentences with two plausible readings (main text)

| # | Location | Sentence | Readings | Fix |
|---|----------|----------|----------|-----|
| 56 | abs:14-15, p1 | "gates it against keeping each feature in place" | gate applied to transport vs. gate keeps features | See #2. |
| 57 | abs:21, p1 | "under a shared frozen backbone" | whole model frozen vs. only encoder frozen | See #15. |
| 58 | exp:23 / concl:25 | "surgical training tasks" | ML training vs. surgeon practice | See #26. |
| 59 | teaser (c), abs:35 | "% lower error than the best learned competitor" | for plug-in bars the competitor is the backbone alone (stated in the figure, not the caption) | Caption: "(c) % lower error than the best matched predictor, or than the backbone alone for plug-in heads". |
| 60 | intro:22-23 | "every matched regression predictor we train" | "matched" to what is not yet said | "every regression predictor we train with the same encoder and architecture". |
| 61 | res:9-11 | "whose teacher-forced absolute predictions are worse than persistence once rolled out" | "absolute" (vs residual) unexplained | "which predicts the full next feature rather than a change, and is worse than persistence once rolled out". |
| 62 | res:60-61 | "leaves the head one observed frame to move and, in rollouts, only DINO-WM's own predictions" | whether the head moves predictions or observations | "...one observed frame, and after the first rollout step the head moves DINO-WM's own predictions instead of observed features". |
| 63 | res:143-144 | "loses the entire gain (+X%, as much as Direct)" | "as much as" error increase vs. same error as Direct | "its error rises by X%, to the level of Direct". |

## 6. Appendix (first occurrences not already covered)

| # | Term | First occurrence | Defined? | Proposed fix |
|---|------|------------------|----------|--------------|
| 64 | **end-effector** | Fig. 7 (setting.pdf) axis label, p14 | No. | Glossary. Figure agent may relabel "gripper position". |
| 65 | **training phantom** | app:25, p14 | No; "training" ambiguity again. | "a practice phantom (artificial tissue model)"; glossary. |
| 66 | **stable-worldmodel, kinematics-conditioned** | app:72-73, p15 | No. | "conditioned on the robot's joint and tool commands (kinematics)"; stable-worldmodel: "the LeWM authors' planning benchmark library". |
| 67 | **AdaLN-zero** | app:79, p15 | See #55. | - |
| 68 | **proprioceptive channels** | app:168, p16 | No. | "proprioceptive (robot state) channels"; glossary *proprioception*. |
| 69 | **ViTPredictor, shards, salted hash** | app:165, 184-185, p16-17 | Software jargon. | "shards (data files)"; "salted hash" fine for ML. Low priority. |
| 70 | **RealSense scene camera; stream labelled "endoscope"** | app:190, p17 | Partly. | Keep; glossary *scene camera*. |
| 71 | **expert datasets** | app:204, p17 | No. | "datasets of successful demonstrations". |
| 72 | **CEM, elites, warm start, replan, env steps** | tab/planning_protocol, app:317, p20 | No. | Glossary: *CEM* (cross-entropy method: sample action sequences, keep the best "elites", refit, repeat). |
| 73 | **success criteria / joint match** | app:318; tab/planning_protocol | Given in table; "joint match" vague. | "Reacher: joint angles within tolerance of the goal". |
| 74 | **planning regret** | app:325, p20 | No. | "regret, the true-cost gap to the best plan". |
| 75 | **open-loop error, closed-loop planning** | app:297, 313, p19-20 | No. | Glossary: *open-loop* (forecast without new observations); *closed-loop* (re-plans after each executed chunk). |
| 76 | **Grounding DINO, SAM 2.1, IoU, placement** | app:443, p21 | Tool names; "placement" defined app:455. | "Grounding DINO, an open-vocabulary detector, ... SAM 2.1, a video segmentation model". |
| 77 | **PSNR/SSIM/LPIPS** | app:284, p19 | Standard for vision readers. | OK. |
| 78 | **Spearman, quintile, PCA, total-variation, tower property, Lipschitz** | app | Standard ML/maths. | OK. |
| 79 | **expected source offset** | app:505 (qualitative) and software.tex:33 | Defined where used. | Unify with "transport field" (#21). |

### Non-jargon defects found in passing (report only)
- app:335-336: a line break leaves a sentence starting with ", whereas the LeWM paper reports ..." (period then comma). Join into one sentence.
- tab/datasets row "PushT / TwoRoom / Reacher ... (LeWM planning suites)" lists 10-D actions and "5 env steps" but the main-text plug-in study uses DINO-WM's PushT and Wall, which have no row; caption says "DINO-WM PushT and Wall use the official releases". The reader may confuse the two PushT datasets (LeWM vs DINO-WM). Glossary entry for PushT should say both exist.

---

## 7. Recommended global structure

1. **Inline, main text (12 clauses, most important first):** #1 transport, #2 gate (abstract rephrase), #6 persistence (define once, drop synonyms), #7 test window, #8 moving patches (intro + teaser caption), #9 Direct/AR/AR-TF in contribution 4, #10 teacher forcing, #3 + #5 "identity" and "innovation" replaced by plain words, #26 "surgical practice tasks" + #27 dVRK, #31 action block (hand pose + gripper), #33/#34/#35 episode, session split and unseen camera, #39 PushT/Wall + #24 teacher-forced latent error. Plus the symbol fixes #48-#51 (renames, no space cost) and #46 Holm (3 words).
2. **Glossary table in Appendix A** (new \S A.2 "Terms and benchmarks", placed after A.1 "The task and its uses"), `\scriptsize`, two term/definition column pairs side by side, one line per entry. Entries: episode, session, window, horizon, end-effector, gripper, action block, proprioception, Franka, xArm, dVRK, phantom, scene camera vs. endoscope, surgical practice task, DROID, Open-H Hamlyn, Language-Table (play data), PushT (LeWM and DINO-WM versions), Wall, TwoRoom, Reacher, CEM/MPC, open- vs closed-loop, planning success, teacher forcing, persistence, skill, oracle move. (About 28 entries -> 14 table rows.) Benchmark scenes already sit in Tab. 5 (datasets), so glossary benchmark rows stay to five words.
3. **Pointer from main text:** extend exp:31 "\Cref{tab:datasets} lists sizes, splits and licences" with ", and \cref{tab:glossary} defines robotics and surgical terms" (no new line if it fits on the existing line; otherwise one line).

## 8. Space estimate

- Main text: most fixes replace words; net additions about 9 short clauses, roughly +6 to +8 lines (abstract +1, intro +3, experiments +2 to +3, results +1, conclusion 0 once "training" -> "practice"). Pointer +0 to +1 line. Total +7 to +9 of the ~12 lines of slack on p9. Buffer: dropping the optional #14/#17/#42 keeps it at +6.
- Appendix: glossary table of 14 rows at \scriptsize is about 0.3 page. p25 is nearly full, so to stay at 25 pages, offset it by cutting A.2 "Why current latent world models fall short" (app:42-57, about 11 lines, which repeats intro para 2 and Props. 2-3) or folding it into one sentence of A.1. Net appendix change about +0.1 page, within 25 pages but should be checked after the figure agent's changes land.
