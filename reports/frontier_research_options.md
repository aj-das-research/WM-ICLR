# Frontier research options: VLA, agents and world models

Audit: 18 September 2026. Primary paper/repository sources checked; no benchmark or memory-fit claims have been reproduced locally. Existing TTA proposal is preserved as an earlier candidate, not silently replaced by a validated new method. Hardware constraint remains three independent RTX5000 Ada 32GB jobs; no multi-GPU training allocation.

## Recommendation

Best application-led pilot: **When Does Replanning Help? Learning Intervention Value for Frozen Vision–Language–Action Policies.** The practical objective is fewer failed manipulation episodes with a limited intervention/compute budget. Train a small intervention-value head using paired simulator branches; freeze the underlying VLA. World-model-style latent transitions may improve the head, but must beat a direct predictor before being retained. This is a proposed research hypothesis, not established novelty or a promise of acceptance.

A VLA can fail after a missed grasp, disturbance, stale action chunk or visual distraction. A failure detector estimates whether the task will fail, but that alone does not establish whether a particular intervention helps. An interruption can be unnecessary or actively harmful during successful contact. Learn the incremental value of the available intervention relative to continuing from the same starting state.

## Verified VLA foundations and latest comparisons

| Work | Date / status | Primary paper and implementation | Evidence and five-day implication |
|---|---|---|---|
| VLAct / StarVLA | August 27 release; preprint | https://arxiv.org/abs/2608.27550 ; https://github.com/starVLA/VLAct ; https://github.com/starVLA/starVLA | Public backbone/model collection; modular VLA infrastructure. Use released models, not continued pretraining. Main README notes optimizer state is not saved: exact resume needs explicit handling. |
| LangForce | ICML2026 per official repo; July28 VLA-Arena release | https://arxiv.org/abs/2601.15197 ; https://github.com/ZGC-EmbodyAI/LangForce | Weights linked for VLA-Arena and integration into StarVLA. Training recipe uses 8 H100; unsuitable to reproduce full training here. Optional inference comparator after fit check. |
| StreamingVLA | March2026 preprint, active released source | https://arxiv.org/abs/2603.28565 ; https://github.com/gen-robot/StreamingVLA | Actual implementation; LIBERO policy and predictor HF endpoints verified ungated. Adaptive early observation already exists. Single-GPU script does not establish 32GB training fit. |
| PDF | CVPR2026; April preprint | https://arxiv.org/abs/2604.18107 ; https://github.com/zhoujiahuan1991/CVPR2026-PDF | Eight Python files and evaluation runner found; OpenVLA suite checkpoints specified. Delayed-feedback test-time perturbation adaptation is existing work. |
| VLA-Corrector | July2 preprint | https://arxiv.org/abs/2607.01804 ; https://github.com/ZJU-OmniAI/vla-corrector | Code for latent dynamics detection/correction and evaluation exists. README explicitly excludes trained corrector and fine-tuned policy checkpoints; base-policy links are not substitutes. Reproduction requires training. |
| SmolVLA / LeRobot | Established lightweight foundation, maintained2026 | https://github.com/huggingface/lerobot ; https://huggingface.co/HuggingFaceVLA/smolvla_libero | Preferred first policy. HF model/config/normalization assets verified ungated. This checkpoint is task-adapted; do not assume the generic base solves LIBERO zero-shot. |
| Small-VLA deployment study | September12 preprint | https://arxiv.org/abs/2609.14146 ; https://github.com/rafiqul713/smolvla-libero-onnx | Relevant warning that runtime and interface changes alter closed-loop behavior. Repository linked by author; source not deeply audited here. Useful reproduction reference, not a novelty foundation. |

Raw README snapshots, source tree commit hashes and model metadata are in `references/frontier/`. They establish source/artifact availability, not successful reproduction. Missing top-level licenses in some tree audits need checking before redistributing derivatives.

## Mandatory novelty comparisons, including non-runnable work

- SAFECAST (Aug4): https://arxiv.org/abs/2608.04246 — contrast-set training/calibration for failure probes. Perturbation-robust monitoring is already studied; no official implementation verified in this pass.
- Foresight (June): https://arxiv.org/abs/2606.23085 — action-conditioned world-model latents for failure detection. Adding a latent world model to a monitor is not enough.
- CoRe (Aug14): https://arxiv.org/abs/2608.14822 — counterfactual imagined continuations for frozen-policy recovery. Imagining recovery is not new; no usable official repo verified in this pass.
- B2FF (June8): https://arxiv.org/abs/2606.09258 — pre-imagined milestones for recovery; evaluate trigger timing fairly, not using injected-event knowledge.
- FAR (July1): https://arxiv.org/abs/2607.01111 — failure-aware retries and continual improvement. Avoid presenting retry learning as new.
- Human-in-the-loop modular recovery (HRI2026): https://emprise.cs.cornell.edu/modularhil/ — uncertainty and intervention-cost selection; value/cost tradeoffs already exist.
- Counterfactual-regret policy repair implementation discussion: https://amohan.dev/blog/2026/repairing-frozen-visuomotor-policy-cfr-flow-matching/ . This author technical article is a collision lead, not evidence of conference acceptance. Paired regret labels alone cannot be claimed novel.

Potential distinction to investigate: a policy-conditioned estimate of the **incremental utility of a specific intervention**, evaluated under both helpful and harmful interruptions and generalizing across held-out disturbance types. This might still reduce to existing value-of-computation, options, advantage-learning or selective-control methods. Before a paper claim, examine those literatures and closest method sections. Do not invent a name and call that novelty.

## Minimal algorithm and decisive experiment

1. Reproduce a frozen SmolVLA on a fixed LIBERO task subset. Verify action units, normalization, gripper semantics, camera orientation, control rate and simulator state restore.
2. At selected training rollout states, snapshot the complete simulator plus policy RNG/state. Branch into (a) continuing the queued action chunk, (b) discarding the queue and replanning from the current observation, optionally (c) a separately trained short-horizon policy. Do not silently give one arm extra environment steps.
3. Use multiple matched random continuations when stochastic. Record success/progress, collisions if available, elapsed steps, inference cost and any intervention-induced failures. Supervision is the paired outcome difference, not simply whether failure occurred.
4. Fit a small head to recent policy features, action queue, proprioception and visual-transition residuals. Its target for intervention k is expected return difference minus declared compute/interruption cost. Compare direct features against a compact action-conditioned predictor; retain the predictor only if useful.
5. At test time use observations only. No simulator forks, true state, reward, future labels or event timestamps enter the controller. Branch rollouts are training supervision and evaluator-only oracle diagnostics. This is simulation-supervised control, not label-free TTA.
6. Intervene under a fixed budget when estimated incremental utility is positive; lock threshold on validation tasks. Any uncertainty bound needs actual calibration and explicit assumptions, not a heuristic called a guarantee.

Baselines: original queue; always replan; periodic replan; random matched-rate interruption; entropy/action disagreement; world-model prediction-error threshold; failure-probability probe; VLA-Corrector/StreamingVLA when reproducible; direct intervention-value head without a world model. Cost-match forward passes, environment steps and intervention counts. Report successes versus interventions and wall time, harmful-intervention rate, missed beneficial opportunities, paired intervals and per-task results.

A decisive negative result: failure detection and intervention value make essentially the same choices, or a fixed replan interval matches the learned controller. In either case, do not proceed with a novelty claim. Another failure is that paired-rollout labels need too much simulation to generalize; report sample cost explicitly.

## Benchmarks

- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO — nominal reproduction and controlled paired branches.
- LIBERO-Plus: https://github.com/sylvestf/LIBERO-plus — existing visual/layout/instruction perturbations. Prespecify a balanced subset, report it as a subset; do not claim the full ~10000-variant benchmark. Related official integration: https://github.com/huggingface/lerobot/blob/main/docs/source/libero_plus.mdx .
- VLA-Arena: https://github.com/PKU-Alignment/VLA-Arena — source includes 170 tasks, distractors/safety/extrapolation/long-horizon suites and model evaluators. Use selected suites only if a compatible policy reaches a meaningful baseline; weak OOD policy performance can confound recovery evaluation.
- A controlled missed-grasp or mid-episode disturbance suite would be an additional experimental protocol, not an established benchmark. Report exact interventions, simulator state handling and held-out split.

These suites share substantial simulation lineage. They do not establish real-robot generalization or three independent embodiments. A second simulator is desirable after core success but cannot be promised in five days.

## Alternative tracks

**Small world models: best compute fit.** LeWorldModel is ~15M parameters with released training/evaluation code and checkpoints: https://github.com/lucas-maes/le-wm . Authors report single-GPU few-hour training, not a measured runtime on our machine. Study online sensor-versus-dynamics fault diagnosis and selective model repair only after comparing AdaJEPA: https://github.com/agentic-learning-ai-lab/adajepa . Plain online world-model adaptation is already covered. Other close work: RC-aux https://arxiv.org/abs/2605.07278 ; SCALE https://arxiv.org/abs/2608.16287 ; LEAP https://arxiv.org/abs/2609.03294 ; amortized planning https://arxiv.org/abs/2605.08732 . Merely changing latent distance, adding a critic or accelerating planning is crowded.

**Agents: easier direct software application, but crowded and environment-dependent.** See `agent_world_scout.md` for latest primary sources and runnable-status checks. Prefer local environments with deterministic success checks; no paid API requirement. A latest paper with only a project landing page is not a usable codebase. Building OSWorld/WebArena infrastructure from scratch can consume the sprint, and a generic reflection/retry wrapper is unlikely to be sufficient novelty.

**Large video world models: poor five-day choice.** Reuse for inference only if weights, license and 32GB fit are confirmed. Training a foundation video simulator is outside this budget. Compact latent dynamics can test the substantive scientific question without video-generation infrastructure.

## Five-day conditional budget

Plan 240 GPU-hours, not a guarantee: day1 setup/reproduction/paired-branch diagnostic30; day2 collect paired data and train heads50; day3 fixed benchmark comparisons70; day4 second policy, held-out perturbations, seeds70; day5 reproducibility and final reruns20. Three workers run independent episodes/experiments. No cross-node DDP.

Within the first 8–12 hours require EGL/MuJoCo rendering, one successful nominal policy rollout, restore-equivalent branches and measured episodes/hour. Then choose exact task counts. A planning example of 20 tasks × 10 initial states × 6 methods × 2 policies equals2400 evaluation episodes before training branches, ablations and repeats; at one minute/episode this is40GPU-hours, at five minutes it is200. Measure rather than assume. If simulator setup fails, LeWorldModel's smaller tasks are the fallback; that changes the scientific scope and must be declared.

Release target: trained intervention head(s), optional compact dynamics predictor, pinned policy references, paired training trajectories, evaluation manifests and a synchronized rollout demo showing helpful and harmful intervention cases. Base VLA remains frozen. No real-robot or production-safety claims without corresponding experiments. Visibility may follow a useful reproducible artifact, but traction and conference acceptance cannot be forecast reliably.
