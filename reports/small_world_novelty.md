# Small world models: contribution audit and two candidate projects

Research audit: 2026-09-18. This is a proposal, not a result or a claim of established novelty. No implementation or experiment was performed for this report. Scope: three independent 32 GB GPUs, approximately five days, reusable trained world-model checkpoints.

## Recommendation

Use the official [LeWorldModel](https://github.com/lucas-maes/le-wm) and its existing simulation stack. The approximately 15M-parameter model and reported single-GPU training make this a plausible starting point, but local throughput and full experiment duration remain unmeasured. Train a compact world model that can recalibrate to unknown changes in both image formation and physical response. Release actual encoder, context-inference and predictor weights, with task-family-specific compatibility documented.

The clearest motivating question is: **Did the camera change, did the physical system change, or did both change?** A small model useful for planning should preserve its physical knowledge across harmless visual shifts while changing its predictions when the response to an action really changes.

This is not a claim that appearance/dynamics factorization or system identification is new. Several extremely recent papers overlap directly; the candidate contribution must be a training mechanism and a demonstrated advantage on unknown mixed shifts and unseen combinations.

## Closest primary work and what is already taken

| Work | Established direction and implication |
|---|---|
| [LeWM, March 13](https://arxiv.org/abs/2603.19312) | Small action-conditioned latent prediction from pixels with anti-collapse regularization; baseline architecture and code to reuse. |
| [RC-aux, May 8](https://arxiv.org/abs/2605.07278), [code](https://github.com/Guang000/RC-aux) | Adds multi-horizon prediction and budget-conditioned reachability. A reachability head, longer training rollouts or a better latent goal score are insufficient novelty. |
| [Amortized planning, May 9](https://arxiv.org/abs/2605.08732) | A compact goal-conditioned inverse dynamics model replaces repeated search. A fast learned planner alone is crowded. |
| [UWM-JEPA, May 25](https://arxiv.org/abs/2605.25313) | Belief-space dynamics and counterfactual training for action sensitivity. Uncertainty or counterfactual targets alone are not new. |
| [ICWM, June 24](https://arxiv.org/abs/2606.26025) | Infers system configuration from short task-agnostic interaction histories to condition robotic control. Learning a context vector from past actions is already established. |
| [Fast-LeWM, June 24](https://arxiv.org/abs/2606.26217), [code](https://github.com/Yuntian-Gao/Fast-LeWorldModel) | Efficient world-model rollouts; current repository contains implementation, configs and [checkpoint link](https://huggingface.co/naiverer/fast-leworldmodel). Earlier search-index claims of README-only code are stale. |
| [AdaJEPA, June 30](https://arxiv.org/abs/2606.32026), [code](https://github.com/agentic-learning-ai-lab/adajepa) | Closed-loop self-supervised adaptation inside MPC. Generic online world-model adaptation is not a contribution. |
| [Controlled WM identifiability, July 24](https://arxiv.org/abs/2607.22430) | Under explicit Gaussian assumptions, representation identification and action excitation are distinct requirements. Good on-policy prediction does not identify every counterfactual action effect. |
| [SCALE, August 17](https://arxiv.org/abs/2608.16287) | Uses training-time task-state distances to improve the geometry consumed by planning. Better latent distances alone are crowded and supervision must be matched. |
| [LEAP, September 3](https://arxiv.org/abs/2609.03294) | Refines actions through frozen LeWM with a goal score and terminal state energy. Merely combining a state head and gradient planner is insufficient. |
| [Habit, Physics, and Nuisance, September 6](https://arxiv.org/abs/2609.09210) | Explicitly separates operator action selection, shared physics and observation nuisance, using interventions and thin-interface adaptation. This is the closest overlap with the motivating explanation. |
| [SG-JEPA, September 9](https://arxiv.org/abs/2609.10464) | Conditions on a supplied gravity scalar and optimizes recursive latent rollouts. Unknown-parameter inference is listed as future work. Its control experiments train a separate gravity-conditioned diffusion policy on frozen encoder features; the predictor is not queried online. |

The table is a targeted audit, not an exhaustive novelty certificate. Repository existence is not reproduction. Most listed 2026 items are preprints unless venue acceptance is separately established.

## Candidate A — Compact world models that recalibrate appearance and dynamics separately

### Model and learned artifact

Start with LeWM. Add small history encoders producing an observation-calibration context and a dynamics context. The first conditions visual encoding; the second conditions the action-dependent transition predictor. Parameter count should stay near the original small-model regime; the actual count will be measured rather than advertised in advance.

Use simulated episodes with independently randomized image formation and physical settings. Initial changes should be photometric appearance and genuinely changed dynamics such as friction, damping or action gain. Camera rotations are a later extension because they can also change the apparent action coordinate frame, making a naive nuisance-invariance objective incorrect.

The model receives image/action history and no true physics parameters at test time. It then predicts latent consequences for candidate actions, enabling ordinary MPC. A lightweight decoder may be trained solely for demonstrations; quantitative control evidence must not rely on prettier decoded videos.

### Proposed training mechanism

1. Generate episode groups crossing appearance and dynamics factors. Re-render the same simulator trajectory under different appearances. For altered dynamics, roll out anew from a saved state with matched actions; do not falsely treat these diverging trajectories as identical states.
2. Infer contexts from separate support snippets. Use a distinct query snippet for prediction so contexts cannot simply memorize the queried state.
3. Train regularized latent prediction with crossed support/query pairings. Support clips with the same physical configuration but different appearance should produce interchangeable dynamics contexts. Appearance support should remain interchangeable across physical configurations when the rendering transformation is actually shared.
4. Add context-swap training only where the data generation provides a valid target. Pairing metadata supplies weak supervision during training; disclose this extra supervision and simulation budget. Do not describe the method as entirely unsupervised.
5. At deployment, infer/update contexts from a short sliding history. Test a simple amortized version first. Selective gradient adaptation or active diagnostic actions are optional extensions, not required features of the initial contribution.

The key hypothesis is that independently crossed training prevents a model from using appearance as a shortcut for physical context, improving planning on new combinations of familiar factors.

### Specific distinction from closest work

Against Habit/Physics/Nuisance, **physics itself is allowed to change**, so permanently freezing one dynamics function is not assumed correct. Against ICWM, the proposed claim is not context inference: it is the benefit of intervention-paired, separate context learning for explicit predictive rollouts under compositional mixed shifts. Against SG-JEPA, physical parameters are not supplied and the model is used directly in planning. Against AdaJEPA, the primary mechanism is learned inference of changing factors, tested against unconstrained online updates.

These differences define a testable research opportunity, not proof that no prior paper implements the combination.

### Essential evaluations

- One canonical condition, appearance-only shifts, dynamics-only shifts, and mixed shifts. Include abrupt changes within an episode, after stationary performance is established.
- Hold out appearance×dynamics combinations, with each component seen separately during training. Distinguish combination generalization from extrapolation beyond all trained values.
- Compare standard LeWM; domain-randomized LeWM; one joint context; two contexts without pairing; paired training with a shared context; our full training; AdaJEPA-style adaptation; and a supplied-parameter oracle. Keep total parameters, trajectories, image views and simulation queries matched where possible.
- Evaluate prediction at multiple horizons, action ranking, actual MPC success and interaction count. Latent MSE alone can improve through altered representation scale and is insufficient evidence.
- Use simulator state only for generation and clearly separated diagnostic probes. No hidden access to it at deployment.
- Release complete trained checkpoints per demonstrated action schema/environment family, plus context adapters, configuration, data generator, episode splits and normalization metadata. Do not promise one universal checkpoint for arbitrary robots or medical devices.

### First-day falsification test

On one cheap environment, create small independently crossed appearance/dynamics variations. Compare joint adaptation, perception-only adaptation, dynamics-only adaptation and an oracle choice. If which component changes makes no material difference, the project lacks the proposed mechanism's headroom. Then compare joint versus separated context models on a held-out combination before scaling.

### Theory boundary

Independent factors are not automatically identifiable from passive observations. Different observation transformations and physical systems can produce the same observed history. Limited action excitation compounds this ambiguity. We should aim for explicit counterexamples, measurable uncertainty and a restricted toy analysis, not claim universal causal disentanglement. A truthful result may be that adequate diagnostic actions are necessary in some conditions.

## Candidate B — Train world models on action effects, not only recorded transitions

This is a simpler fallback that produces a reusable predictor checkpoint. A model can fit observed transitions while providing misleading local action sensitivities to a planner. Collect paired simulator continuations from the exact same state under nearby action alternatives; train both predicted successors and the **difference between their encoded outcomes**. Shared rendering eliminates some visual nuisance in the difference target. A predictor trained on actual alternative outcomes is more informative than one trained to reject a shuffled action/observation pair.

Potential implementation: continue training the official encoder/predictor with the original objective plus a bounded action-effect loss. Begin with a fixed encoder to prevent changing representation scale from masquerading as improved action sensitivity. Then consider joint training only if it adds measured value. Include multiple action perturbation magnitudes because contacts may be discontinuous and a single local derivative may be misleading.

The candidate contribution would be a budgeted branch-selection and training objective that improves action-effect prediction and planning per extra simulator interaction. Compare with the same extra transitions sampled independently, random matched branches, wrong-action contrastive training, plain continuation training, multi-horizon training, and RC-aux. Evaluate under both sampling and gradient planners to establish whether the effect is general predictive improvement or optimizer-specific.

Risks: counterfactual supervision, Sobolev/derivative supervision and action sensitivity are established ideas. UWM-JEPA explicitly studies counterfactual training; the controlled-identifiability paper studies action coverage; the Habit/Physics/Nuisance paper includes wrong-action sensitivity. This fallback is computationally feasible but may prove more incremental than Candidate A. Do not adopt it merely because it is easy to code.

## Five-day prioritization

Spend the initial half-day on baseline and simulator/data integrity, and the following half-day on a mechanism diagnostic. Run one independent task per GPU; training across nodes is unnecessary for these small models. Two or three thoroughly checked environments with repeated seeds and matched controls are more persuasive than a wide unverified benchmark list.

Choose Candidate A if its diagnostic exposes distinct appearance and dynamics repair needs and the cross-combination pilot supports separate context learning. Otherwise reassess before a large grid. Keep medical video as an optional offline action-conditioned prediction test if actual synchronized actions are available. Static image datasets do not establish controllable world modeling, and offline medical prediction alone does not demonstrate successful medical planning or safe clinical intervention.
