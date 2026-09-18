# Focused proposal: reusable compact world models

18 September 2026. User direction: world models only; prioritize reusable trained checkpoints; medical optional. “Lord model” is interpreted as LeWorldModel. This supersedes the browser-agent/VLA-controller recommendation as the current research direction, while preserving earlier reports. This is the original brainstorming document, preserved as design history. Implementation and full training have since started; [execution notes](world_execution.md), [the current manuscript](../paper/main.tex), and machine-readable run records define the implemented scope and measured status. Candidate features below are not all implemented or evaluated.

## Recommendation

Build a **small context-conditioned latent world model that can distinguish changes in observation from changes in physical dynamics**, starting from LeWorldModel. Train a reusable model with separately inferred observation and dynamics contexts. Test whether it plans successfully when familiar changes appear in unfamiliar combinations and when the mechanism changes during an episode.

Working title: *What Changed? Compact World Models for Unseen Combinations of Observation and Dynamics Shifts.* Title is provisional. Novelty remains a hypothesis; factorization, context inference, domain randomization and adaptation are established ideas.

Example: a pushing system fails after either a camera/lighting change, a friction/mass change, or both. A useful predictor should recalibrate observation processing without inventing new physics in the first case, revise dynamics in the second, and handle both in the third. We should not hard-code that physics is always invariant.

The trained product must predict action-conditioned future latents and support planning. A standalone intervention classifier would not satisfy this project direction.

## Architecture and training candidate

Reuse LeWM's encoder, action embedder, autoregressive predictor, existing latent objective and anti-collapse regularization. Introduce only small context modules initially:

- Observation context from recent observations and fixed-view/appearance evidence.
- Dynamics context from a short chronological history of observations and **executed** actions.
- Observation context modulates the encoder or its adapter; dynamics context modulates the action-conditioned predictor. Start with small FiLM/low-rank modules rather than enlarging the backbone.
- Maintain uncertainty or an explicit insufficient-evidence state when the two causes cannot be distinguished. Avoid translating a heuristic confidence score into a formal guarantee.

At deployment, the model receives recent images/actions and updates its contexts; initially no gradient-based weight updates. For a novel target goal, an existing CEM/MPC planner uses the resulting world model. Optional online weight updates are a separate ablation; they must not obscure the primary contribution.

Training data would independently vary appearance and physical factors in an existing simulator. Preserve complete trajectory identity and intervention metadata. Paired renders of the *same* physical trajectory provide an appearance-invariance signal; separate rollouts under changed physics provide dynamics evidence. Do not claim that a camera transform preserves visibility or information when it introduces occlusion.

Candidate training terms: upstream predictive/anti-collapse losses; consistency of dynamics context across paired renderings; consistency of observation context across physically different rollouts sharing the observation setup; and held-out transition prediction under swapped compatible contexts. Avoid forcing physical-state latents to agree for actually different states. Fixed targets or shared reference coordinates are needed to prevent changing encoder geometry from creating a trivial win. Whether these losses help is the core experiment, not a presumed result.

Use intervention IDs to construct training pairs, but do not feed true simulator physics parameters, shift type, goal reward, or future frames to the inference context module. Report any privileged training information explicitly. Matching baselines receive the same data and interventions. Contexts need not recover uniquely identifiable physical variables; call them functional factors unless identified quantities are actually evaluated.

## What could be new, and what is already known

| Primary work | Existing contribution | What our study would still have to establish |
|---|---|---|
| LeWM https://arxiv.org/abs/2603.19312 | Small end-to-end latent predictive model with planning | Base architecture is reused, not ours |
| AdaJEPA https://arxiv.org/abs/2606.32026 | Closed-loop encoder/predictor TTA from observed transitions | Benefit of a trained reusable context mechanism over equal-budget updates |
| ICWM https://arxiv.org/abs/2606.26025 | Inferring system properties from short interaction histories | Separating observation and dynamics must add value beyond a single history context |
| SG-JEPA https://arxiv.org/abs/2609.10464 | Known physical-parameter conditioning and rollout-trained features | Unknown shift inference and actual world-model planning, rather than supplying gravity |
| Habit, Physics, Nuisance https://arxiv.org/abs/2609.09210 | Separate physics/observation/operator effects, thin-interface adaptation | Handle genuinely changing dynamics and unknown mixtures, not claim first separation |
| MoVie, NeurIPS2023 | Adapt visual encoding while freezing dynamics | Perception-only adaptation is an essential baseline, not a new idea |

The potential claim is **compositional transfer of a compact learned world model to unseen observation–dynamics combinations**, under an equal data/model/compute budget. A larger benchmark count or two extra heads alone cannot support this claim. Review conditional world models, hidden-parameter MDPs, modular dynamics and meta-learning before claiming novelty.

## Benchmark design

Primary natural visual-control environments: PushT and Reacher. Add TwoRooms or Cube after mechanism validation. These use rendered image–action sequences with environment interactions, not static ImageNet classification.

Split environments into no shift, observation-only, dynamics-only and mixed shifts. Train on a sparse, prespecified set of factor combinations and test on held-out combinations, unseen factor magnitudes, new goal states and new initial states. Keep entire trajectories in one split. Separate interpolation from extrapolation. Add within-episode switches only after stationary-shift reproduction succeeds. Changing appearance and physics simultaneously must not accidentally expose the shift label through timestamps/config IDs.

Baselines: original frozen LeWM; continued training on identical randomized data; single-context model with matched parameters; separate-context model without pairing constraints; fixed encoder-only/dynamics-only/joint adaptation; AdaJEPA under its own and matched protocol; oracle factor/context access as a diagnostic ceiling. Same planner and candidate budget are the primary comparison. A second solver is a portability check, not a replacement for the main control.

Main metrics: closed-loop success, steps to goal, adaptation/recovery interaction cost, total latency and memory. Supporting metrics: fixed-coordinate multi-step prediction, action sensitivity, context swap effects and selective adaptation interference. Latent loss alone can be incomparable across encoders; use frozen probes/coordinates and downstream success. Three seeds on core comparisons with paired initial states and intervals. All extra data collection and context identification steps count against the method.

Day-one diagnostic: determine whether privileged knowledge of the shift/adaptation locus improves planning enough over the best fixed strategy to motivate a learned mechanism. If no headroom exists, stop this hypothesis. Day-two: separate contexts must outperform single context and ordinary domain randomization on held-out development combinations. Preserve negative findings. Broad claims need both environments, not one favorable condition.

## Reusable checkpoint release

Release **full model weights** for successful environment families, not just a test-time cache:

1. Encoder, action embedder, predictor and context-inference weights in a weights-only format, plus exact configuration and upstream revision.
2. Fine-tuning/adaptation-only weights where useful, including the required base-model hash.
3. Preprocessing, image size, action units/ranges, frame rate, observation-history length and context reset rules.
4. A model card defining tested environment family, unseen conditions, failures, training data and allowed uses.
5. A loader and examples for encoding observations, updating context, rolling out candidate actions and planning to a goal image. The API is a planned deliverable, not existing functionality.

Practical reuse: new goals, new initial states and tested variations within a compatible environment/action interface. Do not claim a PushT checkpoint works directly on endoscopy, arbitrary robots, or different action dimensions. The framework can transfer; a new domain generally needs new data and trained weights. Upstream approximately15M parameter size is a reference; our final size and hardware costs will be measured.

## Healthcare option

A meaningful optional extension exists: Open-H CUHK robotic endoscopy contains synchronized images, motor actions and recorded state in a phantom setup. The `find_greater_curvature` subset has462 episodes/107488 frames according to its metadata, with2D motor actions. Obtain only the selected subset, not the full multi-terabyte collection. Source: https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment . Detailed access/metadata evidence: `medical_world_fit.md`.

Offline endoscopy supports action-conditioned prediction and representation transfer, but not online planning claims without an interactive environment or actual robot. Appearance perturbations can be synthesized, but genuine changed tissue/mechanical dynamics cannot be invented by relabeling images. An optional endoscopy checkpoint therefore has a separate, narrow forecasting/representation claim. CathSim or LapGym may support interactive medical-simulation evidence, at additional installation cost.

Static chest X-rays/CT slices/histology are poor direct fits for this action-conditioned method. Cine echo or surgical video without actions can support temporal prediction, but not causal predictions about treatment or instrument control. Do not convert classification into a purported world model by renaming it.

Cosmos-H-Dreams is a real medical world-model release, not an absent possibility. The scout reports official minimum12GBVRAM but driverR580+/CUDA13; our audited driver570 does not meet the documented setup. Its2B model also changes the project scale. Exclude from the five-day critical path unless compatibility is established without relying on an administrator upgrade.

## Five-day implementation plan, after the research design is fixed

Budget240 GPU-hours (planning allowance, not measured demand): day1 reproduction/data/diagnostics30; day2 context modules and minimal training50; day3 core held-out-combination experiments70; day4 seeds, ablations and second environment70; day5 reproducible checkpoint packaging and reruns20. Three independent workers within verified account quotas. Download and benchmark small released checkpoints first; no foundation-model pretraining.

Healthcare offline transfer is optional and cannot displace the controls needed to establish the main mechanism. If all three natural environments plus medical cannot fit, keep two natural environments with rigorous controls. Paper, model cards, demo and source package can be built while GPU jobs run. Five days can test this hypothesis; neither full scope nor ICLR acceptance is guaranteed.
