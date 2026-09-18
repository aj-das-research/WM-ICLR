# Novelty boundary: planning geometry and real-video extension

Primary sources checked on 19 September 2026. These findings constrain claims;
they are not reproduced comparison results.

**DA-LeWM (19 August 2026)** already studies the gap between informative latent
features and useful planning distances. It introduces Plan-Real and CEM-stage
rank diagnostics, analyzes conditions for preserving real-cost ordering, and
adds inverse-dynamics and goal-action objectives to LeWM. Our recorded-branch
rank audit must acknowledge this precedent; rank correlation between imagined
and executed plans is not a new contribution. Its published control rates are
not directly comparable to our different budgets, inputs, shifts and splits.
[Primary paper](https://arxiv.org/abs/2608.18746),
[method and experiments](https://arxiv.org/html/2608.18746v1).

**Monotone Planning Costs (10 August 2026)** already combines a frozen DINO
representation, autoregressive rollout training and a cost-ranking objective
for image-goal navigation. It evaluates on real navigation data and describes
physical robot deployment. Therefore, neither switching to DINO, adding
recursive training, ranking candidate costs nor merely adding real footage
establishes novelty.
[Primary paper](https://arxiv.org/abs/2608.09073),
[method and experiments](https://arxiv.org/html/2608.09073v1).

The inspected primary pages and targeted repository searches did not establish
a runnable official repository/checkpoint for either paper. This is an
unverified-release status, not proof that no release exists. Keep them in
related work and scope comparisons carefully; do not list them as reproduced.

Our defensible research question remains whether **causal support-dependent
adaptation improves future prediction and decisions under naturally varying
observation/dynamics conditions**, beyond a learned shared-context control and
an action-free history model. The current simulation evidence is mixed. The
observation-gain experiment is a capacity ablation; the new DROID study tests
real captured imagery. Neither is automatically a novel algorithm or a
state-of-the-art result. The paper's contribution statement must track the
eventual matched measurements and cannot promise universal applicability or
acceptance.
