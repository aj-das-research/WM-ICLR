# Drone development failure-mode audit

**Settling is secondary in the recorded failures.** Of 114 failed task-instances, 111 (97.4%) never enter the 4 cm XY goal region at a recorded native step. Three (2.6%) enter but fail solely the speed criterion, for 4 native samples total. All occur after support. No near-goal sample is blocked by altitude, crash, or workspace escape.

The audit covers all 18 registered transformer/GRU models: 144 evaluations of the same 8 development tasks, with 30 successes. None succeeds during the 10 support commands; all 30 successes occur after planning begins, and all 144 evaluations reach planned control. Counts are repeated task-instances, not 144 independent goals. Only appearance 1 / dynamics 1 is represented.

Across failed trajectories, 12 task-instances eventually escape the workspace; none crashes or violates altitude tolerance. One of the 3 speed-blocked trajectories later escapes. Native metrics are recorded at 10 Hz; crossings between samples are unobserved.

| Family | Mode | Seed | Failed | Never XY<4cm | Speed only: planned tasks / steps | Speed only: support tasks / steps | Any escape |
|---|---|---:|---:|---:|---:|---:|---:|
| gru | constant dynamics | 0 | 7 | 6 | 1 / 1 | 0 / 0 | 1 |
| gru | constant dynamics | 1 | 4 | 4 | 0 / 0 | 0 / 0 | 0 |
| gru | constant dynamics | 2 | 6 | 6 | 0 / 0 | 0 / 0 | 3 |
| gru | ShiftWM (ours) | 0 | 7 | 7 | 0 / 0 | 0 / 0 | 1 |
| gru | ShiftWM (ours) | 1 | 6 | 5 | 1 / 1 | 0 / 0 | 0 |
| gru | ShiftWM (ours) | 2 | 7 | 6 | 1 / 2 | 0 / 0 | 3 |
| gru | framewise | 0 | 7 | 7 | 0 / 0 | 0 / 0 | 1 |
| gru | framewise | 1 | 5 | 5 | 0 / 0 | 0 / 0 | 0 |
| gru | framewise | 2 | 6 | 6 | 0 / 0 | 0 / 0 | 3 |
| transformer | constant dynamics | 0 | 7 | 7 | 0 / 0 | 0 / 0 | 0 |
| transformer | constant dynamics | 1 | 7 | 7 | 0 / 0 | 0 / 0 | 0 |
| transformer | constant dynamics | 2 | 6 | 6 | 0 / 0 | 0 / 0 | 0 |
| transformer | ShiftWM (ours) | 0 | 7 | 7 | 0 / 0 | 0 / 0 | 0 |
| transformer | ShiftWM (ours) | 1 | 6 | 6 | 0 / 0 | 0 / 0 | 0 |
| transformer | ShiftWM (ours) | 2 | 6 | 6 | 0 / 0 | 0 / 0 | 0 |
| transformer | framewise | 0 | 7 | 7 | 0 / 0 | 0 / 0 | 0 |
| transformer | framewise | 1 | 6 | 6 | 0 / 0 | 0 / 0 | 0 |
| transformer | framewise | 2 | 7 | 7 | 0 / 0 | 0 / 0 | 0 |

The three speed-blocked cases are:

| Run | Task seed | Native steps | XY distances (cm) | Speeds (cm/s) |
|---|---:|---|---|---|
| drone_gru_constant_dynamics_s0 | 2053002 | 46 | 3.953 | 7.252 |
| drone_gru_factorized_s1 | 2053002 | 41 | 3.880 | 10.965 |
| drone_gru_factorized_s2 | 2053010 | 62, 63 | 3.814, 3.818 | 10.866, 9.287 |

The first 10 native commands are support; subsequent commands are planner-selected. Speed-only blockers occur at steps 41, 46, 62, 63, briefly within the boundary at 3.81–3.95 cm, with speeds 7.25–10.97 cm/s against a strict 6 cm/s limit. Waiting or braking has not been tested and is not established to recover these tasks.

CEM minimizes final predicted latent goal MSE (`src/shiftwm/evaluate.py`, `LatentGoalCost.get_cost`) without an explicit speed penalty. The mismatch exists, but these traces do not support it as the principal explanation of failure. Prioritize goal-region geometry and action ranking; evaluate a settling-aware cost as a secondary hypothesis without changing success criteria.

All native success flags were recomputed from unchanged strict criteria. Native counts, decision boundaries, final metrics, first-success/support flags, matched task sets and identical support prefixes passed consistency checks. Source hashes match all 18 stored identities. [Evidence JSON](evidence/drone_failure_modes.json) includes SHA256 hashes of all 18 input files, source hashes, phase/family/mode/seed counts and every failed task. NPZ hashes are retained as references; the audit uses JSON native measurements and launches no simulations.
