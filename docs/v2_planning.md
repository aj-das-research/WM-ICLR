# v2 planning harness (E4): official LeWM protocol

Code: `scripts/v2/planning_eval.py`, `src/shiftwm/v2/planning/{interface,lewm,protocol,compat}.py`.
Slurm: `slurm_jebel/planning_eval.sbatch` (standalone) and lane tasks
`slurm_jebel/tasks/planning_lewm_repro.sh`, `slurm_jebel/tasks/planning_eval_model.sh`
(`sbatch -J NAME slurm_jebel/job.sbatch slurm_jebel/tasks/X.sh`).
Results: `results/v2/planning/<env>/<model>/<seed>[_<tag>].json`.

## Official protocol (le-wm@8edfeb33 `config/eval/*.yaml`, swm@4821c8e6)

| | PushT | TwoRoom | Reacher |
|---|---|---|---|
| env | `swm/PushT-v1` | `swm/TwoRoom-v1` | `swm/ReacherDMControl-v0`, task `qpos_match` |
| dataset (start/goal source) | `pusht_expert_train.h5` | `tworoom.h5` | `reacher.h5` (= `dmc/reacher_random`) |
| state setters | `_set_state(state)`, `_set_goal_state(goal_state)` | `_set_state(proprio)`, `_set_goal_state(goal_proprio)` | `set_state(qpos,qvel)`, `set_target_qpos(goal_qpos)` |
| success | block pos+agent err <20 px and angle <pi/9 (any step) | agent within 16 px of target | joints match target qpos |

Shared settings: 50 episodes; seed 42 selects the start rows and seeds CEM. Start rows are drawn uniformly
without replacement from all dataset rows that have at least 25 more steps in the same episode, using
`rng.choice(len(valid)-1)` exactly as le-wm `eval.py` does. The goal is the dataset frame and state
**25 native steps later** in the same trajectory. The eval budget is **50 env steps**
(`max_episode_steps = 100`), and an episode counts as a success if the env terminates at any step
within the budget. CEM uses **300 samples × 30 iterations**, top-30 elites, initial var 1, batch 1
(one env at a time), and the first sample is the current mean. It plans **horizon 5** blocks with
**action_block 5** (25 native steps) and **receding horizon 5**, so the controller replans every 25
steps (≤2 plans per episode), and warm-starts from the unexecuted tail. Context is `history_len = 1`
(one frame). Cost is the summed squared error between the last predicted latent and the goal latent.
Images are rendered at 224×224, turned to float in [0, 1], ImageNet-normalised and resized to 224.
Actions are z-scored with a StandardScaler fit on the full dataset's action column (population std).

The paper (arXiv 2603.19312, App. D/F.1) states two deviations from the released configs. First,
TwoRoom uses budget **150** and goal **+100**. Second, CEM uses **10** iterations outside PushT. Run
these with `--variant paper`. It also says "history length 3 for PushT/Cube, 1 for TwoRoom". The
released checkpoints have predictor `num_frames=3`, and the released eval uses `history_len=1`. Set
it with `--history-len 3`.

The datasets **are** required. They supply the start states, goal images and goal states, and the
action scaler. The released HDF5 pixels use the Blosc filter (id 32001), so `hdf5plugin` is needed. It is
installed outside `.venv` in `.cache/v2_planning_pydeps`, which is gitignored, and is added to
`sys.path` automatically.

## Reproduction of the released LeWM checkpoints (H200 NVL)

| env | paper LeWM (Fig. 6) | ours seed 42 (official) | ours seeds 42/43/44 | mean |
|---|---|---|---|---|
| PushT | 96.0 ± 2.8 (3 train seeds, Tab. 5) | **45/50 = 90%** | 90 / 88 / 86 | 88.0 |
| TwoRoom | 87 | pending (`planning_lewm_repro.sh`) | | |
| Reacher | 86 | pending | | |

Other PushT results: `--backend native` (upstream `ShootingCostEvaluator(LeWM, GoalMSE)`) gives 45/50,
identical to the adapter. `--history-len 3` gives 45/44/43, the same totals, with 9 episodes differing
in success step. The earlier historical-snapshot reproduction (swm@abdced49, 2026-09-18) got 46/50
on PushT and 39/50 on Reacher. The paper's 96% averages three training seeds; the HF checkpoint is one.
Paper baselines for reference: PushT DINO-WM 74 (92 with proprio), PLDM 78. Reacher DINO-WM 79,
PLDM 78. TwoRoom DINO-WM 100, PLDM 97.

Timing (PushT): **0.41 s per CEM plan per env** (9,000 candidate rollouts; paper reports 0.98 s). That
is 0.50 s of planning per episode and 0.63 s of total eval wall time per episode. One 50-episode eval
takes about 32 s. Peak GPU memory is 0.14 GB.

## Plugging in a new predictor

1. Subclass `shiftwm.v2.planning.interface.PlanningPredictor` (an `nn.Module`):
   ```python
   class MyPredictor(PlanningPredictor):
       name = "shiftwm_v2"
       def encode(self, pixels):            # (N,T,3,224,224) ImageNet-normalised -> (N,T,*latent)
           ...
       def rollout(self, latent_hist, action_hist, actions):
           # latent_hist (N,H,*latent), action_hist (N,H-1,A) or None, actions (N,T,A)
           # A = env_action_dim*5 (PushT/TwoRoom/Reacher: 10), z-scored by the dataset StandardScaler;
           # actions[:,0] is the block executed from the current frame.
           return preds                     # (N,T,*latent): state after actions[:, :t+1]
       def cost(self, pred_latents, goal_latent):   # optional; default = LeWM last-step SSE
           ...                              # (N,T,*latent), (N,*latent) -> (N,)
   ```
   `N` = one env × 300 candidates. Latents may have any trailing shape, such as a `(256, 384)`
   patch grid. Context pixels and the goal are encoded once per env per plan and cached across CEM
   iterations.
2. Provide a factory `build(env, ckpt=None, device="cuda", **_) -> (predictor, meta_dict)`.
   See `shiftwm.v2.planning.lewm:build`.
3. Run `python scripts/v2/planning_eval.py --env pusht --predictor my.module:build --ckpt PATH --model NAME --seed 42`,
   or on a lane:
   `sbatch -J v2-plan-NAME --export=ALL,PREDICTOR=my.module:build,CKPT=PATH,MODEL=NAME slurm_jebel/job.sbatch slurm_jebel/tasks/planning_eval_model.sh`.
   Train with the same action convention: 5-step action blocks, z-scored with the dataset
   StandardScaler (`shiftwm.v2.planning.compat.StandardScaler` fit on the action column).
4. `--check-adapter` checks the interface against native LeWM rollout+GoalMSE at H=1 and H=3.
   Both give max|Δcost| = 0.

The solver, MPC loop, env reset and success accounting are the unmodified upstream
`CEMSolver`, `WorldModelPolicy` and `World.evaluate`. The harness adds only a timing subclass and a
per-step success recorder. Any result with `protocol.overrides` non-empty (for example a reduced
`--num-samples`) has `protocol.is_official = false`.

## Notes / caveats
- swm@4821c8e6 computes the CEM elite std with `correction=0`. Older swm used the unbiased std, a
  factor of about 0.98 at k=30. The stack is otherwise the pinned upstream.
- Solve timings include CUDA synchronisation. The first solve includes warm-up. The number of envs in
  the second solve can differ by ±1 between identical runs because of GPU non-determinism.
- The old in-repo evaluator (`src/shiftwm/evaluate.py`) defaults to 128 samples × 5 iterations and
  16 elites. That reduced budget explains the low v1 paper numbers. It is not used here.
