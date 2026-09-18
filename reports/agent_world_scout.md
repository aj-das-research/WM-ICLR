# Agent and world-model research scout — 18 September 2026

Scope: primary-source research plus GitHub metadata/tree/README inspection; no model weights downloaded, simulator installed, or experiment executed. Compute assumed from the completed audit: three independent 32 GB GPU jobs, not a distributed three-GPU allocation. Repository/API snapshots are in `references/frontier_agent_scout/`.

## Recommendation

The strongest five-day world-model foundation is **LeWorldModel plus stable-worldmodel**, with **AdaJEPA as the mandatory adaptive baseline**. These provide small models, simulation, released artifacts, and a genuine deployment problem. GUI world-model RL is newer but has substantially greater infrastructure, memory, judge, and emulator risks. Frontier relevance is not the same as foundational model scale.

## Six inspected foundations

| Work | Verified release and implementation | Assets and practical assessment |
|---|---|---|
| [LeWorldModel, March 2026](https://arxiv.org/abs/2603.19312) | [Official MIT repository](https://github.com/lucas-maes/le-wm), real `train.py`, `eval.py`, `jepa.py`, configs | Authors report approximately 15M parameters and single-GPU training in hours. [HF PushT checkpoint](https://huggingface.co/quentinll/lewm-pusht) API confirms ungated `weights.pt` and `config.json`, approximately 72 MB total storage; corresponding Cube, TwoRooms, and Reacher checkpoint links exist. Best compute fit, still requires local throughput verification. |
| [stable-worldmodel, May 2026](https://arxiv.org/abs/2605.21800) | [Official MIT repository](https://github.com/galilai-group/stable-worldmodel), 414 tree entries; environment wrappers, solvers, data pipeline, baseline scripts | Supplies LeWM and DINO-WM implementations, CEM/iCEM/MPPI/gradient planners and controlled visual/physical factors across PushT, TwoRoom, DMControl, OGBench, etc. Do not count framework as a new algorithm. Active APIs require compatible revision pinning. |
| [AdaJEPA, June 30 2026](https://arxiv.org/abs/2606.32026) | [Official MIT repository](https://github.com/agentic-learning-ai-lab/adajepa), 256 entries, planning/model/environment code | Released checkpoint and evaluation-data Google Drive links documented, including separate medium-maze data. Applies observed-transition self-supervision after each MPC action chunk, then replans. Evaluates visual, shape, dynamics and layout shifts. Drive assets linked, not downloaded or checksum verified. Direct novelty collision with generic world-model TTA. |
| [DINO-WM, ICML 2025](https://proceedings.mlr.press/v267/zhou25t.html) | [Official MIT repository](https://github.com/gaoyuezhou/dino_wm), 236 entries, real planning/train/model code | OSF pretrained PointMaze, PushT and Wall checkpoint/data links documented. Valuable second architecture; older MuJoCo installation instructions add friction. LeWM framework's reproduction is easier but must be distinguished from original code. |
| [WM-R1, August 27 2026](https://arxiv.org/abs/2608.27508) | [Official Apache-2.0 repository](https://github.com/genalyu/WM-R1), 804 entries, veRL/Ray code and Code2World integration | README defaults to two GPUs per node and multi-node training: one world-model server and one actor/rollout worker. Qwen2.5-VL-3B agent and 8B Code2World model links exist; a trained WM-R1 agent checkpoint was not verified. Reproduction on our single-GPU-per-job allocation is not established. Do not select full RL replication as five-day core. |
| [Qwen-AgentWorld, June 24 2026](https://arxiv.org/abs/2606.24597) | [Official Apache-2.0 repository](https://github.com/QwenLM/Qwen-AgentWorld), 43 tree entries and `eval/eval.py` | Released [35B-A3B model](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B) and [AgentWorldBench](https://huggingface.co/datasets/Qwen/AgentWorldBench), seven language-environment domains. 35B total parameters still occupy memory despite 3B active; BF16 weights alone approximately 70 GB. Quantized/offloaded deployment unverified. Evaluator uses model inference plus LLM judge; entirely local evaluation requires provisioning both. Not a small-model critical path. |

## Newest relevant work that is not a verified runnable foundation

[Discriminative World Models for Web Agents](https://arxiv.org/abs/2609.02885), September 2: predicted-state matching over alternative actions rather than conventional generative next-state supervision; action ranking and WebArena-Lite evaluation. The [project page](https://dhruvpendharkar.github.io/dwm/) was fetched directly: only paper external link found, no official GitHub/model release verified. It is important novelty context, but does not pass our code-availability requirement.

The parent audit also identifies September LEAP and August SCALE plus reachability and amortized-planning methods. Improving a latent distance or adding a faster planner alone is crowded.

## Concrete hypothesis: diagnose the cause of mismatch before adapting

Working concept: **Did the camera change, or did the world change?**

A pushing robot can miss its target because the camera moved or because friction changed. Both create large prediction error. Updating the same encoder/predictor parameters in both cases may produce an apparently improved training loss while corrupting useful physical knowledge. This is a testable hypothesis, not a finding we have established.

Proposed algorithmic direction:

1. Maintain a compact belief over observation change, dynamics change, both, or insufficient evidence.
2. Compare lightweight encoder-only and dynamics-only residual updates using held-out *observed transitions*. Score in a fixed reference space or use an explicitly justified invariant target, to prevent a moving latent target from making either update trivially win.
3. If ambiguous, select a short, bounded action probe whose candidate predicted outcomes most distinguish the hypotheses, subject to a task-progress budget.
4. Route updates to the identified component and return to task execution. Count probe actions, latency, and failed episodes against the method.

Potential contribution: **active identification of the adaptation locus under unknown mixed shifts**, rather than world-model adaptation, confidence gating, or generic information-gain exploration themselves. Could release small adapters/checkpoints and a reproducible mixed-shift suite. Actual benefit: recover from changed sensing/physics without retraining the full controller or relying on an engineer to label the shift type.

Critical scientific limitation: sensory and dynamical changes can be observationally indistinguishable. No universal identification claim is legitimate. Theory, if included, should specify when actions distinguish hypotheses and when the algorithm must abstain. All candidate states must be judged with comparable representations. Simulator privileged shift labels may be used for evaluation/oracle controls only, not by the deployed controller.

## Closest collisions and required controls

- **AdaJEPA (2026)** already performs lightweight closed-loop TTA and ablates encoder/predictor adaptation choices. We must demonstrate benefit over its best *fixed* subset and equal-budget updates, especially mixed and changing shifts.
- **[MoVie, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/43b77cef2a83a25aa27d3271d209e4fd-Paper-Conference.pdf)** adapts visual encoding using frozen dynamics for new views. Observation-only adaptation is established.
- **[LACE repository](https://github.com/bourkefloyd/lace-tta)** already studies anchored targets around a miniature AdaJEPA reproduction. This is collision evidence, not a verified accepted-paper baseline. Anchoring a target is not our novelty.
- Classical dual control, active system identification, causal modular adaptation, and ensemble information-gain exploration must be reviewed before asserting originality. The present search does not establish that active routing is absent from prior art.
- Oracle routing, fixed encoder-only, fixed predictor-only, joint adaptation, no adaptation, residual-error routing, random probes, matched-cost information-gain probes, and adaptation without probes are mandatory controls.

## Minimal five-day decision experiment

Use pretrained LeWM PushT and Reacher, plus one AdaJEPA released task if assets load. Independently vary appearance and physical factors; hold out factor values and their combinations. Start with four groups: no shift, observation-only, dynamics-only, mixed. Add within-episode switching only after reproduction. Evaluate success, extra action count, recovery time, adaptation compute, and shift diagnosis against withheld simulator labels. Second architecture only after the first mechanism passes.

Day-one kill criteria: baseline cannot run reliably within the budget, or oracle selection of adaptation locus provides negligible improvement over fixed joint adaptation. If oracle routing has no useful headroom, there is no reason to learn a router. Day-two kill criterion: learned routing/probing fails to outperform cheaper controls on held-out development shifts. This would be a pilot, not evidence of guaranteed ICLR acceptance.

## Evidence boundary

Repository source exists and asset links were checked; runtime correctness and published results are not independently reproduced. LeWM's single-GPU time is an author report. There is no basis to claim general compatibility with arbitrary VLAs, agents or world models. This route offers smaller-scale, causal-control research with plausible fast iteration; photorealistic video-world-model pretraining and general-agent RL are outside the verified five-day compute envelope.
