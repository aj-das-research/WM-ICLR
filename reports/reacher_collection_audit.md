# Reacher collection termination audit

**Finding: the current 320-native-step collection does not cross a termination
or time-limit reset. No dataset or generator changes were made for this audit.**

## Source-level reasoning

`external/stable-worldmodel/stable_worldmodel/envs/dmcontrol/custom_tasks/reacher.py`
initializes `ReacherQPosMatchTask.target_qpos` to `None`. Its overridden
`get_termination` immediately returns `None` until a target configuration is
explicitly assigned. The collector's `make_env` and `collect_episode` never call
`set_target_qpos` and never provide a target through reset options. Reaching the
random target geometry inherited from ordinary Reacher can affect reward, but
does not terminate this overridden qpos-match task during collection.

The wrapper repeats each native action twice. The installed Reacher model's
physics and control timestep are both 0.02 seconds; its 20-second time limit is
therefore 1,000 control steps. Our 64 grouped transitions contain five native
actions each: 64 x 5 x 2 = 640 control steps, or 12.8 simulated seconds. The
installed dm_control environment auto-resets only on the next step after a
termination or after reaching that 1,000-step limit. Neither trigger occurs in
the current collection.

## Empirical checks

Four full-length new rollouts, one per physical condition, each reached exactly
640 control steps and simulated time 12.8 seconds. Their target configurations
remained `None`; no time rollback or termination was observed.

Four stored test trajectories (`test-s3031000-d0` through `d3`) were then replayed
using their recorded action arrays. Every saved physical state matched the
replayed state exactly: maximum absolute error **0.0** in all four conditions.
No auto-reset or termination was observed in these complete replays. Evidence:
[`evidence/reacher_collection_replay_audit.json`](evidence/reacher_collection_replay_audit.json).

All 1,216 stored episodes, comprising 77,824 grouped transitions, were inspected
for nonfinite physical states and state-jump statistics:

| Quantity | Observed value |
|---|---:|
| Episodes with nonfinite states | 0 |
| Median maximum per-joint position change per grouped transition | 0.228300 rad |
| 95th percentile | 0.487519 rad |
| 99th percentile | 0.599013 rad |
| Maximum | 0.863656 rad |
| Position changes exceeding pi radians | 0 |

These jump statistics alone would not prove the absence of resets. The decisive
evidence is the disabled task-termination condition, the collection duration
being below the time limit, and exact full-trajectory replays with a monitored
simulation clock. Evidence:
[`evidence/reacher_collection_state_audit.json`](evidence/reacher_collection_state_audit.json).

This conclusion applies to the current 64-group, 320-native-step configuration.
Extending collection beyond 100 groups would cross the environment's time
limit; such a change needs explicit episode-boundary handling first.
