"""Official LeWM planning-evaluation protocols.

Source of truth: ``external/le-wm/config/eval/{pusht,tworoom,reacher,cube}.yaml`` +
``solver/cem.yaml`` (@8edfeb33) and the equivalent pinned stable-worldmodel
``scripts/plan/config`` (@4821c8e6). ``released`` = those YAMLs exactly.
``paper`` = the deviations stated in LeWM arXiv 2603.19312 App. D/F.1
(TwoRoom budget 150 / goal +100; CEM 10 iterations outside PushT).
"""

from __future__ import annotations

import copy

CEM = dict(num_samples=300, n_steps=30, topk=30, var_scale=1.0, batch_size=1)
PLAN = dict(horizon=5, receding_horizon=5, action_block=5, history_len=1)
EVAL = dict(num_eval=50, goal_offset_steps=25, eval_budget=50, img_size=224, seed=42)

RELEASED = {
    "pusht": dict(
        world=dict(env_name="swm/PushT-v1"),
        dataset="pusht/pusht_expert_train.h5",
        keys_to_cache=["action", "proprio", "state"],
        callables=[
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
        ],
    ),
    "tworoom": dict(
        world=dict(env_name="swm/TwoRoom-v1"),
        dataset="tworoom/tworoom.h5",
        keys_to_cache=["action", "proprio"],
        # le-wm@8edfeb33 uses proprio/goal_proprio; swm@4821c8e6 renamed it to
        # pos_agent/goal_pos_agent. The harness picks whichever column exists.
        callables=[
            {"method": "_set_state", "args": {"state": {"value": "proprio"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_proprio"}}},
        ],
        callables_alt=[
            {"method": "_set_state", "args": {"state": {"value": "pos_agent"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_pos_agent"}}},
        ],
    ),
    "reacher": dict(
        world=dict(env_name="swm/ReacherDMControl-v0", task="qpos_match"),
        dataset="reacher/reacher.h5",  # == dmc/reacher_random.h5 in the configs
        keys_to_cache=["action"],
        callables=[
            {"method": "set_state", "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
            {"method": "set_target_qpos", "args": {"target_qpos": {"value": "goal_qpos"}}},
        ],
    ),
}


def get_protocol(env: str, variant: str = "released") -> dict:
    p = copy.deepcopy(RELEASED[env])
    p["cem"] = dict(CEM)
    p["plan"] = dict(PLAN)
    p["eval"] = dict(EVAL)
    if variant == "paper":
        if env == "tworoom":
            p["eval"].update(eval_budget=150, goal_offset_steps=100)
        if env != "pusht":
            p["cem"]["n_steps"] = 10
    elif variant != "released":
        raise ValueError(variant)
    p["variant"] = variant
    return p
