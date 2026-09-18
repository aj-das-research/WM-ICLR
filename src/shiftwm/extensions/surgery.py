"""LapGym tissue-positioning adapter, isolated from frozen experiment sources.

The controller receives RGB and commanded actions only. SOFA state and the
hidden gain are restricted to collection/scoring. No clinical claims are made.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

UPSTREAM_COMMIT = "85bf7e05dd088b824794dda0046679df13b13e6e"


@dataclass(frozen=True)
class SurgeryConfig:
    image_size: int = 128
    time_step: float = 0.1
    maximum_robot_velocity: float = 10.0  # upstream mm/s, not m/s
    actuator_gain: float = 1.0
    seed: int = 4100000
    settle_steps: int = 20

    def __post_init__(self):
        if self.image_size < 32 or not 0 < self.actuator_gain <= 2:
            raise ValueError("Invalid image size or actuator gain")


class SurgeryAdapter:
    """Commanded two-axis actions -> bounded hidden gain -> native simulator.

    ``executed_action`` is the bounded normalized action *supplied* to LapGym.
    Workspace rejection can further affect actual gripper displacement; that
    displacement is recorded separately and is not equated with a command.
    """

    def __init__(self, config: SurgeryConfig):
        from sofa_env.base import RenderMode
        from sofa_env.scenes.tissue_manipulation.tissue_manipulation_env import (
            ActionType, ObservationType, TissueManipulationEnv,
        )
        self.config = config
        self.env = TissueManipulationEnv(
            image_shape=(config.image_size, config.image_size),
            observation_type=ObservationType.RGB, action_type=ActionType.CONTINUOUS,
            render_mode=RenderMode.HEADLESS, time_step=config.time_step, frame_skip=1,
            settle_steps=config.settle_steps, end_position_threshold=0.002,
            maximum_robot_velocity=config.maximum_robot_velocity,
            log_episode_env_info=False, end_episode_criteria=[],
        )
        # Upstream _get_info unconditionally uses this dictionary, even when
        # its filesystem logging is disabled. Initialize without changing source.
        self.env._initial_env_state = {}
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs, _ = self.env.reset(seed=self.config.seed if seed is None else seed)
        self.env.set_initial_env_state()
        return np.array(obs, dtype=np.uint8, copy=True)

    def observe(self) -> np.ndarray:
        return np.array(self.env._get_observation(self.env._maybe_update_rgb_buffer()), copy=True)

    def diagnostics(self) -> dict:
        e = self.env
        tissue = np.array(e._tissue.get_manipulation_target_pose()[:3], copy=True)
        goal = np.array(e._visual_target.get_pose()[:3], copy=True)
        distance = float(np.linalg.norm((tissue - goal)[[0, 2]]))
        return {
            "tissue_target_xyz_m": tissue,
            "goal_xyz_m": goal,
            "gripper_pose": np.array(e._gripper.get_pose(), copy=True),
            "distance_m": distance,
            "success": distance <= 0.002,
            "deformation_step_m": float(e._tissue.get_displacement_norm()),
        }

    def set_reachable_goal(self, position_xyz: np.ndarray) -> np.ndarray:
        """Assign a previously observed tissue point as the visible goal.

        The caller must record its provenance. This defines a reachable-image
        goal protocol and does not reproduce LapGym's random target sampler.
        """
        position = np.asarray(position_xyz, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Expected finite xyz goal")
        self.env._visual_target.reset(new_position=position.copy())
        self.env._visual_target.set_visibility(False)
        self.env._visual_target_position = position.copy()
        self.env.set_initial_env_state()
        return self.observe()

    def step(self, commanded_action: np.ndarray) -> tuple[np.ndarray, dict]:
        command = np.asarray(commanded_action, dtype=np.float32)
        if command.shape != (2,) or not np.isfinite(command).all() or np.any(np.abs(command) > 1):
            raise ValueError("Expected two finite commanded actions in [-1,1]")
        executed = np.clip(command * self.config.actuator_gain, -1, 1).astype(np.float32)
        before = np.array(self.env._gripper.get_pose()[:3], copy=True)
        obs, _, terminated, truncated, info = self.env.step(executed.astype(np.float64))
        diagnostic = self.diagnostics()
        diagnostic.update({
            "commanded_action": command.copy(), "executed_action": executed,
            "gripper_delta_xyz_m": diagnostic["gripper_pose"][:3] - before,
            "valid_action": bool(info["valid_action"]),
            "stable_deformation": bool(info["stable_deformation"]),
            "terminated": bool(terminated), "truncated": bool(truncated),
        })
        if bool(info["is_success"]) != diagnostic["success"]:
            raise AssertionError("Native goal criterion disagrees with diagnostic")
        return np.array(obs, dtype=np.uint8, copy=True), diagnostic

    def close(self):
        self.env.close()


def exploration_actions(seed: int, calls: int) -> np.ndarray:
    """Open-loop correlated exploration; no hidden state or oracle feedback."""
    rng = np.random.default_rng(seed + 9000000)
    actions = np.empty((calls, 2), dtype=np.float32)
    previous = np.zeros(2)
    for start in range(0, calls, 5):
        previous = np.clip(0.25 * previous + rng.uniform(-0.8, 0.8, 2), -1, 1)
        actions[start:start + 5] = previous
    return actions
