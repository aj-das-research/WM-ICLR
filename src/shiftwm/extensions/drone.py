"""Visual planar drone control using pinned, unmodified gym-pybullet-drones.

The learned policy receives only RGB, its previous commanded actions, and a
recorded goal image. An upstream PID uses physical state as the low-level flight
controller; it is identical for every learned method. This is external-camera
visual waypoint control, not onboard navigation or learned attitude control.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import asdict, dataclass

import numpy as np

UPSTREAM_COMMIT = "7ebad1ecabd28a7000add2d05f888aa2e837c2cc"
DYNAMICS_GAINS = {0: 1.0, 1: 0.75, 2: 1.25, 3: 1.5}


@dataclass(frozen=True)
class DroneConfig:
    image_size: int = 128
    physics_hz: int = 240
    controller_hz: int = 120
    command_hz: int = 10
    altitude_m: float = 0.65
    velocity_scale_m_s: float = 0.18
    workspace_radius_m: float = 0.36
    position_tolerance_m: float = 0.04
    speed_tolerance_m_s: float = 0.06
    altitude_tolerance_m: float = 0.05
    max_steps: int = 200

    def __post_init__(self):
        if self.image_size < 32 or self.max_steps < 1:
            raise ValueError("Invalid image size or episode length")
        if self.physics_hz % self.controller_hz or self.controller_hz % self.command_hz:
            raise ValueError("Physics, controller, and command frequencies must divide")


class VisualDroneEnv:
    """Two commanded XY velocity components in [-1,1], canonical RGB output.

    The hidden response gain multiplies the velocity request *before* the
    upstream flight controller. It models command calibration, not a change in
    the quadcopter mass, motor thrust, wind, or aerodynamic coefficients.
    """

    def __init__(self, dynamics_id: int = 0, config: DroneConfig | None = None):
        if dynamics_id not in DYNAMICS_GAINS:
            raise ValueError(f"Unknown dynamics condition {dynamics_id}")
        self.config = config or DroneConfig()
        self.dynamics_id = dynamics_id
        self.response_gain = DYNAMICS_GAINS[dynamics_id]
        self.env = None
        self.goal_state = None
        self.last_rpm = None
        self.last_executed_action = None
        self.steps = 0

    def reset(self, seed: int = 0):
        import pybullet as p
        from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
        from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
        from gym_pybullet_drones.utils.enums import DroneModel, Physics

        self.close()
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.initial_xy = rng.uniform(-0.12, 0.12, 2)
        initial_xyz = np.array([[*self.initial_xy, self.config.altitude_m]])
        # Suppress upstream parameter printout only; simulator errors propagate.
        with contextlib.redirect_stdout(io.StringIO()):
            self.env = CtrlAviary(drone_model=DroneModel.CF2X, initial_xyzs=initial_xyz,
                                 physics=Physics.PYB, pyb_freq=self.config.physics_hz,
                                 ctrl_freq=self.config.controller_hz, gui=False,
                                 obstacles=False, user_debug_gui=False, record=False)
        self.pid = DSLPIDControl(drone_model=DroneModel.CF2X)
        self.client = self.env.CLIENT
        self.steps = 0
        self.goal_state = None
        self.last_rpm = np.empty((0, 4), dtype=np.float32)
        self.last_executed_action = np.zeros(2, dtype=np.float32)
        p.changeVisualShape(self.env.DRONE_IDS[0], -1, rgbaColor=[0.10, 0.16, 0.25, 1],
                            physicsClientId=self.client)
        # Static visual-only floor tiles supply scale/orientation without
        # collision surfaces, task labels, goal locations, or state overlays.
        for x in range(-3, 4):
            for y in range(-3, 4):
                color = [0.77, 0.80, 0.82, 1] if (x + y) % 2 else [0.91, 0.92, 0.89, 1]
                shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.10, 0.10, 0.001],
                                            rgbaColor=color, physicsClientId=self.client)
                p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape,
                                  basePosition=[0.20 * x, 0.20 * y, 0.003],
                                  physicsClientId=self.client)
        self.view = p.computeViewMatrix([0, 0, 1.65], [0, 0, 0.65], [0, 1, 0])
        self.projection = p.computeProjectionMatrixFOV(50, 1, 0.05, 3.0)
        return self.render(), {"seed": self.seed}

    def simulator_state(self) -> np.ndarray:
        """Privileged audit/collector/scoring state; never a learned policy input."""
        if self.env is None:
            raise RuntimeError("reset() must precede state access")
        return self.env._getDroneStateVector(0).copy()

    def set_goal(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (20,) or not np.isfinite(state).all():
            raise ValueError("Goal must be a finite recorded native 20-D state")
        self.goal_state = state.copy()

    def metrics(self) -> dict:
        state = self.simulator_state()
        speed = float(np.linalg.norm(state[10:13]))
        altitude_error = float(abs(state[2] - self.config.altitude_m))
        escaped = bool(np.max(np.abs(state[:2])) > self.config.workspace_radius_m + 0.08)
        crash = bool(state[2] < 0.20 or np.max(np.abs(state[7:9])) > 0.8)
        distance = None if self.goal_state is None else float(np.linalg.norm(state[:2] - self.goal_state[:2]))
        success = bool(distance is not None and distance < self.config.position_tolerance_m
                       and speed < self.config.speed_tolerance_m_s
                       and altitude_error < self.config.altitude_tolerance_m and not escaped and not crash)
        return {"goal_distance_m": distance, "speed_m_s": speed,
                "altitude_error_m": altitude_error, "workspace_escape": escaped,
                "crash": crash, "success": success}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Action must be finite shape (2,)")
        if np.any(np.abs(action) > 1.000001):
            raise ValueError("Commanded action outside [-1,1]")
        if self.steps >= self.config.max_steps:
            raise RuntimeError("Episode exhausted; reset before further steps")
        action = np.clip(action, -1, 1)
        executed = action * self.response_gain
        velocity = np.array([*executed, 0.0]) * self.config.velocity_scale_m_s
        rpms = []
        for _ in range(self.config.controller_hz // self.config.command_hz):
            state = self.simulator_state()
            rpm, _, _ = self.pid.computeControlFromState(
                control_timestep=self.env.CTRL_TIMESTEP, state=state,
                target_pos=np.array([state[0], state[1], self.config.altitude_m]),
                target_vel=velocity, target_rpy=np.zeros(3))
            self.env.step(rpm.reshape(1, 4))
            rpms.append(self.env.last_clipped_action[0].copy())
        self.steps += 1
        self.last_rpm = np.asarray(rpms, dtype=np.float32)
        self.last_executed_action = executed.astype(np.float32)
        info = self.metrics()
        reward = 0.0 if info["goal_distance_m"] is None else -info["goal_distance_m"]
        return self.render(), reward, bool(info["crash"] or info["workspace_escape"]), self.steps >= self.config.max_steps, info

    def render(self) -> np.ndarray:
        import pybullet as p
        size = self.config.image_size
        _, _, rgba, _, _ = p.getCameraImage(size, size, viewMatrix=self.view,
                                            projectionMatrix=self.projection,
                                            renderer=p.ER_TINY_RENDERER,
                                            flags=p.ER_NO_SEGMENTATION_MASK,
                                            physicsClientId=self.client)
        return np.asarray(rgba, dtype=np.uint8).reshape(size, size, 4)[..., :3].copy()

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def protocol(self):
        return {"environment": "visual_drone", "upstream_commit": UPSTREAM_COMMIT,
                "config": asdict(self.config), "dynamics_id": self.dynamics_id,
                "response_gain": self.response_gain, "camera": "fixed_external_top_down",
                "action_semantics": "normalized_xy_velocity_before_hidden_response_gain",
                "state_access": "upstream PID plus collection/scoring only"}
