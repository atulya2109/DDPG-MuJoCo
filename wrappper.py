import gymnasium as gym
import numpy as np
from gymnasium import Env


class HumanoidPDWrapper(gym.ActionWrapper):
    def __init__(self, env: Env, kp=10.0, kd=0.1, action_scale=0.5):
        super().__init__(env)
        self.kp = kp  # Stiffness (Proportional gain)
        self.kd = kd  # Damping (Derivative gain)
        self.action_scale = (
            action_scale  # How far can the agent move joints (in radians)
        )

        # We need to access the internal MuJoCo physics data
        # 'unwrapped' gets us to the base environment, bypassing other wrappers
        self.data = env.unwrapped.data  # type: ignore
        self.model = env.unwrapped.model  # type: ignore

        # Capture the "Default Standing Pose"
        # In MuJoCo, qpos[0:7] is the root (position/rotation) of the whole body
        # qpos[7:] are the actual joints (hips, knees, elbows) that we control
        self.default_qpos = (
            self.model.key_qpos[0][7:].copy() if self.model.nkey > 0 else np.zeros(17)
        )

    def action(self, action):
        """
        1. Receive 'Target Angle' from Agent
        2. Calculate necessary Torque using PD formula
        3. Send Torque to Physics Engine
        """

        # --- Step 1: Interpret the Agent's Action ---
        # Agent outputs [-1, 1]. We map this to a range around the default pose.
        # e.g., if scale is 0.5, the joint can move +/- 0.5 radians (approx 30 degrees)
        target_q = self.default_qpos + (action * self.action_scale)

        # --- Step 2: Get Current Physics State (q and q_dot) ---
        # Note the indices! We skip the root body (first 7 pos, first 6 vel)
        current_q = self.data.qpos[7:]
        current_qdot = self.data.qvel[6:]

        # --- Step 3: The PD Equation ---
        # Torque = Kp * (Target - Current) - Kd * (Velocity)
        # This is the "Spring - Damper" logic
        error = target_q - current_q
        torque = (self.kp * error) - (self.kd * current_qdot)

        # --- Step 4: Safety Clipping ---
        # Ensure we don't send values larger than the motors can physically handle
        lb, ub = self.env.action_space.low, self.env.action_space.high  # type: ignore
        torque = np.clip(torque, lb, ub)

        return torque
