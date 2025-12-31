import gymnasium as gym
import numpy as np
from gymnasium import Env


class HumanoidPDWrapper(gym.ActionWrapper):
    def __init__(self, env: Env, kp=10.0, kd=0.1):
        super().__init__(env)
        self.kp = kp  # Stiffness (Proportional gain)
        self.kd = kd  # Damping (Derivative gain)

        # Lazy initialization - capture reference pose after first action call
        self._reference_qpos = None

    def action(self, action):
        """
        1. Receive 'Target Angle' from Agent
        2. Calculate necessary Torque using PD formula
        3. Send Torque to Physics Engine
        """

        # Access MuJoCo data
        data = self.env.unwrapped.data  # type: ignore

        # Lazy initialization - capture the balanced standing pose
        # This happens on the first action call, right after env.reset()
        if self._reference_qpos is None:
            # Use the current joint positions as the reference (balanced) pose
            self._reference_qpos = data.qpos[7:].copy()

        # --- Step 1: Interpret the Agent's Action ---
        # Map action from [-1, 1] to target position as offset from reference pose
        # action = 0 means "maintain reference pose"
        # Maximum offset is ±1 radian (±57 degrees) from reference
        action_scale = 1.0  # Max deviation from reference in radians
        target_q = self._reference_qpos + (action * action_scale)

        # --- Step 2: Get Current Physics State (q and q_dot) ---
        # Note the indices! We skip the root body (first 7 pos, first 6 vel)
        current_q = data.qpos[7:]
        current_qdot = data.qvel[6:]

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
