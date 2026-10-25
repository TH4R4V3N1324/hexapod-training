from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


class HexapodLocomotionEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv that keeps a physics blow-up from reaching the learner.

    Rewards are computed before terminated envs reset, so on the step an env blows up its velocity
    penalties can reach 1e19 and turn the critic's loss to inf. The physics_blowup termination resets
    that env; this drops its reward for the step.
    """

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, extras = super().step(action)
        blown_up = self.termination_manager.get_term("physics_blowup")
        return obs, torch.where(blown_up, 0.0, reward), terminated, truncated, extras
