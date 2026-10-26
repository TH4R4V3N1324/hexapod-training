from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TrackedVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand that also keeps, per episode, how far it asked the robot to go and how
    far the robot went along what it was asked, for the terrain curriculum.

    commanded_distance sums |command xy| * dt. progress sums the body's velocity along the command
    each step, capped at the commanded speed so a fast stretch can't make up for a stuck one.
    Commands change every resampling period and heading commands turn the robot, so neither the
    straight-line distance from the start nor the final command says how much walking was asked for.
    """

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.commanded_distance = torch.zeros(self.num_envs, device=self.device)
        self.progress = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        super()._update_metrics()
        dt = self._env.step_dt
        command = self.vel_command_b[:, :2]
        speed = command.norm(dim=-1)
        along = (self.robot.data.root_lin_vel_b.torch[:, :2] * command).sum(dim=-1) / speed.clamp_min(1.0e-6)
        self.commanded_distance += speed * dt
        self.progress += torch.minimum(along, speed).clamp_min(-speed) * dt

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        # the curriculum reads both before the command manager resets
        extras = super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self.commanded_distance[env_ids] = 0.0
        self.progress[env_ids] = 0.0
        return extras

