from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

from hexapod_training.gait.tripod_gait import TripodGait, cap_command

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
        # The sampled command, before the gait's speed cap. vel_command_b is rebuilt from it every
        # step, since the cap would otherwise compound on the x and y it only sets at resampling.
        self.sampled_command = torch.zeros_like(self.vel_command_b)
        gait = TripodGait(1, self.device)
        self._max_body_radius = gait.max_body_radius
        self._max_foot_speed = gait.max_foot_speed

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        self.sampled_command[env_ids] = self.vel_command_b[env_ids]

    def _update_command(self):
        # Heading control rewrites yaw, and standing envs are zeroed, then the gait's cap applies, so
        # the reward, the curriculum and the gait all see the same reachable command
        self.vel_command_b[:] = self.sampled_command
        super()._update_command()
        self.vel_command_b[:], _ = cap_command(self.vel_command_b, self._max_body_radius, self._max_foot_speed)

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

