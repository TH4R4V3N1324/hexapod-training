# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Implement the task's reward term."""
    raise NotImplementedError("Implement the reward term.")


def feet_air_time(env, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float, min_command: float = 0.02):
    """Isaac Lab's feet_air_time, with the 'is it walking' cutoff as a parameter."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > min_command
    return reward


def feet_air_time_variance(env, command_name: str, sensor_cfg: SceneEntityCfg, max_time: float = 0.5, min_command: float = 0.02):
    """Penalise feet that swing or stand for different lengths of time (limping)."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    air = torch.clip(contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids], max=max_time)
    stance = torch.clip(contact_sensor.data.last_contact_time.torch[:, sensor_cfg.body_ids], max=max_time)
    penalty = torch.var(air, dim=1) + torch.var(stance, dim=1)
    penalty *= torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > min_command
    return penalty
