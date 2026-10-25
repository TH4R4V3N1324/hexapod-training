# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Implement the task's termination term."""
    raise NotImplementedError("Implement the termination term.")


def physics_blowup(
    env: ManagerBasedRLEnv,
    max_lin_vel: float = 4.0,
    max_ang_vel: float = 20.0,
    max_joint_vel: float = 50.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008  (Isaac Lab term idiom)
) -> torch.Tensor:
    """The simulation has blown up: the robot's state is not finite, or moving faster than it physically can.

    The limits sit far above anything the robot reaches (it walks at 0.17 m/s, pushes reach 0.5 m/s and
    the servos top out at 4.7 rad/s), so this only catches a solver blow-up. HexapodLocomotionEnv drops
    the reward of the step it fires on.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_lin_vel = asset.data.root_lin_vel_w.torch
    root_ang_vel = asset.data.root_ang_vel_w.torch
    joint_vel = asset.data.joint_vel.torch
    root_pos = asset.data.root_pos_w.torch
    state = torch.cat((root_pos, root_lin_vel, root_ang_vel, asset.data.joint_pos.torch, joint_vel), dim=-1)
    return (
        ~torch.isfinite(state).all(dim=-1)
        | (root_lin_vel.norm(dim=-1) > max_lin_vel)
        | (root_ang_vel.norm(dim=-1) > max_ang_vel)
        | (joint_vel.abs() > max_joint_vel).any(dim=-1)
    )
