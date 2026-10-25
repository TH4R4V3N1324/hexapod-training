# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Implement the task's observation term."""
    raise NotImplementedError("Implement the observation term.")


def gait_phase(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Per leg: sin and cos of the leg's own cycle phase, and whether it is swinging, (N, 18) in leg 1-6 order.

    Read from the GaitResidualAction term's gait. The per-leg form reads the same for tripod,
    ripple and wave, so one policy can run any gait.
    """
    gait = env.action_manager.get_term(action_name).gait
    angle = 2.0 * math.pi * gait.leg_cycle_phase()
    # A frozen phase start (idle command) has its group marked but not yet lifted, so it isn't swinging
    swinging = gait.swinging & (gait.phase_progress > 0.0).unsqueeze(-1)
    return torch.stack((torch.sin(angle), torch.cos(angle), swinging.float()), dim=-1).view(env.num_envs, -1)
