from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

from hexapod_training.gait import JOINT_NAMES, KinematicSolver, TripodGait

if TYPE_CHECKING:
    from isaaclab.assets.articulation import Articulation
    from isaaclab.envs import ManagerBasedEnv


class GaitResidualAction(ActionTerm):
    """The tripod gait drives the walking; the policy shifts each foot target before IK.

    The action is an (x, y, z) offset per leg in its leg frame, 18 values in leg 1-6 order, scaled by
    cfg.scale_xy / cfg.scale_z (m). The gait steps once per policy step, and the joint targets are held across the
    decimated physics steps.
    """

    cfg: GaitResidualActionCfg
    _asset: Articulation

    def __init__(self, cfg: GaitResidualActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        # Isaac sorts joints its own way, so look them up in leg order
        joint_ids, self._joint_names = self._asset.find_joints(JOINT_NAMES, preserve_order=True)
        self._joint_ids = torch.tensor(joint_ids, device=self.device)

        self.gait = TripodGait(self.num_envs, self.device)
        self.ik = KinematicSolver()
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, len(JOINT_NAMES), device=self.device)
        self.foot_targets = torch.zeros(self.num_envs, 6, 3, device=self.device)  # leg frame, after offsets
        # Foot offset per unit action (m), per leg-frame axis; zero_residual gives the gait-alone baseline
        scale = 0.0 if cfg.zero_residual else 1.0
        self._scale = torch.tensor([cfg.scale_xy, cfg.scale_xy, cfg.scale_z], device=self.device) * scale

    @property
    def action_dim(self) -> int:
        return 18

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        commands = self._env.command_manager.get_command(self.cfg.command_name)[:, :3]
        feet = self.gait.step(commands, self._env.step_dt)
        self.foot_targets = feet + self._raw_actions.view(self.num_envs, 6, 3) * self._scale
        self._processed_actions = self.ik.solveIK(self.foot_targets).view(self.num_envs, -1)

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self.gait.reset(env_ids)


@configclass
class GaitResidualActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = GaitResidualAction
    asset_name: str = "robot"

    command_name: str = "base_velocity"
    """The velocity command (forward, left, yaw rate) the gait follows."""
    scale_xy: float = 0.015
    """Horizontal foot offset per unit action (m)."""
    scale_z: float = 0.04
    """Vertical foot offset per unit action (m)."""
    zero_residual: bool = False
    """Ignore the policy's actions, leaving the gait alone: the baseline and first test."""
