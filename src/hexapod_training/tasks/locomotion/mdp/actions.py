from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

from hexapod_training.gait import JOINT_NAMES, KinematicSolver, TripodGait
from hexapod_training.gait.gait_config import NOMINAL_BODY_HEIGHT, SWING_LIFT

if TYPE_CHECKING:
    from isaaclab.assets.articulation import Articulation
    from isaaclab.envs import ManagerBasedEnv


class GaitResidualAction(ActionTerm):
    """The tripod gait drives the walking; the policy adjusts how it steps and where each foot goes.

    The action has 25 values:

    - 0-17: an (x, y, z) offset per leg in its leg frame, in leg 1-6 order, scaled by cfg.scale_xy /
      cfg.scale_z (m), added to each foot target before IK.
    - 18-23: each leg's swing lift, SWING_LIFT plus cfg.swing_lift_scale (m) per unit, within
      cfg.swing_lift_range. Unlike a z offset this raises the middle of the swing and still lands
      on target.
    - 24: the body height, NOMINAL_BODY_HEIGHT plus cfg.body_height_scale (m) per unit, within
      cfg.body_height_range. The gait glides to it at BODY_HEIGHT_RATE, so a taller stance takes time.

    Zero is the hand-written gait. The gait steps once per policy step, and the joint targets are
    held across the decimated physics steps.
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
        # Foot offset per unit action (m), per leg-frame axis
        self._scale = torch.tensor([cfg.scale_xy, cfg.scale_xy, cfg.scale_z], device=self.device)
        # zero_residual ignores the policy, leaving the gait alone: the baseline
        self._gain = 0.0 if cfg.zero_residual else 1.0
        self.swing_lift = torch.full((self.num_envs, 6), SWING_LIFT, device=self.device)  # m, as commanded
        self.body_height = torch.full((self.num_envs,), NOMINAL_BODY_HEIGHT, device=self.device)  # m, as commanded

    @property
    def action_dim(self) -> int:
        return 25

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        actions = self._raw_actions * self._gain
        offsets = actions[:, :18].view(self.num_envs, 6, 3) * self._scale
        self.swing_lift = torch.clamp(
            SWING_LIFT + actions[:, 18:24] * self.cfg.swing_lift_scale, *self.cfg.swing_lift_range
        )
        self.body_height = torch.clamp(
            NOMINAL_BODY_HEIGHT + actions[:, 24] * self.cfg.body_height_scale, *self.cfg.body_height_range
        )
        commands = self._env.command_manager.get_command(self.cfg.command_name)[:, :3]
        feet = self.gait.step(commands, self._env.step_dt, swing_lift=self.swing_lift, body_height=self.body_height)
        self.foot_targets = feet + offsets
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
    swing_lift_scale: float = 0.04
    """Swing lift per unit action (m), about the default SWING_LIFT of 0.06 m."""
    swing_lift_range: tuple[float, float] = (0.03, 0.10)
    """Swing lift limits (m). The foot peaks at about 0.88x this, so 0.10 clears about 8.8 cm."""
    body_height_scale: float = 0.026
    """Body height per unit action (m), about the nominal 0.134 m."""
    body_height_range: tuple[float, float] = (0.10, 0.16)
    """Body height limits (m), hips above the ground; 0.16 is set_height's maximum."""
    zero_residual: bool = False
    """Ignore the policy's actions, leaving the gait alone: the baseline and first test."""
