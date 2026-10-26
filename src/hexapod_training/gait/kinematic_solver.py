"""Leg inverse and forward kinematics, ported from kinematic_solver.cpp."""

import torch

from .gait_config import (
    COXA_LENGTH,
    FEMUR_LENGTH,
    FEMUR_MAX,
    FEMUR_MIN,
    MAX_COXA_ANGLE,
    REACH_TOLERANCE,
    TIBIA_LENGTH,
    TIBIA_MAX,
    TIBIA_MIN,
)


class KinematicSolver:
    """
    A class to solve the inverse kinematics for a hexapod robot.
    Positions are (..., 3) tensors in the leg frame (X outward); angles are (..., 3) as (coxa, femur, tibia).
    """

    def __init__(self, coxa_length=COXA_LENGTH, femur_length=FEMUR_LENGTH, tibia_length=TIBIA_LENGTH):
        self.coxa_length = coxa_length
        self.femur_length = femur_length
        self.tibia_length = tibia_length

    def solveIK(self, target_position, mirror=False):
        """
        Solve the inverse kinematics for the given target position.

        Args:
            target_position (torch.Tensor): The target position (..., 3) for the end effector (leg frame).
            mirror (bool): Whether to mirror the y-coordinate.

        Returns:
            torch.Tensor: The joint angles (..., 3) as (coxa, femur, tibia).

        Note:
            If the target is outside the workspace, angles are clamped to the nearest valid solution.
        """
        a1 = self.coxa_length
        a2 = self.femur_length
        a3 = self.tibia_length

        x = target_position[..., 0]
        y = target_position[..., 1]
        z = target_position[..., 2]

        if mirror:
            y = -y

        coxa_angle = torch.atan2(y, x)
        r1 = torch.hypot(x, y) - a1
        r2 = z
        q2 = torch.atan2(r2, r1)
        r3 = torch.hypot(r1, r2)
        q1 = torch.acos(torch.clamp((a3**2 - a2**2 - r3**2) / (-2 * a2 * r3), -1.0, 1.0))
        femur_angle = -(q1 + q2)
        q3 = torch.acos(torch.clamp((r3**2 - a2**2 - a3**2) / (-2 * a2 * a3), -1.0, 1.0))
        tibia_angle = torch.pi / 2 - q3

        return torch.stack((coxa_angle, femur_angle, tibia_angle), dim=-1)

    def solveFK(self, joint_angles):
        """
        Forward kinematics, the foot position (..., 3) for joint angles (..., 3).
        Leg frame, X outward, Z negative below the hip.
        """
        a1 = self.coxa_length
        a2 = self.femur_length
        a3 = self.tibia_length

        coxa_angle = joint_angles[..., 0]
        # Elevation of each link above horizontal, matching solveIK:
        # a negative femur angle raises the femur, and tibia = 0 puts the knee at 90°
        femur_elevation = -joint_angles[..., 1]
        tibia_elevation = femur_elevation - joint_angles[..., 2] - torch.pi / 2

        horizontal_reach = a1 + a2 * torch.cos(femur_elevation) + a3 * torch.cos(tibia_elevation)
        z = a2 * torch.sin(femur_elevation) + a3 * torch.sin(tibia_elevation)
        x = horizontal_reach * torch.cos(coxa_angle)
        y = horizontal_reach * torch.sin(coxa_angle)

        return torch.stack((x, y, z), dim=-1)

    def reachable(self, target_position):
        """
        Whether the leg can reach each target (..., 3) inside its joint limits, ported from
        GaitController::reachable. Returns a (...) bool tensor.
        """
        angles = self.solveIK(target_position)
        coxa, femur, tibia = angles.unbind(-1)
        within_limits = (
            (coxa.abs() <= MAX_COXA_ANGLE)
            & (femur >= FEMUR_MIN)
            & (femur <= FEMUR_MAX)
            & (tibia >= TIBIA_MIN)
            & (tibia <= TIBIA_MAX)
        )
        # The IK clamps an out-of-reach target to the nearest pose, which FK then misses it by
        return within_limits & (torch.linalg.norm(self.solveFK(angles) - target_position, dim=-1) <= REACH_TOLERANCE)
