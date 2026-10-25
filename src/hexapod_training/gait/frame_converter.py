"""Converts positions between each leg's frame and the body frame, ported from frame_converter.hpp.

The body frame follows REP 103 (+x forward, +y left, +z up) and matches the URDF.
Each leg frame has +x pointing out along the leg from its coxa joint.
Positions are (..., 6, 3) tensors, one row per leg 1-6.
"""

import math

import torch

# Coxa joint yaw (rad) and position (m), matching the URDF: the middle legs 0.11836 m out
# to the side, the corner legs at 45 degrees from (0.11916, 0.08952) on the slimmer body
_M = 0.11836
_CX = 0.11916
_CY = 0.08952

LEG_YAWS = (
    -math.pi / 4.0,  # 1 front right
    -math.pi / 2.0,  # 2 middle right
    -3.0 * math.pi / 4.0,  # 3 rear right
    3.0 * math.pi / 4.0,  # 4 rear left
    math.pi / 2.0,  # 5 middle left
    math.pi / 4.0,  # 6 front left
)
LEG_OFFSETS = (
    (_CX, -_CY, 0.0),
    (0.0, -_M, 0.0),
    (-_CX, -_CY, 0.0),
    (-_CX, _CY, 0.0),
    (0.0, _M, 0.0),
    (_CX, _CY, 0.0),
)


def rotate_z(v: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate (..., 3) vectors about z by angle (broadcast against v's leading dims)."""
    c = torch.cos(angle)
    s = torch.sin(angle)
    x = v[..., 0] * c - v[..., 1] * s
    y = v[..., 0] * s + v[..., 1] * c
    return torch.stack((x, y, v[..., 2]), dim=-1)


class FrameConverter:
    def __init__(self, device="cpu"):
        self.yaw = torch.tensor(LEG_YAWS, device=device)  # (6,)
        self.offset = torch.tensor(LEG_OFFSETS, device=device)  # (6, 3)

    def legToBodyFrame(self, position):
        """Leg frame (..., 6, 3) to body frame: rotate by the leg yaw, then translate by the coxa offset."""
        return rotate_z(position, self.yaw) + self.offset

    def bodyToLegFrame(self, position):
        """Body frame (..., 6, 3) to leg frame: remove the coxa offset, then rotate by minus the leg yaw."""
        return rotate_z(position - self.offset, -self.yaw)
