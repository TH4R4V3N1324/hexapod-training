"""Foot displacement, stride clamp and swing curve, ported from trajectory_generator.cpp.

Foot positions are (num_envs, 6, 3) tensors in each leg's frame.
"""

import torch

from .frame_converter import FrameConverter, rotate_z
from .gait_config import MAX_STRIDE_LENGTH, NEUTRAL_FOOT, SWING_EASING, SWING_LIFT


class TrajectoryGenerator:
    def __init__(self, device="cpu"):
        self.converter = FrameConverter(device)
        self.neutral = torch.tensor(NEUTRAL_FOOT, device=device).expand(6, 3).clone()  # (6, 3) leg frame
        self.swing_lift = SWING_LIFT
        self.swing_easing = SWING_EASING  # 0 moves along the swing at an even pace, 1 eases in and out

    def neutralFootInBody(self):
        """Position of each leg's neutral foot relative to the body centre, (6, 3) body frame with z = 0."""
        foot = self.converter.legToBodyFrame(self.neutral)
        foot[..., 2] = 0.0
        return foot

    def maxBodyRadius(self):
        """Largest horizontal distance from the body centre to a neutral foot (m)."""
        return self.neutralFootInBody().norm(dim=-1).max().item()

    def footDisplacementFrom(self, forward, left, yaw, foot_pos, duration):
        """
        Foot displacement in each leg's frame for a body velocity held for a duration.

        Args:
            forward, left, yaw: (num_envs,) body velocity (m/s, m/s, rad/s positive anticlockwise).
            foot_pos: (num_envs, 6, 3) foot position the rotation starts from (leg frame).
            duration: (num_envs,) tensor or float, how long the velocity is held (s).

        Returns:
            (num_envs, 6, 3) displacement (leg frame, z = 0), in the direction the foot swings.
            A stance foot moves by the negative of this.
        """
        duration = torch.as_tensor(duration, device=foot_pos.device).expand_as(forward)
        flat = foot_pos.clone()
        flat[..., 2] = 0.0
        foot = self.converter.legToBodyFrame(flat)
        foot[..., 2] = 0.0
        linear = torch.stack((forward * duration, left * duration, torch.zeros_like(forward)), dim=-1).unsqueeze(1)
        rotation = rotate_z(foot, (yaw * duration).unsqueeze(1)) - foot
        return rotate_z(linear + rotation, -self.converter.yaw)

    def footDisplacement(self, forward, left, yaw, duration):
        """footDisplacementFrom, turning about the body centre from the neutral foot positions."""
        return self.footDisplacementFrom(forward, left, yaw, self.neutral.expand(forward.shape[0], 6, 3), duration)

    def clampToStride(self, target):
        """Keep foot targets (num_envs, 6, 3) within MAX_STRIDE_LENGTH of neutral, with z at the neutral height."""
        delta = target[..., :2] - self.neutral[:, :2]
        distance = torch.linalg.norm(delta, dim=-1, keepdim=True)
        delta = delta * torch.clamp(MAX_STRIDE_LENGTH / distance.clamp_min(1e-9), max=1.0)
        z = self.neutral[:, 2].expand(target.shape[:-1])
        return torch.cat((self.neutral[:, :2] + delta, z.unsqueeze(-1)), dim=-1)

    def swingTarget(self, displacement):
        """Where a swinging foot should land, at most MAX_STRIDE_LENGTH from neutral."""
        return self.clampToStride(self.neutral + displacement)

    def stanceTarget(self, current_pos, displacement):
        """Where a stance foot should be once the body has moved by displacement."""
        return self.clampToStride(current_pos - displacement)

    def genSwingSpline(self, start, end, lift=None):
        """
        Control points of a swing-phase quartic Bezier, as a (5, ..., 3) tensor.

        The middle three points are all at the same height, so the foot rises most of the way
        before moving forward and comes down nearly vertically.
        """
        lift = self.swing_lift if lift is None else lift
        p1 = start.clone()
        p1[..., 2] += lift
        p2 = 0.5 * (start + end)
        p2[..., 2] = torch.maximum(start[..., 2], end[..., 2]) + lift
        p3 = end.clone()
        p3[..., 2] += lift
        return torch.stack((start, p1, p2, p3, end))

    def evalBezier(self, points, s):
        """
        Evaluate a quartic Bezier with smoothstep easing, blended with even progress by swing_easing.

        Args:
            points: (5, ..., 3) control points from genSwingSpline.
            s: normalised progress [0, 1], broadcastable against points[0][..., 0].
        """
        s = torch.as_tensor(s, device=points.device)
        # Smootherstep starts and ends the swing at rest, which holds the foot nearly still around
        # every lift-off and touchdown; blending in even progress keeps it moving through them
        eased = s * s * s * (s * (6.0 * s - 15.0) + 10.0)
        st = (self.swing_easing * eased + (1.0 - self.swing_easing) * s).unsqueeze(-1)
        u = 1.0 - st
        return (
            points[0] * (u * u * u * u)
            + points[1] * (4.0 * u * u * u * st)
            + points[2] * (6.0 * u * u * st * st)
            + points[3] * (4.0 * u * st * st * st)
            + points[4] * (st * st * st * st)
        )
