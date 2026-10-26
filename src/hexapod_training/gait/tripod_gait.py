"""Batched tripod gait, ported from GaitController::walk in gait_controller.cpp.

Drives the walking for every environment at once and returns leg-frame foot targets. A policy can
shift those targets before IK without changing the gait's own foot tracking, like the C++ load
balancing offsets (see tasks.locomotion.mdp.actions.GaitResidualAction).
"""

import torch

from .gait_config import (
    BODY_HEIGHT_RATE,
    IDLE_DEADZONE,
    MAX_BODY_HEIGHT,
    MAX_STRIDE_LENGTH,
    MAX_TARGET_SPEED,
    MIN_BODY_HEIGHT,
    MIN_PHASE_DURATION,
    NOMINAL_BODY_HEIGHT,
    PHASE_DURATION,
    TRIPOD_GROUPS,
)
from .kinematic_solver import KinematicSolver
from .trajectory_generator import TrajectoryGenerator


def cap_command(commands, max_body_radius, max_foot_speed):
    """
    Scale body velocity commands down to what the gait can walk, matching GaitController::walk.

    Args:
        commands: (num_envs, 3) body velocity (forward m/s, left m/s, yaw rad/s).
        max_body_radius: largest distance from the body centre to a neutral foot (m).
        max_foot_speed: fastest a foot can travel relative to the body (m/s).

    Returns:
        The capped (num_envs, 3) commands, and each one's foot speed (num_envs,), at most max_foot_speed.
    """
    forward, left, yaw = commands.unbind(-1)
    foot_speed = torch.hypot(forward, left) + yaw.abs() * max_body_radius
    scale = torch.clamp(max_foot_speed / foot_speed.clamp_min(1e-9), max=1.0)
    return commands * scale.unsqueeze(-1), torch.clamp(foot_speed, max=max_foot_speed)


class TripodGait:
    def __init__(self, num_envs, device="cpu"):
        self.num_envs = num_envs
        self.device = device
        self.trajectory = TrajectoryGenerator(device)
        self.ik = KinematicSolver()

        self.groups = len(TRIPOD_GROUPS)
        self.group_mask = torch.zeros(self.groups, 6, dtype=torch.bool, device=device)
        self.leg_group = torch.zeros(6, dtype=torch.long, device=device)  # which group each leg swings with
        for i, group in enumerate(TRIPOD_GROUPS):
            self.group_mask[i, list(group)] = True
            self.leg_group[list(group)] = i
        self.max_body_radius = self.trajectory.maxBodyRadius()
        # A stance foot sweeps up to 2 x MAX_STRIDE_LENGTH over (groups - 1) phases,
        # and phases can be no shorter than MIN_PHASE_DURATION
        self.max_foot_speed = 2.0 * MAX_STRIDE_LENGTH / ((self.groups - 1) * MIN_PHASE_DURATION)

        self.foot_pos = torch.zeros(num_envs, 6, 3, device=device)  # leg frame
        self.swing_start = torch.zeros_like(self.foot_pos)
        self.swing_target = torch.zeros_like(self.foot_pos)
        self.swinging = torch.zeros(num_envs, 6, dtype=torch.bool, device=device)
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)  # index into TRIPOD_GROUPS
        self.phase_progress = torch.zeros(num_envs, device=device)  # [0, 1)
        self.phase_duration = torch.full((num_envs,), PHASE_DURATION, device=device)
        self.body_height = torch.full((num_envs,), NOMINAL_BODY_HEIGHT, device=device)  # m, hips above ground
        self.reset()

    def reset(self, env_ids=None):
        """Put the feet back at neutral and restart the gait clock."""
        if env_ids is None:
            env_ids = slice(None)
        self.foot_pos[env_ids] = self.trajectory.neutral
        self.swing_start[env_ids] = self.trajectory.neutral
        self.swing_target[env_ids] = self.trajectory.neutral
        self.swinging[env_ids] = False
        self.phase[env_ids] = 0
        self.phase_progress[env_ids] = 0.0
        self.phase_duration[env_ids] = PHASE_DURATION
        self.body_height[env_ids] = NOMINAL_BODY_HEIGHT

    def limit_swing_lift(self, swing, ground_z):
        """
        Lower swinging feet that a high lift would take outside the leg's reach, keeping as much of
        the lift as fits. Ported from GaitController::limitSwingLift.

        Args:
            swing: (num_envs, 6, 3) swing foot targets (leg frame).
            ground_z: (num_envs, 6) the ground height under each foot at this point of the swing.
        """
        lift = swing[..., 2] - ground_z
        keep = (lift <= 0.0) | self.ik.reachable(swing)
        low = torch.zeros_like(lift)  # share of the lift kept
        high = torch.ones_like(lift)
        lowered = swing.clone()
        for _ in range(10):
            mid = 0.5 * (low + high)
            lowered[..., 2] = ground_z + lift * mid
            fits = self.ik.reachable(lowered)
            low = torch.where(fits, mid, low)
            high = torch.where(fits, high, mid)
        lowered[..., 2] = ground_z + lift * low
        return torch.where(keep.unsqueeze(-1), swing, lowered)

    def leg_cycle_phase(self):
        """
        Each leg's position in its own gait cycle, (num_envs, 6) in [0, 1).

        0 is lift-off: the leg swings over [0, 1 / groups) and stands for the rest, so it reads the
        same for any gait whatever its number of groups.
        """
        phases_since_lift_off = (self.phase.unsqueeze(-1) - self.leg_group) % self.groups
        return (phases_since_lift_off + self.phase_progress.unsqueeze(-1)) / self.groups

    def step(self, commands, dt, swing_lift=None, body_height=None):
        """
        Advance the gait by one tick. Step at the policy rate (dt = 0.02 s in the velocity task).

        Args:
            commands: (num_envs, 3) body velocity (forward m/s, left m/s, yaw rad/s).
            dt: the time since the last tick (s).
            swing_lift: optional (num_envs, 6) swing lift per leg (m), the step_height of each leg's
                swing; SWING_LIFT if not given. Lowered where it would take a foot out of reach.
            body_height: optional (num_envs,) body height to move towards (m), hips above the ground,
                within MIN_BODY_HEIGHT to MAX_BODY_HEIGHT; the body rises or falls at BODY_HEIGHT_RATE.

        Returns:
            (num_envs, 6, 3) foot targets per leg 1-6, each in its own leg frame.
        """
        idle = (commands.abs() < IDLE_DEADZONE).all(dim=-1)

        # The body glides to its new height, as set_height does; feet stand that far below the hips
        if body_height is not None:
            height_step = torch.clamp(body_height, MIN_BODY_HEIGHT, MAX_BODY_HEIGHT) - self.body_height
            max_step = BODY_HEIGHT_RATE * dt
            self.body_height = self.body_height + torch.clamp(height_step, -max_step, max_step)
        ground_z = (-self.body_height).view(-1, 1)  # (num_envs, 1), foot height on level ground
        if swing_lift is None:
            swing_lift = self.trajectory.swing_lift

        # Current command, capped at what this gait can do
        commands, foot_speed = cap_command(commands, self.max_body_radius, self.max_foot_speed)
        forward, left, yaw = commands.unbind(-1)

        # Step faster once the stride would exceed MAX_STRIDE_LENGTH
        stride_limited = 2.0 * MAX_STRIDE_LENGTH / ((self.groups - 1) * foot_speed.clamp_min(1e-9))
        self.phase_duration = torch.where(
            foot_speed > 1e-6,
            torch.clamp(stride_limited, MIN_PHASE_DURATION, PHASE_DURATION),
            torch.full_like(foot_speed, PHASE_DURATION),
        )
        stance_time = (self.groups - 1) * self.phase_duration

        # Start of a phase: the next group lifts off from where it stands
        starting = self.phase_progress == 0.0
        self.swinging = torch.where(starting.unsqueeze(-1), self.group_mask[self.phase], self.swinging)
        self.swing_start = torch.where(starting.view(-1, 1, 1), self.foot_pos, self.swing_start)
        self.swing_target = torch.where(starting.view(-1, 1, 1), self.foot_pos, self.swing_target)

        # Swing landing targets follow the command, at a limited rate. Both ends stay inside the
        # stride circle, so every point in between does too.
        desired = self.trajectory.swingTarget(self.trajectory.footDisplacement(forward, left, yaw, stance_time / 2.0))
        desired[..., 2] = ground_z
        target_step = desired - self.swing_target
        step_norm = torch.linalg.norm(target_step, dim=-1, keepdim=True)
        target_step = target_step * torch.clamp(MAX_TARGET_SPEED * dt / step_norm.clamp_min(1e-9), max=1.0)
        self.swing_target = torch.where(self.swinging.unsqueeze(-1), self.swing_target + target_step, self.swing_target)

        # Advance; an idle command holds the gait where it is
        ds = torch.where(idle, 0.0, dt / self.phase_duration)
        move_dt = (~idle).float() * dt
        s = torch.clamp(self.phase_progress + ds, max=1.0)

        # Swing feet follow a curve from their lift-off point to their landing target,
        # stance feet move back by the body's motion
        spline = self.trajectory.genSwingSpline(self.swing_start, self.swing_target, swing_lift)
        swing = self.trajectory.evalBezier(spline, s.unsqueeze(-1))
        # A high lift can fold the leg past its reach, worst near the hip: keep what fits
        start_z = self.swing_start[..., 2]
        swing = self.limit_swing_lift(swing, start_z + (self.swing_target[..., 2] - start_z) * s.unsqueeze(-1))
        swing = torch.where((s >= 1.0).view(-1, 1, 1), self.swing_target, swing)
        stance = self.trajectory.stanceTarget(
            self.foot_pos, self.trajectory.footDisplacementFrom(forward, left, yaw, self.foot_pos, move_dt)
        )
        stance[..., 2] = ground_z  # on the ground, at the body's height
        # Only once its swing is under way: an idle gait holds at the start of a phase with the next
        # group marked swinging, and those feet stay on the ground (and follow the body height) until it moves
        in_air = self.swinging & (s > 0.0).unsqueeze(-1)
        self.foot_pos = torch.where(in_air.unsqueeze(-1), swing, stance)

        # Phase transition
        self.phase_progress = s
        done = s >= 1.0
        self.phase = torch.where(done, (self.phase + 1) % self.groups, self.phase)
        self.phase_progress = torch.where(done, 0.0, self.phase_progress)
        self.swinging = self.swinging & ~done.unsqueeze(-1)

        return self.foot_pos
