"""Tests for the gait port, checked against the hexapod-ros2 C++ behaviour."""

import math

import torch

from hexapod_training.gait import FrameConverter, KinematicSolver, TrajectoryGenerator, TripodGait
from hexapod_training.gait.gait_config import MAX_STRIDE_LENGTH, NEUTRAL_FOOT, SWING_LIFT


def test_ik_fk_round_trip():
    solver = KinematicSolver()
    targets = torch.tensor([[0.177, 0.0, -0.134], [0.20, 0.04, -0.10], [0.15, -0.05, -0.16]])
    assert torch.allclose(solver.solveFK(solver.solveIK(targets)), targets, atol=1e-6)


def test_neutral_foot_matches_stand_pose():
    # The stand pose in hexapod_cfg.py is the neutral foot
    angles = KinematicSolver().solveIK(torch.tensor(NEUTRAL_FOOT))
    assert torch.allclose(angles, torch.tensor([0.0, -0.523, 0.376]), atol=2e-3)


def test_frame_round_trip():
    converter = FrameConverter()
    position = torch.rand(4, 6, 3)
    assert torch.allclose(converter.bodyToLegFrame(converter.legToBodyFrame(position)), position, atol=1e-6)
    # Leg 2 points straight right: +x in its frame is -y in the body
    body = converter.legToBodyFrame(torch.tensor([1.0, 0.0, 0.0]).expand(6, 3))
    assert torch.allclose(body[1], torch.tensor([0.0, -1.0 - 0.11836, 0.0]), atol=1e-6)


def test_foot_displacement():
    trajectory = TrajectoryGenerator()
    one = torch.ones(1)
    zero = torch.zeros(1)
    # Forward 0.1 m in the body frame, seen from leg 5 (pointing left) that's -y
    forward = trajectory.footDisplacement(0.1 * one, zero, zero, 1.0)
    assert torch.allclose(forward[0, 4], torch.tensor([0.0, -0.1, 0.0]), atol=1e-6)
    # Turning anticlockwise moves every foot sideways the same way round its leg, by the yaw arc
    turn = trajectory.footDisplacement(zero, zero, 0.1 * one, 1.0)
    radius = trajectory.neutralFootInBody().norm(dim=-1)
    expected = 2 * radius * math.sin(0.05)
    assert torch.allclose(turn[0].norm(dim=-1), expected, atol=1e-6)
    assert (turn[0, :, 1] > 0).all()


def test_clamp_to_stride():
    trajectory = TrajectoryGenerator()
    target = trajectory.neutral + torch.tensor([0.1, 0.0, 0.05])
    clamped = trajectory.clampToStride(target.unsqueeze(0))[0]
    offset = clamped - trajectory.neutral
    assert torch.allclose(offset[:, :2].norm(dim=-1), torch.full((6,), MAX_STRIDE_LENGTH))
    assert torch.allclose(offset[:, 2], torch.zeros(6))


def test_swing_curve():
    trajectory = TrajectoryGenerator()
    start = torch.tensor([0.177, -0.05, -0.134])
    end = torch.tensor([0.177, 0.05, -0.134])
    points = trajectory.genSwingSpline(start, end)
    assert torch.allclose(trajectory.evalBezier(points, 0.0), start)
    assert torch.allclose(trajectory.evalBezier(points, 1.0), end)
    peak = trajectory.evalBezier(points, 0.5)
    assert abs((peak[2] - start[2]).item() - 0.875 * SWING_LIFT) < 1e-6


def test_tripod_gait_walks():
    num_envs = 3
    gait = TripodGait(num_envs)
    commands = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [1.0, 0.0, 0.0]])
    dt = 0.02
    neutral = gait.trajectory.neutral
    swung = torch.zeros(num_envs, 6, dtype=torch.bool)
    solver = KinematicSolver()
    for _ in range(200):
        feet = gait.step(commands, dt)
        assert feet.shape == (num_envs, 6, 3)
        assert torch.isfinite(solver.solveIK(feet)).all()
        offset = gait.foot_pos - neutral
        assert (offset[..., :2].norm(dim=-1) <= MAX_STRIDE_LENGTH + 1e-6).all()
        swung |= gait.swinging
        # Only one tripod is ever in the air
        assert (gait.swinging.sum(dim=-1) <= 3).all()

    # Idle holds still at neutral, with the gait clock frozen at the start of the first phase
    assert torch.allclose(gait.foot_pos[0], neutral)
    assert gait.phase_progress[0] == 0.0
    # Slow walking uses the full phase, fast walking is capped at the minimum
    assert abs(gait.phase_duration[1].item() - 1.0) < 1e-6
    assert abs(gait.phase_duration[2].item() - 0.7) < 1e-6
    assert swung[1:].all()


def test_tripod_gait_reset():
    gait = TripodGait(2)
    for _ in range(30):
        gait.step(torch.tensor([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]), 0.02)
    gait.reset(torch.tensor([1]))
    assert torch.allclose(gait.foot_pos[1], gait.trajectory.neutral)
    assert gait.phase_progress[1] == 0.0
    assert gait.phase_progress[0] > 0.0


def test_leg_cycle_phase():
    gait = TripodGait(1)
    commands = torch.tensor([[0.1, 0.0, 0.0]])
    gait.step(commands, 0.02)
    phase = gait.leg_cycle_phase()[0]
    # The swinging tripod is just past lift-off, in the first half of its cycle; the other is half a cycle on
    assert ((phase < 0.5) == gait.swinging[0]).all()
    assert torch.allclose(phase[[1, 3, 5]] - phase[[0, 2, 4]], torch.full((3,), 0.5))
    # Phase only moves forward, and wraps once per cycle
    previous = phase.clone()
    wraps = 0
    for _ in range(210):
        gait.step(commands, 0.02)
        phase = gait.leg_cycle_phase()[0]
        wraps += int((phase[0] < previous[0]).item())
        previous = phase.clone()
    assert wraps == 2  # 210 steps x 0.02 s = 4.2 s, a cycle is two 1 s phases


def test_swing_easing_blend():
    # Matches hexapod-ros2 a8064dc: st = easing * smootherstep(s) + (1 - easing) * s
    trajectory = TrajectoryGenerator()
    start = torch.tensor([0.0, 0.0, 0.0])
    end = torch.tensor([1.0, 0.0, 0.0])
    points = torch.stack((start, start, start, end, end))  # x along the curve is a plain function of st
    s = torch.tensor(0.1)
    smootherstep = s**3 * (s * (6 * s - 15) + 10)
    for easing in (0.0, 0.5, 1.0):
        trajectory.swing_easing = easing
        st = easing * smootherstep + (1 - easing) * s
        u = 1 - st
        expected_x = 4 * u * st**3 + st**4
        assert torch.allclose(trajectory.evalBezier(points, s)[0], expected_x, atol=1e-6)


def test_cap_command():
    from hexapod_training.gait.tripod_gait import cap_command

    gait = TripodGait(1)
    radius, top = gait.max_body_radius, gait.max_foot_speed
    commands = torch.tensor([[0.05, 0.0, 0.0], [0.17, 0.1, 0.5], [0.0, 0.0, 0.0]])
    capped, foot_speed = cap_command(commands, radius, top)
    # Within reach: untouched. Too fast: scaled down evenly onto the limit. Standing: still zero.
    assert torch.allclose(capped[0], commands[0])
    assert torch.allclose(capped[1] / commands[1], (capped[1, 0] / commands[1, 0]).expand(3))
    assert abs(torch.hypot(capped[1, 0], capped[1, 1]) + capped[1, 2].abs() * radius - top) < 1e-6
    assert torch.allclose(capped[2], torch.zeros(3)) and foot_speed[2] == 0.0
    # Capping again changes nothing, so the gait sees the same command the reward does
    again, _ = cap_command(capped, radius, top)
    assert torch.allclose(again, capped)
