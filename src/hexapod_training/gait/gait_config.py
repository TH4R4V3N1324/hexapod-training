"""Gait constants, ported from gait_config.hpp, kinematic_solver.hpp and trajectory_generator.hpp."""

MAX_LEGS = 6
MAX_STRIDE_LENGTH = 0.06  # m, furthest a foot moves from its neutral position
SWING_LIFT = 0.06  # m, height of the swing's middle control points (peak ~0.88x this)
# How much a swing eases in and out (0 even pace, 1 smootherstep). Full easing left the feet nearly
# still around every lift-off and touchdown, a visible pause at each phase change; half keeps them
# moving through it while still landing at half the even-pace speed.
SWING_EASING = 0.5

# Neutral stance from the Hexapod_V6.1 stand pose: each foot straight out from its coxa
# (so the corner feet sit at 45 degrees), 0.177 m out and 0.134 m below the hips
NEUTRAL_FOOT = (0.177, 0.0, -0.134)  # leg frame

# Hexapod_V6.1, matching the URDF. The tibia is curved: its length is the straight line
# from the knee axis to the bottom of the foot.
COXA_LENGTH = 0.063  # m
FEMUR_LENGTH = 0.100  # m
TIBIA_LENGTH = 0.186  # m

# Tripod groups, as leg indices 0-5 (legs 1-6): {1, 3, 5} and {2, 4, 6}
TRIPOD_GROUPS = ((0, 2, 4), (1, 3, 5))

PHASE_DURATION = 1.0  # s per gait phase at low speeds
MIN_PHASE_DURATION = 0.7  # s, shortest phase allowed, which sets the gait's top speed
MAX_TARGET_SPEED = 0.5  # m/s, how fast a swing foot's landing target can follow the command
IDLE_DEADZONE = 0.01  # command below this on every axis holds the gait still

# Joint names in leg order (coxa, femur, tibia per leg), matching the (num_envs, 6, 3) angle tensors
JOINT_NAMES = [f"leg{leg}_{joint}_joint" for leg in range(1, MAX_LEGS + 1) for joint in ("coxa", "femur", "tibia")]
