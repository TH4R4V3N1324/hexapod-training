from isaaclab.utils import configclass, replace
from isaaclab_tasks.core.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
from hexapod_training.assets.hexapod_cfg import HEXAPOD_CFG
from hexapod_training.terrains import HEXAPOD_ROUGH_CFG
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_tasks.core.velocity import mdp
from hexapod_training.tasks.locomotion import mdp as hexapod_mdp
import math

@configclass
class HexapodRewards(RewardsCfg):
    """The gait owns the step pattern: keep going where you were told, stay level and steady, and
    don't touch the ground with anything but feet."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.01)},  # 0.1 m/s
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.0625)},  # 0.25 rad/s, half of 0.5
    )
    # -- body: attitude is the main thing the residual should improve
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-4.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # -- contacts
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*_femur"]), "threshold": 1.0},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_tibia"),
        },
    )
    # -- residual: stay close to the gait unless deviating pays
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    # -- the gait's own step pattern and motion would be penalised
    feet_air_time = None
    dof_acc_l2 = None


@configclass
class HexapodObservations(ObservationsCfg):
    """Only what the real robot can measure."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        base_lin_vel = None  # nothing reliable on the robot; leg odometry over-reads
        height_scan = None  # off until stage 4 (depth / ToF height map)

        # ST3215 load readback, noisy and slow on the real servos. The noise is in N·m (added before
        # the scale), and the scale makes full stall (2.94 N·m) read 1
        joint_effort = ObsTerm(func=mdp.joint_effort, noise=Unoise(n_min=-0.3, n_max=0.3), scale=1.0 / 2.94)
        # per leg sin/cos of its cycle phase and a swing flag, from the gait in the action term
        gait_phase = ObsTerm(func=hexapod_mdp.gait_phase)

    policy: PolicyCfg = PolicyCfg()


@configclass
class HexapodTerminations(TerminationsCfg):
    # resets an env whose simulation has blown up; HexapodLocomotionEnv drops that step's reward
    physics_blowup = DoneTerm(func=hexapod_mdp.physics_blowup)


@configclass
class HexapodRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: HexapodRewards = HexapodRewards()
    observations: HexapodObservations = HexapodObservations()
    terminations: HexapodTerminations = HexapodTerminations()
    
    def __post_init__(self):
        super().__post_init__()

        # scene: your robot and your terrain go into the template's slots
        self.scene.robot = replace(HEXAPOD_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_generator = HEXAPOD_ROUGH_CFG
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/base_link"
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/.*"

        # the template names ANYmal's links; rename them to yours
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_com.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"

        # the template's disturbances are sized for ANYmal: a 0.5 m/s push is 3x this robot's walking
        # speed and a ±5 cm CoM shift is most of its 16 cm wide body. Start small, raise once it trains.
        self.events.push_robot.params["velocity_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}
        self.events.base_com.params["com_range"] = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01)}

        # the tripod gait tops out at 2 x 0.06 / 0.7 = 0.171 m/s of foot speed
        self.commands.base_velocity.ranges.lin_vel_x = (-0.17, 0.17)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.10, 0.10)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        # terrain levels follow how much of its commanded walking each robot did: the template's 2 m
        # from the start is more than a robot this slow covers, so it only ever demoted
        self.commands.base_velocity.class_type = hexapod_mdp.TrackedVelocityCommand
        self.curriculum.terrain_levels.func = hexapod_mdp.terrain_levels_progress
        
        # the hand-written tripod gait walks; the policy only shifts each foot target
        self.actions.joint_pos = hexapod_mdp.GaitResidualActionCfg()
        self.actions.joint_pos.zero_residual = False
