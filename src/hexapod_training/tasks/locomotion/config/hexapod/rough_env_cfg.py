from isaaclab.utils import configclass, replace
from isaaclab.sensors import RayCasterCfg
from isaaclab_tasks.core.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
from hexapod_training.assets.hexapod_cfg import HEXAPOD_CFG
from hexapod_training.terrains import HEXAPOD_ROUGH_CFG
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_tasks.core.velocity import mdp
from hexapod_training.tasks.locomotion import mdp as hexapod_mdp
import math

TOF_PITCH = 0.0  # rad, down; the V6.1 pod faces straight ahead (hexapod-ros2's README says 10 deg)


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
        # the ground truth all around the body isn't available on the robot; the ToF camera's map is
        height_scan = None
        # what hexapod-ros2's HeightMap holds: ground the front camera has seen, remembered as the
        # robot walks onto it, with a mask for cells it hasn't (noise is added per point, inside)
        height_map = ObsTerm(func=hexapod_mdp.tof_height_map, params={"sensor_cfg": SceneEntityCfg("tof_camera")})

        # ST3215 load readback, noisy and slow on the real servos. The noise is in N·m (added before
        # the scale), and the scale makes full stall (2.94 N·m) read 1
        joint_effort = ObsTerm(func=mdp.joint_effort, noise=Unoise(n_min=-0.3, n_max=0.3), scale=1.0 / 2.94)
        # per leg sin/cos of its cycle phase and a swing flag, from the gait in the action term
        gait_phase = ObsTerm(func=hexapod_mdp.gait_phase)

    @configclass
    class CriticCfg(ObsGroup):
        """Ground truth only the simulator has, for the critic on top of the policy's observations."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # the terrain all around the body, seen or not, on the same grid as the policy's height map
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.145},
            clip=(-0.5, 0.5),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class HexapodTerminations(TerminationsCfg):
    # resets an env whose simulation has blown up; HexapodLocomotionEnv drops that step's reward
    physics_blowup = DoneTerm(func=hexapod_mdp.physics_blowup)


@configclass
class HexapodSceneCfg(MySceneCfg):
    # Arducam ToF (B0410) on the front camera pod, as front_depth_camera_link in hexapod-ros2's
    # hexapod.cameras.xacro: 0.138 m ahead of and 0.0124 m below base_link, facing straight ahead
    # (tof_pitch; positive tilts it down). Only the terrain is cast against, so the robot's own legs
    # are never seen, as the real map's 6 cm self-filter around the feet intends.
    tof_camera = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.138, 0.0, -0.0124), rot=(0.0, math.sin(TOF_PITCH / 2), 0.0, math.cos(TOF_PITCH / 2))
        ),
        ray_alignment="base",
        pattern_cfg=hexapod_mdp.ToFPatternCfg(),
        max_distance=4.0,  # far mode
        update_period=0.2,  # map_update_rate: 5 Hz
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
        debug_vis=False,
    )


@configclass
class HexapodRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: HexapodSceneCfg = HexapodSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: HexapodRewards = HexapodRewards()
    observations: HexapodObservations = HexapodObservations()
    terminations: HexapodTerminations = HexapodTerminations()
    
    def __post_init__(self):
        super().__post_init__()

        # scene: your robot and your terrain go into the template's slots
        self.scene.robot = replace(HEXAPOD_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_generator = HEXAPOD_ROUGH_CFG
        # the critic's ground-truth scan: the policy's 0.7 x 0.7 m grid at 0.05 m
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/base_link"
        self.scene.height_scanner.pattern_cfg.resolution = 0.05
        self.scene.height_scanner.pattern_cfg.size = [0.7, 0.7]
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
