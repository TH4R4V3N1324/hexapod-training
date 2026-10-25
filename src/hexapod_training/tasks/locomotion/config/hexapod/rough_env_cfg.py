from isaaclab.utils import configclass, replace
from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg
from hexapod_training.assets.hexapod_cfg import HEXAPOD_CFG
from hexapod_training.terrains import HEXAPOD_ROUGH_CFG
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab_tasks.core.velocity import mdp
from hexapod_training.tasks.locomotion import mdp as hexapod_mdp
import math

@configclass
class HexapodRewards(RewardsCfg):
    """The walking task's rewards, plus a pull back toward the stand pose."""

    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_femur_joint", ".*_tibia_joint"])},
    )
    
    feet_air_time_variance = RewTerm(
        func=hexapod_mdp.feet_air_time_variance,
        weight=-1.0,
        params={"command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia")},
    )


@configclass
class HexapodRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: HexapodRewards = HexapodRewards()
    
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
        
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_tibia"
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_femur"
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.01)   # 0.1 m/s, half of 0.2
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.0625)   # 0.25 rad/s, half of 0.5
        self.rewards.feet_air_time.func = hexapod_mdp.feet_air_time
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.params["threshold"] = 0.5

        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 0.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.10, 0.10)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        
        self.actions.joint_pos.scale = 0.25
