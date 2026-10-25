from isaaclab.utils import configclass, replace
from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from hexapod_training.assets.hexapod_cfg import HEXAPOD_CFG
from hexapod_training.terrains import HEXAPOD_ROUGH_CFG


@configclass
class HexapodRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
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