from isaaclab.utils import configclass

from .rough_env_cfg import HexapodRoughEnvCfg


@configclass
class HexapodFlatEnvCfg(HexapodRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # plain ground, no terrain, no ToF camera
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.tof_camera = None
        self.scene.height_scanner = None
        self.observations.policy.height_map = None
        self.observations.critic = None  # flat runs keep the shared actor/critic of PPORunnerCfg
        self.curriculum.terrain_levels = None