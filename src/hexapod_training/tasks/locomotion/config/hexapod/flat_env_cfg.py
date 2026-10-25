from isaaclab.utils import configclass

from .rough_env_cfg import HexapodRoughEnvCfg


@configclass
class HexapodFlatEnvCfg(HexapodRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # plain ground, no terrain, no height scanner
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None