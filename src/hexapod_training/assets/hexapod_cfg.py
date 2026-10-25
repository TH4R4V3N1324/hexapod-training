import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent   # if hexapod_cfg.py sits in assets/
HEXAPOD_USD_PATH = ASSETS_DIR / "hexapod" / "usd" / "hexapod" / "hexapod.usda"

HEXAPOD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(HEXAPOD_USD_PATH),
        activate_contact_sensors=True,   # needed for foot-contact rewards
        articulation_props=[
            PhysxArticulationCfg(enabled_self_collisions=True),
            NewtonArticulationCfg(self_collision_enabled=True),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.145),
        joint_pos={".*_coxa_joint": 0.0, ".*_femur_joint": -0.523, ".*_tibia_joint": 0.376},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "st3215": DelayedPDActuatorCfg(
            joint_names_expr=[".*_joint"],
            effort_limit_sim=2.94,       # N·m, as in the URDF
            velocity_limit_sim=4.7,      # rad/s
            stiffness=20.0,              # placeholder until the servo is measured
            damping=0.5,
            armature=0.026,
            friction=0.05,
            min_delay=0, 
            max_delay=2,
        ),
    },
)