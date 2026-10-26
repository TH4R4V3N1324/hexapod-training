from isaaclab.terrains import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen

HEXAPOD_ROUGH_CFG = TerrainGeneratorCfg(
    # rows in order of difficulty, for the terrain_levels curriculum. The template turns this on in
    # its __post_init__, but the env cfg swaps the whole generator in afterwards, which reset it to
    # False: every tile got a random difficulty and the levels meant nothing.
    curriculum=True,
    size=(4.0, 4.0),         # one tile, m (ANYmal uses 8 x 8)
    border_width=10.0,
    num_rows=10,             # difficulty levels
    num_cols=10,             # tiles per level
    horizontal_scale=0.05,   # height-field cell size, m
    vertical_scale=0.002,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(0.0, 0.03), noise_step=0.005),
        "slopes": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.35), platform_width=1.0),
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15, step_height_range=(0.01, 0.06),
            step_width=0.2, platform_width=1.0),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.3,
            grid_height_range=(0.01, 0.05), platform_width=1.0),
    },
)