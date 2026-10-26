from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainImporter

    from .commands import TrackedVelocityCommand


def terrain_levels_progress(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    min_distance: float = 1.0,
    promote: float = 0.75,
    demote: float = 0.5,
) -> torch.Tensor:
    """Move each robot up or down the terrain levels by how much of its commanded walking it did.

    The template's terrain_levels_vel promotes only robots that end 2 m (half a tile) from where they
    started. At 0.17 m/s top speed, with random commands resampled mid-episode, almost none do, while
    any that cover less than half their command's distance are demoted, so every robot drifts down
    to the easiest terrain. Here a robot moves up after covering promote of what it was asked to,
    and down below demote. Episodes that asked for less than min_distance (m) of walking (standing
    or very slow commands) say too little either way and leave the level as it is.

    Requires the command to be a TrackedVelocityCommand.

    Returns:
        The mean terrain level over all robots.
    """
    terrain: TerrainImporter = env.scene.terrain
    command: TrackedVelocityCommand = env.command_manager.get_term(command_name)
    asked = command.commanded_distance[env_ids]
    done = command.progress[env_ids]
    judged = asked >= min_distance
    move_up = judged & (done >= promote * asked)
    move_down = judged & (done < demote * asked)
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
