from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_yaw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


def tof_pattern(cfg: ToFPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """One ray per (down-sampled) pixel of a pinhole depth camera looking along +x, rows top to bottom."""
    half_w = math.tan(math.radians(cfg.horizontal_fov) / 2.0)
    half_h = math.tan(math.radians(cfg.vertical_fov) / 2.0)
    # pixel centres on the image plane at x = 1; image left is +y, image top is +z
    u = ((torch.arange(cfg.width, device=device) + 0.5) / cfg.width * 2.0 - 1.0) * half_w
    v = ((torch.arange(cfg.height, device=device) + 0.5) / cfg.height * 2.0 - 1.0) * half_h
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    directions = torch.stack((torch.ones_like(uu), -uu, -vv), dim=-1).view(-1, 3)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    return torch.zeros_like(directions), directions


@configclass
class ToFPatternCfg(PatternBaseCfg):
    """Arducam ToF camera (B0410): 240 x 180 depth, 70 deg diagonal, so 58.5 x 45.6 deg at 4:3.

    width x height is the down-sampled image the map is built from (hexapod-ros2 uses every
    depth_stride-th pixel); 40 x 30 still puts rays ~2 cm apart on the ground a metre ahead.
    """

    func: Callable = tof_pattern
    width: int = 40
    height: int = 30
    horizontal_fov: float = 58.5  # deg
    vertical_fov: float = 45.6  # deg


class tof_height_map(ManagerTermBase):
    """Height map the ToF camera has built so far, sampled on a grid around the body.

    Mirrors hexapod-ros2's HeightMap: camera hits go into cells fixed in the world, at most
    update_period steps apart, each cell moving halfway to the frame's mean height; cells further
    than map_radius from the camera are forgotten. Ground the camera hasn't seen (under the body at
    the start, behind or beside it when backing up or strafing) reads 0 with a mask of 0, which is
    where the policy has to fall back on joint effort.

    Returns (N, 2 * G): for each of the G grid points, body height above the mapped ground minus
    offset (0 on flat ground at standing height), then 1 if that cell is mapped, else 0.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self.sensor = env.scene.sensors[params["sensor_cfg"].name]
        self.asset = env.scene[params["asset_cfg"].name]
        self.resolution = params["resolution"]
        # ring buffer of cells per env, keyed by world cell index: big enough that nothing within
        # map_radius shares a slot, so pruning is just a key that no longer matches
        self.side = int(math.ceil(2.0 * params["map_radius"] / self.resolution)) + 4
        cells = self.side * self.side
        self.heights = torch.zeros(self.num_envs, cells, device=self.device)
        self.keys = torch.full((self.num_envs, cells, 2), torch.iinfo(torch.int64).min, device=self.device)
        self.steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # sample grid in the body's yaw frame, as GridPatternCfg lays it out
        sx, sy = params["size"]
        xs = torch.arange(-sx / 2, sx / 2 + 1.0e-9, self.resolution, device=self.device)
        ys = torch.arange(-sy / 2, sy / 2 + 1.0e-9, self.resolution, device=self.device)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")
        self.grid = torch.stack((gx.flatten(), gy.flatten(), torch.zeros_like(gx.flatten())), dim=-1)

    def reset(self, env_ids=None):
        # the real map starts empty once the robot is standing
        if env_ids is None:
            env_ids = slice(None)
        self.keys[env_ids] = torch.iinfo(torch.int64).min
        self.steps[env_ids] = 0

    def _slots(self, ix: torch.Tensor, iy: torch.Tensor) -> torch.Tensor:
        return (ix % self.side) * self.side + (iy % self.side)

    def _insert(self, env_ids: torch.Tensor, map_radius: float, point_noise: float, frame_bias: float):
        hits = self.sensor.data.ray_hits_w.torch[env_ids]
        camera = self.sensor.data.pos_w.torch[env_ids]
        valid = torch.isfinite(hits).all(dim=-1)
        hits = torch.where(valid.unsqueeze(-1), hits, camera.unsqueeze(1))
        valid &= (hits[..., :2] - camera[:, None, :2]).norm(dim=-1) <= map_radius
        # ToF precision is ~2 cm a point; the body's pitch skews a whole frame by up to ~1 cm
        z = hits[..., 2] + point_noise * (2.0 * torch.rand_like(hits[..., 2]) - 1.0)
        z = z + frame_bias * (2.0 * torch.rand(len(env_ids), 1, device=self.device) - 1.0)
        ix = torch.floor(hits[..., 0] / self.resolution).long()
        iy = torch.floor(hits[..., 1] / self.resolution).long()

        # average each cell's points in this frame
        cells = self.side * self.side
        local = (torch.arange(len(env_ids), device=self.device).unsqueeze(1) * cells + self._slots(ix, iy))[valid]
        total = torch.zeros(len(env_ids) * cells, device=self.device).index_add_(0, local, z[valid])
        count = torch.zeros(len(env_ids) * cells, device=self.device).index_add_(0, local, torch.ones_like(z[valid]))
        new_keys = torch.empty(len(env_ids) * cells, 2, dtype=torch.long, device=self.device)
        new_keys[local] = torch.stack((ix[valid], iy[valid]), dim=-1)
        seen = count.nonzero().squeeze(-1)
        mean = total[seen] / count[seen]
        new_keys = new_keys[seen]

        # a cell already holding this world cell moves halfway to the frame's mean, others take it
        rows = env_ids[seen // cells]
        slots = seen % cells
        old = self.heights[rows, slots]
        same = (self.keys[rows, slots] == new_keys).all(dim=-1)
        self.heights[rows, slots] = torch.where(same, old + 0.5 * (mean - old), mean)
        self.keys[rows, slots] = new_keys

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        resolution: float = 0.05,
        size: tuple[float, float] = (0.7, 0.7),
        map_radius: float = 1.0,
        update_period: int = 10,
        point_noise: float = 0.02,
        frame_bias: float = 0.01,
        offset: float = 0.145,
        clip: float = 0.5,
    ) -> torch.Tensor:
        due = (self.steps % update_period == 0).nonzero().squeeze(-1)
        self.steps += 1
        if len(due) > 0:
            self._insert(due, map_radius, point_noise, frame_bias)

        pos = self.asset.data.root_pos_w.torch
        quat = self.asset.data.root_quat_w.torch
        num_points = len(self.grid)
        points = quat_apply_yaw(quat.repeat_interleave(num_points, dim=0), self.grid.repeat(self.num_envs, 1))
        points = points.view(self.num_envs, num_points, 3) + pos.unsqueeze(1)
        ix = torch.floor(points[..., 0] / self.resolution).long()
        iy = torch.floor(points[..., 1] / self.resolution).long()
        slots = self._slots(ix, iy)
        keys = torch.gather(self.keys, 1, slots.unsqueeze(-1).expand(-1, -1, 2))
        known = (keys == torch.stack((ix, iy), dim=-1)).all(dim=-1)
        ground = torch.gather(self.heights, 1, slots)
        heights = torch.where(known, (pos[:, 2:3] - ground - offset).clamp(-clip, clip), 0.0)
        return torch.cat((heights, known.float()), dim=-1)
