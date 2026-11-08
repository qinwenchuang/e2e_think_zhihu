"""连续占据查询与 proposal 相关的 BEV 几何特征。

本模块只依赖 Config 和 PyTorch，便于独立做张量/几何单元测试。
坐标约定与全项目一致：ego x 前、y 左；BEV 行=x、列=y。
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..md_common import Config


def xy_to_grid(xy: torch.Tensor, cfg: Config) -> torch.Tensor:
    """物理 xy → grid_sample(align_corners=False) 的归一化 (gx=y, gy=x)。"""
    gx = (xy[..., 1] - cfg.pc_y_min) / (cfg.pc_y_max - cfg.pc_y_min) * 2.0 - 1.0
    gy = (xy[..., 0] - cfg.pc_x_min) / (cfg.pc_x_max - cfg.pc_x_min) * 2.0 - 1.0
    return torch.stack([gx, gy], dim=-1)


def xy_in_bounds(xy: torch.Tensor, cfg: Config) -> torch.Tensor:
    return ((xy[..., 0] >= cfg.pc_x_min) & (xy[..., 0] <= cfg.pc_x_max)
            & (xy[..., 1] >= cfg.pc_y_min) & (xy[..., 1] <= cfg.pc_y_max))


class ContinuousOccDecoder(nn.Module):
    """在规则 BEV 特征上查询任意连续三维物理坐标的占据 logit。

    BEV 特征按物理坐标双线性采样；坐标 Fourier 编码补充高度与亚 cell 位置。
    额外预测少量局部 xy 偏移，缓解 1m cell 对表面位置的量化。
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        bands = cfg.occ_fourier_bands
        coord_dim = 3 + 6 * bands
        self.register_buffer("frequencies",
                             (2.0 ** torch.arange(bands).float()) * math.pi)
        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_dim, d), nn.GELU(), nn.Linear(d, d))
        self.offset_head = nn.Linear(d, cfg.occ_decoder_offsets * 2)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        self.input_norm = nn.LayerNorm(2 * d)
        self.input_proj = nn.Linear(2 * d, d)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(),
                          nn.Linear(2 * d, d))
            for _ in range(2)
        ])
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)

    def _coord_features(self, queries: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        x = (queries[..., 0] - cfg.pc_x_min) / (cfg.pc_x_max - cfg.pc_x_min) * 2 - 1
        y = (queries[..., 1] - cfg.pc_y_min) / (cfg.pc_y_max - cfg.pc_y_min) * 2 - 1
        z = queries[..., 2] / max(abs(cfg.occ_z_min), abs(cfg.occ_z_max), 1.0)
        normalized = torch.stack([x, y, z], dim=-1)
        angles = normalized[..., None] * self.frequencies
        encoded = torch.cat([normalized, torch.sin(angles).flatten(-2),
                             torch.cos(angles).flatten(-2)], dim=-1)
        return self.coord_mlp(encoded)

    def _sample_bev(self, feat2d: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """feat2d (B,D,G,G)，xy (B,...,2) → (B,...,D)，支持额外偏移维。"""
        B = xy.shape[0]
        leading = xy.shape[1:-1]
        flat = xy.reshape(B, -1, 2)
        grid = xy_to_grid(flat, self.cfg)[:, None].to(feat2d.dtype)  # (B,1,Q,2)
        sampled = F.grid_sample(feat2d, grid, mode="bilinear", padding_mode="zeros",
                                align_corners=False)             # (B,D,1,Q)
        sampled = sampled[:, :, 0].transpose(1, 2)
        return sampled.reshape(B, *leading, feat2d.shape[1])

    def forward(self, bev_features: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """
        :param bev_features: (B,G*G,D)
        :param queries: (B,Q,3)，当前 ego 系物理坐标
        :return: (B,Q) occupancy logits
        """
        B, Q = queries.shape[:2]
        d = bev_features.shape[-1]
        g = self.cfg.bev_grid
        feat2d = bev_features.transpose(1, 2).reshape(B, d, g, g)
        coord = self._coord_features(queries)
        bev = self._sample_bev(feat2d, queries[..., :2])
        if self.cfg.occ_decoder_offsets > 0:
            offsets = torch.tanh(self.offset_head(coord)).reshape(
                B, Q, self.cfg.occ_decoder_offsets, 2) * self.cfg.bev_pixel_size
            neighbor = self._sample_bev(feat2d, queries[..., None, :2] + offsets).mean(dim=2)
            bev = (bev + neighbor) / 2.0
        x = self.input_proj(F.gelu(self.input_norm(torch.cat([bev, coord], dim=-1))))
        for block in self.blocks:
            x = x + block(x)
        return self.out(self.out_norm(x)).squeeze(-1)


class BEVProposalFeatures(nn.Module):
    """为每条候选轨迹提取地图、检测和可选连续占据特征。

    每条 proposal 输出 11 维：
      agent_max/mean、road_min/mean、occ_max/mean、progress、abs_lateral、
      comfort、final_x、final_y。
    地图/检测/占据默认 detach， scorer 不会反向污染感知监督。
    """

    feature_dim = 11

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # 中心 + 四个带 margin 的 footprint 角点。
        hl = cfg.ego_len / 2 + cfg.col_margin
        hw = cfg.ego_wid / 2 + cfg.col_margin
        points = torch.tensor([[0.0, 0.0], [hl, hw], [hl, -hw],
                               [-hl, hw], [-hl, -hw]], dtype=torch.float32)
        self.register_buffer("footprint", points)

    def _footprint_xy(self, proposals: torch.Tensor) -> torch.Tensor:
        xy = proposals[..., :2]
        yaw = proposals[..., 2]
        c, s = torch.cos(yaw), torch.sin(yaw)
        rot = torch.stack([torch.stack([c, -s], dim=-1),
                           torch.stack([s, c], dim=-1)], dim=-2)  # (B,N,T,2,2)
        local = self.footprint.to(dtype=proposals.dtype)
        offset = torch.einsum("bntij,fj->bntfi", rot, local)
        return xy[..., None, :] + offset                         # (B,N,T,F,2)

    @staticmethod
    def _sample(field: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """field (B,C,H,W)，grid (B,N,W_out,2) → (B,N,W_out,C)。"""
        value = F.grid_sample(field, grid.to(field.dtype), mode="bilinear",
                              padding_mode="zeros", align_corners=False)
        return value.permute(0, 2, 3, 1)

    def forward(self, proposals: torch.Tensor, bev_output: Dict[str, torch.Tensor],
                occ_decoder: Optional[ContinuousOccDecoder] = None,
                ) -> torch.Tensor:
        cfg = self.cfg
        B, N, T = proposals.shape[:3]
        footprint = self._footprint_xy(proposals.detach())
        Fp = footprint.shape[3]
        flat_xy = footprint.reshape(B, N, T * Fp, 2)
        grid = xy_to_grid(flat_xy, cfg)

        map_logits = bev_output["map_logits"]
        det_heatmap = bev_output["det_heatmap"]
        if cfg.planner_bev_detach:
            map_logits = map_logits.detach()
            det_heatmap = det_heatmap.detach()
        map_prob = torch.softmax(map_logits.float(), dim=1)
        road = (map_prob[:, 1:2] + map_prob[:, 2:3]).contiguous()
        agent = torch.sigmoid(det_heatmap.float()).amax(dim=1, keepdim=True).contiguous()
        road_samples = self._sample(road, grid)[..., 0].reshape(B, N, T, Fp)
        agent_samples = self._sample(agent, grid)[..., 0].reshape(B, N, T, Fp)

        if occ_decoder is not None and "bev_features" in bev_output:
            queries = torch.cat([flat_xy.reshape(B, N * T * Fp, 2),
                                 torch.full((B, N * T * Fp, 1), cfg.planner_occ_z,
                                            device=flat_xy.device, dtype=flat_xy.dtype)],
                                dim=-1)
            bev_features = bev_output["bev_features"]
            if cfg.planner_bev_detach:
                bev_features = bev_features.detach()
                with torch.no_grad():
                    occ_logits = occ_decoder(bev_features, queries)
            else:
                occ_logits = occ_decoder(bev_features, queries)
            occ = torch.sigmoid(occ_logits.float())
            # 已知 BEV 范围之外按有风险处理，避免模型利用边界外零特征获得安全分。
            occ = torch.where(xy_in_bounds(queries[..., :2], cfg), occ,
                              torch.ones_like(occ))
            occ = occ.reshape(B, N, T, Fp)
        else:
            occ = torch.zeros_like(agent_samples)

        props = proposals.detach().float()
        if T >= 3:
            # yaw 差分必须先回绕：±π 边界上直接二阶差分会产生虚假的 6.28 大值
            dyaw = torch.atan2(torch.sin(torch.diff(props[..., 2], dim=-1)),
                               torch.cos(torch.diff(props[..., 2], dim=-1)))
            comfort = torch.diff(dyaw, n=1, dim=-1).abs().mean(dim=-1)
        else:
            comfort = props.new_zeros(B, N)
        features = torch.stack([
            agent_samples.amax(dim=(2, 3)),
            agent_samples.mean(dim=(2, 3)),
            road_samples.amin(dim=(2, 3)),
            road_samples.mean(dim=(2, 3)),
            occ.amax(dim=(2, 3)),
            occ.mean(dim=(2, 3)),
            props[:, :, -1, 0] / 20.0,
            props[:, :, -1, 1].abs() / 20.0,
            comfort,
            props[:, :, -1, 0],
            props[:, :, -1, 1],
        ], dim=-1)
        return features