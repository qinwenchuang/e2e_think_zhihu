"""BEV 地图的 PA 对比学习与 BANE 边界困难负样本（训练专用）。

借鉴 Contextrast++（arXiv:2608.22679）：
  * Pixel-to-Anchor（PA）：同类像素向类别均值 Anchor 聚拢，异类像素被推远；
  * Boundary-Aware Negative（BANE）sampling：从误分类区域提取靠近边界的样本，
    作为其他类别 Anchor 的困难负样本。

本实现是最小优先组合，不包含 AA 记忆库和多层自适应融合。投影头只在
LiteBEVHead.training 且 cfg.map_ccl_enabled 时执行，验证/推理没有附加前向。
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..md_common import Config


@torch.no_grad()
def bane_negative_mask(logits: torch.Tensor, gt: torch.Tensor,
                       ratio: float = 0.5) -> torch.Tensor:
    """返回 BANE 困难负样本并集 (B,H,W)，不改变梯度。

    对每个 GT 类 n，误分类区域为 ``(gt == n) & (pred != n)``。距离变换后，
    选取距误差区域边界最近的 ratio 部分。对类别 c 计算 PA 时，调用方还会
    加上 ``gt != c``，因此这些误差 embedding 会作为其他类别 Anchor 的负样本；
    对 GT=c 的误分类像素自身则仍是正样本，语义与论文一致。
    """
    if logits.ndim != 4 or gt.ndim != 3:
        raise ValueError(f"logits/gt 形状错误：{tuple(logits.shape)} / {tuple(gt.shape)}")
    if logits.shape[0] != gt.shape[0] or logits.shape[-2:] != gt.shape[-2:]:
        raise ValueError("logits 与 gt 的 batch/空间尺寸必须一致")
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"BANE ratio 必须在 (0,1]，实际 {ratio}")
    pred = logits.detach().argmax(dim=1)
    pred_np = pred.cpu().numpy().astype(np.uint8)
    gt_np = gt.detach().cpu().numpy().astype(np.uint8)
    selected = np.zeros_like(gt_np, dtype=bool)
    for b in range(gt_np.shape[0]):
        for c in range(logits.shape[1]):
            error = ((gt_np[b] == c) & (pred_np[b] != c)).astype(np.uint8)
            if not error.any():
                continue
            # OpenCV 对非零像素计算到最近零像素（误差区边界）的欧氏距离。
            dist = cv2.distanceTransform(error, cv2.DIST_L2, 3)
            values = dist[error > 0]
            threshold = float(np.quantile(values, ratio))
            selected[b] |= (error > 0) & (dist <= threshold + 1e-6)
    return torch.from_numpy(selected).to(device=gt.device)


def _sample_flat_indices(mask: torch.Tensor, limit: int) -> torch.Tensor:
    """从二维布尔 mask 中取展平下标；超过上限时随机子采样。"""
    idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    if limit > 0 and idx.numel() > limit:
        perm = torch.randperm(idx.numel(), device=idx.device)[:limit]
        idx = idx[perm]
    return idx


def pixel_to_anchor_bane_loss(embedding: torch.Tensor, logits: torch.Tensor,
                              gt: torch.Tensor, cfg: Config,
                              ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """计算单尺度 PA+BANE 损失。

    :param embedding: (B,C,H,W) 训练专用投影特征；
    :param logits: (B,4,H,W) 地图分类 logits，仅用于无梯度地选择 BANE；
    :param gt: (B,H,W) 整型地图标签；
    :return: 标量损失与无梯度诊断（BANE 占全部误分类像素比例、有效 PA 项数）。
    """
    if embedding.ndim != 4 or embedding.shape[0] != gt.shape[0]:
        raise ValueError(f"embedding 应为 (B,C,H,W)，实际 {tuple(embedding.shape)}")
    if embedding.shape[-2:] != gt.shape[-2:]:
        raise ValueError("embedding 与 gt 的空间尺寸必须一致")
    B, _, H, W = embedding.shape
    z = F.normalize(embedding.float(), dim=1, eps=1e-6)
    hard = bane_negative_mask(logits, gt, cfg.map_ccl_bane_ratio)
    gt_long = gt.long()
    errors = logits.detach().argmax(dim=1) != gt_long
    info = {
        "map_bane_fraction": (
            hard.sum().float() / errors.sum().clamp_min(1).float()).detach(),
        "map_pa_terms": z.new_zeros(()),
    }
    losses = []
    tau = max(float(cfg.map_ccl_temperature), 1e-4)
    for b in range(B):
        zb = z[b].reshape(z.shape[1], H * W).transpose(0, 1)  # (HW,C)
        gb = gt_long[b].reshape(-1)
        hb = hard[b].reshape(-1)
        for c in range(logits.shape[1]):
            pos_mask = gb == c
            neg_mask = (gb != c) & hb
            # 当前 batch 没有误分类时退化为普通异类负样本，避免 PA 消失。
            if not neg_mask.any():
                neg_mask = gb != c
            if not pos_mask.any() or not neg_mask.any():
                continue
            pos_idx = _sample_flat_indices(pos_mask, cfg.map_ccl_pos_samples)
            neg_idx = _sample_flat_indices(neg_mask, cfg.map_ccl_neg_samples)
            pos = zb[pos_idx]
            neg = zb[neg_idx]
            # 用该类全部像素估计均值 Anchor，采样仅限制两两相似度矩阵规模。
            anchor = F.normalize(zb[pos_mask].mean(dim=0), dim=0, eps=1e-6)
            pos_logit = (pos * anchor).sum(dim=-1, keepdim=True) / tau
            neg_logits = pos @ neg.transpose(0, 1) / tau
            # 每行的第 0 列均为对应的正 Anchor logit。
            losses.append(F.cross_entropy(
                torch.cat([pos_logit, neg_logits], dim=1),
                torch.zeros(pos.shape[0], dtype=torch.long, device=z.device)))
    if losses:
        info["map_pa_terms"] = z.new_tensor(float(len(losses)))
        return torch.stack(losses).mean(), info
    # 单类别全图等极端情况下没有异类，保持计算图并返回零损失。
    return embedding.sum() * 0.0, info
