"""my_drive 损失与指标。

损失：
  * traj：逐层 winner-takes-all L1（只对每层离 GT 最近的 proposal 回传）
  * score：质量分数回归「负 ADE」（让分数排序与轨迹误差一致）
  * bev_map：4 类语义分割交叉熵（背景类降权）
  * det_heatmap：CenterNet penalty-reduced focal loss
  * det_reg：正样本位置上的 L1（offset/log-size/sin-cos yaw）

指标（随 loss 一并返回，进 TensorBoard）：
  * traj_ade / traj_fde：轨迹 xy 平均/末点位移误差（米）
  * map_acc / map_miou：地图分割像素准确率 / 前景 mIoU
  * det_center_err：GT 正样本位置上预测中心与真值中心的平均距离（米）
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F
from ..md_common import Config


def focal_loss(pred_logits: torch.Tensor, target: torch.Tensor,
               alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss。

    必须 fp32 计算：bf16 里 sigmoid 饱和后 clamp(1-1e-4) 会舍入成 1.0，
    log(0) → NaN（训练日志中实测出现过）。
    """
    pred = torch.sigmoid(pred_logits.float()).clamp(1e-4, 1 - 1e-4)
    target = target.float()
    pos = target.eq(1.0)
    neg = ~pos
    neg_weights = (1 - target) ** beta
    pos_loss = -torch.log(pred) * (1 - pred) ** alpha * pos
    neg_loss = -torch.log(1 - pred) * pred ** alpha * neg_weights * neg
    num_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """在 mask>0 的位置计算 L1。pred/target (B,C,H,W)，mask (B,H,W)。"""
    mask = mask[:, None].expand_as(pred)
    num = mask.sum().clamp(min=1.0)
    return (torch.abs(pred - target) * mask).sum() / num


def compute_losses(pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                   cfg: Config, with_planner: bool, with_bev: bool,
                   ) -> Dict[str, torch.Tensor]:
    """汇总损失：with_planner/with_bev 与 set_train_stage 的前向开关对应。"""
    losses: Dict[str, torch.Tensor] = {}
    total = batch["traj"].new_zeros(())
    if with_planner:
        gt = batch["traj"]
        # ---- 逐层 WTA L1：每层只监督「离 GT 最近的那条 proposal」（min over N），
        # 保住 proposal 多样性（对全部 proposal 回归 GT 容易塌缩成一条）；
        # 首尾层权重 1.0，中间精修层 0.3（深监督）。
        n_layers = len(pred["proposal_list"])
        for i, props in enumerate(pred["proposal_list"]):
            # heading 拆项 + 角度回绕：atan2(sin,cos) 把误差 wrap 到 [-π,π]
            # （直接 L1 会把 3.13 vs -3.13 当成 6.26 的错误梯度方向）；
            # heading 权重 0.5：否则被 xy（米）淹没
            diff = props - gt[:, None]
            diff_h = torch.atan2(torch.sin(diff[..., 2]), torch.cos(diff[..., 2])).abs()
            l1_per = (diff[..., :2].abs().mean(-1) + 0.5 * diff_h).mean(-1)
            w = 1.0 if (i == 0 or i == n_layers - 1) else 0.3
            losses[f"traj_l1_L{i}"] = l1_per.min(dim=1).values.mean()
            total = total + cfg.w_traj * w * losses[f"traj_l1_L{i}"]
        losses["traj_l1"] = losses[f"traj_l1_L{n_layers - 1}"]
        # 质量分数监督：回归「负 ADE/10」——让分数排序与轨迹误差一致
        # （硬 CE 用 best-proposal 独热标签在 WTA 噪声下学不动；归一化后回归更稳）
        final_props = pred["proposal_list"][-1]
        err_per = torch.linalg.norm(
            final_props[..., :2] - gt[:, None, ..., :2], dim=-1).mean(-1)
        score_target = -err_per.detach() / 10.0
        losses["score_reg"] = F.smooth_l1_loss(pred["proposal_scores"], score_target)
        total = total + cfg.w_score_ce * losses["score_reg"]
        with torch.no_grad():
            losses["traj_ade_oracle"] = err_per.min(dim=1).values.mean()
            losses["score_hit"] = (pred["proposal_scores"].argmax(dim=1)
                                   == err_per.argmin(dim=1)).float().mean()
            # 选中轨迹（scorer argmax）的指标
            losses["traj_l1_sel"] = F.l1_loss(pred["trajectory"], gt)
            err = torch.linalg.norm(pred["trajectory"][..., :2] - gt[..., :2], dim=-1)
            losses["traj_ade"] = err.mean()
            losses["traj_fde"] = err[:, -1].mean()
    if with_bev:
        gt_map = batch["bev_map"].long()
        logits = pred["map_logits"]
        # 背景类降权（背景占 45~70% 像素，否则前景类学不动）
        ce_w = torch.tensor([cfg.map_bg_weight, 1.0, 1.0, 1.0],
                            device=logits.device, dtype=torch.float32)
        losses["bev_map_ce"] = F.cross_entropy(logits.float(), gt_map, weight=ce_w)
        total = total + cfg.w_bev_map * losses["bev_map_ce"]
        hm_loss = focal_loss(pred["det_heatmap"], batch["det_heatmap"])
        reg_loss = masked_l1(pred["det_reg"], batch["det_reg"],
                             batch["det_reg_mask"].float())
        losses["det_heatmap"] = hm_loss
        losses["det_reg"] = reg_loss
        total = total + cfg.w_det_heatmap * hm_loss + cfg.w_det_reg * reg_loss
        with torch.no_grad():
            pred_cls = logits.argmax(dim=1)
            losses["map_acc"] = (pred_cls == gt_map).float().mean()
            # 前景 mIoU（只统计 GT 或预测中出现的类）
            ious = []
            for c in range(1, 4):
                p_c = pred_cls == c
                g_c = gt_map == c
                union = (p_c | g_c).sum()
                if union > 0:
                    ious.append(((p_c & g_c).sum().float() / union.float()))
            losses["map_miou"] = (torch.stack(ious).mean() if ious
                                  else torch.zeros((), device=logits.device))
            # 检测中心误差：GT 正样本 cell 上预测 offset 与真值 offset 的欧氏距离（米）
            mask = batch["det_reg_mask"] > 0
            if mask.any():
                reg = pred["det_reg"]
                gt_reg = batch["det_reg"]
                off_err = torch.linalg.norm(reg[:, :2] - gt_reg[:, :2], dim=1)
                m = mask.float()
                losses["det_center_err"] = ((off_err * m).sum()
                                            / m.sum().clamp(min=1.0)
                                            * cfg.det_pixel_size)
            else:
                losses["det_center_err"] = torch.zeros((), device=logits.device)
    losses["loss"] = total
    return losses
