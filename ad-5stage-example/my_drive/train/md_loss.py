"""my_drive 损失与指标。

损失：
  * traj：逐层 top-k 加权 L1（k=1 退化为旧 WTA；bezier 模式航向项为切线余弦距离）
  * score：质量分数回归「负 ADE」（让分数排序与轨迹误差一致）
  * bev_map：4 类语义分割交叉熵（背景类降权）
  * det_heatmap：CenterNet penalty-reduced focal loss
  * det_reg：正样本位置上的 L1（offset/log-size/sin-cos yaw）

指标（随 loss 一并返回，进 TensorBoard）：
  * traj_ade / traj_fde：选中轨迹 xy 平均/末点位移误差（米；geo 开启时是 geo 口径）
  * traj_ade_argmax / traj_fde_argmax：纯 argmax(scorer) 口径（与 geo 口径对照）
  * geo_flip_rate / geo_win_rate / geo_ade_delta / geo_oob_rate：几何重打分诊断
  * toad_ade / toad_fde / toad_win_rate / toad_accept_rate / toad_comfort_cost：
    TOAD 独立轨迹指标（仅 eval 开启 TOAD 时出现）
  * map_acc / map_miou：地图分割像素准确率 / 前景 mIoU
  * det_center_err：GT 正样本位置上预测中心与真值中心的平均距离（米）
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F
from ..md_common import Config
from .md_map_ccl import pixel_to_anchor_bane_loss


def _boundary_band(gt: torch.Tensor, width: int) -> torch.Tensor:
    """GT (B,H,W) → 类别边界带 (B,H,W)，带宽由局部窗口半径控制。"""
    onehot = F.one_hot(gt.long(), num_classes=4).permute(0, 3, 1, 2).float()
    kernel = max(1, int(width)) * 2 + 1
    local_max = F.max_pool2d(onehot, kernel, stride=1, padding=kernel // 2)
    local_min = -F.max_pool2d(-onehot, kernel, stride=1, padding=kernel // 2)
    return (local_max > local_min).any(dim=1)


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
    """在 mask>0 的位置计算 L1。pred/target (B,C,H,W)，mask (B,H,W)。

    autocast 下 pred 可能是 bf16；L1 求和前统一转 fp32，避免大 mask 上精度损失。
    """
    mask = mask[:, None].expand_as(pred).float()
    num = mask.sum().clamp(min=1.0)
    return (torch.abs(pred.float() - target.float()) * mask).sum() / num


def compute_losses(pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                   cfg: Config, with_planner: bool, with_bev: bool,
                   score_only: bool = False,
                   ) -> Dict[str, torch.Tensor]:
    """汇总损失：with_planner/with_bev 与 set_train_stage 的前向开关对应。

    score_only=True 时只把 score loss 计入 total（指标照常计算）；B 实验冻结
    候选/BEV 后，traj/map/det/occ 反传没有可训练路径，不应再浪费时间。
    """
    losses: Dict[str, torch.Tensor] = {}
    total = batch["traj"].new_zeros(())
    if with_planner:
        gt = batch["traj"]
        # ---- 逐层 top-k 加权 L1（借鉴 TopoCurve CARS 的一对多监督思想）：
        # 旧 WTA 每层 64 条 proposal 只回传离 GT 最近的 1 条——监督极稀疏，
        # scorer 的排序信号也缺学习来源。改为每层对最近的 k 条按 decay^(j-1)
        # 加权回传（归一化），既保多样性又加密监督；k=1 严格退化为旧行为。
        # 首尾层权重 1.0，中间精修层 0.3（深监督）。
        # 航向项：raw 模式沿用 atan2 角度回绕差；bezier 模式用切线余弦距离
        # 1 − cos(t̂, u_gt)（|切线| clamp(min=1m)：低速/停车时梯度有界，
        # 规避 atan2 在近零切线下的奇点；见 md_model._ctrl_to_waypoints）。
        n_layers = len(pred["proposal_list"])
        tangent_list = pred.get("tangent_list") or [None] * n_layers
        k_sup = max(1, cfg.w_traj_topk)
        gt_dir = torch.stack([torch.cos(gt[..., 2]), torch.sin(gt[..., 2])],
                             dim=-1)                       # (B,8,2)
        for i, props in enumerate(pred["proposal_list"]):
            k = min(k_sup, props.shape[1])
            diff_xy = (props[..., :2] - gt[:, None, :, :2]).abs().mean(-1)  # (B,N,8)
            tan = tangent_list[i]
            if tan is not None:
                # 切线余弦航向项：t̂ = tan/max(|tan|,1m)，h = 1 − t̂·u_gt ∈ [0,2]
                tan_n = tan / tan.norm(dim=-1, keepdim=True).clamp(min=1.0)
                head_term = (1.0 - (tan_n * gt_dir[:, None]).sum(-1))       # (B,N,8)
            else:
                diff_h = props[..., 2] - gt[:, None, :, 2]
                head_term = torch.atan2(torch.sin(diff_h),
                                        torch.cos(diff_h)).abs()            # (B,N,8)
            l1_per = (diff_xy + 0.5 * head_term).mean(-1)              # (B,N)
            if k == 1:
                layer_loss = l1_per.min(dim=1).values.mean()
            else:
                topv = l1_per.topk(k, dim=1, largest=False).values     # (B,k)
                wk = cfg.topk_weight_decay ** torch.arange(
                    k, device=l1_per.device, dtype=l1_per.dtype)
                wk = wk / wk.sum()
                layer_loss = (topv * wk).sum(-1).mean()
            w = 1.0 if (i == 0 or i == n_layers - 1) else 0.3
            losses[f"traj_l1_L{i}"] = layer_loss
            if not score_only:
                total = total + cfg.w_traj * w * layer_loss
        losses["traj_l1"] = losses[f"traj_l1_L{n_layers - 1}"]
        # 质量分数监督：回归「负 ADE/10」——让分数排序与轨迹误差一致
        # （硬 CE 用 best-proposal 独热标签在 WTA 噪声下学不动；归一化后回归更稳）
        final_props = pred["proposal_list"][-1]
        err_per = torch.linalg.norm(
            final_props[..., :2] - gt[:, None, ..., :2], dim=-1).mean(-1)
        score_target = -err_per.detach() / 10.0
        losses["score_reg"] = F.smooth_l1_loss(pred["proposal_scores"], score_target)
        # score_only 也要保留点值校准；它只屏蔽 traj/map/det/occ。
        total = total + cfg.w_score_ce * losses["score_reg"]
        if cfg.score_listwise:
            # listwise：把 softmax(-ADE/T) 当软标签，直接优化 top-1 排序；
            # 与点值回归叠加，既保留分数校准，又给相对排序更直接的梯度。
            temp = max(float(cfg.score_listwise_temp), 1e-3)
            soft = torch.softmax(-err_per.detach().float() / temp, dim=1)
            logp = torch.log_softmax(pred["proposal_scores"].float(), dim=1)
            losses["score_listwise"] = -(soft * logp).sum(1).mean()
            total = total + cfg.w_score_listwise * losses["score_listwise"]
        with torch.no_grad():
            losses["traj_ade_oracle"] = err_per.min(dim=1).values.mean()
            losses["score_hit"] = (pred["proposal_scores"].argmax(dim=1)
                                   == err_per.argmin(dim=1)).float().mean()
            # 选中轨迹：默认是 argmax(scorer)；geo 开启时被几何重打分覆盖
            losses["traj_l1_sel"] = F.l1_loss(pred["trajectory"], gt)
            err = torch.linalg.norm(pred["trajectory"][..., :2] - gt[..., :2], dim=-1)
            losses["traj_ade"] = err.mean()
            losses["traj_fde"] = err[:, -1].mean()
            # 双口径：始终保留纯 argmax(scorer) 的指标，便于与旧 ckpt/训练口径对齐
            base = pred["proposal_scores"].argmax(dim=1)
            base_traj = final_props[torch.arange(final_props.shape[0],
                                                 device=final_props.device), base]
            err_base = torch.linalg.norm(base_traj[..., :2] - gt[..., :2], dim=-1)
            losses["traj_ade_argmax"] = err_base.mean()
            losses["traj_fde_argmax"] = err_base[:, -1].mean()
            losses["traj_l1_argmax"] = F.l1_loss(base_traj, gt)
            # 选轨 regret / oracle 排名：只做诊断，不参与反传。
            base_per = err_base.mean(dim=-1)                         # (B,)
            oracle_per = err_per.min(dim=1).values                     # (B,)
            oracle_idx = err_per.argmin(dim=1)                         # (B,)
            losses["traj_regret"] = (base_per - oracle_per).mean()
            rank = (pred["proposal_scores"] >
                    pred["proposal_scores"].gather(1, oracle_idx[:, None])).sum(1)
            losses["oracle_rank_mean"] = rank.float().mean()
            top8 = pred["proposal_scores"].topk(
                min(8, pred["proposal_scores"].shape[1]), dim=1).indices
            losses["oracle_at_8"] = (
                top8 == oracle_idx[:, None]).any(1).float().mean()
            if "geo_sel" in pred:
                losses["geo_flip_rate"] = (pred["geo_sel"] != base).float().mean()
                losses["geo_oob_rate"] = (
                    pred["geo_oob"].mean() if "geo_oob" in pred
                    else torch.zeros((), device=final_props.device))
                err_geo = torch.linalg.norm(
                    pred["trajectory"][..., :2] - gt[..., :2], dim=-1)
                losses["geo_win_rate"] = (
                    err_geo.mean(1) < err_base.mean(1)).float().mean()
                losses["geo_ade_delta"] = err_geo.mean() - err_base.mean()
            # TOAD 独立轨迹的指标；训练态不会产生这些 key，仅 eval/推理出现。
            if "toad_trajectory" in pred:
                toad = pred["toad_trajectory"]
                err_toad = torch.linalg.norm(
                    toad[..., :2] - gt[..., :2], dim=-1)
                losses["toad_ade"] = err_toad.mean()
                losses["toad_fde"] = err_toad[:, -1].mean()
                losses["toad_ade_delta"] = err_toad.mean() - err_base.mean()
                losses["toad_win_rate"] = (
                    err_toad.mean(1) < err_base.mean(1)).float().mean()
                losses["toad_accept_rate"] = pred["toad_accept"].float().mean()
                losses["toad_objective_gain"] = pred["toad_objective_gain"].mean()
                losses["toad_model_score_gain"] = pred["toad_model_score_gain"].mean()
                losses["toad_comfort_cost"] = pred["toad_comfort_cost"].mean()
                losses["toad_anchor_cost"] = pred["toad_anchor_cost"].mean()
                if "toad_mean_trajectory" in pred:
                    err_mean = torch.linalg.norm(
                        pred["toad_mean_trajectory"][..., :2] - gt[..., :2], dim=-1)
                    losses["toad_mean_ade"] = err_mean.mean()
                if "geo_toad_selected" in pred:
                    losses["geo_toad_select_rate"] = pred["geo_toad_selected"].mean()
    if with_bev:
        gt_map = batch["bev_map"].long()
        logits = pred["map_logits"]
        # 背景类降权（背景占 45~70% 像素，否则前景类学不动）
        ce_w = torch.tensor([cfg.map_bg_weight, 1.0, 1.0, 1.0],
                            device=logits.device, dtype=torch.float32)
        losses["bev_map_ce"] = F.cross_entropy(logits.float(), gt_map, weight=ce_w)
        if not score_only:
            total = total + cfg.w_bev_map * losses["bev_map_ce"]
        # 仅 model.train() 且显式开启时存在该 embedding；val/test 无额外前向或损失。
        if cfg.map_ccl_enabled and "map_ccl_embedding" in pred:
            ccl_loss, ccl_info = pixel_to_anchor_bane_loss(
                pred["map_ccl_embedding"], logits, gt_map, cfg)
            losses["map_pa_bane"] = ccl_loss
            losses.update(ccl_info)
            if not score_only:
                total = total + cfg.w_map_ccl * ccl_loss
        hm_loss = focal_loss(pred["det_heatmap"], batch["det_heatmap"])
        reg_loss = masked_l1(pred["det_reg"], batch["det_reg"],
                             batch["det_reg_mask"].float())
        losses["det_heatmap"] = hm_loss
        losses["det_reg"] = reg_loss
        if not score_only:
            total = total + cfg.w_det_heatmap * hm_loss + cfg.w_det_reg * reg_loss
        with torch.no_grad():
            pred_cls = logits.argmax(dim=1)
            losses["map_acc"] = (pred_cls == gt_map).float().mean()
            # 逐类 IoU + 前景 mIoU（均只统计 GT 或预测中出现的类）。
            ious = []
            class_names = ("bg", "lane", "roadblock", "walkway")
            for c, name in enumerate(class_names):
                p_c = pred_cls == c
                g_c = gt_map == c
                union = (p_c | g_c).sum()
                if union > 0:
                    iou = (p_c & g_c).sum().float() / union.float()
                    losses[f"map_iou_{name}"] = iou
                    if c > 0:
                        ious.append(iou)
            losses["map_miou"] = (torch.stack(ious).mean() if ious
                                  else torch.zeros((), device=logits.device))
            # GT 类别边界带上的四类 mIoU，专门监控 PA+BANE 的目标区域。
            band = _boundary_band(gt_map, cfg.map_boundary_width)
            boundary_ious = []
            for c in range(4):
                p_c = (pred_cls == c) & band
                g_c = (gt_map == c) & band
                union = (p_c | g_c).sum()
                if union > 0:
                    boundary_ious.append((p_c & g_c).sum().float() / union.float())
            losses["map_boundary_miou"] = (
                torch.stack(boundary_ious).mean() if boundary_ious
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
        # 当前帧连续占据 BCE（仅有有效射线的样本参与）
        if "occ_logits" in pred and "occ_valid" in batch:
            valid = batch["occ_valid"].bool()
            if valid.any():
                logits_occ = pred["occ_logits"].float()
                target_occ = batch["occ_labels"].float()
                occ_bce = F.binary_cross_entropy_with_logits(
                    logits_occ[valid], target_occ[valid])
                losses["occ_bce"] = occ_bce
                if not score_only:
                    total = total + cfg.w_occ * occ_bce
                with torch.no_grad():
                    occ_prob = torch.sigmoid(logits_occ)
                    occ_pred = occ_prob >= 0.5
                    occ_true = target_occ >= 0.5
                    losses["occ_acc"] = (
                        occ_pred[valid] == occ_true[valid]).float().mean()
                    pos = valid & occ_true
                    neg = valid & ~occ_true
                    losses["occ_pos_recall"] = (
                        (occ_pred[pos] == occ_true[pos]).float().mean()
                        if pos.any() else occ_bce.new_zeros(()))
                    losses["occ_neg_recall"] = (
                        (occ_pred[neg] == occ_true[neg]).float().mean()
                        if neg.any() else occ_bce.new_zeros(()))
    losses["loss"] = total
    return losses
