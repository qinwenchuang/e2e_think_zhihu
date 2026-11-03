"""TOAD：冻结 Planner/scorer 的测试时轨迹优化（arXiv:2606.07170）。

在 ProposalPlanner 已生成的 64 条轨迹之上，使用 Cross-Entropy Method（CEM）
在控制空间搜索新轨迹。视觉/时序上下文只计算一次，每轮仅重复 scorer 头
（V2 有 BEV 时还包括 proposal_bev/cost_mlp 的质量支路）。

与训练期 Top-k Loss 的区别：Top-k 用 GT 改善未来模型的 proposals；本模块在
推理时无 GT、不更新模型，以当前 proposals 为 trust-region 先验连续优化轨迹。

控制模型采用离散后轮轴 unicycle/bicycle 等价形式：
    v_k     = v_{k-1} + a_k Δt
    x_k     = x_{k-1} + v_k cos(yaw_{k-1}) Δt
    y_k     = y_{k-1} + v_k sin(yaw_{k-1}) Δt
    yaw_k   = yaw_{k-1} + ω_k Δt
该离散模型可逆；对非运动学可行的 Bézier 输入，inverse 将位移投影到上一时刻
heading，再转换成最接近的控制序列。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ..md_common import Config, FUTURE_INTERVAL


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    """可微角度回绕到 [-π, π)。"""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _expand_initial(v0: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """把 (B,) 的初始标量扩展成 controls/trajectory 的前导形状。"""
    out = v0.reshape(-1).to(device=ref.device, dtype=ref.dtype)
    target_ndim = ref.ndim - 2 if ref.ndim >= 3 else 1
    while out.ndim < target_ndim:
        out = out.unsqueeze(-1)
    return out


def bicycle_rollout(controls: torch.Tensor, v0: torch.Tensor,
                    dt: float = FUTURE_INTERVAL) -> torch.Tensor:
    """控制序列 (..., K, 2) → 轨迹 (..., K, 3)，控制顺序为 (accel, yaw_rate)。

    :param controls: 最后两维为 K×2；通常形状为 (B,M,K,2)。
    :param v0: (B,) 当前纵向速度，单位 m/s。
    :param dt: 控制步长，项目默认 0.5s。
    """
    if controls.shape[-1] != 2:
        raise ValueError(f"controls 末维必须为 2，实际 {tuple(controls.shape)}")
    v = _expand_initial(v0, controls)
    x, y, yaw = torch.zeros_like(v), torch.zeros_like(v), torch.zeros_like(v)
    poses = []
    for k in range(controls.shape[-2]):
        accel, omega = controls[..., k, 0], controls[..., k, 1]
        v = v + accel * dt
        x = x + v * torch.cos(yaw) * dt
        y = y + v * torch.sin(yaw) * dt
        yaw = yaw + omega * dt
        poses.append(torch.stack([x, y, yaw], dim=-1))
    return torch.stack(poses, dim=-2)


def bicycle_inverse(trajectory: torch.Tensor, v0: torch.Tensor,
                    dt: float = FUTURE_INTERVAL) -> torch.Tensor:
    """轨迹 (..., K, 3) → 控制序列 (..., K, 2)。

    对本模块 rollout 生成的轨迹严格可逆。对外部 Bézier/raw 轨迹，横向不可行
    位移会被投影掉，因此 inverse→rollout 是运动学可行投影，不保证逐点相等。
    """
    if trajectory.shape[-1] != 3:
        raise ValueError(f"trajectory 末维必须为 3，实际 {tuple(trajectory.shape)}")
    v_prev = _expand_initial(v0, trajectory)
    x_prev, y_prev = torch.zeros_like(v_prev), torch.zeros_like(v_prev)
    yaw_prev = torch.zeros_like(v_prev)
    controls = []
    for k in range(trajectory.shape[-2]):
        x, y, yaw = trajectory[..., k, 0], trajectory[..., k, 1], trajectory[..., k, 2]
        dx, dy = x - x_prev, y - y_prev
        # rollout 的位移沿 yaw_prev，故纵向投影可精确恢复本步末速度。
        v = (dx * torch.cos(yaw_prev) + dy * torch.sin(yaw_prev)) / dt
        accel = (v - v_prev) / dt
        omega = _wrap(yaw - yaw_prev) / dt
        controls.append(torch.stack([accel, omega], dim=-1))
        v_prev, x_prev, y_prev, yaw_prev = v, x, y, yaw
    return torch.stack(controls, dim=-2)


def _speed_from_controls(controls: torch.Tensor, v0: torch.Tensor,
                         dt: float) -> torch.Tensor:
    speed0 = v0.reshape(-1).to(device=controls.device, dtype=controls.dtype)
    while speed0.ndim < controls.ndim - 1:
        speed0 = speed0.unsqueeze(-1)
    return speed0 + controls[..., 0].cumsum(dim=-1) * dt


def trajectory_comfort_cost(controls: torch.Tensor, v0: torch.Tensor,
                            prev_accel: torch.Tensor, prev_yaw_rate: torch.Tensor,
                            cfg: Config, dt: float = FUTURE_INTERVAL) -> torch.Tensor:
    """闭式舒适约束，返回 (...,) 的无量纲超限平方代价。

    每项先按对应限值归一化，仅惩罚超过限值的部分；覆盖纵向/横向加速度、
    纵向 jerk、yaw rate、yaw acceleration，并包含上一执行控制到第一步规划的跳变。
    """
    accel, omega = controls[..., 0], controls[..., 1]
    speed = _speed_from_controls(controls, v0, dt)

    pa = prev_accel.reshape(-1).to(device=controls.device, dtype=controls.dtype)
    po = prev_yaw_rate.reshape(-1).to(device=controls.device, dtype=controls.dtype)
    while pa.ndim < accel.ndim - 1:
        pa, po = pa.unsqueeze(-1), po.unsqueeze(-1)
    accel_prev = torch.cat([
        pa.unsqueeze(-1).expand_as(accel[..., :1]), accel[..., :-1]], dim=-1)
    omega_prev = torch.cat([
        po.unsqueeze(-1).expand_as(omega[..., :1]), omega[..., :-1]], dim=-1)
    jerk = (accel - accel_prev) / dt
    yaw_accel = (omega - omega_prev) / dt

    def violation(value: torch.Tensor, limit: float) -> torch.Tensor:
        return torch.relu(value.abs() / max(limit, 1e-6) - 1.0).square()

    terms = (
        violation(accel, cfg.toad_max_accel),
        violation(speed * omega, cfg.toad_max_lat_accel),
        violation(jerk, cfg.toad_max_jerk),
        violation(omega, cfg.toad_max_yaw_rate),
        violation(yaw_accel, cfg.toad_max_yaw_accel),
    )
    # 各约束在时间维取均值后累加，避免 K 改变时总权值随步数变化。
    return torch.stack(terms, dim=-1).sum(dim=-1).mean(dim=-1)


def _anchor_cost(controls: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """相对 base 的逐步控制 L²；支持 (B,K,2) 或 (B,M,K,2)。"""
    ref = base[:, None] if controls.ndim == base.ndim + 1 else base
    return (controls - ref).square().sum(dim=-1).mean(dim=-1)


def _initial_previous_control(ego_status: torch.Tensor, ego_motion: torch.Tensor,
                              frame_dt: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """从缓存 ego 状态提取 v0、上一纵向加速度和上一 yaw rate。"""
    # ego_dynamic_state 的 vel2 为 ego 系 (vx,vy)；保留 vx 符号以支持倒车，
    # 不能用二维速度模长（会把负纵向速度错误折成正向）。
    v0 = ego_status[:, -1, 3].float()
    prev_accel = ego_status[:, -1, 5].float()
    dt_prev = frame_dt[:, -1] - frame_dt[:, -2]
    dt_prev = torch.where(dt_prev.abs() > 1e-3, dt_prev,
                          torch.full_like(dt_prev, FUTURE_INTERVAL))
    dyaw = _wrap(ego_motion[:, -1, 2] - ego_motion[:, -2, 2])
    prev_yaw_rate = dyaw / dt_prev
    return v0, prev_accel.float(), prev_yaw_rate.float()


@torch.no_grad()
def toad_search(out: Dict[str, torch.Tensor], planner, ego_status: torch.Tensor,
                ego_motion: torch.Tensor, frame_dt: torch.Tensor,
                cfg: Config, bev_output: Optional[Dict[str, torch.Tensor]] = None,
                occ_decoder=None) -> Dict[str, torch.Tensor]:
    """执行 TOAD CEM 搜索并原地补充诊断字段。

    输入 out 必须包含 ProposalPlanner 的 proposal_list、proposal_scores、
    planner_context、ego_token。最终返回规则与论文一致：比较 CEM 均值轨迹与
    scorer 原始 argmax 轨迹，最终比较去掉 anchor 项、保留 scorer+comfort。

    注意：回退只保证当前学习目标的分数不退化，不保证真实碰撞/PDMS 不退化。
    """
    props = out["proposal_list"][-1].float()           # (B,N,K,3)
    scores = out["proposal_scores"].float()            # (B,N)
    ctx, ego_token = out["planner_context"], out["ego_token"]

    def _score(trajectories: torch.Tensor) -> torch.Tensor:
        # V2 的 proposal score 含 BEV/occ 质量支路；TOAD 也走同一 scorer 口径，
        # 否则 toad_score 与 proposal_scores 不同尺度，geo top-m 会系统性排错。
        if bev_output is None and occ_decoder is None:
            return planner.score_trajectories(trajectories, ctx, ego_token)
        return planner.score_trajectories(
            trajectories, ctx, ego_token,
            bev_output=bev_output, occ_decoder=occ_decoder)

    B, N, K = props.shape[:3]
    device = props.device
    v0, prev_accel, prev_yaw_rate = _initial_previous_control(
        ego_status, ego_motion, frame_dt)

    controls_all = bicycle_inverse(props, v0)           # (B,N,K,2)
    base_idx = scores.argmax(dim=1)                     # (B,)
    batch_idx = torch.arange(B, device=device)
    base_traj = props[batch_idx, base_idx]
    base_controls = controls_all[batch_idx, base_idx]
    base_score = scores[batch_idx, base_idx]

    # proposal 自身离散程度决定 trust-region 初始范围。
    sigma0 = cfg.toad_std_scale * controls_all.std(dim=1, unbiased=False)
    floor = torch.tensor([cfg.toad_std_floor_accel, cfg.toad_std_floor_yaw_rate],
                         device=device, dtype=props.dtype)
    sigma0 = torch.maximum(sigma0, floor)
    mean, sigma = base_controls.clone(), sigma0

    iterations = max(0, int(cfg.toad_iterations))
    num_samples = max(1, int(cfg.toad_samples))
    num_elites = min(max(1, int(cfg.toad_elites)), num_samples)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg.toad_seed))

    for _ in range(iterations):
        noise = torch.randn((B, num_samples, K, 2), generator=generator,
                            device=device, dtype=props.dtype)
        candidates = mean[:, None] + sigma[:, None] * noise
        candidates[:, 0] = mean                         # 每轮显式评价当前均值
        trajectories = bicycle_rollout(candidates, v0)
        cand_score = _score(trajectories).float()
        cand_anchor = _anchor_cost(candidates, base_controls)
        cand_comfort = trajectory_comfort_cost(
            candidates, v0, prev_accel, prev_yaw_rate, cfg)
        objective = (cand_score - cfg.toad_anchor_weight * cand_anchor
                     - cfg.toad_comfort_weight * cand_comfort)
        elite_idx = objective.topk(num_elites, dim=1).indices
        elites = candidates.gather(
            1, elite_idx[:, :, None, None].expand(-1, -1, K, 2))
        mean = elites.mean(dim=1)
        sigma = elites.std(dim=1, unbiased=False).clamp_min(1e-4)

    mean_traj = bicycle_rollout(mean, v0)
    mean_score = _score(mean_traj[:, None])[:, 0].float()
    mean_comfort = trajectory_comfort_cost(
        mean, v0, prev_accel, prev_yaw_rate, cfg)
    base_comfort = trajectory_comfort_cost(
        base_controls, v0, prev_accel, prev_yaw_rate, cfg)
    mean_anchor = _anchor_cost(mean, base_controls)

    # 论文 Eq.4：最终比较不再惩罚 mean 离开 anchor，只保留 scorer+comfort。
    base_objective = base_score - cfg.toad_comfort_weight * base_comfort
    mean_objective = mean_score - cfg.toad_comfort_weight * mean_comfort
    accept = mean_objective > base_objective
    best_traj = torch.where(accept[:, None, None], mean_traj, base_traj)
    best_score = torch.where(accept, mean_score, base_score)
    best_objective = torch.where(accept, mean_objective, base_objective)
    best_comfort = torch.where(accept, mean_comfort, base_comfort)
    best_anchor = torch.where(accept, mean_anchor, torch.zeros_like(mean_anchor))

    out["trajectory"] = best_traj
    out["toad_base_trajectory"] = base_traj
    out["toad_mean_trajectory"] = mean_traj
    out["toad_trajectory"] = best_traj
    out["toad_score"] = best_score                       # 供 geo 做联合候选评价
    out["toad_accept"] = accept.float()
    out["toad_objective_gain"] = best_objective - base_objective
    out["toad_model_score_gain"] = best_score - base_score
    out["toad_base_objective"] = base_objective
    out["toad_mean_objective"] = mean_objective
    out["toad_comfort_cost"] = best_comfort
    out["toad_anchor_cost"] = best_anchor
    out["toad_iterations"] = torch.full(
        (B,), float(iterations), device=device, dtype=props.dtype)
    return out
