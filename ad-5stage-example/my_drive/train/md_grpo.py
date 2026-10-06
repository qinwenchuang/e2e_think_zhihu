"""md_grpo：stage 1.5 —— GRPO 强化微调（ProposalPlanner 的选择策略）。

结构映射：
  组（group） = 每帧从策略采样 k 条 proposal（π = softmax(quality_scores/τ)，64 类）
  奖励         = 规则奖励（碰撞/车道合规/进度/舒适 + 小权重 GT 锚定）
  优势         = 组内归一化 Â = (r − mean) / (std + eps)
  参考策略     = SFT（stage 1）scorer 的冻结副本，KL 正则防漂移
  可训         = 仅 scorer（pos_embed + scorer blocks + quality_head）；
                 生成器与编码器全部冻结

碰撞 reward 用「未来帧障碍物真值」：轨迹第 k 步（t+0.5k）对应 pickle 帧的
GT 框（缓存时已变换到当前 ego 系）——真·动态日志回放，无静止/外推近似。

用法：
    python -m my_drive.train.md_grpo --init-from runs/stage1/best_val.pt \
        --cache-root workspace/cache --out-dir workspace/runs/grpo --epochs 8
"""
from __future__ import annotations
import argparse
import copy
import math
import time
from pathlib import Path
from typing import Dict, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ..md_common import WORK_ROOT, Config
from ..data.md_data import OverfitDataset
from .md_model import MyDriveModel

#: 可行驶区类别（地图标签：1=lane、2=roadblock）
DRIVABLE_CLASSES = (1, 2)


def _footprint_corners(x, y, yaw, l, w) -> np.ndarray:
    """(x,y,yaw,l,w) → 2D 有向框 (4,2) 角点。"""
    c, s = np.cos(yaw), np.sin(yaw)
    loc = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    return loc @ np.array([[c, -s], [s, c]]).T + np.array([x, y])


def _poly_to_grid(corners: np.ndarray, cfg: Config) -> np.ndarray:
    """(N,2) ego 坐标多边形 → (100,100)@0.5m 的 0/1 栅格掩码（行=x、列=y）。"""
    ps = cfg.bev_pixel_size / cfg.bev_map_stride
    col = (corners[:, 1] - cfg.pc_y_min) / ps
    row = (corners[:, 0] - cfg.pc_x_min) / ps
    pts = np.stack([col, row], -1).reshape(-1, 1, 2).astype(np.int32)
    m = np.zeros(cfg.bev_map_size, dtype=np.uint8)
    cv2.fillConvexPoly(m, pts, 1)                            # cv2 自动裁剪出界部分
    return m


def _boxes_occupancy(boxes: np.ndarray, valid: np.ndarray, cfg: Config) -> np.ndarray:
    """(M,7)+(M,) → (100,100) 障碍物占用图。"""
    occ = np.zeros(cfg.bev_map_size, dtype=np.uint8)
    for box, v in zip(boxes, valid):
        if not v:
            continue
        x, y, yaw = float(box[0]), float(box[1]), float(box[6])
        l, w = float(box[3]), float(box[4])
        occ |= _poly_to_grid(_footprint_corners(x, y, yaw, l, w), cfg)
    return occ


def compute_rewards(sampled: np.ndarray, future_boxes: np.ndarray,
                    future_valid: np.ndarray, bev_map: np.ndarray,
                    traj_gt: np.ndarray, cfg: Config,
                    ego_len: float, ego_wid: float, margin: float) -> np.ndarray:
    """逐条采样轨迹的规则奖励（numpy，全部在 ego 系）。

    :param sampled: (B,k,8,3) 采样轨迹；future_boxes (B,8,M,7) 各时间步障碍物真值；
        bev_map (B,100,100) 地图标签；traj_gt (B,8,3)
    :return: (B,k) 奖励矩阵
    """
    B, k = sampled.shape[:2]
    rewards = np.zeros((B, k), dtype=np.float32)
    for b in range(B):
        # 每时间步的障碍物占用图（k 条轨迹共享）
        occs = [_boxes_occupancy(future_boxes[b, s], future_valid[b, s], cfg)
                for s in range(8)]
        for i in range(k):
            traj = sampled[b, i]                             # (8,3)
            # ---- 碰撞：自车 footprint（+margin）逐时间步 vs 该时刻障碍物 ----
            # 一票否决（0/1），且作为乘性门控把进度/舒适全部归零
            #（软惩罚会被进度奖励劫持：scorer 学会「冲得快」对冲碰撞扣分）
            n_col = 0
            for s in range(8):
                x, y, yaw = traj[s]
                ego_mask = _poly_to_grid(
                    _footprint_corners(x, y, yaw, ego_len + 2 * margin,
                                       ego_wid + 2 * margin), cfg)
                if (ego_mask & occs[s]).any():
                    n_col += 1
            r_col = 1.0 if n_col == 0 else 0.0
            # ---- 车道合规：8 步轨迹点落在可行驶区内的比例 ----
            ps = cfg.bev_pixel_size / cfg.bev_map_stride
            rows = ((traj[:, 0] - cfg.pc_x_min) / ps).astype(np.int64)
            cols = ((traj[:, 1] - cfg.pc_y_min) / ps).astype(np.int64)
            in_grid = (rows >= 0) & (rows < 100) & (cols >= 0) & (cols < 100)
            drivable = np.zeros(8, dtype=bool)
            for s in range(8):
                if in_grid[s]:
                    drivable[s] = bev_map[b, rows[s], cols[s]] in DRIVABLE_CLASSES
            r_lane = drivable.mean()
            # ---- 进度：末端前向位移（限幅 20m；倒退得 0） ----
            r_prog = float(np.clip(traj[-1, 0] / 20.0, 0.0, 1.0))
            # ---- 舒适：yaw 二阶差（jerk 代理） ----
            ddy = np.abs(np.diff(traj[:, 2], n=2))
            r_comf = float(np.exp(-ddy.mean()))              # ∈ (0,1]
            # ---- GT 锚定（小权重防 reward hacking） ----
            ade = float(np.linalg.norm(traj[:, :2] - traj_gt[b, :, :2], axis=-1).mean())
            r_gt = float(np.exp(-ade / 2.0))
            rewards[b, i] = (r_col * (cfg.w_r_lane * r_lane + cfg.w_r_prog * r_prog
                                     + cfg.w_r_comf * r_comf)
                             + cfg.w_r_gt * r_gt)           # 碰撞门控 + GT 锚定
    return rewards


def train_grpo(cfg: Config, init_from: Path, cache_root: Path, out_dir: Path,
               epochs: int = 8, limit_batches: int = 0, tb: bool = True) -> Path:
    """GRPO 主循环：采样 proposal → 规则奖励 → 组内归一化优势 → KL 正则策略更新。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tb")) if tb else None
    train_ds = OverfitDataset(cache_root, cfg, split="train", augment=True)
    val_ds = OverfitDataset(cache_root, cfg, split="val", augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True,
                              persistent_workers=cfg.num_workers > 0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    model = MyDriveModel(cfg).to(device)
    ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
    # 形状匹配的键才载入（同 md_train 的初始化策略）
    own = model.state_dict()
    compatible = {kk: vv for kk, vv in ckpt["model"].items()
                  if kk in own and own[kk].shape == vv.shape}
    model.load_state_dict(compatible, strict=False)
    print(f"载入 SFT {init_from}（stage {ckpt.get('stage')} epoch {ckpt.get('epoch')}）"
          f" 匹配 {len(compatible)} 键")
    # 只训 scorer（pos_embed / scorer / quality_head），其余全部冻结
    scorer_prefixes = ("planner.pos_embed.", "planner.scorer.", "planner.quality_head.")
    for n, p in model.named_parameters():
        p.requires_grad = any(n.startswith(pre) for pre in scorer_prefixes)
    # 参考策略：SFT scorer 的冻结深拷贝（KL 锚定目标）
    ref_planner = copy.deepcopy(model.planner)
    for p in ref_planner.parameters():
        p.requires_grad = False
    model.eval()
    ref_planner.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params) / 1e6
    optimizer = torch.optim.AdamW(params, lr=cfg.grpo_lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    from .md_train import WarmupCosine
    scheduler = WarmupCosine(optimizer, steps_per_epoch * epochs,
                             max(1, int(steps_per_epoch * epochs * 0.1)), cfg.grpo_lr)
    print(f"[GRPO] 可训 {n_train:.2f}M（仅 scorer）| τ={cfg.grpo_temperature} "
          f"k={cfg.grpo_samples} β={cfg.grpo_beta} | {steps_per_epoch} opt-steps/epoch",
          flush=True)
    def _forward(batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp):
            pred = model(batch, with_planner=True, with_bev=False)
        scores = pred["proposal_scores"].float()
        props = pred["proposal_list"][-1].float()
        return scores, props
    def _ref_forward(batch):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=cfg.amp):
            scene_cur, _ = model._encode(batch["img_cur"], cams=[0, 1, 2, 3],
                                         need_patch=False)
            scene_hist, _ = model._encode(batch["img_hist"], cams=[0, 0, 0],
                                          need_patch=False)
            r_all = scene_cur.reshape(scene_cur.shape[0], -1, scene_cur.shape[-1])
            queue = scene_hist.reshape(scene_hist.shape[0], -1, scene_hist.shape[-1])
            ref = ref_planner(r_all, queue, batch["ego_status"],
                              batch["ego_motion"], batch["frame_dt"])
        return ref["proposal_scores"].float()
    best_val_reward = -float("inf")
    for epoch in range(epochs):
        t0 = time.time()
        loss_sum, rew_sum, count = 0.0, 0.0, 0
        n_batches = min(limit_batches, len(train_loader)) if limit_batches else len(train_loader)
        for i, batch in enumerate(train_loader):
            if limit_batches and i >= limit_batches:
                break
            batch = {kk: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for kk, v in batch.items()}
            scores, props = _forward(batch)
            scores_ref = _ref_forward(batch)
            tau = cfg.grpo_temperature
            pi = torch.softmax(scores / tau, dim=-1)               # 当前策略 (B,64)
            pi_ref = torch.softmax(scores_ref / tau, dim=-1)       # 冻结参考策略
            idx = torch.multinomial(pi, cfg.grpo_samples, replacement=False)  # 组采样 k 条
            logp = torch.log(pi + 1e-8).gather(1, idx)            # (B,k)
            sampled = props.gather(1, idx[:, :, None, None]
                                   .expand(-1, -1, 8, 3))         # (B,k,8,3)
            r = compute_rewards(sampled.detach().cpu().numpy(),
                                batch["future_boxes"].cpu().numpy(),
                                batch["future_valid"].cpu().numpy(),
                                batch["bev_map"].cpu().numpy(),
                                batch["traj"].cpu().numpy(),
                                cfg, cfg.ego_len, cfg.ego_wid, cfg.col_margin)
            r_t = torch.from_numpy(r).to(device)                   # (B,k)
            # 组内归一化优势 + KL(π‖π_ref) 防漂移
            adv = (r_t - r_t.mean(dim=1, keepdim=True)) / (r_t.std(dim=1, keepdim=True) + 1e-8)
            kl = (pi * (torch.log(pi + 1e-8) - torch.log(pi_ref + 1e-8))).sum(-1).mean()
            loss = -(adv * logp).mean() + cfg.grpo_beta * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            loss_sum += float(loss.detach()) * batch["traj"].shape[0]
            rew_sum += float(r_t.mean()) * batch["traj"].shape[0]
            count += batch["traj"].shape[0]
        tr = {"loss": loss_sum / max(1, count), "reward": rew_sum / max(1, count),
              "sec": time.time() - t0}
        print(f"[GRPO e{epoch:03d}] loss={tr['loss']:.4f} mean_r={tr['reward']:.3f} "
              f"({tr['sec']:.0f}s)", flush=True)
        if writer:
            writer.add_scalar("train/loss", tr["loss"], epoch)
            writer.add_scalar("train/mean_reward", tr["reward"], epoch)
            writer.flush()
        # 每 epoch 验证：π 选择 vs SFT(π_ref) 选择的规则奖励对比
        vm = evaluate_selection(model, ref_planner, val_loader, cfg, device)
        line = " ".join(f"{kk}={vv:.3f}" for kk, vv in vm.items())
        print(f"    → val {line}", flush=True)
        if writer:
            for kk, vv in vm.items():
                writer.add_scalar(f"val/{kk}", vv, epoch)
            writer.flush()
        state = {"model": model.state_dict(), "epoch": epoch, "cfg": vars(cfg)}
        torch.save(state, out_dir / "last.pt")
        if vm["mean_reward"] > best_val_reward:
            best_val_reward = vm["mean_reward"]
            torch.save(state, out_dir / "best.pt")
            print("    → ★best", flush=True)
    if writer:
        writer.close()
    print(f"GRPO 完成：best val mean_reward = {best_val_reward:.3f} → {out_dir}")
    return out_dir / "best.pt"
@torch.no_grad()
def evaluate_selection(model, ref_planner, val_loader, cfg, device) -> Dict[str, float]:
    """验证：π（GRPO）与 π_ref（SFT）各自 argmax 选择的轨迹的规则奖励指标对比。"""
    sums = {"mean_reward": 0.0, "sft_reward": 0.0, "col_rate": 0.0, "sft_col_rate": 0.0,
            "lane_rate": 0.0, "sft_lane_rate": 0.0, "prog": 0.0, "sft_prog": 0.0,
            "ade": 0.0, "sft_ade": 0.0}
    n = 0
    for batch in val_loader:
        batch = {kk: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for kk, v in batch.items()}
        scene_cur, _ = model._encode(batch["img_cur"], cams=[0, 1, 2, 3], need_patch=False)
        scene_hist, _ = model._encode(batch["img_hist"], cams=[0, 0, 0], need_patch=False)
        r_all = scene_cur.reshape(scene_cur.shape[0], -1, scene_cur.shape[-1])
        queue = scene_hist.reshape(scene_hist.shape[0], -1, scene_hist.shape[-1])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp):
            out = model.planner(r_all, queue, batch["ego_status"],
                                batch["ego_motion"], batch["frame_dt"])
            out_ref = ref_planner(r_all, queue, batch["ego_status"],
                                  batch["ego_motion"], batch["frame_dt"])
        props = out["proposal_list"][-1].float()
        sel = out["proposal_scores"].float().argmax(1)
        sel_ref = out_ref["proposal_scores"].float().argmax(1)
        B = props.shape[0]
        for b in range(B):
            traj_pi = props[b, sel[b]].cpu().numpy()[None, None]
            traj_sft = props[b, sel_ref[b]].cpu().numpy()[None, None]
            fb = batch["future_boxes"][b].cpu().numpy()[None]
            fv = batch["future_valid"][b].cpu().numpy()[None]
            bm = batch["bev_map"][b].cpu().numpy()[None]
            gt = batch["traj"][b].cpu().numpy()[None]
            r_pi = compute_rewards(traj_pi, fb, fv, bm, gt, cfg,
                                   cfg.ego_len, cfg.ego_wid, cfg.col_margin)[0, 0]
            r_sft = compute_rewards(traj_sft, fb, fv, bm, gt, cfg,
                                    cfg.ego_len, cfg.ego_wid, cfg.col_margin)[0, 0]
            d_pi = _decompose(traj_pi, fb, fv, bm, gt, cfg, cfg.ego_len, cfg.ego_wid, cfg.col_margin)
            d_sft = _decompose(traj_sft, fb, fv, bm, gt, cfg, cfg.ego_len, cfg.ego_wid, cfg.col_margin)
            sums["mean_reward"] += r_pi
            sums["sft_reward"] += r_sft
            sums["col_rate"] += d_pi["col"]
            sums["sft_col_rate"] += d_sft["col"]
            sums["lane_rate"] += d_pi["lane"]
            sums["sft_lane_rate"] += d_sft["lane"]
            sums["prog"] += d_pi["prog"]
            sums["sft_prog"] += d_sft["prog"]
            sums["ade"] += d_pi["ade"]
            sums["sft_ade"] += d_sft["ade"]
            n += 1
    return {k: v / n for k, v in sums.items()}
def _decompose(traj, fb, fv, bm, gt, cfg, ego_len, ego_wid, margin) -> Dict[str, float]:
    """单条轨迹的奖励拆项（碰撞率/车道率/进度/ADE，用于验证指标）。"""
    tr = traj[0, 0]
    n_col = 0
    for s in range(8):
        occ = _boxes_occupancy(fb[0, s], fv[0, s], cfg)
        ego_mask = _poly_to_grid(
            _footprint_corners(float(tr[s, 0]), float(tr[s, 1]), float(tr[s, 2]),
                               ego_len + 2 * margin, ego_wid + 2 * margin), cfg)
        if (ego_mask & occ).any():
            n_col += 1
    ps = cfg.bev_pixel_size / cfg.bev_map_stride
    rows = ((tr[:, 0] - cfg.pc_x_min) / ps).astype(np.int64)
    cols = ((tr[:, 1] - cfg.pc_y_min) / ps).astype(np.int64)
    ok = (rows >= 0) & (rows < 100) & (cols >= 0) & (cols < 100)
    lane = sum(bm[0, rows[s], cols[s]] in DRIVABLE_CLASSES for s in range(8)
               if ok[s]) / 8.0
    prog = float(np.clip(tr[-1, 0] / 20.0, 0.0, 1.0))
    ade = float(np.linalg.norm(tr[:, :2] - gt[0, :, :2], axis=-1).mean())
    return {"col": float(n_col > 0), "lane": lane, "prog": prog, "ade": ade}
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", type=str, required=True, help="SFT 检查点（stage1 best_val）")
    ap.add_argument("--cache-root", type=str, default=str(WORK_ROOT / "cache_13k"))
    ap.add_argument("--out-dir", type=str, default=str(WORK_ROOT / "runs_v5" / "grpo"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit-batches", type=int, default=0)
    ap.add_argument("--no-tb", action="store_true")
    args = ap.parse_args()
    cfg = Config(batch_size=args.batch_size, grad_accum=args.grad_accum,
                 num_workers=args.num_workers)
    train_grpo(cfg, Path(args.init_from), Path(args.cache_root), Path(args.out_dir),
               epochs=args.epochs, limit_batches=args.limit_batches, tb=not args.no_tb)
if __name__ == "__main__":
    main()
