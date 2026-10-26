"""md_train：stage 1-3 训练主循环（纯 PyTorch + TensorBoard）。

阶段冻结日程（stage 内恒定，跨阶段从上一阶段 checkpoint 继续）：
  stage 1: 冻结 DINO 主干；LoRA(+scene tokens) 与 Planner 可训（无 BEV 前向）
  stage 2: 冻结 DINO + LoRA + Planner；仅 BEV 头可训（DINO 前向 no_grad）
  stage 3: 冻结 DINO 主干；LoRA + Planner + BEV 联合可训

用法：
    python -m my_drive.train.md_train --stage 1 --epochs 30
    python -m my_drive.train.md_train --stage 2 --epochs 30 --init-from runs/stage1/last.pt
    python -m my_drive.train.md_train --stage 3 --epochs 30 --init-from runs/stage2/last.pt

快速自检：--limit-batches 4 --epochs 1
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ..md_common import WORK_ROOT, Config
from ..data.md_data import OverfitDataset
from .md_loss import compute_losses
from .md_model import MyDriveModel


_PLANNER_STRUCT_PREFIXES = (
    "planner.init_head.",
    "planner.refine_heads.",
    "planner.ego_motion_mlp.",
    "planner.bern",
    "planner.anchor_offsets",
)


def check_ckpt_compat(ckpt: Dict, cfg: Config, skipped: List[str],
                      source: str = "checkpoint",
                      missing: Optional[List[str]] = None,
                      own_keys: Optional[List[str]] = None) -> None:
    """结构演进时 fail-fast：禁止把旧结构 checkpoint 静默半载入。

    shape 过滤（strict=False）保证代码能跑，但 planner 头 / 运动编码 / Bernstein
    基座 / anchor 的形状变化会让“换表征”的实验静默退化成部分随机初始化。
    这里把已知结构键与 cfg 元信息显式对账：旧 raw 头(24↔8)、旧 Fourier 编码
    (39↔3)、旧 Bernstein 基座（控制点数变化）、带 anchor_offsets 的旧 ckpt
    都直接报错，避免训出一个无法解释的模型。完全不带 planner 的 ckpt（如
    stage0 预训练）例外：允许 planner 随机初始化。
    """
    # stage0 是 pretext（planner 未训练，权重随机）：下游换表征时 planner 头本就
    # 该重新初始化，不能拿 stage0 的随机 planner 结构做 fail-fast。
    if ckpt.get("stage") == 0:
        return
    missing_set = set(missing or [])
    if own_keys is None:
        has_planner = True
    else:
        has_planner = (
            any(k.startswith("planner.") for k in skipped)
            or any(k.startswith("planner.") and k not in missing_set
                   for k in own_keys))
    meta = ckpt.get("cfg") or {}
    if has_planner and "traj_repr" in meta and meta["traj_repr"] != cfg.traj_repr:
        raise ValueError(
            f"{source} 的 cfg.traj_repr={meta['traj_repr']!r} 与当前 "
            f"cfg.traj_repr={cfg.traj_repr!r} 不一致；换表征请从原始 DINO/"
            "干净 ckpt 重训，不要从旧 planner 头热启动")
    if (has_planner and cfg.traj_repr == "bezier" and "bezier_n_ctrl" in meta
            and meta["bezier_n_ctrl"] != cfg.bezier_n_ctrl):
        raise ValueError(
            f"{source} 的 cfg.bezier_n_ctrl={meta['bezier_n_ctrl']!r} 与当前 "
            f"cfg.bezier_n_ctrl={cfg.bezier_n_ctrl!r} 不一致；控制点数变化会"
            "改变 Bernstein 基座与头维度，请从干净 ckpt 重训")
    bad = [k for k in skipped if k.startswith(_PLANNER_STRUCT_PREFIXES)]
    # 完全不带 planner 的 ckpt（如 stage0 预训练）允许 planner 随机初始化；
    # 只有 ckpt 已带部分 planner 结构、却又缺当前结构键时，才视为半截/旧版 ckpt。
    missing_struct = [k for k in (missing or [])
                      if k.startswith(_PLANNER_STRUCT_PREFIXES)]
    if missing_struct:
        partial = any(k.startswith("planner.") and k not in missing_set
                      for k in (own_keys or []))
        if partial:
            bad += missing_struct
    if bad:
        raise ValueError(
            f"{source} 有 {len(bad)} 个 planner 结构键形状/存在性不匹配"
            f"（如 {bad[0]}）；拒绝静默部分载入。请用 --init-from 指向同表征/"
            "同控制点数的 ckpt，或从零训练 stage1")


class WarmupCosine:
    """线性 warmup + cosine 学习率（按 optimizer step）。

    初始 lr 直接置为 warmup 第一步的值：optimizer 以全量 lr 创建，而
    scheduler.step() 在首个 optimizer.step() 之后才调用——不设初值的话
    首个 step 会用全量 lr 冲出。
    """

    def __init__(self, optimizer: torch.optim.Optimizer, total_steps: int,
                 warmup_steps: int, base_lr: float):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(1, warmup_steps)
        self.base_lr = base_lr
        self.step_num = 0
        for g in optimizer.param_groups:
            g["lr"] = base_lr / max(1, warmup_steps)
    def step(self) -> float:
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.base_lr * self.step_num / self.warmup_steps
        else:
            t = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr
def run_epoch(model: MyDriveModel, loader: DataLoader, cfg: Config, flags: Dict[str, bool],
              device: torch.device, optimizer: Optional[torch.optim.Optimizer] = None,
              scheduler: Optional[WarmupCosine] = None, limit_batches: int = 0,
              ) -> Dict[str, float]:
    """一个 epoch 的训练（optimizer 非空）或评估（optimizer=None）。"""
    training = optimizer is not None
    model.train(training)
    # bf16 不需要 GradScaler（指数位与 fp32 相同，不会下溢；GradScaler 是 fp16 时代的产物）
    sums: Dict[str, float] = {}
    count = 0
    micro = 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        n_batches = min(limit_batches, len(loader)) if limit_batches else len(loader)
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        def _forward():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp):
                pred = model(batch, with_planner=flags["with_planner"],
                             with_bev=flags["with_bev"],
                             dino_no_grad=flags["dino_no_grad"])
                losses = compute_losses(pred, batch, cfg,
                                        flags["with_planner"], flags["with_bev"])
            return losses
        if training:
            losses = _forward()
            # 梯度累积：loss 不预除 accum，更新时按本组实际微批数归一化
            # （若 loss/accum，尾批不足 accum 个微批时梯度会被欠缩放）
            losses["loss"].backward()
            micro += 1
            if (i + 1) % cfg.grad_accum == 0 or (i + 1) == n_batches:
                if micro > 1:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.div_(micro)
                micro = 0
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
        else:
            with torch.no_grad():
                losses = _forward()
        bs = batch["traj"].shape[0]
        count += bs
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach()) * bs
    metrics = {k: v / max(1, count) for k, v in sums.items()}
    metrics["sec"] = time.time() - t0
    return metrics
def evaluate(model, loader, cfg, flags, device) -> Dict[str, float]:
    return run_epoch(model, loader, cfg, flags, device, optimizer=None)
def select_viz_indices(viz_ds, n: int = 10, max_per_log_scan: int = 60) -> List[int]:
    """从数据集精选 n 帧作为 TB 固定可视化帧（确定性，跨 epoch 可对比）。

    策略：① 按日志均匀配额（场景多样性）；② 每日志内按 GT 检测框数排序，
    优先车辆密集的复杂场景（能同时看出 BEV/检测与轨迹质量差异）。
    """
    entries = viz_ds.entries
    if n >= len(entries):
        return list(range(len(entries)))
    by_log: Dict[str, List[int]] = {}
    for i, e in enumerate(entries):
        by_log.setdefault(e["log_name"], []).append(i)
    quota = max(1, math.ceil(n / len(by_log)))
    picked: List[int] = []
    for log in sorted(by_log):
        idxs = by_log[log][:max_per_log_scan]
        scored = []
        for i in idxs:
            with np.load(viz_ds.cache_root / "labels" / f"{entries[i]['token']}.npz") as z:
                n_box = len(z["boxes"])
            scored.append((n_box, i))
        scored.sort(reverse=True)
        picked += [i for _, i in scored[:quota]]
        if len(picked) >= n:
            break
    return sorted(picked[:n])
def _make_loader(cache_root: Path, cfg: Config, split: str, shuffle: bool,
                 required: bool = False) -> Optional[DataLoader]:
    try:
        ds = OverfitDataset(cache_root, cfg, split=split, augment=shuffle)
    except RuntimeError:
        if required:
            raise
        return None
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers if shuffle else 0, pin_memory=True,
                      drop_last=False, persistent_workers=shuffle and cfg.num_workers > 0)
def train(stage: int, cfg: Config, cache_root: Path, out_dir: Path,
          init_from: Optional[Path] = None, epochs: int = 30,
          limit_batches: int = 0,
          tb: bool = True, device_str: str = "cuda", eval_every: int = 2,
          viz_frames: int = 10, patience: int = 6) -> Path:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tb")) if tb else None
    train_loader = _make_loader(cache_root, cfg, "train", shuffle=True, required=True)
    # 验证集缺省时回退训练集（纯过拟合模式）
    val_loader = _make_loader(cache_root, cfg, "val", shuffle=False) or \
        _make_loader(cache_root, cfg, "train", shuffle=False)
    test_loader = _make_loader(cache_root, cfg, "test", shuffle=False)
    # 固定可视化帧：优先验证集
    viz_ds = val_loader.dataset if val_loader is not None else train_loader.dataset
    viz_indices = select_viz_indices(viz_ds, n=viz_frames)
    model = MyDriveModel(cfg).to(device)
    if init_from is not None:
        ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
        # 形状匹配的键才载入；随后 check_ckpt_compat 对旧 planner 结构 fail-fast
        own = model.state_dict()
        compatible = {k: v for k, v in ckpt["model"].items()
                      if k in own and own[k].shape == v.shape}
        skipped = [k for k in ckpt["model"] if k not in compatible]
        missing = [k for k in own if k not in compatible]
        check_ckpt_compat(ckpt, cfg, skipped, source=str(init_from),
                          missing=missing, own_keys=list(own))
        model.load_state_dict(compatible, strict=False)
        print(f"载入 {init_from}（stage {ckpt.get('stage')} epoch {ckpt.get('epoch')}）"
              f" 匹配 {len(compatible)}/{len(own)} 键，跳过 {len(skipped)} 个不匹配键"
              + (f"（如 {skipped[0]}）" if skipped else ""))
    flags = model.set_train_stage(stage)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(
        (min(limit_batches, len(train_loader)) if limit_batches else len(train_loader))
        / cfg.grad_accum)
    scheduler = WarmupCosine(optimizer, total_steps=steps_per_epoch * epochs,
                             warmup_steps=max(1, int(steps_per_epoch * epochs
                                                     * cfg.warmup_ratio)),
                             base_lr=cfg.lr)
    from ..eval.md_viz import render_fixed_frames
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset) if val_loader else 0
    n_test = len(test_loader.dataset) if test_loader else 0
    print(f"stage {stage} | train {n_train} / val {n_val} / test {n_test} "
          f"| batch {cfg.batch_size}×accum{cfg.grad_accum} "
          f"| {steps_per_epoch} opt-steps/epoch | device={device}", flush=True)
    history: List[dict] = []
    best_val = float("inf")
    bad_evals = 0
    stopped_early = False
    for epoch in range(epochs):
        train_m = run_epoch(model, train_loader, cfg, flags, device, optimizer=optimizer,
                            scheduler=scheduler, limit_batches=limit_batches)
        do_eval = (epoch % eval_every == 0) or (epoch == epochs - 1)
        eval_m = evaluate(model, val_loader, cfg, flags, device) if do_eval else {}
        lr = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch, "lr": lr,
               **{f"train_{k}": v for k, v in train_m.items()},
               **{f"val_{k}": v for k, v in eval_m.items()}}
        history.append(row)
        line = " ".join(f"{k}={v:.4f}" for k, v in sorted(train_m.items())
                        if k != "sec")
        eval_str = f" | val_loss={eval_m['loss']:.4f}" if eval_m else ""
        print(f"[s{stage} e{epoch:03d}] lr={lr:.2e} {line}{eval_str} ({train_m['sec']:.1f}s)",
              flush=True)
        # checkpoint：last 恒存；best_val 只在评估 epoch 按 val_loss 更新
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "stage": stage, "epoch": epoch, "cfg": vars(cfg), "history": history}
        torch.save(state, out_dir / "last.pt")
        marker = ""
        if do_eval:
            # P0 只补双口径指标；best_val 仍按 val_loss 选模——改判据会改变训练行为，
            # 等 traj_ade_argmax 与 geo 口径都跑过一轮再决定是否切换。
            crit = eval_m.get("loss", train_m["loss"])
            if crit < best_val:
                best_val = crit
                bad_evals = 0
                torch.save(state, out_dir / "best_val.pt")
                marker = " ★best"
            else:
                bad_evals += 1
                marker = f" (bad {bad_evals}/{patience})"
            print(f"    → val 判据 {crit:.4f}{marker}", flush=True)
            if bad_evals >= patience:
                print(f"    → 早停：val 连续 {patience} 次评估未改善（best={best_val:.4f}）",
                      flush=True)
                stopped_early = True
        if writer is not None:
            writer.add_scalar("train/lr", lr, epoch)
            for k, v in train_m.items():
                if k != "sec":
                    writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in eval_m.items():
                if k != "sec":
                    writer.add_scalar(f"val/{k}", v, epoch)
            # 固定帧可视化：每个 epoch 写 BEV / 轨迹 / 相机图
            figs = render_fixed_frames(model, viz_ds, viz_indices, cfg, device, flags)
            for fi, fig in enumerate(figs):
                writer.add_image(f"viz/frame_{fi:02d}",
                                 np.transpose(fig, (2, 0, 1)), epoch)
            writer.flush()
        if stopped_early:
            break
    # 测试集最终评估：用 best_val 权重而非 last
    best_ckpt = out_dir / "best_val.pt"
    if best_ckpt.is_file():
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"测试评估使用 best_val.pt（epoch {ckpt.get('epoch')}）", flush=True)
    if test_loader is not None:
        test_m = evaluate(model, test_loader, cfg, flags, device)
        row = {"stage": stage, **{f"test_{k}": v for k, v in test_m.items()}}
        print(f"[s{stage} TEST] " + " ".join(f"{k}={v:.4f}" for k, v in sorted(test_m.items())
                                             if k != "sec"), flush=True)
        with open(out_dir / "test_metrics.json", "w") as f:
            json.dump(row, f, indent=1)
        if writer is not None:
            for k, v in test_m.items():
                if k != "sec":
                    writer.add_scalar(f"test/{k}", v, epochs - 1)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=1)
    if writer is not None:
        writer.close()
    print(f"stage {stage} 完成：best val 判据 = {best_val:.4f}"
          f"{'（早停）' if stopped_early else ''} → {out_dir}")
    return best_ckpt if best_ckpt.is_file() else out_dir / "last.pt"
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--init-from", type=str, default=None)
    ap.add_argument("--cache-root", type=str, default=str(WORK_ROOT / "cache"))
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--viz-frames", type=int, default=10,
                    help="每个 epoch 写 TensorBoard 的固定可视化帧数")
    ap.add_argument("--patience", type=int, default=6,
                    help="val 早停耐心（连续 N 次评估不改善则停）")
    ap.add_argument("--limit-batches", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-tb", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    # ---- 轨迹表征 / top-k 监督 / 几何重打分（默认开；raw / topk=1 / --geo-rescore-off 可回退消融） ----
    ap.add_argument("--traj-repr", type=str, default="bezier", choices=["bezier", "raw"],
                    help="轨迹表征：bezier=Bézier 控制点（默认）/ raw=逐点回归（旧行为）")
    ap.add_argument("--bezier-ctrl", type=int, default=5,
                    help="Bézier 控制点数（含固定起点，默认 5；4=三次曲线）")
    ap.add_argument("--topk", type=int, default=5,
                    help="每层监督的 top-k proposal 数（默认 5；1=旧 WTA）")
    ap.add_argument("--topk-decay", type=float, default=0.5,
                    help="top-k 第 j 名权重衰减（decay^(j-1)，归一化）")
    ap.add_argument("--geo-rescore-off", action="store_true",
                    help="关闭推理时几何重打分（旧 argmax(scorer) 行为）")
    args = ap.parse_args()
    cfg = Config(batch_size=args.batch_size, grad_accum=args.grad_accum,
                 lr=args.lr, num_workers=args.num_workers, amp=not args.no_amp,
                 traj_repr=args.traj_repr, bezier_n_ctrl=args.bezier_ctrl,
                 w_traj_topk=args.topk, topk_weight_decay=args.topk_decay,
                 geo_rescore=not args.geo_rescore_off)
    epochs = args.epochs or {1: cfg.epochs_stage1, 2: cfg.epochs_stage2,
                             3: cfg.epochs_stage3}[args.stage]
    out_dir = Path(args.out_dir or (WORK_ROOT / "runs" / f"stage{args.stage}"))
    train(stage=args.stage, cfg=cfg, cache_root=Path(args.cache_root), out_dir=out_dir,
          init_from=Path(args.init_from) if args.init_from else None,
          epochs=epochs, limit_batches=args.limit_batches, tb=not args.no_tb,
          device_str="cpu" if args.cpu else "cuda", eval_every=args.eval_every,
          viz_frames=args.viz_frames, patience=args.patience)
if __name__ == "__main__":
    main()
