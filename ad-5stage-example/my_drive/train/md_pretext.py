"""md_pretext：stage 0 —— 未来特征预测（Foresight pretext，自监督，无标签）。

在冻结 DINOv3 特征空间上，用 masked transformer 从 t-3..t 的 patch 特征预测
t+1 帧同位置的原始 patch 特征（SmoothL1，β=0.1）。目标特征由「冻结目标分支」
计算（见 MyDriveModel.make_frozen_target_branch）——目标不漂移、无塌缩、无需 EMA。

数据：全部日志中「5 帧连续前视图像」的窗口（不需要 4 相机/路线/标签）。

用法：
    # 1) 数据准备（扫描 + 图像本地化）
    python -m my_drive.train.md_pretext prepare --num-windows 50000
    # 2) 预训练
    python -m my_drive.train.md_pretext train --epochs 10
    # 之后：md_train --stage 1 --init-from <pretext>/best_val.pt
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ..md_common import WORK_ROOT, Config
from ..data.md_data import PretextDataset, prepare_pretext_dataset
from .md_model import ForesightPredictor, MyDriveModel
from .md_train import WarmupCosine


def run_epoch(model, predictor, loader, cfg, device, optimizer=None,
              scheduler=None, limit_batches: int = 0) -> Dict[str, float]:
    """一个 epoch 的训练或评估；返回 loss / cos_sim（预测与目标的余弦相似度）。"""
    training = optimizer is not None
    model.train(training)
    predictor.train(training)
    # bf16 无需 GradScaler（同 md_train）
    loss_sum, cos_sum, count, micro, t0 = 0.0, 0.0, 0, 0, time.time()
    n_batches = min(limit_batches, len(loader)) if limit_batches else len(loader)
    for i, batch in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        def _fwd():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp):
                pred, tgt = model.forward_foresight(batch, predictor)
                loss = F.smooth_l1_loss(pred.float(), tgt.float(), beta=0.1)
                cos = F.cosine_similarity(pred.float(), tgt.float(), dim=-1).mean()
            return loss, cos
        if training:
            loss, cos = _fwd()
            # 梯度累积：按本组实际微批数归一化（同 md_train）
            loss.backward()
            micro += 1
            if (i + 1) % cfg.grad_accum == 0 or (i + 1) == n_batches:
                if micro > 1:
                    for p in list(model.parameters()) + list(predictor.parameters()):
                        if p.grad is not None:
                            p.grad.div_(micro)
                micro = 0
                torch.nn.utils.clip_grad_norm_(
                    [p for p in list(model.parameters()) + list(predictor.parameters())
                     if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        else:
            with torch.no_grad():
                loss, cos = _fwd()
        bs = batch["imgs"].shape[0]
        count += bs
        loss_sum += float(loss) * bs
        cos_sum += float(cos) * bs
    return {"loss": loss_sum / max(1, count), "cos_sim": cos_sum / max(1, count),
            "sec": time.time() - t0}
def train_pretext(cfg: Config, cache_root: Path, out_dir: Path, epochs: int,
                  patience: int = 3, eval_every: int = 1, tb: bool = True,
                  limit_batches: int = 0) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tb")) if tb else None
    train_ds = PretextDataset(cache_root, cfg, "train", augment=True)
    val_ds = PretextDataset(cache_root, cfg, "val", augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True,
                              persistent_workers=cfg.num_workers > 0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    model = MyDriveModel(cfg).to(device)
    model.make_frozen_target_branch()
    predictor = ForesightPredictor(cfg).to(device)
    # 可训：LoRA 组（含 neck/scene_embeds）+ predictor；DINO 主干/planner/BEV 冻结
    groups = model.param_groups()
    for p in groups["dino"] + groups["planner"] + groups["bev"]:
        p.requires_grad = False
    for p in groups["lora"]:
        p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad] + \
        list(predictor.parameters())
    n_train = sum(p.numel() for p in params) / 1e6
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(
        (min(limit_batches, len(train_loader)) if limit_batches else len(train_loader))
        / cfg.grad_accum)
    scheduler = WarmupCosine(optimizer, steps_per_epoch * epochs,
                             max(1, int(steps_per_epoch * epochs * cfg.warmup_ratio)),
                             cfg.lr)
    print(f"[stage 0] train {len(train_ds)} / val {len(val_ds)} 窗口 | "
          f"可训 {n_train:.2f}M | {steps_per_epoch} opt-steps/epoch", flush=True)
    best_val, bad = float("inf"), 0
    history = []
    for epoch in range(epochs):
        tr = run_epoch(model, predictor, train_loader, cfg, device, optimizer, scheduler,
                       limit_batches=limit_batches)
        do_eval = (epoch % eval_every == 0) or (epoch == epochs - 1)
        va = run_epoch(model, predictor, val_loader, cfg, device) if do_eval else {}
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
               **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)
        print(f"[s0 e{epoch:03d}] train_loss={tr['loss']:.4f} cos={tr['cos_sim']:.3f}"
              + (f" | val_loss={va['loss']:.4f} cos={va['cos_sim']:.3f}" if va else "")
              + f" ({tr['sec']:.0f}s)", flush=True)
        if writer:
            for k, v in tr.items():
                if k != "sec":
                    writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in va.items():
                if k != "sec":
                    writer.add_scalar(f"val/{k}", v, epoch)
            writer.flush()
        state = {"model": model.state_dict(), "predictor": predictor.state_dict(),
                 "stage": 0, "epoch": epoch, "cfg": vars(cfg)}
        torch.save(state, out_dir / "last.pt")
        # best_val 只在评估 epoch 按 val_loss 更新（若回退 train_loss 会让早停永远被重置）
        if do_eval:
            crit = va["loss"]
            if crit < best_val:
                best_val, bad = crit, 0
                torch.save(state, out_dir / "best_val.pt")
                print(f"    → ★best val {crit:.4f}", flush=True)
            else:
                bad += 1
                if bad >= patience:
                    print(f"    → 早停（best val {best_val:.4f}）", flush=True)
                    break
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=1)
    if writer:
        writer.close()
    print(f"stage 0 完成：best val = {best_val:.4f} → {out_dir}")
    return out_dir / "best_val.pt"
def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--num-windows", type=int, default=50000)
    p_prep.add_argument("--max-logs", type=int, default=2000)
    p_prep.add_argument("--cache-root", type=str,
                        default=str(WORK_ROOT / "cache_pretext"))
    p_tr = sub.add_parser("train")
    p_tr.add_argument("--epochs", type=int, default=10)
    p_tr.add_argument("--cache-root", type=str, default=str(WORK_ROOT / "cache_pretext"))
    p_tr.add_argument("--out-dir", type=str, default=str(WORK_ROOT / "runs_13k" / "stage0"))
    p_tr.add_argument("--batch-size", type=int, default=2)
    p_tr.add_argument("--grad-accum", type=int, default=2)
    p_tr.add_argument("--lr", type=float, default=2.5e-4)
    p_tr.add_argument("--num-workers", type=int, default=4)
    p_tr.add_argument("--patience", type=int, default=3)
    p_tr.add_argument("--limit-batches", type=int, default=0)
    p_tr.add_argument("--no-tb", action="store_true")
    args = ap.parse_args()
    cfg = Config(batch_size=getattr(args, "batch_size", 2),
                 grad_accum=getattr(args, "grad_accum", 2),
                 lr=getattr(args, "lr", 2.5e-4),
                 num_workers=getattr(args, "num_workers", 4))
    if args.cmd == "prepare":
        prepare_pretext_dataset(cfg, num_windows=args.num_windows,
                                cache_root=Path(args.cache_root),
                                max_logs=args.max_logs)
    else:
        train_pretext(cfg, Path(args.cache_root), Path(args.out_dir),
                      epochs=args.epochs, patience=args.patience,
                      tb=not args.no_tb, limit_batches=args.limit_batches)
if __name__ == "__main__":
    main()
