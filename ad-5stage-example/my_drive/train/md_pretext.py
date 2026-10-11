"""md_pretext：stage 0 —— 未来特征预测（Foresight pretext，自监督，无标签）。

DINO-Foresight 化版本（相对旧版四项升级，均可经 Config 回退消融）：
  1) 预测器：时空可分离注意力（_ForesightBlock，cfg.foresight_spatial）；
  2) 预测目标：冻结目标分支的多层特征 concat（第 3/6/9/12 层，1536 维）经 PCA
     压缩到 cfg.foresight_pca_dim 维（cfg.foresight_target="pca"），"last" 回退
     为仅末层 384 维旧行为。PCA 由 prepare-pca 离线预计算；
  3) 多步长：8 帧窗口（4 历史 + 4 未来候选），训练随机采 k∈[1, foresight_max_step]，
     验证按 k=1..4 分别报告（早停准则 = 各时距 val_loss 均值）；
  4) 分辨率课程：--image-size 可先低分辨率训练、再 --init-from 热启动高分辨率
     （patch 网格随 cfg.image_size 自适应，缓存与分辨率无关，无需重新 prepare）。

数据：全部日志中「8 帧连续前视图像」的窗口（不需要 4 相机/路线/标签）。

用法：
    # 0) 数据准备（扫描 + 图像本地化；与旧 5 帧缓存不兼容，需重建）
    python -m my_drive.train.md_pretext prepare --num-windows 50000
    # 1) PCA 目标矩阵预计算（foresight_target="pca" 时必需）
    python -m my_drive.train.md_pretext prepare-pca
    # 2) 预训练（可选低分辨率课程：--image-size 320,192 → 再 --init-from 续训 640,384）
    python -m my_drive.train.md_pretext train --epochs 10
    # 之后：md_train --stage 1 --init-from <pretext>/best_val.pt
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from ..md_common import WORK_ROOT, Config
from ..data.md_data import PretextDataset, prepare_pretext_dataset
from .md_model import ForesightPredictor, MyDriveModel
from .md_train import WarmupCosine


def _build_model_with_target(cfg: Config, device: torch.device) -> MyDriveModel:
    """构建模型并建立冻结目标分支（train 与 prepare-pca 共用）。"""
    model = MyDriveModel(cfg).to(device)
    model.make_frozen_target_branch()
    return model


def prepare_pca(cfg: Config, cache_root: Path, out_path: Path,
                num_windows: int, patches_per_window: int) -> Path:
    """离线预计算 PCA 目标矩阵（DINO-Foresight 的关键组件：压缩 + 去噪 + 正则）。

    在冻结目标分支的第 3/6/9/12 层 concat 特征（1536 维）上，对训练窗口的目标帧
    随机采样 patch，累计一阶/二阶统计量后特征值分解，取前 pca_dim 个主成分。
    结果存 pca.pt：{"mean": (1536,), "components": (pca_dim,1536), ...}。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model_with_target(cfg, device)
    model.eval()
    ds = PretextDataset(cache_root, cfg, "train", augment=False)  # 统计用原图，不开增强
    n_win = min(num_windows, len(ds))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True,
                        sampler=SubsetRandomSampler(
                            torch.randperm(len(ds))[:n_win].tolist()))
    in_dim = model.encoder.embed_dim * 4                          # 1536
    s = torch.zeros(in_dim, dtype=torch.float64)
    q = torch.zeros(in_dim, in_dim, dtype=torch.float64)
    n = 0
    k_max = cfg.foresight_max_step
    for i, batch in enumerate(loader):
        imgs = batch["imgs"].to(device, non_blocking=True)         # (B,8,3,H,W)
        k = int(torch.randint(1, k_max + 1, ()).item())            # 各时距目标都入统计
        B = imgs.shape[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=cfg.amp and device.type == "cuda"):
            st = model._tgt_scene_embeds[:, 0].expand(B, -1, -1)
            final, inters = model._tgt_encoder.model.forward_features(imgs[:, 3 + k], st)
            n_pre = model._tgt_encoder.n_prefix
            raw = torch.cat([t[:, n_pre:] for t in inters] + [final[:, n_pre:]],
                            dim=-1)                                # (B,P,1536)
        x = raw.float().reshape(-1, in_dim)
        take = min(patches_per_window * imgs.shape[0], x.shape[0])
        x = x[torch.randperm(x.shape[0], device=x.device)[:take]].double().cpu()
        s += x.sum(0)
        q += x.T @ x
        n += x.shape[0]
        if (i + 1) % 50 == 0:
            print(f"[prepare-pca] {i + 1} 批 / 累计 {n} patch", flush=True)
    mean = s / n
    cov = q / n - torch.outer(mean, mean)
    evals, evecs = torch.linalg.eigh(cov)                          # 升序
    pca_dim = min(cfg.foresight_pca_dim, in_dim)
    comps = evecs[:, -pca_dim:].flip(-1).T.contiguous()            # (pca_dim,1536) 降序
    explained = float(evals[-pca_dim:].sum() / evals.sum())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean.float(), "components": comps.float(),
                "in_dim": in_dim, "pca_dim": pca_dim, "n_samples": n,
                "explained_var": explained}, out_path)
    print(f"[prepare-pca] 完成：{n} patch，前 {pca_dim} 主成分解释方差 {explained:.3f}"
          f" → {out_path}")
    return out_path


def run_epoch(model, predictor, loader, cfg, device, optimizer=None,
              scheduler=None, limit_batches: int = 0,
              horizon: Optional[int] = None) -> Dict[str, float]:
    """一个 epoch 的训练或评估。

    训练（optimizer 非 None）：每 batch 随机采时距 k∈[1, foresight_max_step]，
    返回 loss / cos_sim（所采时距上的值）。
    评估（optimizer 为 None）：逐 batch 跑全部 k=1..max_step，返回
    loss（各时距均值，早停准则）/ cos_sim（均值）及逐时距 loss_k1..kN / cos_k1..kN。
    可用 horizon=k 固定时距（消融/复现旧单步行为用）。
    """
    training = optimizer is not None
    model.train(training)
    predictor.train(training)
    k_max = horizon or cfg.foresight_max_step
    k_min = horizon or 1
    # bf16 无需 GradScaler（同 md_train）
    loss_sum, cos_sum, count, micro, t0 = 0.0, 0.0, 0, 0, time.time()
    per_k_loss: Dict[int, float] = {}
    per_k_cos: Dict[int, float] = {}
    per_k_n: Dict[int, int] = {}
    n_batches = min(limit_batches, len(loader)) if limit_batches else len(loader)
    for i, batch in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        def _fwd(k: int):
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=cfg.amp and device.type == "cuda"):
                pred, tgt = model.forward_foresight(batch, predictor, future_step=k)
                loss = F.smooth_l1_loss(pred.float(), tgt.float(), beta=0.1)
                cos = F.cosine_similarity(pred.float(), tgt.float(), dim=-1).mean()
            return loss, cos

        bs = batch["imgs"].shape[0]
        if training:
            k = int(torch.randint(k_min, k_max + 1, ()).item())   # 随机多步长
            loss, cos = _fwd(k)
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
            count += bs
            loss_sum += float(loss) * bs
            cos_sum += float(cos) * bs
        else:
            with torch.no_grad():
                for k in range(k_min, k_max + 1):                 # 逐时距评估
                    loss, cos = _fwd(k)
                    per_k_loss[k] = per_k_loss.get(k, 0.0) + float(loss) * bs
                    per_k_cos[k] = per_k_cos.get(k, 0.0) + float(cos) * bs
                    per_k_n[k] = per_k_n.get(k, 0) + bs
            count += bs
    out: Dict[str, float] = {"sec": time.time() - t0}
    if training:
        out["loss"] = loss_sum / max(1, count)
        out["cos_sim"] = cos_sum / max(1, count)
    else:
        ks = sorted(per_k_loss)
        out["loss"] = sum(per_k_loss[k] for k in ks) / max(1, sum(per_k_n.values()))
        out["cos_sim"] = sum(per_k_cos[k] for k in ks) / max(1, sum(per_k_n.values()))
        for k in ks:
            out[f"loss_k{k}"] = per_k_loss[k] / max(1, per_k_n[k])
            out[f"cos_k{k}"] = per_k_cos[k] / max(1, per_k_n[k])
    return out


def train_pretext(cfg: Config, cache_root: Path, out_dir: Path, epochs: int,
                  patience: int = 3, eval_every: int = 1, tb: bool = True,
                  limit_batches: int = 0, pca_path: Optional[Path] = None,
                  init_from: Optional[Path] = None,
                  horizon: Optional[int] = None) -> Path:
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
    model = _build_model_with_target(cfg, device)
    if cfg.foresight_target == "pca":
        model.load_foresight_pca(pca_path or (Path(cache_root) / "pca.pt"))
    predictor = ForesightPredictor(cfg).to(device)
    if init_from is not None:
        # 分辨率课程热启动：低分辨率 stage-0 checkpoint → 高分辨率续训。
        # 模型无尺寸相关参数（predictor 对 token 序列操作、scene token 按相机数），
        # 直接 strict=False 加载；缺失/多余键打印出来便于核查。
        ckpt = torch.load(str(init_from), map_location="cpu", weights_only=True)
        m_missing, m_unexp = model.load_state_dict(ckpt["model"], strict=False)
        p_missing, p_unexp = predictor.load_state_dict(ckpt["predictor"], strict=False)
        print(f"[init-from] {init_from} | model missing={m_missing} "
              f"unexpected={m_unexp} | predictor missing={p_missing} "
              f"unexpected={p_unexp}")
        model.make_frozen_target_branch()          # init-from 后重建冻结目标分支
        if cfg.foresight_target == "pca" and not bool(model.foresight_pca_ready):
            model.load_foresight_pca(pca_path or (Path(cache_root) / "pca.pt"))
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
          f"可训 {n_train:.2f}M | {steps_per_epoch} opt-steps/epoch | "
          f"target={cfg.foresight_target} spatial={cfg.foresight_spatial} "
          f"max_step={horizon or cfg.foresight_max_step}", flush=True)
    best_val, bad = float("inf"), 0
    history = []
    for epoch in range(epochs):
        tr = run_epoch(model, predictor, train_loader, cfg, device, optimizer, scheduler,
                       limit_batches=limit_batches, horizon=horizon)
        do_eval = (epoch % eval_every == 0) or (epoch == epochs - 1)
        va = run_epoch(model, predictor, val_loader, cfg, device,
                       limit_batches=limit_batches, horizon=horizon) if do_eval else {}
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
               **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)
        msg = (f"[s0 e{epoch:03d}] train_loss={tr['loss']:.4f} cos={tr['cos_sim']:.3f}"
               + (f" | val_loss={va['loss']:.4f} cos={va['cos_sim']:.3f}" if va else "")
               + f" ({tr['sec']:.0f}s)")
        if va and any(k.startswith("loss_k") for k in va):
            msg += " | " + " ".join(f"k{k[-1]}={va[k]:.4f}"
                                    for k in sorted(va) if k.startswith("loss_k"))
        print(msg, flush=True)
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
        # best_val 只在评估 epoch 按 val_loss（各时距均值）更新
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


def _parse_size(s: str):
    w, h = s.split(",")
    return (int(w), int(h))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--num-windows", type=int, default=50000)
    p_prep.add_argument("--max-logs", type=int, default=2000)
    p_prep.add_argument("--cache-root", type=str,
                        default=str(WORK_ROOT / "cache_pretext"))
    p_pca = sub.add_parser("prepare-pca")
    p_pca.add_argument("--cache-root", type=str,
                       default=str(WORK_ROOT / "cache_pretext"))
    p_pca.add_argument("--out", type=str, default="")
    p_pca.add_argument("--num-windows", type=int, default=0,
                       help="默认 cfg.foresight_pca_windows")
    p_pca.add_argument("--patches-per-window", type=int, default=0,
                       help="默认 cfg.foresight_pca_patches")
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
    p_tr.add_argument("--pca", type=str, default="",
                      help="PCA 矩阵路径（默认 <cache_root>/pca.pt）")
    p_tr.add_argument("--init-from", type=str, default="",
                      help="低分辨率 stage-0 checkpoint，分辨率课程热启动用")
    p_tr.add_argument("--image-size", type=str, default="",
                      help="W,H 覆盖 cfg.image_size（分辨率课程，如 320,192）")
    p_tr.add_argument("--horizon", type=int, default=0,
                      help="固定时距 k（消融/复现旧单步行为；默认 0=随机多步长）")
    p_tr.add_argument("--no-tb", action="store_true")
    args = ap.parse_args()
    cfg = Config(batch_size=getattr(args, "batch_size", 2),
                 grad_accum=getattr(args, "grad_accum", 2),
                 lr=getattr(args, "lr", 2.5e-4),
                 num_workers=getattr(args, "num_workers", 4))
    if getattr(args, "image_size", ""):
        cfg.image_size = _parse_size(args.image_size)
    if args.cmd == "prepare":
        prepare_pretext_dataset(cfg, num_windows=args.num_windows,
                                cache_root=Path(args.cache_root),
                                max_logs=args.max_logs)
    elif args.cmd == "prepare-pca":
        prepare_pca(cfg, Path(args.cache_root),
                    Path(args.out) if args.out
                    else Path(args.cache_root) / "pca.pt",
                    num_windows=args.num_windows or cfg.foresight_pca_windows,
                    patches_per_window=(args.patches_per_window
                                        or cfg.foresight_pca_patches))
    else:
        train_pretext(cfg, Path(args.cache_root), Path(args.out_dir),
                      epochs=args.epochs, patience=args.patience,
                      tb=not args.no_tb, limit_batches=args.limit_batches,
                      pca_path=Path(args.pca) if args.pca else None,
                      init_from=Path(args.init_from) if args.init_from else None,
                      horizon=args.horizon or None)
if __name__ == "__main__":
    main()