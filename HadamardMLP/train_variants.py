import os
import sys
import json
import math
import time
import argparse
import contextlib

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from gpt_variants import GPT, make_config, param_breakdown, flops_per_token


def get_batch(split, block_size, batch_size, device, data_dir):
    path = os.path.join(data_dir, f"{split}.bin")
    if split == "train":
        data = np.memmap(path, dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64))
                         for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64))
                         for i in ix])
    else:
        data = np.memmap(path, dtype=np.uint16, mode="r")
        n_blocks = len(data) // block_size
        ix = torch.randint(n_blocks - 1, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i * block_size:(i + 1) * block_size]).astype(np.int64))
                         for i in ix])
        y = torch.stack([torch.from_numpy((data[i * block_size + 1:(i + 1) * block_size + 1]).astype(np.int64))
                         for i in ix])
    if device.startswith("cuda"):
        x, y = x.pin_memory(), y.pin_memory()
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def chunked_ce(logits, targets, chunk=4096):
    V = logits.size(-1)
    logits = logits.reshape(-1, V).float()
    targets = targets.reshape(-1)
    total = targets.numel()
    loss_sum = logits.new_zeros(())
    num = 0
    for i in range(0, total, chunk):
        t = targets[i:i + chunk]
        loss_sum = loss_sum + F.cross_entropy(logits[i:i + chunk], t,
                                              ignore_index=-1, reduction="sum")
        num = num + (t != -1).sum()
    return loss_sum / num.clamp(min=1)


@torch.no_grad()
def estimate_val_loss(model, block_size, batch_size, device, data_dir, split="val"):
    model.eval()
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
    n_blocks = len(data) // block_size
    losses, total, correct1, correct5 = [], 0, 0, 0
    for start in range(0, n_blocks - 1, batch_size):
        bs = min(batch_size, n_blocks - 1 - start)
        ix = np.arange(start, start + bs) * block_size
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)
        with torch.autocast(device, torch.bfloat16) if device.startswith("cuda") else contextlib.nullcontext():
            logits = model(x, y)
        losses.append(chunked_ce(logits, y).item() * (bs * block_size))
        lf = logits.reshape(-1, logits.size(-1))
        yv = y.reshape(-1)
        correct1 += (lf.argmax(-1) == yv).sum().item()
        _, top5 = lf.topk(5, dim=-1)
        correct5 += (top5 == yv[:, None]).sum().item()
        total += bs * block_size
    model.train()
    return sum(losses) / total, correct1 / total, correct5 / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices="AB")
    ap.add_argument("--scale", default="s", choices=["s", "m"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--ckpt", action="store_true", help="activation checkpointing per block")
    ap.add_argument("--out", default=None, help="output dir (default runs/<scale><variant>)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=0, help="cap total steps (for smoke tests)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    meta = json.load(open(os.path.join(args.data, "meta.json"), encoding="utf-8"))
    vocab_size = meta["vocab_size"]
    out_dir = args.out or f"runs/{args.scale}{args.variant}"
    os.makedirs(out_dir, exist_ok=True)

    cfg = make_config(args.scale, args.variant)
    cfg.vocab_size = vocab_size
    cfg.use_ckpt = args.ckpt
    model = GPT(cfg).to(device)
    model.train()

    pb = param_breakdown(model)
    fl = flops_per_token(model, cfg.block_size)
    n_params = pb["total"]
    print(json.dumps({**{k: (round(v, 2) if isinstance(v, float) else v) for k, v in pb.items()},
                      **fl}, ensure_ascii=False))
    print(f"variant={args.variant} scale={args.scale} layers={cfg.n_layer} "
          f"params={n_params/1e6:.2f}M  fwd={fl['forward_flops']/1e6:.1f}M FLOPs/tok")

    train_tokens = meta["train_tokens"]
    steps_per_epoch = max(1, train_tokens // (args.batch * cfg.block_size))
    max_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        max_steps = min(max_steps, args.max_steps)
    print(f"train_tokens={train_tokens:,}  steps/epoch={steps_per_epoch}  max_steps={max_steps}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                              weight_decay=args.weight_decay)
    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        if step >= max_steps:
            return args.min_lr
        p = (step - args.warmup_steps) / max(1, max_steps - args.warmup_steps)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * p))

    best_val = float("inf")
    log_path = os.path.join(out_dir, "log.txt")
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    logf = open(log_path, "a", buffering=1)
    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")

    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
    hist_names = [n for n, p in model.named_parameters()
                  if p.dim() >= 2 and any(k in n for k in
                  ("c_attn", "c_proj", "c_fc",
                   "w1a", "w2a", "w3a", "w1b", "w2b", "w3b",
                   "cond_v", "cond_u", "d1", "d2", "d3", "d4", "b2"))][:14]
    log(f"tensorboard: {os.path.join(out_dir, 'tb')}  hist_tensors={len(hist_names)}")

    t0 = time.time()
    step = 0
    epoch = 0
    while step < max_steps:
        epoch += 1
        epoch_loss, n = 0.0, 0
        for _ in range(min(steps_per_epoch, max_steps - step)):
            lr = lr_at(step)
            for g in optim.param_groups:
                g["lr"] = lr
            x, y = get_batch("train", cfg.block_size, args.batch, device, args.data)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device, torch.bfloat16) if device.startswith("cuda") else contextlib.nullcontext():
                logits = model(x, y)
                loss = chunked_ce(logits, y)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            epoch_loss += loss.item()
            n += 1
            step += 1

            if step % 10 == 0 or step == 1:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", lr, step)
            if step % 50 == 0 or step == 1:
                dt = time.time() - t0
                writer.add_scalar("train/grad_norm", gnorm.item(), step)
                writer.add_scalar("train/tok_per_s", step * args.batch * cfg.block_size / dt, step)
                with torch.no_grad():
                    lf = logits.reshape(-1, logits.size(-1))
                    writer.add_scalar("train/acc_top1",
                                      (lf.argmax(-1) == y.reshape(-1)).float().mean().item(), step)
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        writer.add_scalar(f"grad_norm/{name}", p.grad.detach().norm().item(), step)
                log(f"step {step}/{max_steps} loss {epoch_loss/n:.4f} "
                    f"lr {lr:.2e} grad_norm {gnorm.item():.2f} "
                    f"{step*args.batch*cfg.block_size/dt:.0f} tok/s")
            if step % 200 == 0:
                for name in hist_names:
                    p = dict(model.named_parameters())[name]
                    writer.add_histogram(f"grad_hist/{name}", p.grad.detach(), step)
                    writer.add_histogram(f"weight_hist/{name}", p.detach(), step)
        val_loss, val_acc1, val_acc5 = estimate_val_loss(model, cfg.block_size, 32, device, args.data)
        dt = time.time() - t0
        writer.add_scalar("val/loss", val_loss, step)
        writer.add_scalar("val/ppl", math.exp(val_loss), step)
        writer.add_scalar("val/acc_top1", val_acc1, step)
        writer.add_scalar("val/acc_top5", val_acc5, step)
        log(f"epoch {epoch} train_loss {epoch_loss/n:.4f} "
            f"val_loss {val_loss:.4f} "
            f"val_ppl {math.exp(val_loss):.1f} val_acc1 {val_acc1:.4f} val_acc5 {val_acc5:.4f} "
            f"elapsed {dt/60:.1f}min")
        rec = {"epoch": epoch, "step": step, "tokens": step * args.batch * cfg.block_size,
               "train_loss": round(epoch_loss / n, 4),
               "val_loss": round(val_loss, 4),
               "val_ppl": round(math.exp(val_loss), 2),
               "val_acc_top1": round(val_acc1, 4), "val_acc_top5": round(val_acc5, 4),
               "lr": lr, "elapsed_s": round(dt, 1)}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        writer.flush()
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch,
                        "step": step, "val_loss": val_loss},
                       os.path.join(out_dir, "best.pt"))

    torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch,
                "step": step, "val_loss": val_loss},
               os.path.join(out_dir, "last.pt"))

    ckpt_path = os.path.join(out_dir, "best.pt")
    ckpt = (torch.load(ckpt_path, map_location=device, weights_only=False)
            if os.path.exists(ckpt_path) else None)
    test_loss, test_ppl, test_acc1, test_acc5 = None, None, None, None
    if ckpt is not None and os.path.exists(os.path.join(args.data, "test.bin")):
        model.load_state_dict(ckpt["model"])
        test_loss, test_acc1, test_acc5 = estimate_val_loss(model, cfg.block_size, 32,
                                                            device, args.data, "test")
        test_ppl = math.exp(test_loss)
        writer.add_scalar("test/loss", test_loss, ckpt["step"])
        writer.add_scalar("test/ppl", test_ppl, ckpt["step"])
        writer.add_scalar("test/acc_top1", test_acc1, ckpt["step"])
        writer.add_scalar("test/acc_top5", test_acc5, ckpt["step"])
        log(f"test_loss {test_loss:.4f} test_ppl {test_ppl:.1f} "
            f"test_acc1 {test_acc1:.4f} test_acc5 {test_acc5:.4f} (best.pt @ epoch {ckpt['epoch']})")
    writer.close()

    summary = {
        "variant": args.variant, "scale": args.scale,
        "config": {k: v for k, v in cfg.__dict__.items()},
        "params": {k: int(v) if isinstance(v, int) else v for k, v in pb.items()},
        "flops_per_token": fl,
        "final_train_loss": round(epoch_loss / n, 4),
        "best_val_loss": round(best_val, 4),
        "best_val_ppl": round(math.exp(best_val), 2),
        "test_loss": round(test_loss, 4) if test_loss is not None else None,
        "test_ppl": round(test_ppl, 2) if test_ppl is not None else None,
        "test_acc_top1": round(test_acc1, 4) if test_acc1 is not None else None,
        "test_acc_top5": round(test_acc5, 4) if test_acc5 is not None else None,
        "best_epoch": ckpt["epoch"] if ckpt is not None else epoch,
        "total_steps": step, "total_tokens": step * args.batch * cfg.block_size,
        "wall_seconds": round(time.time() - t0, 1),
        "avg_tok_per_s": round(step * args.batch * cfg.block_size / (time.time() - t0), 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"done. best_val_loss={best_val:.4f} best_ppl={math.exp(best_val):.1f}")
    logf.close()


if __name__ == "__main__":
    main()
