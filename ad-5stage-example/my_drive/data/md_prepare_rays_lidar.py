"""从 nuPlan 的 MergedPointCloud 生成占据射线缓存。

每个 cache 样本读取其锚点帧的 ``lidar_path``，把 LiDAR 点变换到当前 ego 系，
随机降采样后写成 ``<cache_root>/rays/<token>.npz``（origins/points，float32）。
这是 OpenScene 连续占据 GT 缺失时的替代监督：表面点 + 传感器原点即一条射线，
``md_occupancy.sample_occupancy_queries`` 会在射线上采正/负查询。

用法：
  python -m my_drive.data.md_prepare_rays_lidar \
      --cache-root /media/.../md_work/cache \
      --dataset-root /media/.../dataset/sensor_blobs/trainval \
      --workers 4 --max-rays 20000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from .md_data import load_log_frames


def _pcd_xyz(path: Path) -> np.ndarray:
    """读取 PCD 的 x/y/z（支持 ascii / binary；binary_compressed 返回空）。"""
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline()
            if not line:
                return np.zeros((0, 3), dtype=np.float32)
            text = line.decode("ascii", errors="ignore").strip()
            if not text or text.startswith("#"):
                continue
            key, _, val = text.partition(" ")
            header[key.upper()] = val
            if key.upper() == "DATA":
                data_kind = val.strip().lower()
                break
        fields = header.get("FIELDS", "x y z").split()
        sizes = [int(x) for x in header.get("SIZE", "4 4 4").split()]
        types = header.get("TYPE", "F F F").split()
        counts = [int(x) for x in header.get("COUNT", " ".join(["1"] * len(fields))).split()]
        n = int(header.get("POINTS", header.get("WIDTH", "0")))
        if data_kind == "ascii":
            raw = np.loadtxt(f, dtype=np.float64, max_rows=n)
            if raw.ndim == 1:
                raw = raw[None]
            cols = {}
            off = 0
            for name, cnt in zip(fields, counts):
                cols[name] = raw[:, off:off + cnt]
                off += cnt
            return np.stack([cols["x"][:, 0], cols["y"][:, 0], cols["z"][:, 0]],
                            axis=1).astype(np.float32)
        if data_kind != "binary":
            return np.zeros((0, 3), dtype=np.float32)
        dtype = []
        for name, size, typ, cnt in zip(fields, sizes, types, counts):
            np_t = {"F": "f", "U": "u", "I": "i"}.get(typ.upper())
            if np_t is None:
                return np.zeros((0, 3), dtype=np.float32)
            dtype.append((name, f"<{np_t}{size}", (cnt,)) if cnt > 1
                         else (name, f"<{np_t}{size}"))
        buf = np.frombuffer(f.read(n * np.dtype(dtype).itemsize), dtype=dtype)
        if not len(buf):
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack([buf["x"], buf["y"], buf["z"]], axis=1).astype(np.float32)


def _transform(points: np.ndarray, T_ego_lidar: np.ndarray) -> np.ndarray:
    """LiDAR 点 → 当前 ego 系。"""
    p = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return (p @ T_ego_lidar.T)[:, :3]


def _process_log(task):
    log_name, entries, args = task
    cache_root = Path(args["cache_root"])
    dataset_root = Path(args["dataset_root"])
    ray_root = cache_root / "rays"
    ray_root.mkdir(parents=True, exist_ok=True)
    frames = load_log_frames(log_name)
    stats = {"written": 0, "missing": 0, "empty": 0, "points": 0}
    rng = np.random.default_rng(20261101)
    for token, anchor in entries:
        dst = ray_root / f"{token}.npz"
        if dst.is_file() and not args["overwrite"]:
            stats["written"] += 1
            continue
        try:
            frame = frames[anchor]
            raw_path = str(frame.get("lidar_path", ""))
            if not raw_path:
                stats["missing"] += 1
                continue
            # cache 里是相对 sensor_blobs/trainval 的路径；也兼容已带前缀的路径
            p = Path(raw_path)
            if not p.is_absolute():
                p = dataset_root / p
            if not p.is_file():
                stats["missing"] += 1
                continue
            pts = _pcd_xyz(p)
            if len(pts) < 64:
                stats["empty"] += 1
                continue
            pts = pts[np.isfinite(pts).all(axis=1)]
            r = np.linalg.norm(pts[:, :2], axis=1)
            pts = pts[(r > 1.0) & (r < 90.0) & (pts[:, 2] > -3.0) & (pts[:, 2] < 8.0)]
            if len(pts) < 64:
                stats["empty"] += 1
                continue
            if len(pts) > args["max_rays"]:
                idx = rng.choice(len(pts), args["max_rays"], replace=False)
                pts = pts[idx]
            # 当前帧 ego 系：T_ego_lidar = inv(ego2global) @ lidar2global
            ego2global = np.asarray(frame["ego2global"], dtype=np.float32)
            lidar2global = np.asarray(frame.get("lidar2global", ego2global),
                                      dtype=np.float32)
            T_ego_lidar = np.linalg.inv(ego2global) @ lidar2global
            pts = _transform(pts, T_ego_lidar)
            origin = T_ego_lidar[:3, 3].astype(np.float32)
            origins = np.broadcast_to(origin, pts.shape).copy()
            tmp = dst.with_suffix(f".{os.getpid()}.tmp.npz")
            np.savez_compressed(tmp, origins=origins, points=pts.astype(np.float32))
            os.replace(tmp, dst)
            stats["written"] += 1
            stats["points"] += len(pts)
        except Exception:
            stats["empty"] += 1
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--dataset-root", required=True,
                    help=".../dataset/sensor_blobs/trainval")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-rays", type=int, default=20000)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cache_root = Path(args.cache_root)
    index = json.loads((cache_root / "index.json").read_text())
    want = set(args.splits.split(","))
    groups = {}
    for e in index:
        if e["split"] not in want:
            continue
        groups.setdefault(e["log_name"], []).append((e["token"], e["anchor"]))
    tasks = [(log, ents, vars(args)) for log, ents in groups.items()]
    print(f"rays: {sum(len(v) for v in groups.values())} samples / {len(tasks)} logs",
          flush=True)
    from multiprocessing import Pool
    total = {"written": 0, "missing": 0, "empty": 0, "points": 0}
    with Pool(args.workers) as pool:
        for i, st in enumerate(pool.imap_unordered(_process_log, tasks), 1):
            for k in total:
                total[k] += st[k]
            if i % 20 == 0:
                print(f"  logs {i}/{len(tasks)} samples {total['written']} "
                      f"missing {total['missing']} empty {total['empty']} "
                      f"points {total['points']/1e6:.1f}M", flush=True)
    print("done", total, flush=True)


if __name__ == "__main__":
    sys.exit(main())
