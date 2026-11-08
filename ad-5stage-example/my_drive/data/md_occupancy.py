"""当前帧连续占据监督：射线缓存校验、导入与在线查询采样。

射线文件格式（每个样本一个 ``<token>.npz``）：
  * origins / ray_origins：float32 (N,3)，传感器原点；
  * points  / ray_points ：float32 (N,3)，射线观测表面点；
  * confidence（可选）：float32 (N,)，当前版本仅校验并保留。

所有坐标必须已经位于当前帧 ego 系（x 前、y 左、z 上）。本模块不猜测
不同数据集的 LiDAR/深度字段；请先用对应数据集标定完成坐标变换。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from ..md_common import Config


def _ray_arrays(data: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, np.ndarray]:
    origins = data["origins"] if "origins" in data else data["ray_origins"]
    points = data["points"] if "points" in data else data["ray_points"]
    return (np.asarray(origins, dtype=np.float32),
            np.asarray(points, dtype=np.float32))


def validate_rays(origins: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """校验并过滤非有限、形状错误或长度过短的射线。"""
    origins = np.asarray(origins, dtype=np.float32).reshape(-1, 3)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if origins.shape != points.shape:
        raise ValueError(f"origins/points 形状不一致：{origins.shape} vs {points.shape}")
    if len(origins) == 0:
        return origins, points
    valid = (np.isfinite(origins).all(axis=1) & np.isfinite(points).all(axis=1)
             & (np.linalg.norm(points - origins, axis=1) > 1e-3))
    return origins[valid], points[valid]


def import_ray_cache(source_root: Path, cache_root: Path,
                     tokens: Iterable[str]) -> Dict[str, int]:
    """将通用 npz 射线文件导入 ``cache_root/rays``，返回导入统计。

    source_root 中文件名必须为 ``<token>.npz``。缺失文件只计数、不报错，
    因而允许只在部分日志上启用占据监督。
    """
    source_root = Path(source_root)
    ray_root = Path(cache_root) / "rays"
    ray_root.mkdir(parents=True, exist_ok=True)
    stats = {"imported": 0, "missing": 0, "invalid": 0, "points": 0}
    for token in tokens:
        src = source_root / f"{token}.npz"
        if not src.is_file():
            stats["missing"] += 1
            continue
        try:
            with np.load(src) as data:
                origins, points = validate_rays(*_ray_arrays(data))
            if len(points) == 0:
                stats["invalid"] += 1
                continue
            dst = ray_root / f"{token}.npz"
            tmp = dst.with_suffix(f".{os.getpid()}.tmp.npz")
            np.savez_compressed(tmp, origins=origins, points=points)
            os.replace(tmp, dst)
            stats["imported"] += 1
            stats["points"] += len(points)
        except (OSError, ValueError, KeyError):
            stats["invalid"] += 1
    return stats


def load_sample_rays(cache_root: Path, token: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """读取独立射线缓存；不存在时返回 None。"""
    path = Path(cache_root) / "rays" / f"{token}.npz"
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            return validate_rays(*_ray_arrays(data))
    except (OSError, ValueError, KeyError):
        return None


def _choose_indices(rng: np.random.RandomState, near: np.ndarray, count: int,
                    near_ratio: float) -> np.ndarray:
    """按近场配额有放回采样，同时避免近/远场一侧为空时崩溃。"""
    n = len(near)
    near_idx = np.flatnonzero(near)
    far_idx = np.flatnonzero(~near)
    n_near = int(round(count * near_ratio)) if len(near_idx) else 0
    n_near = min(n_near, count)
    n_far = count - n_near
    picked = []
    if n_near:
        picked.append(rng.choice(near_idx, n_near, replace=True))
    if n_far:
        source = far_idx if len(far_idx) else near_idx
        picked.append(rng.choice(source, n_far, replace=True))
    if not picked:  # 近场配额为 0 且远场为空等边界情况
        return rng.choice(n, count, replace=True)
    return np.concatenate(picked)


def sample_occupancy_queries(origins: Optional[np.ndarray], points: Optional[np.ndarray],
                             cfg: Config, rng: np.random.RandomState,
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从当前帧射线生成固定数量的连续查询、标签和有效掩码。

    返回：
      queries (Q,3)：一半表面正样本、一半射线自由空间负样本；
      labels  (Q,)：正样本 1、负样本 0；
      valid   (Q,)：无射线或超出当前 BEV/高度范围时为 False。

    无射线样本也返回固定形状，保证 DataLoader 可与其他样本任意混合。
    """
    q_num = int(cfg.occ_num_queries)
    queries = np.zeros((q_num, 3), dtype=np.float32)
    labels = np.zeros(q_num, dtype=np.float32)
    valid = np.zeros(q_num, dtype=bool)
    if q_num == 0 or origins is None or points is None or len(points) == 0:
        return queries, labels, valid

    origins, points = validate_rays(origins, points)
    if len(points) == 0:
        return queries, labels, valid
    near = ((points[:, 0] >= cfg.occ_near_x_min)
            & (points[:, 0] <= cfg.occ_near_x_max)
            & (points[:, 1] >= cfg.occ_near_y_min)
            & (points[:, 1] <= cfg.occ_near_y_max))
    n_pos = q_num // 2
    n_neg = q_num - n_pos
    pos_idx = _choose_indices(rng, near, n_pos, cfg.occ_near_ratio) if n_pos else np.array([], int)
    neg_idx = _choose_indices(rng, near, n_neg, cfg.occ_near_ratio) if n_neg else np.array([], int)

    if n_pos:
        direction = points[pos_idx] - origins[pos_idx]
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        delta = rng.uniform(0.0, cfg.occ_delta, size=(n_pos, 1)).astype(np.float32)
        queries[:n_pos] = points[pos_idx] + delta * direction
        labels[:n_pos] = 1.0
    if n_neg:
        ratio = rng.uniform(0.05, 0.95, size=(n_neg, 1)).astype(np.float32)
        queries[n_pos:] = origins[neg_idx] + ratio * (points[neg_idx] - origins[neg_idx])

    in_range = ((queries[:, 0] >= cfg.pc_x_min) & (queries[:, 0] <= cfg.pc_x_max)
                & (queries[:, 1] >= cfg.pc_y_min) & (queries[:, 1] <= cfg.pc_y_max)
                & (queries[:, 2] >= cfg.occ_z_min) & (queries[:, 2] <= cfg.occ_z_max))
    # 正负样本都来自同一批射线端点，不存在「无来源」查询 → valid 就是 in_range
    valid = in_range
    order = rng.permutation(q_num)
    return queries[order], labels[order], valid[order]