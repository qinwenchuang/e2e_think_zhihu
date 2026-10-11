"""my_drive 数据管线：纯 pickle 场景加载（无 navsim 依赖）→ 本地自包含缓存 → Dataset。

数据来源（只读）：
  * navsim_logs/trainval/*.pkl —— 每日志一个 pickle，内容为「帧 dict 列表」；
  * sensor_blobs/trainval/<log>/<CAM>/<frame>.jpg —— 相机图像；
  * 地图：nuplan-devkit 地图 API（$NUPLAN_MAPS_ROOT）。

缓存产物（$MD_WORK_ROOT/cache/，构建一次后训练不碰只读盘）：
  * index.json            样本清单（token、日志、anchor 帧下标、split）
  * images/<token>/*.jpg  每样本 7 张图（历史 3×前视 t3/t2/t1 + 当前 4 路）
  * labels/<token>.npz    轨迹 / ego 状态 / 相机参数 / 检测框 / BEV 地图标签 / 未来帧障碍物
"""
from __future__ import annotations
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
from ..md_common import (
    CAMS_CURRENT, CAM_HISTORY, HISTORY_INTERVAL, MAPS_ROOT, NAVSIM_LOGS,
    NUM_FUTURE_POSES, NUM_HISTORY, SENSOR_BLOBS, Config,
    absolute_to_relative_se2, cam_extrinsic_ego2cam, quaternion_yaw,
)
from .md_bev import DET_NAME_TO_CLASS, MapLabelCache, build_det_targets, build_map_label


def list_logs() -> List[str]:
    return sorted(p.name[:-4] for p in NAVSIM_LOGS.glob("*.pkl"))


def load_log_frames(log_name: str) -> List[dict]:
    """读一个日志的全部帧（帧 dict + numpy，可直接 pickle.load）。"""
    with open(NAVSIM_LOGS / f"{log_name}.pkl", "rb") as f:
        return pickle.load(f)


def frame_global_pose(frame: dict) -> np.ndarray:
    """帧 dict → 全局 ego 位姿 (x, y, yaw)。"""
    t = frame["ego2global_translation"]
    return np.array([t[0], t[1], quaternion_yaw(frame["ego2global_rotation"])], dtype=np.float64)


def _image_path(log_name: str, frame: dict, cam: str) -> Path:
    return SENSOR_BLOBS / log_name / cam / Path(frame["cams"][cam]["data_path"]).name


def frame_has_image(log_name: str, frame: dict, cam: str) -> bool:
    return _image_path(log_name, frame, cam).is_file()


def find_anchors(log_name: str, frames: Optional[List[dict]] = None,
                 verbose: bool = False) -> List[int]:
    """在日志里找合法 anchor i：i-3..i 四帧窗口连续（0.5s±0.25s）、图像齐全、
    未来 8 帧时间连续且当前帧有路线（roadblock_ids 非空）。返回帧下标列表。"""
    if frames is None:
        frames = load_log_frames(log_name)
    n = len(frames)
    if n < NUM_HISTORY + NUM_FUTURE_POSES:
        return []
    # 历史窗口用前视图，当前帧需要 4 路相机齐全
    has_img = np.zeros(n, dtype=bool)
    has_img_cur = np.zeros(n, dtype=bool)
    for i, fr in enumerate(frames):
        has_img[i] = frame_has_image(log_name, fr, CAM_HISTORY)
        has_img_cur[i] = all(frame_has_image(log_name, fr, c) for c in CAMS_CURRENT)
    anchors = []
    for i in range(NUM_HISTORY - 1, n - NUM_FUTURE_POSES):
        if not (has_img[i - NUM_HISTORY + 1: i + 1].all() and has_img_cur[i]):
            continue
        # 历史/未来帧时间连续（允许 ±0.25s 浮动）
        ts = [frames[j]["timestamp"] for j in range(i - NUM_HISTORY + 1, i + 1)]
        dts = np.diff(ts) / 1e6
        if not np.all(np.abs(dts - HISTORY_INTERVAL) < 0.25):
            continue
        fts = [frames[j]["timestamp"] for j in range(i, i + NUM_FUTURE_POSES + 1)]
        if not np.all(np.abs(np.diff(fts) / 1e6 - HISTORY_INTERVAL) < 0.25):
            continue
        if len(frames[i]["roadblock_ids"]) == 0:
            continue
        anchors.append(i)
    if verbose:
        print(f"  {log_name}: {len(frames)} 帧 → {len(anchors)} 个 anchor")
    return anchors


_MAP_APIS: Dict[str, object] = {}   # 每张地图只加载一次 API（进程级缓存）


def _get_map_api(map_name: str):
    if map_name not in _MAP_APIS:
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
        _MAP_APIS[map_name] = get_maps_api(str(MAPS_ROOT), "nuplan-maps-v1.0", map_name)
    return _MAP_APIS[map_name]
def build_sample(log_name: str, frames: List[dict], anchor: int, cfg: Config,
                 cache_root: Path, map_cache: MapLabelCache,
                 split: str = "train") -> Optional[dict]:
    """构建单个样本：复制 7 张图 + 写 labels npz。已存在则直接返回索引条目。"""
    token = str(frames[anchor]["token"])
    img_dir = cache_root / "images" / token
    label_path = cache_root / "labels" / f"{token}.npz"
    if label_path.is_file() and img_dir.is_dir() and len(list(img_dir.glob("*.jpg"))) >= 7:
        return {"token": token, "log_name": log_name, "anchor": anchor, "split": split}
    cur = frames[anchor]
    ego_now = frame_global_pose(cur)
    # ---- 时序帧 ego 状态：(4, 11) = Δpose3 + vel2 + acc2 + cmd4 ----
    ego_poses_global = np.stack([frame_global_pose(frames[j])
                                 for j in range(anchor - NUM_HISTORY + 1, anchor + 1)])
    ego_poses_local = absolute_to_relative_se2(ego_now, ego_poses_global)
    frame_dt = np.array([(frames[j]["timestamp"] - cur["timestamp"]) / 1e6
                         for j in range(anchor - NUM_HISTORY + 1, anchor + 1)],
                        dtype=np.float32)
    ego_status = []
    for j in range(anchor - NUM_HISTORY + 1, anchor + 1):
        fr = frames[j]
        dyn = np.asarray(fr["ego_dynamic_state"], dtype=np.float32)
        cmd = np.asarray(fr["driving_command"], dtype=np.float32)
        ego_status.append(np.concatenate([ego_poses_local[j - anchor + NUM_HISTORY - 1],
                                          dyn, cmd]).astype(np.float32))
    ego_status = np.stack(ego_status)
    # ---- 未来轨迹（当前 ego 系，8 步） ----
    fut_global = np.stack([frame_global_pose(frames[anchor + k])
                           for k in range(1, NUM_FUTURE_POSES + 1)])
    traj = absolute_to_relative_se2(ego_now, fut_global).astype(np.float32)
    # ---- 当前帧相机参数：内参（原图口径，加载时按 resize 缩放）+ ego→cam 外参 ----
    cam_K, cam_e2c = [], []
    img0 = None
    for cam in CAMS_CURRENT:
        entry = cur["cams"][cam]
        cam_K.append(np.asarray(entry["cam_intrinsic"], dtype=np.float64))
        cam_e2c.append(cam_extrinsic_ego2cam(entry))
        if img0 is None:
            p = _image_path(log_name, cur, cam)
            with Image.open(p) as im:
                img0 = im.size
    cam_K = np.stack(cam_K).astype(np.float32)
    cam_e2c = np.stack(cam_e2c).astype(np.float32)
    # ---- 当前帧检测框（自车系 (x,y,z,l,w,h,yaw) + 类别 id） ----
    anns = cur["anns"]
    boxes = np.asarray(anns["gt_boxes"], dtype=np.float32).reshape(-1, 7)
    box_cls = np.array([DET_NAME_TO_CLASS.get(str(nm), -1) for nm in anns["gt_names"]],
                       dtype=np.int64)
    # ---- 未来帧障碍物真值（GRPO 碰撞 reward 用） ----
    # 模型轨迹第 k 步对应帧 anchor+k 的 anns；该帧框在其自身 ego 系，
    # 两步变换到当前 ego 系（局部→全局→当前 ego），yaw 加 Δyaw，pad 到 (8,256,7)。
    future_box_max = 256
    future_boxes = np.zeros((NUM_FUTURE_POSES, future_box_max, 7), dtype=np.float32)
    future_valid = np.zeros((NUM_FUTURE_POSES, future_box_max), dtype=np.uint8)
    for k in range(1, NUM_FUTURE_POSES + 1):
        fr = frames[anchor + k]
        fboxes = np.asarray(fr["anns"]["gt_boxes"], dtype=np.float32).reshape(-1, 7)
        pose_k = frame_global_pose(fr)
        c, s = np.cos(pose_k[2]), np.sin(pose_k[2])
        lx, ly = fboxes[:, 0], fboxes[:, 1]
        gx = pose_k[0] + c * lx - s * ly
        gy = pose_k[1] + s * lx + c * ly
        cn, sn = np.cos(ego_now[2]), np.sin(ego_now[2])
        dx, dy = gx - ego_now[0], gy - ego_now[1]
        fb = fboxes.copy()
        fb[:, 0] = cn * dx + sn * dy
        fb[:, 1] = -sn * dx + cn * dy
        fb[:, 6] = fb[:, 6] + (pose_k[2] - ego_now[2])
        m = min(len(fb), future_box_max)
        future_boxes[k - 1, :m] = fb[:m]
        future_valid[k - 1, :m] = 1
    # ---- BEV 地图标签（逐 token 磁盘缓存） ----
    bev = map_cache.get(log_name, token)
    if bev is None:
        map_api = _get_map_api(str(cur["map_location"]))
        bev = build_map_label(map_api, ego_now, cfg)
        map_cache.put(log_name, token, bev)
    # ---- 复制图像：历史 t3/t2/t1（前视）+ 当前 4 路 ----
    img_dir.mkdir(parents=True, exist_ok=True)
    copy_jobs = [(anchor - k, CAM_HISTORY, f"t{k}") for k in range(NUM_HISTORY - 1, 0, -1)]
    copy_jobs += [(anchor, cam, cam) for cam in CAMS_CURRENT]
    for fr_idx, cam, tag in copy_jobs:
        dst = img_dir / f"{tag}.jpg"
        if not dst.is_file():
            shutil.copyfile(_image_path(log_name, frames[fr_idx], cam), dst)
    # ---- 原子写 npz（tmp + rename，避免并发/中断损坏） ----
    label_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = label_path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(
        tmp,
        traj=traj, ego_status=ego_status, ego_poses_local=ego_poses_local.astype(np.float32),
        frame_dt=frame_dt,
        cam_K=cam_K, cam_ego2cam=cam_e2c, img_size_orig=np.array(img0, dtype=np.int64),
        boxes=boxes, box_cls=box_cls, bev_map=bev.astype(np.uint8),
        future_boxes=future_boxes, future_valid=future_valid,
    )
    os.replace(tmp, label_path)
    return {"token": token, "log_name": log_name, "anchor": anchor, "split": split}


def prepare_dataset(cfg: Config, split_counts: Dict[str, int], cache_root: Path,
                    max_logs: int = 2000, log_names: Optional[List[str]] = None,
                    ) -> List[dict]:
    """扫描日志 → 按「日志级不重叠」划分 train/val/test → 构建样本缓存。

    同一日志的所有 anchor 只进入一个 split（避免连续帧窗口跨 split 泄漏）；
    日志顺序先打散，保证子集的场景多样性；split 顺序分配（train→val→test）。
    """
    cache_root = Path(cache_root)
    (cache_root / "labels").mkdir(parents=True, exist_ok=True)
    map_cache = MapLabelCache(cache_root / "map_label_cache")
    logs = log_names if log_names else list_logs()
    rng = np.random.RandomState(cfg.seed)
    if log_names is None:
        logs = list(logs)
        rng.shuffle(logs)
    targets = dict(split_counts)
    index: List[dict] = []
    scanned = 0
    for log_name in logs:
        if scanned >= max_logs or all(v <= 0 for v in targets.values()):
            break
        split = next((s for s in ("train", "val", "test") if targets.get(s, 0) > 0), None)
        if split is None:
            break
        try:
            frames = load_log_frames(log_name)
        except Exception as exc:
            print(f"  !! {log_name} 读取失败: {exc}")
            continue
        scanned += 1
        anchors = find_anchors(log_name, frames)
        if not anchors:
            continue
        anchors = list(anchors)
        rng.shuffle(anchors)
        print(f"[{scanned}] {log_name}: {len(anchors)} anchors → {split}"
              f"（{ {s: len([e for e in index if e['split'] == s]) for s in targets} }）",
              flush=True)
        for anchor in anchors:
            if targets[split] <= 0:
                break
            try:
                entry = build_sample(log_name, frames, anchor, cfg, cache_root,
                                     map_cache, split=split)
            except Exception as exc:
                print(f"  !! 样本构建失败 {log_name}@{anchor}: {exc}")
                continue
            if entry is not None:
                index.append(entry)
                targets[split] -= 1
    with open(cache_root / "index.json", "w") as f:
        json.dump(index, f, indent=1)
    summary = {s: len([e for e in index if e["split"] == s]) for s in split_counts}
    print(f"准备完成：{summary} → {cache_root}")
    return index
def _color_jitter(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """0-1 图像上的亮度/对比度/饱和度抖动（仅 train 用）。"""
    img = img * rng.uniform(0.75, 1.25)
    m = img.mean(axis=(0, 1), keepdims=True)
    img = (img - m) * rng.uniform(0.75, 1.25) + m
    g = img.mean(axis=2, keepdims=True)
    img = (img - g) * rng.uniform(0.75, 1.25) + g
    return np.clip(img, 0, 1).astype(np.float32)


def _grid_mask(img: np.ndarray, rng: np.random.RandomState,
               prob: float = 0.5, ratio: float = 0.5) -> np.ndarray:
    """简化 GridMask：随机网格单元挖洞（填 0）。"""
    if rng.rand() > prob:
        return img
    h, w = img.shape[:2]
    unit = rng.randint(48, 97)
    ox, oy = rng.randint(0, unit), rng.randint(0, unit)
    mask = np.ones((h, w, 1), dtype=np.float32)
    for y in range(oy, h, unit):
        for x in range(ox, w, unit):
            if rng.rand() < ratio:
                mask[y:y + unit, x:x + unit] = 0.0
    return (img * mask).astype(np.float32)


def find_pretext_windows(log_name: str, frames: Optional[List[dict]] = None) -> List[int]:
    """pretext 无标签窗口：返回所有合法「当前帧」下标 j（窗口 = j-3..j+4，8 帧连续前视）。

    8 帧 = 4 帧历史（j-3..j，输入）+ 4 帧未来候选目标（j+1..j+4，对应 0.5/1.0/1.5/2.0s），
    训练时随机采时距 k∈[1, foresight_max_step]（DINO-Foresight 式多步长监督）。
    """
    if frames is None:
        frames = load_log_frames(log_name)
    n = len(frames)
    if n < 8:
        return []
    has_img = np.zeros(n, dtype=bool)
    for i, fr in enumerate(frames):
        has_img[i] = frame_has_image(log_name, fr, CAM_HISTORY)
    out = []
    for j in range(3, n - 4):
        if not has_img[j - 3: j + 5].all():
            continue
        ts = [frames[k]["timestamp"] for k in range(j - 3, j + 5)]
        if not np.all(np.abs(np.diff(ts) / 1e6 - HISTORY_INTERVAL) < 0.25):
            continue
        out.append(j)
    return out


def prepare_pretext_dataset(cfg: Config, num_windows: int, cache_root: Path,
                            val_fraction: float = 0.05, max_logs: int = 2000,
                            ) -> List[dict]:
    """扫描日志 → 收集 8 帧窗口 → 复制唯一前视图像 → 写 index.json（日志级划分）。

    窗口定义：历史 j-3..j（4 帧输入），未来候选目标 j+1..j+4（多步长，训练时随机采 k）；
    存相对当前帧 j 的 Δt/Δpose。与旧 5 帧缓存不兼容，需重新 prepare。
    """
    cache_root = Path(cache_root)
    (cache_root / "frames").mkdir(parents=True, exist_ok=True)
    logs = list_logs()
    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(logs)
    index: List[dict] = []
    val_target = max(1, int(num_windows * val_fraction))
    n_train, n_val = 0, 0
    scanned = 0
    for log_name in logs:
        if n_train >= num_windows and n_val >= val_target:
            break
        if scanned >= max_logs:
            break
        try:
            frames = load_log_frames(log_name)
        except Exception:  # noqa: BLE001
            continue
        scanned += 1
        windows = find_pretext_windows(log_name, frames)
        if not windows:
            continue
        for j in windows:
            if n_train < num_windows:
                split = "train"
            elif n_val < val_target:
                split = "val"
            else:
                break
            # 复制 8 帧图像（按帧文件名去重，同一物理图可被多个窗口共享）
            ok = True
            for k in range(j - 3, j + 5):
                src = _image_path(log_name, frames[k], CAM_HISTORY)
                dst = cache_root / "frames" / src.name
                if not dst.is_file():
                    try:
                        shutil.copyfile(src, dst)
                    except Exception:  # noqa: BLE001
                        ok = False
                        break
            if not ok:
                continue
            # 相对当前帧 j 的位姿 / 时间差 / ego 动态
            ego3 = frame_global_pose(frames[j])
            poses8 = absolute_to_relative_se2(
                ego3, np.stack([frame_global_pose(frames[k]) for k in range(j - 3, j + 5)]))
            dt8 = np.array([(frames[k]["timestamp"] - frames[j]["timestamp"]) / 1e6
                            for k in range(j - 3, j + 5)], dtype=np.float32)
            dyn = np.asarray(frames[j]["ego_dynamic_state"], dtype=np.float32)
            cmd = np.asarray(frames[j]["driving_command"], dtype=np.float32)
            ego8 = np.concatenate([dyn, cmd]).astype(np.float32)[:8]
            tags = [_image_path(log_name, frames[k], CAM_HISTORY).name
                    for k in range(j - 3, j + 5)]
            index.append({"log_name": log_name, "anchor": int(j), "tags8": tags,
                          "dt8": [float(v) for v in dt8],
                          "poses8": [[float(x) for x in p] for p in poses8],
                          "ego8": [float(v) for v in ego8],
                          "split": split})
            if split == "train":
                n_train += 1
            else:
                n_val += 1
        if scanned % 50 == 0:
            print(f"[{scanned}] 累计窗口 {len(index)}", flush=True)
    # 边界日志的 val 窗口若与 train 共享日志，全部归回 train（相邻帧窗口跨 split 即泄漏）
    train_logs = {e["log_name"] for e in index if e["split"] == "train"}
    n_moved = 0
    for e in index:
        if e["split"] == "val" and e["log_name"] in train_logs:
            e["split"] = "train"
            n_moved += 1
    n_train = sum(1 for e in index if e["split"] == "train")
    n_val = len(index) - n_train
    with open(cache_root / "index.json", "w") as f:
        json.dump(index, f)
    print(f"pretext 准备完成：train {n_train} / val {n_val}（边界日志回迁 {n_moved} 窗口）→ {cache_root}")
    return index


class PretextDataset:
    """8 帧前视窗口（stage 0 未来特征预测用，多步长版）。

    返回 imgs (8,3,H,W)、dt8 (8,)、motion8 (8,3)（均相对当前帧 f3）、ego8 (8,)。
    历史 = f0..f3；目标 = f[3+k]，k∈[1, cfg.foresight_max_step] 由训练循环采样。
    """

    def __init__(self, cache_root: Path, cfg: Config, split: str = "train",
                 augment: bool = False):
        self.cache_root = Path(cache_root)
        self.cfg = cfg
        self.augment = augment
        self._rng = np.random.RandomState(os.getpid() & 0xFFFF)
        with open(self.cache_root / "index.json") as f:
            index = json.load(f)
        self.entries = [e for e in index if e["split"] == split]
        if not self.entries:
            raise RuntimeError(f"pretext 数据为空：{cache_root} split={split}")
    def __len__(self) -> int:
        return len(self.entries)
    def _load_image(self, name: str) -> np.ndarray:
        w, h = self.cfg.image_size
        with Image.open(self.cache_root / "frames" / name) as im:
            im = im.convert("RGB").resize((w, h), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        if self.augment:
            arr = _color_jitter(arr, self._rng)
            arr = _grid_mask(arr, self._rng)
        # ImageNet 归一化（DINOv3 预训练口径）
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return ((arr - mean) / std).transpose(2, 0, 1)

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        import torch
        e = self.entries[idx]
        imgs = np.stack([self._load_image(t) for t in e["tags8"]])      # (8,3,H,W)
        return {
            "imgs": torch.from_numpy(imgs),                              # f0..f3 输入 + f4..f7 候选目标
            "dt8": torch.tensor(e["dt8"], dtype=torch.float32),          # (8,)
            "motion8": torch.tensor(e["poses8"], dtype=torch.float32),   # (8,3) 相对当前帧 Δpose
            "ego8": torch.tensor(e["ego8"], dtype=torch.float32),        # (8,)
        }


class OverfitDataset:
    """读取本地缓存的自包含有标签数据集（训练时不触碰 navsim / 只读数据盘）。

    :param split: "train" / "val" / "test" / None（全部）
    :param augment: 是否开启数据增强（仅 train 用）
    """

    def __init__(self, cache_root: Path, cfg: Config, split: Optional[str] = None,
                 tokens: Optional[List[str]] = None, augment: bool = False):
        self.cache_root = Path(cache_root)
        self.cfg = cfg
        self.augment = augment
        self._rng = np.random.RandomState(os.getpid() & 0xFFFF)
        with open(self.cache_root / "index.json") as f:
            index = json.load(f)
        if split is not None:
            index = [e for e in index if e.get("split", "train") == split]
        self.entries = [e for e in index if tokens is None or e["token"] in set(tokens)]
        if not self.entries:
            raise RuntimeError(f"数据为空：{cache_root} split={split}")

    def __len__(self) -> int:
        return len(self.entries)

    def _load_image(self, token: str, tag: str) -> np.ndarray:
        w, h = self.cfg.image_size
        with Image.open(self.cache_root / "images" / token / f"{tag}.jpg") as im:
            im = im.convert("RGB").resize((w, h), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        if self.augment:
            arr = _color_jitter(arr, self._rng)
            arr = _grid_mask(arr, self._rng)
        # ImageNet 归一化（DINOv3 预训练口径）
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return ((arr - mean) / std).transpose(2, 0, 1)                    # (3,H,W)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """返回 MyDriveModel.forward 所需的全部 batch 键（见其输入约定）。"""
        import torch
        entry = self.entries[idx]
        token = entry["token"]
        with np.load(self.cache_root / "labels" / f"{token}.npz") as z:
            traj = z["traj"].astype(np.float32)
            ego_status = z["ego_status"].astype(np.float32)
            ego_poses_local = z["ego_poses_local"].astype(np.float32)
            # 旧缓存可能没有 frame_dt（回退为理想 0.5s 间隔）
            frame_dt = (z["frame_dt"].astype(np.float32) if "frame_dt" in z
                        else np.array([-1.5, -1.0, -0.5, 0.0], dtype=np.float32))
            cam_K = z["cam_K"].astype(np.float32).copy()
            cam_e2c = z["cam_ego2cam"].astype(np.float32)
            orig_w, orig_h = z["img_size_orig"].astype(np.float32)
            boxes = z["boxes"].astype(np.float32)
            box_cls = z["box_cls"].astype(np.int64)
            bev_map = z["bev_map"].astype(np.int64)
            # 未来 8 帧障碍物真值（GRPO 碰撞 reward 用；旧缓存回退为全零占位）
            if "future_boxes" in z:
                future_boxes = z["future_boxes"].astype(np.float32)
                future_valid = z["future_valid"].astype(np.uint8)
            else:
                future_boxes = np.zeros((8, 1, 7), dtype=np.float32)
                future_valid = np.zeros((8, 1), dtype=np.uint8)
        # 图像：历史 3×前视（t3 最旧在前）+ 当前 4 路
        img_hist = np.stack([self._load_image(token, f"t{k}") for k in (3, 2, 1)])  # (3,3,H,W)
        img_cur = np.stack([self._load_image(token, c) for c in CAMS_CURRENT])      # (4,3,H,W)
        # 内参按 resize 缩放（标签 npz 存的是原图口径内参）
        w, h = self.cfg.image_size
        sx, sy = w / orig_w, h / orig_h
        for K in cam_K:
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy
        # 检测目标（CenterPoint 热力图 + 回归，在线生成）
        det = build_det_targets(boxes, box_cls, self.cfg)
        return {
            "img_hist": torch.from_numpy(img_hist),                     # (3,3,H,W)
            "img_cur": torch.from_numpy(img_cur),                       # (4,3,H,W)
            "cam_K": torch.from_numpy(cam_K),                           # (4,3,3) resize 后内参
            "cam_ego2cam": torch.from_numpy(cam_e2c),                   # (4,4,4) ego→cam
            "ego_status": torch.from_numpy(ego_status),                 # (4,11)
            "ego_motion": torch.from_numpy(ego_poses_local),            # (4,3)
            "frame_dt": torch.from_numpy(frame_dt),                     # (4,)
            "traj": torch.from_numpy(traj),                             # (8,3)
            "bev_map": torch.from_numpy(bev_map),                       # (100,100) 4 类
            "future_boxes": torch.from_numpy(future_boxes),             # (8,256,7)
            "future_valid": torch.from_numpy(future_valid),             # (8,256)
            "det_heatmap": torch.from_numpy(det["heatmap"]),            # (2,100,100)
            "det_reg": torch.from_numpy(det["reg"]),                    # (6,100,100)
            "det_reg_mask": torch.from_numpy(det["reg_mask"]),          # (100,100)
            "token": token,
        }