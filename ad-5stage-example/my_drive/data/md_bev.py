"""BEV 标签生成：地图语义分割（4 类）+ CenterPoint 式 agent 检测目标。

只依赖 nuplan-devkit 的地图 API（nuplan.common.maps）与 cv2/numpy。

* 地图分割：nuPlan 地图多边形栅格化到 BEV 栅格（默认 100×100 @ 0.5m/px），
  类别：0 背景 / 1 车道+路口 / 2 路块 / 3 人行道+斑马线+停止线。
* agent 检测：标注框（自车系 x/y/z/l/w/h/yaw）→ CenterNet 高斯热力图（2 类
  vehicle/vru）+ 中心偏移 / log 尺寸 / (sin,cos) yaw 回归目标（100×100 @ 0.5m/px）。
"""
from __future__ import annotations
import logging
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from nuplan.common.maps.abstract_map import SemanticMapLayer
from ..md_common import Config

logger = logging.getLogger(__name__)

# (label, 地图图层)——按绘制顺序排列，后者覆盖前者：
# roadblock 垫底（大路面），lane/路口覆盖之，人行设施优先级最高。
MAP_LAYER_SPEC = (
    (2, (SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR)),
    (1, (SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION)),
    (3, (SemanticMapLayer.CROSSWALK, SemanticMapLayer.STOP_LINE, SemanticMapLayer.WALKWAYS)),
)

# 检测类别：标注名 → 类 id（pedestrian/bicycle 合并为 vru；其余为忽略类 -1）
DET_NAME_TO_CLASS = {
    "vehicle": 0,
    "pedestrian": 1,
    "bicycle": 1,
}


def _to_px(coords_xy: np.ndarray, cfg: Config, stride: int) -> np.ndarray:
    """(N,2) 自车坐标 (x 前, y 左) → cv2 像素 (N,1,2) = (col, row)。

    约定：行 = 前向 x、列 = 横向 y（与 BEV 栅格全链路一致）。
    stride=1 → BEV query 栅格；stride=bev_map_stride → 地图/检测栅格。
    """
    ps = cfg.bev_pixel_size / stride
    col = (coords_xy[:, 1] - cfg.pc_y_min) / ps
    row = (coords_xy[:, 0] - cfg.pc_x_min) / ps
    return np.stack([col, row], axis=-1).reshape((-1, 1, 2)).astype(np.int32)


def _geometry_local(geometry, origin_xyyaw: np.ndarray):
    """shapely 几何：全局系 → origin (x, y, yaw) 的局部系（先平移后旋转）。"""
    from shapely import affinity
    a, b = np.cos(origin_xyyaw[2]), np.sin(origin_xyyaw[2])
    d, e = -np.sin(origin_xyyaw[2]), np.cos(origin_xyyaw[2])
    translated = affinity.affine_transform(geometry, [1, 0, 0, 1, -origin_xyyaw[0], -origin_xyyaw[1]])
    return affinity.affine_transform(translated, [a, b, d, e, 0, 0])


def build_map_label(map_api, ego_pose_xyyaw: np.ndarray, cfg: Config) -> np.ndarray:
    """生成地图语义分割标签 (Hm, Wm) uint8。地图查询失败时该层留空，绝不抛异常。"""
    from nuplan.common.actor_state.state_representation import Point2D
    hm, wm = cfg.bev_map_size
    canvas = np.zeros((hm, wm), dtype=np.uint8)
    dropped = 0
    for label, layers in MAP_LAYER_SPEC:
        try:
            objs = map_api.get_proximal_map_objects(
                point=Point2D(float(ego_pose_xyyaw[0]), float(ego_pose_xyyaw[1])),
                radius=cfg.map_query_radius, layers=list(layers))
        except Exception:  # noqa: BLE001  地图不可用 → 该层留空
            continue
        layer_mask = np.zeros((hm, wm), dtype=np.uint8)
        for layer in layers:
            for obj in objs.get(layer, []):
                polygon = getattr(obj, "polygon", None)
                if polygon is None or polygon.is_empty:
                    continue
                local = _geometry_local(polygon, ego_pose_xyyaw)
                if local.is_empty:
                    dropped += 1
                    continue
                # nuPlan 的 polygon 可能是 MultiPolygon（一个对象多个不相连
                # 区域）。旧代码直接访问 .exterior 会抛 AttributeError，又被下面
                # 的 except 静默吞掉，导致该对象整块从 CE 标签消失且无日志。
                if local.geom_type == "MultiPolygon":
                    parts = list(local.geoms)
                elif local.geom_type == "Polygon":
                    parts = [local]
                else:
                    parts = []
                if not parts:
                    dropped += 1
                for part in parts:
                    if part.is_empty or part.exterior is None:
                        dropped += 1
                        continue
                    try:
                        pts = _to_px(np.asarray(part.exterior.coords, dtype=np.float64),
                                     cfg, cfg.bev_map_stride)
                    except Exception:  # noqa: BLE001  退化多边形
                        dropped += 1
                        continue
                    try:
                        cv2.fillPoly(layer_mask, [pts], 255)
                    except cv2.error:  # 少于 3 个有效点等退化情形
                        dropped += 1
                        continue
        canvas[layer_mask > 0] = label
    if dropped:
        logger.warning("build_map_label: %d 个地图几何体无法栅格化（非 Polygon/"
                       "MultiPolygon、空 exterior 或退化坐标），已跳过", dropped)
    return canvas


def gaussian_radius(det_size_px: Tuple[float, float], min_overlap: float = 0.5) -> float:
    """CornerNet 高斯半径（精确解三次方程，保证 IoU ≥ min_overlap）。"""
    height, width = det_size_px
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(0.0, b1 ** 2 - 4 * a1 * c1))
    r1 = (b1 + sq1) / 2
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(max(0.0, b2 ** 2 - 4 * a2 * c2))
    r2 = (b2 + sq2) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(max(0.0, b3 ** 2 - 4 * a3 * c3))
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int) -> None:
    """在 heatmap 上画 2D 高斯（CenterNet 口径，取 max 不覆盖）。

    center 参数约定是 (col, row)——调用方必须传 (cy, cx)（col=横向 y、row=前向 x），
    否则热力图与 reg/reg_mask 互为转置，解码端会读到未监督的位置。
    """
    diameter = 2 * radius + 1
    sigma = diameter / 6
    x = np.arange(diameter) - radius
    gauss = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = gauss[:, None] * gauss[None, :]
    cx, cy = center
    h, w = heatmap.shape
    left, right = min(cx, radius), min(w - cx - 1, radius)
    top, bottom = min(cy, radius), min(h - cy - 1, radius)
    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return
    region = heatmap[cy - top: cy + bottom + 1, cx - left: cx + right + 1]
    k = kernel[radius - top: radius + bottom + 1, radius - left: radius + right + 1]
    np.maximum(region, k, out=region)


def build_det_targets(boxes: np.ndarray, cls_ids: np.ndarray, cfg: Config
                      ) -> Dict[str, np.ndarray]:
    """当前帧标注框 → CenterPoint 式检测目标（栅格 = bev_grid × det_stride = 100×100 @0.5m）。

    :param boxes: (N, 7+) 自车系 (x, y, z, l, w, h, yaw)
    :param cls_ids: (N,) 检测类 id（0=vehicle, 1=vru；<0 忽略）
    :return: heatmap (2,G,G) float32、reg (6,G,G)（off_x,off_y,log_l,log_w,sin,cos）、
             reg_mask (G,G) uint8（正样本位置）
    """
    g = cfg.bev_grid * cfg.det_stride
    num_cls = 2
    heatmap = np.zeros((num_cls, g, g), dtype=np.float32)
    reg = np.zeros((6, g, g), dtype=np.float32)
    reg_mask = np.zeros((g, g), dtype=np.uint8)
    ps = cfg.det_pixel_size
    for box, cls_id in zip(boxes, cls_ids):
        cls_id = int(cls_id)
        if cls_id < 0 or cls_id >= num_cls:
            continue
        x, y, length, width, yaw = (float(box[0]), float(box[1]),
                                    float(box[3]), float(box[4]), float(box[6]))
        gx = (x - cfg.pc_x_min) / ps
        gy = (y - cfg.pc_y_min) / ps
        # 先浮点范围判断再取整：int() 向零截断，范围外 1px 内的框会被画到边缘 cell
        if not (0 <= gx < g and 0 <= gy < g):
            continue
        cx, cy = int(gx), int(gy)                      # cx=row(前向 x)、cy=col(横向 y)
        l_px, w_px = length / ps, width / ps
        radius = max(1, int(gaussian_radius((w_px, l_px))))
        radius = min(radius, 8)
        draw_gaussian(heatmap[cls_id], (cy, cx), radius)
        reg[:, cx, cy] = [gx - cx, gy - cy,
                          math.log(max(length, 1e-3)), math.log(max(width, 1e-3)),
                          math.sin(yaw), math.cos(yaw)]
        reg_mask[cx, cy] = 1
    return {"heatmap": heatmap, "reg": reg, "reg_mask": reg_mask}


class MapLabelCache:
    """地图分割标签的逐 token 磁盘缓存（tmp + rename 原子写，损坏自动回退重建）。"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, log_name: str, token: str) -> Path:
        return self.root / str(log_name) / f"{token}.npz"

    def get(self, log_name: str, token: str) -> Optional[np.ndarray]:
        p = self.path(log_name, token)
        if not p.is_file():
            return None
        try:
            with np.load(p) as data:
                return data["bev"].astype(np.uint8)
        except Exception:  # noqa: BLE001  损坏文件 → 视为未缓存
            return None

    def put(self, log_name: str, token: str, bev: np.ndarray) -> None:
        p = self.path(log_name, token)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.tmp.npz")
        try:
            np.savez_compressed(tmp, bev=bev.astype(np.uint8))
            os.replace(tmp, p)
        except Exception:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
