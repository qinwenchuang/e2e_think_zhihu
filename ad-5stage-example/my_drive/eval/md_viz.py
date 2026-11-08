"""md_viz：训练期间固定帧的 TensorBoard 可视化。

每个 epoch 对固定若干帧渲染横向拼图（tag = viz/frame_XX，按 epoch 连成动画）：
  1. 当前帧 4 路相机图（2×2）
  2. BEV GT 地图 + GT/预测检测框 + 轨迹对比
  3. 轨迹放大面板（x∈[-5,25]m, y∈[-15,15]m，带 5m 网格 + ADE/FDE 数值）
  4. BEV 预测地图（argmax）
  5. 检测热力图（上 GT / 下预测）

颜色约定（BGR 写入、TB 按 RGB 显示）：
  青色 = GT 轨迹；红色 = 预测轨迹；灰色细线 = 64 条 proposal 候选；
  红/紫框 = GT 检测框；黄/橙框 = 预测检测框。
"""
from __future__ import annotations
from typing import Dict, List, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ..md_common import CAMS_CURRENT, Config, box_corners_bev

#: 地图 4 类配色：背景深灰 / 车道蓝 / 路块橙 / 人行道绿
MAP_COLORS = np.array([
    [40, 40, 40],
    [60, 120, 216],
    [200, 160, 60],
    [90, 200, 90],
], dtype=np.uint8)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
C_GT = (0, 255, 255)      # GT 轨迹：青
C_PRED = (255, 50, 50)    # 预测轨迹：红


@torch.no_grad()
def decode_detections(pred: dict, cfg: Config, topk: int = 30, thresh: float = 0.1,
                      with_box: bool = False):
    """det_heatmap (B,2,Gd,Gd) logits + det_reg (B,6,Gd,Gd) → 每类 [(x, y, score[, l, w, yaw])]。

    栅格约定：row=x(前向)、col=y(横向)。heatmap 峰值在 (row, col)，reg 同位置。
    """
    hm = torch.sigmoid(pred["det_heatmap"].float())
    # 3×3 maxpool NMS（CenterNet 口径）
    pooled = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
    peaks = (hm == pooled) & (hm > thresh)
    reg = pred["det_reg"].float()
    ps = cfg.det_pixel_size
    out = []
    B = hm.shape[0]
    for b in range(B):
        per_cls = []
        for c in range(2):
            scores = hm[b, c][peaks[b, c]]
            idxs = peaks[b, c].nonzero()
            if len(scores) > topk:
                top = scores.topk(topk)
                scores, idxs = top.values, idxs[top.indices]
            pts = []
            for (row, col), s in zip(idxs.tolist(), scores.tolist()):
                r = reg[b, :, row, col]
                x = (row + float(r[0])) * ps + cfg.pc_x_min
                y = (col + float(r[1])) * ps + cfg.pc_y_min
                if with_box:
                    length, width = float(r[2].exp()), float(r[3].exp())
                    yaw = float(torch.atan2(r[4], r[5]))
                    pts.append((x, y, s, length, width, yaw))
                else:
                    pts.append((x, y, s))
            per_cls.append(pts)
        out.append(per_cls)
    return out
def unnormalize(img: np.ndarray) -> np.ndarray:
    """(3,H,W) 归一化图像 → (H,W,3) uint8。"""
    out = img.transpose(1, 2, 0) * _STD + _MEAN
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


class _BevCanvas:
    """BEV 画布：ego 坐标 → 像素，画轨迹/框/网格/ego 标记（行=x 前向朝下）。"""

    def __init__(self, img: np.ndarray, x_min: float, x_max: float,
                 y_min: float, y_max: float):
        self.img = img
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.h, self.w = img.shape[:2]
    def to_px(self, xy: np.ndarray) -> np.ndarray:
        col = (xy[:, 1] - self.y_min) / (self.y_max - self.y_min) * self.w
        row = (xy[:, 0] - self.x_min) / (self.x_max - self.x_min) * self.h
        return np.stack([col, row], -1).reshape(-1, 1, 2).astype(np.int32)
    def polyline(self, xy, color, thickness=3, closed=False):
        cv2.polylines(self.img, [self.to_px(np.asarray(xy).reshape(-1, 2))],
                      closed, color, thickness, cv2.LINE_AA)
    def points(self, xy, color, radius=3):
        for p in self.to_px(np.asarray(xy).reshape(-1, 2)).reshape(-1, 2):
            cv2.circle(self.img, tuple(p), radius, color, -1, cv2.LINE_AA)
    def ego_marker(self):
        cx, cy = self.to_px(np.array([[0.0, 0.0]])).reshape(-1, 2)[0]
        cv2.drawMarker(self.img, (int(cx), int(cy)), (255, 255, 255),
                       cv2.MARKER_TRIANGLE_UP, 14, 2, cv2.LINE_AA)
    def grid(self, step_m: float = 5.0):
        for x in np.arange(np.ceil(self.x_min / step_m) * step_m, self.x_max, step_m):
            p = self.to_px(np.array([[x, self.y_min], [x, self.y_max]])).reshape(-1, 2)
            cv2.line(self.img, tuple(p[0]), tuple(p[1]), (90, 90, 90), 1)
        for y in np.arange(np.ceil(self.y_min / step_m) * step_m, self.y_max, step_m):
            p = self.to_px(np.array([[self.x_min, y], [self.x_max, y]])).reshape(-1, 2)
            cv2.line(self.img, tuple(p[0]), tuple(p[1]), (90, 90, 90), 1)
    def text(self, s, org, color=(255, 255, 255), scale=0.45, thickness=1):
        cv2.putText(self.img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                    thickness + 2, cv2.LINE_AA)
        cv2.putText(self.img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    thickness, cv2.LINE_AA)
def _draw_boxes(canvas: _BevCanvas, boxes: np.ndarray, cls_ids: np.ndarray) -> None:
    for box, c in zip(boxes, cls_ids):
        if c < 0:
            continue
        corners = box_corners_bev(float(box[0]), float(box[1]), float(box[6]),
                                  float(box[3]), float(box[4]))
        color = (255, 80, 80) if c == 0 else (255, 0, 255)
        canvas.polyline(corners, color, thickness=1, closed=True)
def _draw_pred_boxes(canvas: _BevCanvas, det, thickness: int = 2) -> None:
    for cls_id, (pts, color) in enumerate(zip(det, [(255, 220, 0), (255, 140, 0)])):
        for x, y, s, l, w, yaw in pts:
            corners = box_corners_bev(x, y, yaw, l, w)
            canvas.polyline(corners, color, thickness=thickness, closed=True)
def render_sample_figure(sample: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor],
                         cfg: Config, boxes: Optional[np.ndarray] = None,
                         box_cls: Optional[np.ndarray] = None,
                         ) -> np.ndarray:
    """渲染单样本拼图 → (H, W, 3) uint8（H=320）。

    pred 里没有的键（如 stage 1 无 BEV）对应面板显示 GT 或留空提示。
    """
    P = 300
    cams = [unnormalize(sample["img_cur"][i].numpy()) for i in range(4)]
    cw, ch = 280, 160
    cam_panel = np.zeros((ch * 2, cw * 2, 3), dtype=np.uint8)
    for i, im in enumerate(cams):
        im = cv2.resize(im, (cw, ch))
        cam_panel[(i // 2) * ch:(i // 2 + 1) * ch, (i % 2) * cw:(i % 2 + 1) * cw] = im
        cv2.putText(cam_panel, CAMS_CURRENT[i],
                    ((i % 2) * cw + 6, (i // 2) * ch + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    gt_traj = sample["traj"].numpy()[:, :2]
    pred_traj = (pred["trajectory"].detach().cpu().numpy()[:, :2]
                 if "trajectory" in pred else None)
    proposals = None
    if "proposal_list" in pred:
        proposals = pred["proposal_list"][-1][0].detach().float().cpu().numpy()[:, :, :2]
    pred_det = None
    if "det_heatmap" in pred:
        pred_det = decode_detections(
            {"det_heatmap": pred["det_heatmap"][None],
             "det_reg": pred["det_reg"][None]}, cfg, topk=20, thresh=0.15,
            with_box=True)[0]
    gt_map = sample["bev_map"].numpy().astype(np.int64)
    bev_gt = cv2.resize(MAP_COLORS[np.clip(gt_map, 0, 3)], (P, P),
                        interpolation=cv2.INTER_NEAREST).copy()
    c_gt = _BevCanvas(bev_gt, cfg.pc_x_min, cfg.pc_x_max, cfg.pc_y_min, cfg.pc_y_max)
    if boxes is not None:
        _draw_boxes(c_gt, boxes, box_cls)
    if pred_det is not None:
        _draw_pred_boxes(c_gt, pred_det)
    if proposals is not None:
        for prop in proposals:
            c_gt.polyline(prop, (150, 150, 150), 1)
    c_gt.polyline(gt_traj, C_GT, 3)
    c_gt.points(gt_traj, C_GT, 2)
    if pred_traj is not None:
        c_gt.polyline(pred_traj, C_PRED, 3)
        c_gt.points(pred_traj, C_PRED, 2)
    c_gt.ego_marker()
    c_gt.text("GT map | GT box(red) pred box(yellow)", (4, 14))
    c_gt.text("traj GT(cyan) pred(red)", (4, 28))
    zoom = np.full((P, P, 3), 30, dtype=np.uint8)
    zx0, zx1, zy0, zy1 = -5.0, 25.0, -15.0, 15.0
    c_z = _BevCanvas(zoom, zx0, zx1, zy0, zy1)
    c_z.grid(5.0)
    if proposals is not None:
        for prop in proposals:
            c_z.polyline(prop, (140, 140, 140), 1)
    c_z.polyline(gt_traj, C_GT, 4)
    c_z.points(gt_traj, C_GT, 4)
    if pred_traj is not None:
        c_z.polyline(pred_traj, C_PRED, 4)
        c_z.points(pred_traj, C_PRED, 4)
        ade = float(np.linalg.norm(pred_traj - gt_traj, axis=-1).mean())
        fde = float(np.linalg.norm(pred_traj[-1] - gt_traj[-1]))
        c_z.text(f"ADE={ade:.2f}m FDE={fde:.2f}m", (4, P - 10))
    else:
        c_z.text("planner off (stage 2)", (4, P - 10))
    c_z.ego_marker()
    c_z.text("traj zoom x[-5,25] y[-15,15] (gray=proposals)", (4, 14))
    if "map_logits" in pred:
        pred_cls = pred["map_logits"].detach().cpu().numpy().argmax(axis=0)
        bev_pred = MAP_COLORS[np.clip(pred_cls, 0, 3)]
    else:
        bev_pred = np.zeros((cfg.bev_map_size[0], cfg.bev_map_size[1], 3), dtype=np.uint8)
    bev_pred = cv2.resize(bev_pred, (P, P), interpolation=cv2.INTER_NEAREST).copy()
    c_p = _BevCanvas(bev_pred, cfg.pc_x_min, cfg.pc_x_max, cfg.pc_y_min, cfg.pc_y_max)
    if boxes is not None:
        _draw_boxes(c_p, boxes, box_cls)
    if pred_det is not None:
        _draw_pred_boxes(c_p, pred_det)
    c_p.ego_marker()
    c_p.text("pred map + GT(red)/pred(yellow) box", (4, 14))
    if "map_logits" not in pred:
        c_p.text("BEV OFF in this stage", (60, 160), scale=0.7, thickness=2)
    gd = cfg.bev_grid * cfg.det_stride
    gt_hm = sample["det_heatmap"].numpy().sum(axis=0)
    gt_img = (np.clip(gt_hm, 0, 1)[..., None] * np.array([255, 80, 80])).astype(np.uint8)
    if "det_heatmap" in pred:
        pred_hm = torch.sigmoid(pred["det_heatmap"].detach().cpu().float()).numpy().sum(axis=0)
    else:
        pred_hm = np.zeros((gd, gd), dtype=np.float32)
    pred_img = (np.clip(pred_hm, 0, 1)[..., None] * np.array([80, 200, 255])).astype(np.uint8)
    hm = np.concatenate([gt_img, pred_img], axis=0)
    hm = cv2.resize(hm, (P, P), interpolation=cv2.INTER_NEAREST).copy()
    _BevCanvas(hm, 0, 1, 0, 1).text("det GT(top) vs pred(bottom)", (4, 14))
    if "det_heatmap" not in pred:
        _BevCanvas(hm, 0, 1, 0, 1).text("BEV OFF in this stage", (60, 235),
                                        scale=0.55, thickness=2)
    panels = [cam_panel, bev_gt, zoom, bev_pred, hm]
    out = np.concatenate([np.pad(p, ((0, 320 - p.shape[0]), (0, 0), (0, 0)))
                          if p.shape[0] < 320 else p for p in panels], axis=1)
    return out
@torch.no_grad()
def render_fixed_frames(model, entries_ds, indices: List[int], cfg: Config, device,
                        flags: Dict[str, bool]) -> List[np.ndarray]:
    """对固定帧跑前向并渲染（训练循环每个 epoch 调用，写 TB 图像）。"""
    from pathlib import Path
    was_training = model.training
    model.eval()
    figs = []
    for idx in indices:
        sample = entries_ds[idx]
        batch = {k: (v[None].to(device) if torch.is_tensor(v) else v)
                 for k, v in sample.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=cfg.amp and device.type == "cuda"):
            pred = model(batch, with_planner=flags["with_planner"],
                         with_bev=flags["with_bev"], dino_no_grad=True,
                         with_occ=flags.get("with_occ", False))
        pred = {k: (v[0].float() if torch.is_tensor(v) else v) for k, v in pred.items()}
        with np.load(Path(entries_ds.cache_root) / "labels" / f"{sample['token']}.npz") as z:
            boxes, box_cls = z["boxes"], z["box_cls"]
        figs.append(render_sample_figure(
            {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in sample.items()},
            pred, cfg, boxes, box_cls))
    model.train(was_training)
    return figs
