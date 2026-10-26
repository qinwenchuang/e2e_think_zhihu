"""my_drive 公共配置与 SE2 几何工具（全流程共享）。

坐标系约定（与 NAVSIM/nuPlan 一致）：
  * ego 系：x 前、y 左、z 上，原点 = 后轴中心地面，lidar2ego = identity；
  * 相机外参：sensor2lidar（cam→ego）来自场景 pickle，使用时取逆得 ego→cam；
  * BEV 栅格：行 = 前向 x、列 = 横向 y（本项目全链路统一这一约定）。
"""
from __future__ import annotations
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np

# --------------------------------------------------------------------------- #
# 路径（可用环境变量覆盖；数据集只读，输出默认写包内 workspace/）
# --------------------------------------------------------------------------- #
PKG_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PKG_ROOT / "weights" / "vit_small_patch16_dinov3.lvd1689m" / "model.safetensors"
DATASET_ROOT = Path(os.environ.get(
    "MD_DATASET_ROOT", "/media/qwc/66D8D21CD8D1EA77/navsim_workspace/dataset"))
NAVSIM_LOGS = DATASET_ROOT / "navsim_logs" / "trainval"
SENSOR_BLOBS = DATASET_ROOT / "sensor_blobs" / "trainval"
MAPS_ROOT = Path(os.environ.get(
    "NUPLAN_MAPS_ROOT", str(DATASET_ROOT / "maps")))
WORK_ROOT = Path(os.environ.get("MD_WORK_ROOT", str(PKG_ROOT / "workspace")))

# --------------------------------------------------------------------------- #
# 时序规格：历史窗口 t-3..t 共 4 帧，未来轨迹 8 步（4s），间隔均 0.5s
# --------------------------------------------------------------------------- #
NUM_HISTORY = 4
HISTORY_INTERVAL = 0.5
NUM_FUTURE_POSES = 8
FUTURE_INTERVAL = 0.5
# 当前帧用 4 路相机；历史帧只保留前视（省 3/4 编码算力）
CAMS_CURRENT = ("CAM_F0", "CAM_B0", "CAM_L0", "CAM_R0")
CAM_HISTORY = "CAM_F0"


# --------------------------------------------------------------------------- #
# 全局配置
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # ---- 数据 ----
    image_size: Tuple[int, int] = (640, 384)          # (W, H)，16 整除（40×24 patches）
    seed: int = 0
    # ---- BEV 几何（ego 系范围，米） ----
    bev_grid: int = 50                                # BEV query 网格 50×50（1m/px）
    pc_x_min: float = -10.0
    pc_x_max: float = 40.0
    pc_y_min: float = -25.0
    pc_y_max: float = 25.0
    bev_ref_heights: Tuple[float, ...] = (0.0, 0.6, 1.2, 1.8)  # 每 cell 4 个固定高度参考点
    bev_sub_samples: int = 2                          # 每 cell 2×2 子点（缓解欠采样混叠）
    bev_map_stride: int = 2                           # 地图分割 GT 分辨率倍数 → 100×100 @0.5m/px
    det_stride: int = 2                               # 检测头输出分辨率倍数 → 100×100 @0.5m/px
    map_query_radius: float = 60.0                    # 地图对象查询半径（米）
    # ---- 模型 ----
    d_model: int = 256
    lora_rank: int = 32
    num_scene_tokens: int = 16                        # 每相机 register/scene token 数
    bev_layers: int = 3                               # BEV cross-attn 层数
    bev_self_attn: bool = True                        # BEV cell 间自注意力
    bev_heads: int = 8
    planner_heads: int = 8
    # ---- proposal planner ----
    num_proposals: int = 64                           # 候选轨迹条数
    decoder_layers: int = 4                           # 逐层精修层数
    scorer_layers: int = 2                            # 质量 scorer 层数
    # ---- 轨迹表征与监督（借鉴 TopoCurve, arXiv:2610.09118 / MGMP, arXiv:2610.10646） ----
    # traj_repr="bezier"：planner 头输出 Bézier 控制点偏移（P0 固定 ego 原点，
    # heading 由曲线切线解析给出），经 Bernstein 基矩阵投影回 8 个采样位姿——
    # 离散点独立回归 → 结构约束曲线，数学保证光滑性。回归维度 24 → (n_ctrl-1)*2。
    traj_repr: str = "bezier"                         # "bezier" / "raw"（旧行为，消融回退）
    bezier_n_ctrl: int = 5                            # 控制点数（含固定起点；4=TopoCurve 同款三次曲线）
    # top-k 监督（CARS 思想）：每层不再只回传离 GT 最近的 1 条（WTA），
    # 而是 top-k 条按衰减权重加权——缓解 64 选 1 的监督稀疏，给 scorer 更密的信号。
    w_traj_topk: int = 5                              # 1 = 旧 WTA 行为
    topk_weight_decay: float = 0.5                    # 第 j 名权重 ∝ decay^(j-1)，归一化
    # 推理/评估时几何重打分（MGMP 思想：学习先验提议、显式几何裁决）：
    # 对 scorer top-m 的 proposal，用预测 BEV 地图（车道合规）+ 预测检测热力图
    # （碰撞代理）+ 进度做显式校验重选，替代裸 argmax。仅 not training 且
    # with_planner & with_bev 时生效（stage 3 / 最终推理），训练损失不受影响。
    geo_rescore: bool = True
    geo_rescore_m: int = 8                            # 进入几何校验的候选数
    # 权重标定原则：几何项要能在 scorer 过度自信时仍然翻盘（安全壳的意义），
    # 故碰撞与车道权重显著大于 scorer 分（top-m 内 softmax 值域 [0,1]）
    w_g_scorer: float = 0.5                           # 学习 scorer 分（top-m 内 softmax）权重
    w_g_lane: float = 1.5                             # 可行驶区合规权重（预测地图 soft 概率）
    w_g_col: float = 3.0                              # 碰撞代理惩罚（预测热力图，8 步取 max）
    w_g_oob: float = 1.0                              # 越界惩罚：任一步 footprint 出图比例的最大值（未知≠安全）
    w_g_prog: float = 0.3                             # 前向进度权重
    # ---- 损失权重 ----
    w_traj: float = 1.0
    w_bev_map: float = 1.0
    w_det_heatmap: float = 1.0
    w_det_reg: float = 0.5
    w_score_ce: float = 0.5                           # proposal 质量分数回归权重
    map_bg_weight: float = 0.3                        # 地图 CE 背景类降权（背景占比高）
    # ---- GRPO（stage 1.5） ----
    grpo_temperature: float = 0.5                     # 策略 softmax 温度 τ
    grpo_samples: int = 8                             # 每帧采样 proposal 数 k（组大小）
    grpo_beta: float = 0.5                            # KL(π‖π_SFT) 权重
    grpo_lr: float = 1e-4
    w_r_lane: float = 1.0                             # 可行驶区合规权重
    w_r_prog: float = 0.5                             # 前向进度权重（限幅 20m）
    w_r_comf: float = 0.2                             # 舒适度（yaw 二阶差）
    w_r_gt: float = 0.3                               # GT 锚定 exp(-ADE/2)
    ego_len: float = 4.9                              # 自车 footprint（nuPlan Pacifica）
    ego_wid: float = 2.3
    col_margin: float = 0.3                           # 碰撞安全 margin（米）
    # ---- foresight（stage 0，DINO-Foresight 化） ----
    foresight_layers: int = 6                         # 预测器深度（时空可分离块数）
    foresight_heads: int = 8                          # 预测器注意力头数
    foresight_spatial: bool = True                    # False=纯时序注意力（回退旧行为，消融用）
    foresight_target: str = "pca"                     # "pca"=多层concat+PCA；"last"=仅末层384维（旧行为）；
                                                      # "proj"=可学习projector+SIGReg（AD-E2E-JEPA 化，见下）
    foresight_pca_dim: int = 512                      # PCA 目标维度（输入侧 4 层×384=1536）
    foresight_max_step: int = 4                       # 最大预测时距（×0.5s）；1=旧单步行为（消融用）
    foresight_pca_windows: int = 2000                 # prepare-pca 采样窗口数
    foresight_pca_patches: int = 128                  # 每窗口随机采样 patch 数
    # ---- foresight "proj" 模式（AD-E2E-JEPA，arXiv:2609.34085） ----
    # 可学习 patch projector（两层 stride-2 卷积，在线/目标分支共享，论文 Fig.2 a.1）：
    #   patch 数 16×↓（24×40 → 6×10 = 60），嵌入维 4×↓（d_model 256 → proj_dim 64），
    #   预测器在投影空间运行，rollout/规划成本随之 ~64×↓（论文报告 100× 推理加速）。
    foresight_proj_dim: int = 64                      # 投影维度（d_model 的 1/4）
    # SIGReg（论文 Eq.10/11）：Epps–Pulley 检验把投影嵌入的随机 1 维投影拉向
    # 各向同性标准正态，防止可学习 projector 坍缩（target 侧在 pred loss 中 sg，
    # projector 在目标支路的梯度仅来自 SIGReg）。
    foresight_sigreg_weight: float = 0.09             # λ（论文默认 0.09；有效 batch 大时需下调）
    foresight_sigreg_dirs: int = 256                  # 随机投影方向数 M（论文 1024；小 batch 用 256 足够且快）
    foresight_sigreg_knots: int = 17                  # EP 积分结点数（[0,3] 均匀 17 点，LeJEPA 默认）
    # 可选自回归 rollout 一致性损失（论文 Eq.12 / A.2.2，TBPTT 截断反传）：
    # 论文中显著提升 geodesic 精度与 hit rate（FDE 4.2→3.2m，top-1 27%→47%），
    # 代价是每 batch 多 ~1 次历史编码 + F 次预测器前向，默认关。
    foresight_rollout: bool = False
    foresight_rollout_weight: float = 1.0             # rollout 项相对单步 TF 项的权重
    # val 时的规划诊断（借鉴论文 §3.3.2 hit rate）：以真实未来帧投影特征为 goal，
    # 候选 = GT (Δt,Δpose) 查询 + batch 内其他样本的查询（负样本），
    # 统计 GT 候选潜在距离最小的比例（hit@1）。proj 模式专用，B≥2 时有效。
    foresight_plan_eval: bool = True

    # ---- 训练 ----
    batch_size: int = 2
    grad_accum: int = 4
    lr: float = 2.5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs_stage1: int = 30
    epochs_stage2: int = 30
    epochs_stage3: int = 30
    num_workers: int = 2
    amp: bool = True

    # ---- BEV 派生量 ----
    @property
    def bev_map_size(self) -> Tuple[int, int]:
        """地图分割 GT 栅格 (H, W) = 前向 × 横向。"""
        return (self.bev_grid * self.bev_map_stride,) * 2

    @property
    def det_map_size(self) -> Tuple[int, int]:
        """检测 GT 栅格 (H, W)。"""
        return (self.bev_grid * self.det_stride,) * 2

    @property
    def det_pixel_size(self) -> float:
        return self.bev_pixel_size / self.det_stride

    @property
    def bev_pixel_size(self) -> float:
        """BEV query 网格分辨率（米/像素）。"""
        return (self.pc_x_max - self.pc_x_min) / self.bev_grid

    def bev_cell_centers(self) -> np.ndarray:
        """(bev_grid, bev_grid, 2) 的 cell 中心 (x, y) ego 坐标，第一维 = 前向 x。"""
        ps = self.bev_pixel_size
        xs = self.pc_x_min + (np.arange(self.bev_grid) + 0.5) * ps
        ys = self.pc_y_min + (np.arange(self.bev_grid) + 0.5) * ps
        gx, gy = np.meshgrid(xs, ys, indexing="ij")     # [x 行, y 列]
        return np.stack([gx, gy], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------- #
# SE2 工具
# --------------------------------------------------------------------------- #
def normalize_angle(angle):
    """角度归一化到 [-π, π)。"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def absolute_to_relative_se2(origin: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """全局 SE2 位姿序列 (N,3) → origin 局部系 (N,3)，origin = (x, y, heading)。"""
    origin = np.asarray(origin, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    c, s = np.cos(-origin[2]), np.sin(-origin[2])
    rot = np.array([[c, -s], [s, c]])
    xy = (poses[:, :2] - origin[:2]) @ rot.T
    heading = normalize_angle(poses[:, 2] - origin[2])
    return np.stack([xy[:, 0], xy[:, 1], heading], axis=-1)


def quaternion_yaw(q) -> float:
    """四元数 (w, x, y, z) → yaw（已用真实数据校验该顺序）。"""
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rot3d_to_mat(rot) -> np.ndarray:
    """(3,3) 旋转矩阵或四元数 (w,x,y,z) → (3,3) ndarray。"""
    arr = np.asarray(rot, dtype=np.float64)
    if arr.shape == (3, 3):
        return arr
    w, x, y, z = [float(v) for v in arr.flatten()]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def cam_extrinsic_ego2cam(cam_entry: dict) -> np.ndarray:
    """场景 pickle 的相机条目 → (4,4) ego→cam 变换（sensor2lidar 的逆，lidar≡ego）。"""
    rt = np.eye(4, dtype=np.float64)
    rt[:3, :3] = rot3d_to_mat(cam_entry["sensor2lidar_rotation"])
    rt[:3, 3] = np.asarray(cam_entry["sensor2lidar_translation"], dtype=np.float64)
    return np.linalg.inv(rt)


def box_corners_bev(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    """自车系 2D 有向框 → (4,2) 角点（零依赖的 OrientedBox 替代）。"""
    c, s = np.cos(yaw), np.sin(yaw)
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])