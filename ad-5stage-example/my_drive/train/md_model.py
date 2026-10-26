"""my_drive 模型：DINOv3+LoRA 多帧编码器 + 时序 ProposalPlanner + UniAD-lite BEV 头。

数据流（每样本）：
    t-3        t-2        t-1         t            （历史仅前视；当前 4 路相机）
  [img]      [img]      [img]      [img]          （同一冻结 DINO 主干，LoRA 适配）
     │          │          │          │
  R_{t-3}   R_{t-2}    R_{t-1}    R_t             （各 16 个 scene token）
     └──────┬───┴──────────┘          │
        token queue (48)              │
            └───────┬─────────────────┘
                     ▼
  planner 输入 = [R_t(4 路)] + [R_{t-1..t-3}] + [ego_token]
                 每帧 token 加 temporal_PE（Δt 正弦编码）与 ego_motion_PE（Δpose MLP 编码）
  BEV 头仅用当前帧 4 路 patch 特征：50×50 query × 固定 3D 参考点（4 高度 × 2×2 子点）
  投影采样 → 3 层 [self-attn + spatial cross-attn + FFN] → 地图分割 / 检测头。

stage 0（foresight）新增 AD-E2E-JEPA 化的 "proj" 模式（arXiv:2609.34085）：
  可学习 PatchProjector（stride-2 卷积 ×2，patch 16×↓、维度 4×↓，两分支共享）
  + SIGReg 防坍缩 + 目标侧 stop-gradient + 可选 TBPTT rollout 一致性损失。
  "pca" / "last" 模式保留作消融回退，行为与旧版完全一致。

全局符号约定：
  B    batch；M 一次编码图像数（当前帧 M=4 / 历史 M=3 / pretext M=5）
  H,W  图像高宽 = 384, 640；S 每相机 scene token = 16
  E   DINOv3 特征维 = 384；d 模型维 = 256；dp 投影维 = 64（proj 模式）
  gh,gw patch 网格 = 24×40（960 patch）；G BEV 网格 = 50（N = 2500 query）
  n_h BEV 高度采样 = 4；S2 cell 子点 = 2×2；NC 相机数 = 4
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..md_common import Config, DEFAULT_WEIGHTS, NUM_FUTURE_POSES


def _make_eva_class(feat_layers=(2, 5, 8)):
    """返回注入了 scene tokens 的 Eva 子类（DINOv3 用 RoPE，无绝对位置编码）。

    序列顺序：[scene(16), cls(1), reg(4), patch(960)] = 981 tokens。
    关键约束：插入 scene tokens 后必须把每个 attention block 的 num_prefix_tokens
    同步 +16 —— RoPE 只作用于 prefix（cls+reg）之后的 token，不同步 patch 位置角会错位。

    :param feat_layers: 额外收集输出的 block 下标（0 基，2/5/8 = 第 3/6/9 层），
        多层特征融合（冻结主干做稠密任务时单层特征不够）。
    :return: (final_tokens (M,981,E), [inter_tokens (M,981,E) × 3])
    """
    from timm.models.eva import Eva as _EvaBase

    class SceneEvaViT(_EvaBase):
        def forward_features(self, x, scene_tokens=None, attn_mask=None, is_causal=False):
            # x: (M,3,384,640) → patch_embed (M,960,E) → 前拼 [cls(1)|reg(4)]
            x = self.patch_embed(x)
            x, rot_pos_embed = self._pos_embed(x)
            if scene_tokens is not None:
                n_scene = scene_tokens.shape[1]        # 16
                x = torch.cat([scene_tokens, x], dim=1)  # (M,981,E)，scene 放最前
                for blk in self.blocks:
                    # prefix = cls+reg+scene = 21：RoPE 不作用于它们
                    blk.attn.num_prefix_tokens = self.num_prefix_tokens + n_scene
            x = self.norm_pre(x)
            inters = []
            # rope_mixed=True 时每层有不同旋转表（timm Eva 特性），否则共享一张表
            rope_seq = rot_pos_embed if getattr(self, "rope_mixed", False) else None
            for i, blk in enumerate(self.blocks):      # 12 层
                rope_i = rope_seq[i] if rope_seq is not None else rot_pos_embed
                x = blk(x, rope=rope_i, attn_mask=attn_mask, is_causal=is_causal)
                if i in feat_layers:                   # 第 3/6/9 层（0 基 2/5/8）
                    inters.append(x)
            return self.norm(x), inters                # final: (M,981,E)

    return SceneEvaViT


class _LoRAqkv(nn.Module):
    """timm qkv Linear 的 LoRA 包装（只适配 q、v）。

    y = W_qkv·x + [B_q·A_q·x | 0 | B_v·A_v·x]；B 零初始化 → 训练起点等价原 DINOv3，
    微调从恒等出发更稳定。注意用非原地写法（fp32 反向下 inplace 会报错）。
    """

    def __init__(self, qkv: nn.Linear, r: int):
        super().__init__()
        dim = qkv.in_features                          # E = 384
        self.dim = dim
        self.qkv = qkv                                 # 冻结原权重 Linear(384→1152)
        self.linear_a_q = nn.Linear(dim, r, bias=False)
        self.linear_b_q = nn.Linear(r, dim, bias=False)
        self.linear_a_v = nn.Linear(dim, r, bias=False)
        self.linear_b_v = nn.Linear(r, dim, bias=False)
        nn.init.kaiming_uniform_(self.linear_a_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.linear_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear_b_q.weight)
        nn.init.zeros_(self.linear_b_v.weight)

    def forward(self, x):
        # x: (M,981,E)；前 1/3 是 q（加支路），中 1/3 是 k（不动），后 1/3 是 v（加支路）
        qkv = self.qkv(x)                              # (M,981,1152)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        return torch.cat([qkv[..., : self.dim] + new_q,
                          qkv[..., self.dim: 2 * self.dim],
                          qkv[..., 2 * self.dim:] + new_v], dim=-1)


class DinoEncoder(nn.Module):
    """DINOv3 ViT-S/16 + LoRA + scene tokens + neck。

    forward(img, scene_tokens, need_patch) → (scene (M,S,d), patch (M,960,d)|None)
      scene: 最后一层 scene 段过 neck（384→256，planner 用）；
      patch: 第 3/6/9/12 层 patch 段 concat 后 patch_fuse（1536→256，BEV 用）。
    """

    def __init__(self, cfg: Config, weights_path: Optional[str] = None):
        super().__init__()
        import timm
        self.cfg = cfg
        self.num_scene = cfg.num_scene_tokens          # S = 16
        w, h = cfg.image_size                          # (640, 384) → img_size=(H,W)
        wpath = str(weights_path or DEFAULT_WEIGHTS)
        model = timm.create_model(
            "vit_small_patch16_dinov3.lvd1689m",
            pretrained=True,                           # 恒加载预训练权重
            pretrained_cfg_overlay={"file": wpath},
            img_size=(h, w), num_classes=0, dynamic_img_pad=True,
        )
        # 启动校验：抽查关键张量与磁盘 safetensors 是否逐比特一致，
        # 防止「以为加载了预训练权重、实际随机初始化」的静默失败
        try:
            from safetensors import safe_open
            with safe_open(wpath, framework="pt") as f:
                for k in ("patch_embed.proj.weight", "blocks.0.attn.qkv.weight"):
                    disk = f.get_tensor(k)
                    mine = model.state_dict()[k]
                    assert torch.equal(mine, disk), f"{k} 与磁盘不一致"
            print(f"[DinoEncoder] 预训练权重加载并校验通过 ✓（{wpath}）")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"[DinoEncoder] 预训练权重校验失败：{exc}") from exc
        # LoRA 手术：主干全冻结，每个 block 的 qkv 换成 LoRA 包装
        model.__class__ = _make_eva_class(feat_layers=(2, 5, 8))
        for p in model.parameters():
            p.requires_grad = False
        for blk in model.blocks:                       # 12 个 block
            blk.attn.qkv = _LoRAqkv(blk.attn.qkv, cfg.lora_rank)  # r=32
        self.model = model
        self.embed_dim = model.num_features            # E = 384
        self.patch = model.patch_embed.patch_size[0]   # 16
        self.gh = math.ceil(h / self.patch)            # 384/16 = 24
        self.gw = math.ceil(w / self.patch)            # 640/16 = 40
        # prefix 总数 = scene(16) + cls(1) + reg(4) = 21：patch 段从下标 21 开始
        self.n_prefix = self.num_scene + int(model.num_prefix_tokens)
        self.neck = nn.Linear(self.embed_dim, cfg.d_model)      # 384→256（scene 用）
        self.patch_fuse = nn.Linear(self.embed_dim * 4, cfg.d_model)  # 4 层 concat→256

    def forward(self, img: torch.Tensor, scene_tokens: torch.Tensor,
                need_patch: bool = True,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # img: (M,3,H,W)；scene_tokens: (M,S,E)
        final, inters = self.model.forward_features(img, scene_tokens)
        # final: (M,981,E)；inters: [第3,6,9层输出] × 3
        scene = final[:, : self.num_scene]             # (M,16,E) 取 scene 段
        patch = None
        if need_patch:
            # 各层 patch 段（下标 21 起）concat → 1536 维 → 256 维
            parts = [t[:, self.n_prefix:] for t in inters] + [final[:, self.n_prefix:]]
            patch = self.patch_fuse(torch.cat(parts, dim=-1))  # (M,960,256)
        return self.neck(scene), patch


def sinusoidal_dt(dt_seconds: torch.Tensor, dim: int, scale: float = 1.0) -> torch.Tensor:
    """Δt（秒）的正弦位置编码（频率基 10000；dt 可为负：历史帧 < 0）。"""
    half = dim // 2                                    # 128
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=dt_seconds.device,
                                                        dtype=torch.float32) / half)
    args = (dt_seconds[..., None] * scale) * freqs     # (...,128)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (...,256)


def _bernstein_matrices(n_ctrl: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """Bézier Bernstein 基矩阵 A (K,n_ctrl) 与导数基 A′ (K,n_ctrl)。

    采样点 t_k = (k+1)/K（k=0..K-1）：轨迹定义为 ego 当前时刻之后的 K 个未来位姿，
    故 t 从 1/K 起（t=0 即 ego 原点，不占监督位），终点 t=1 对应最后一步。
    A′ 是 A 对 t 的解析导数（B′(t) = d·Σ b_{d-1,j}(t)·(P_{j+1}−P_j)），
    用于从控制点直接求各采样点的切线方向 → heading，无需有限差分。
    """
    from math import comb
    d = n_ctrl - 1
    t = (np.arange(K, dtype=np.float64) + 1.0) / K
    A = np.zeros((K, n_ctrl), dtype=np.float64)
    for j in range(n_ctrl):
        A[:, j] = comb(d, j) * (1.0 - t) ** (d - j) * t ** j
    Ad = np.zeros((K, n_ctrl), dtype=np.float64)
    for j in range(d):
        b = comb(d - 1, j) * (1.0 - t) ** (d - 1 - j) * t ** j
        Ad[:, j] -= d * b
        Ad[:, j + 1] += d * b
    return A.astype(np.float32), Ad.astype(np.float32)


class PatchProjector(nn.Module):
    """可学习 patch projector（AD-E2E-JEPA 核心组件，arXiv:2609.34085 Fig.2 a.1）。

    两个 stride-2 卷积：24×40 patch 网格 → 12×20 → 6×10（16× 更少 patch），
    通道 d_model(256) → proj_dim(64)（4× 降维）。在线分支与目标分支共享
    （论文 Eq.7），预测器/规划全部在投影后的紧凑空间进行 → rollout 成本 ~64×↓。
    GroupNorm 而非 BatchNorm：与 BEV trunk 同理，小 batch 下 BN 统计噪声大。

    forward: (N, gh*gw, d) patch 序列 → (N, (gh//4)*(gw//4), proj_dim)。
    """

    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model                                # 256
        p = cfg.foresight_proj_dim                     # 64
        self.conv1 = nn.Conv2d(d, p * 2, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(8, p * 2)
        self.conv2 = nn.Conv2d(p * 2, p, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(8, p)
        self.act = nn.GELU()

    def forward(self, feats: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        # (N, gh*gw, d) → (N, d, gh, gw)：先 transpose 再 reshape，
        # 避免通道与空间维交错混排（与 LiteBEVHead.forward 中的教训一致）
        x = feats.transpose(1, 2).reshape(feats.shape[0], -1, gh, gw)
        x = self.act(self.norm1(self.conv1(x)))        # (N, 2p, gh/2, gw/2)
        x = self.norm2(self.conv2(x))                  # (N, p, gh/4, gw/4)
        return x.flatten(2).transpose(1, 2)            # (N, gh*gw/16, p)


def _se2_rebase(poses: torch.Tensor, origin: int) -> torch.Tensor:
    """poses (B,N,3)（同一参考系下的相对位姿）→ 以第 origin 个位姿为新原点 (B,N,3)。

    rollout 逐步推进时，dt/motion 条件需要从「相对帧 t」重基到「相对帧 t+j」；
    θ 用 atan2(sin,cos) 包角，等价 normalize_angle 的可微版本。
    """
    o = poses[:, origin]                               # (B,3)
    c, s = torch.cos(-o[:, 2]), torch.sin(-o[:, 2])
    dx = poses[..., 0] - o[:, 0:1]
    dy = poses[..., 1] - o[:, 1:2]
    x = c[:, None] * dx - s[:, None] * dy
    y = s[:, None] * dx + c[:, None] * dy
    dth = poses[..., 2] - o[:, 2:3]
    th = torch.atan2(torch.sin(dth), torch.cos(dth))
    return torch.stack([x, y, th], dim=-1)


class _DecoderBlock(nn.Module):
    """planner/scorer 共用 transformer 块：self-attn → cross-attn → FFN（pre-LN 残差）。

    x: (B,Q,d) 查询（proposal tokens）；ctx: (B,K,d) 上下文（时序 token 队列）。
    """

    def __init__(self, d: int, heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        self.norm_kv = nn.LayerNorm(d)
        self.norm_q = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        self.norm3 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_dim, d))
        self.drop = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                         need_weights=False)[0])
        x = x + self.drop(self.cross_attn(self.norm_q(x), self.norm_kv(ctx),
                                          self.norm_kv(ctx), need_weights=False)[0])
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x
class ProposalPlanner(nn.Module):
    """N 条 proposal + 逐层精修 + 质量 scorer 选最优。

      proposal_embed + ego_token  → 初始 proposal tokens
      init_head                   → 初始 proposals
      decoder（L 层，深监督）     → 每层精修一次 proposals
      pos_embed + scorer blocks   → 每条 proposal 的质量 logit（嵌入用 detach）
      argmax(quality)             → 最终轨迹（推理时可选几何重打分覆盖，见 geo_rescore）
    监督策略（见 md_loss）：top-k 加权 L1（k=1 时退化为 WTA）+ 质量分数回归负 ADE。

    轨迹表征（cfg.traj_repr，借鉴 TopoCurve arXiv:2610.09118 §3.3）：
      "bezier"：头输出 Bézier 控制点偏移 (n_ctrl-1)×2，P0 固定 ego 原点
      （规划场景起点已知，比论文更有利）；经 Bernstein 基投影回 8 个采样位姿，
      heading 由曲线切线解析给出（atan2 导数基），不再逐点独立回归。
      保证轨迹光滑、切线连续，曲率可解析求取；回归维度 24 → (n_ctrl-1)·2。
      "raw"：逐点回归 8×3（旧行为，消融回退）。
    注意：切换 traj_repr 会改变 head 输出维度，旧 checkpoint 的 planner 头键
    形状不匹配会被 check_ckpt_compat 拒绝；换表征请从干净 ckpt 重训 stage1。
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model                                # 256
        N = cfg.num_proposals                          # 64
        self.K = NUM_FUTURE_POSES                      # 8 个未来位姿
        self.bezier = cfg.traj_repr == "bezier"
        if self.bezier:
            self.n_ctrl = cfg.bezier_n_ctrl
            # 头输出维度：控制点 1..n_ctrl-1 的 xy（P0 固定原点不回归）
            head_dim = (self.n_ctrl - 1) * 2
            A, Ad = _bernstein_matrices(self.n_ctrl, self.K)
            # buffer 随 checkpoint 保存/.to() 迁移；旧 ckpt 无此键 → check_ckpt_compat 拒绝
            self.register_buffer("bern", torch.from_numpy(A))        # (K, n_ctrl)
            self.register_buffer("bern_dot", torch.from_numpy(Ad))   # (K, n_ctrl)
        else:
            head_dim = self.K * 3                      # 24 = 8 步 × (x,y,heading)
        self.head_dim = head_dim
        out_dim = self.K * 3                           # scorer pos_embed 恒用 24 维位姿
        self.proposal_embed = nn.Embedding(N, d)
        self.ego_mlp = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))
        self.dt_embed = nn.Linear(d, d)
        self.ego_motion_mlp = nn.Sequential(nn.Linear(3, d), nn.GELU(),
                                            nn.Linear(d, d))
        self.init_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, head_dim))
        self.decoder = nn.ModuleList([
            _DecoderBlock(d, cfg.planner_heads, d * 2) for _ in range(cfg.decoder_layers)])
        self.refine_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, head_dim))
            for _ in range(cfg.decoder_layers)])
        self.pos_embed = nn.Sequential(nn.Linear(out_dim, d * 2), nn.ReLU(),
                                       nn.Linear(d * 2, d))
        self.scorer = nn.ModuleList([
            _DecoderBlock(d, cfg.planner_heads, d * 2) for _ in range(cfg.scorer_layers)])
        self.quality_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def _ctrl_to_waypoints(self, ctrl_delta: torch.Tensor
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bézier 控制点偏移 (B,N,n_ctrl-1,2) → (采样位姿 (B,N,K,3), 切线 (B,N,K,2))。

        P0 固定 (0,0)（ego 原点）；waypoint xy = A·C（Bernstein 基投影，与 loss /
        评估在同一投影空间比较，对应 TopoCurve 的 "matching in projected space"）；
        heading = 解析切线方向 atan2(B′(t))（切向近零时 yaw 无定义但 atan2 安全；
        监督侧用切线余弦项规避 atan2 奇点，见 md_loss）。切线同时返回给损失函数
        做光滑的航向监督。
        """
        B, N = ctrl_delta.shape[:2]
        zeros = ctrl_delta.new_zeros(B, N, 1, 2)
        ctrl = torch.cat([zeros, ctrl_delta], dim=2)   # (B,N,n_ctrl,2)
        xy = torch.einsum("kc,bncd->bnkd", self.bern, ctrl)
        tan = torch.einsum("kc,bncd->bnkd", self.bern_dot, ctrl)
        yaw = torch.atan2(tan[..., 1], tan[..., 0])    # (B,N,K)
        return torch.cat([xy, yaw[..., None]], dim=-1), tan

    def _head_to_waypoints(self, raw: torch.Tensor
                           ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """头输出 (B,N,head_dim) → ((B,N,K,3), 切线|None)（两种表征统一出口）。"""
        if self.bezier:
            return self._ctrl_to_waypoints(
                raw.reshape(*raw.shape[:2], self.n_ctrl - 1, 2))
        return raw.reshape(*raw.shape[:2], self.K, 3), None

    def build_context(self, r_all: torch.Tensor, queue: torch.Tensor,
                      ego_status: torch.Tensor, ego_motion: torch.Tensor,
                      frame_dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """构造时序上下文 tokens（帧 tokens + temporal_PE + ego_motion_PE + ego token）。

        :param r_all: (B,64,d) 当前帧 4 路 scene tokens（4cam×16）
        :param queue: (B,48,d) 历史 3 帧前视 scene tokens（t-3..t-1）
        :return: ctx (B,113,d) = [帧 tokens 112 + ego token 1]；ego_token (B,1,d)
        """
        cfg = self.cfg
        B = r_all.shape[0]
        d = cfg.d_model
        s = cfg.num_scene_tokens                       # 16
        n_cur = r_all.shape[1]                         # 64 = 4 路 × 16
        # ego token：当前帧 vel2+acc2+cmd4（ego_status 后 8 维）
        ego_cur = ego_status[:, -1, 3:]
        ego_token = self.ego_mlp(ego_cur)[:, None]     # (B,1,d)
        frame_tokens = torch.cat([r_all, queue], dim=1)  # (B,112,d)
        # temporal_PE：当前 4 路共享 dt≈0；历史 3 帧用实际 Δt
        dt = torch.cat([
            frame_dt[:, -1:, None].expand(B, 1, n_cur).reshape(B, n_cur),
            frame_dt[:, :3, None].expand(B, 3, s).reshape(B, 3 * s),
        ], dim=1)                                                      # (B,112)
        temporal_pe = self.dt_embed(sinusoidal_dt(dt, d))              # (B,112,d)
        # ego_motion_PE：当前 4 路共享 Δpose=0；历史各帧用实际 Δpose
        motion = torch.cat([
            ego_motion[:, -1:, None].expand(B, 1, n_cur, 3).reshape(B, n_cur, 3),
            ego_motion[:, :3, None].expand(B, 3, s, 3).reshape(B, 3 * s, 3),
        ], dim=1)                                                      # (B,112,3)
        motion_pe = self.ego_motion_mlp(motion)
        frame_tokens = frame_tokens + temporal_pe + motion_pe
        ctx = torch.cat([frame_tokens, ego_token], dim=1)
        return ctx, ego_token
    def forward(self, r_all: torch.Tensor, queue: torch.Tensor,
                ego_status: torch.Tensor, ego_motion: torch.Tensor,
                frame_dt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        :return: proposal_list [(B,N,8,3)]×(L+1)（含初始层，深监督用）、
                 proposal_scores (B,N)、trajectory (B,8,3)（argmax 选中）
        """
        cfg = self.cfg
        B = r_all.shape[0]
        N = cfg.num_proposals
        ctx, ego_token = self.build_context(r_all, queue, ego_status, ego_motion, frame_dt)
        # 初始 proposal tokens：learned embed + ego
        tokens = self.proposal_embed.weight[None] + ego_token        # (B,N,d)
        proposal_list: List[torch.Tensor] = []
        tangent_list: List[torch.Tensor] = []          # 仅 bezier 模式非空（供损失用）
        wp, tan = self._head_to_waypoints(self.init_head(tokens))
        proposal_list.append(wp)
        if tan is not None:
            tangent_list.append(tan)
        # 逐层精修：tokens 过 decoder，每层回归一次 proposals（深监督）
        for block, head in zip(self.decoder, self.refine_heads):
            tokens = block(tokens, ctx)                              # (B,N,d)
            wp, tan = self._head_to_waypoints(head(tokens))
            proposal_list.append(wp)
            if tan is not None:
                tangent_list.append(tan)
        final_props = proposal_list[-1]                              # (B,N,8,3)
        # scorer：proposal 嵌入（detach，scorer 梯度不回流生成器）→ 质量 logit
        embedded = self.pos_embed(final_props.detach().reshape(B, N, -1))  # (B,N,d)
        for block in self.scorer:
            embedded = block(embedded, ctx)
        scores = self.quality_head(embedded + ego_token).squeeze(-1)   # (B,N)
        # argmax 质量 logit 选轨迹
        sel = scores.argmax(dim=1)                                   # (B,)
        trajectory = final_props[torch.arange(B, device=scores.device), sel]  # (B,8,3)
        out = {"trajectory": trajectory, "proposal_list": proposal_list,
               "proposal_scores": scores}
        if tangent_list:
            out["tangent_list"] = tangent_list
        return out


class SpatialCrossAttention(nn.Module):
    """BEV query × 相机采样特征 cross-attention（每个 query 只 attend 自己的采样点）。

    K/V 不是全图 patch，而是每个 query 在投影点采到的 64 个特征
    （4 相机 × 4 高度 × 4 子点）→ 注意力矩阵仅 (B,8,2500,64)，纯 PyTorch 即可。
    """

    def __init__(self, d: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = d // heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
    def forward(self, query: torch.Tensor, sampled: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        B, N, d = query.shape
        S = sampled.shape[2]
        h, hd = self.heads, self.head_dim
        q = self.q_proj(query).reshape(B, N, h, hd)
        k = self.k_proj(sampled).reshape(B, N, S, h, hd)
        v = self.v_proj(sampled).reshape(B, N, S, h, hd)
        scores = torch.einsum("bnhd,bnshd->bhns", q, k) / math.sqrt(hd)
        scores = scores.masked_fill(~mask[:, None, :, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        # 整行全 -inf（该 cell 在所有相机/高度都不可见）→ softmax 出 NaN，
        # 填 0 让该 cell 输出为 0（仅靠残差透传）
        attn = torch.where(torch.isfinite(attn), attn, torch.zeros_like(attn))
        out = torch.einsum("bhns,bnshd->bnhd", attn, v).reshape(B, N, d)
        return self.o_proj(self.drop(out))


class LiteBEVLayer(nn.Module):
    """[cell 间 self-attn + spatial cross-attn + FFN]，pre-LN 残差结构。"""

    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.ln_cross = nn.LayerNorm(d)
        self.cross = SpatialCrossAttention(d, cfg.bev_heads)
        self.use_self = cfg.bev_self_attn
        if self.use_self:
            self.ln_self = nn.LayerNorm(d)
            self.self_attn = nn.MultiheadAttention(d, cfg.bev_heads, batch_first=True)
        self.ln_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
    def forward(self, bev: torch.Tensor, sampled: torch.Tensor, mask: torch.Tensor,
                ) -> torch.Tensor:
        if self.use_self:
            x = self.ln_self(bev)
            bev = bev + self.self_attn(x, x, x, need_weights=False)[0]
        bev = bev + self.cross(self.ln_cross(bev), sampled, mask)
        bev = bev + self.ffn(self.ln_ffn(bev))
        return bev
class LiteBEVHead(nn.Module):
    """UniAD-lite BEV 头：固定 3D 参考点投影采样 + 3 层 cross-attn + 任务头。

    采样策略（每 query 64 点 = 4 相机 × 4 高度 × 4 子点）：单点采样时远处小目标
    会「落在采样点之间」（欠采样混叠），子高度与子点覆盖后显著缓解。
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model                                # 256
        g = cfg.bev_grid                               # 50
        self.bev_embed = nn.Parameter(torch.randn(1, g * g, d) * 0.02)  # 可学习 query
        self.layers = nn.ModuleList([LiteBEVLayer(cfg) for _ in range(cfg.bev_layers)])
        self.out_norm = nn.LayerNorm(d)
        # ---- 固定 3D 参考点（ego 系，buffer 不训练）：(2500, 4, 4, 3) ----
        centers = cfg.bev_cell_centers()               # (50,50,2) cell 中心，1 m/px
        n_sub = cfg.bev_sub_samples                    # 2
        ps = cfg.bev_pixel_size                        # 1.0
        sub_off = np.linspace(-0.25, 0.25, n_sub, dtype=np.float32) * ps  # ±0.25m
        ref = []
        for z in cfg.bev_ref_heights:
            base = np.concatenate([centers, np.full((*centers.shape[:2], 1), z,
                                                    dtype=np.float32)], axis=-1)
            subs = []
            for ox in sub_off:
                for oy in sub_off:
                    p = base.copy()
                    p[..., 0] += ox
                    p[..., 1] += oy
                    subs.append(p)
            ref.append(np.stack(subs, axis=2))
        ref = np.stack(ref, axis=2).reshape(g * g, len(cfg.bev_ref_heights),
                                            n_sub * n_sub, 3)
        self.register_buffer("ref_points", torch.from_numpy(ref).float())  # 40000 个点
        # 共享上采样 trunk：(B,256,50,50) → (B,64,100,100)，地图与检测共用。
        # GroupNorm 而非 BatchNorm：batch=2 时 BN running stats 噪声大，GN 无 batch 统计
        self.trunk = nn.Sequential(
            nn.Conv2d(d, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # ×2 上采样
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
        )
        self.map_out = nn.Conv2d(64, 4, kernel_size=1)  # 4 类：bg/lane/roadblock/walkway
        self.det_heatmap = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1))            # 2 类：vehicle / vru
        self.det_reg = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 6, kernel_size=1))            # off_x,off_y,log_l,log_w,sin,cos
        # CenterNet 惯例：热力图末层偏置 -2.19 ≈ ln(0.1/0.9)，初始 sigmoid ≈ 0.1，
        # 避免初始 focal loss 被大量高置信假阳性主导
        nn.init.constant_(self.det_heatmap[-1].bias, -2.19)
    @torch.no_grad()
    def project_points(self, cam_K: torch.Tensor, cam_ego2cam: torch.Tensor,
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """参考点投影到各相机（纯几何，无梯度）。

        :return: grid (B,NC,P,2) grid_sample 归一化坐标（P=40000）；
                 mask (B,N=2500,NC*n_h*S2=64) 采样点可见性
        """
        cfg = self.cfg
        B, NC = cam_K.shape[:2]
        dev = cam_K.device
        N = self.ref_points.shape[0]
        P = self.ref_points.numel() // 3
        w_img, h_img = cfg.image_size
        pts = self.ref_points.reshape(P, 3)
        pts_h = torch.cat([pts, torch.ones(P, 1, device=dev)], dim=-1)
        # ego→cam：(B,NC,4,4) @ (P,4) → (B,NC,P,4) → xyz
        p_cam = torch.einsum("bcij,pj->bcpi", cam_ego2cam.float(), pts_h)
        p_cam = p_cam[..., :3]                         # (B,NC,P,3)
        # 针孔成像 u = fx·X/Z + cx（Z clamp 防除零）
        z = p_cam[..., 2:3].clamp(min=1e-3)
        uvw = torch.einsum("bcij,bcpj->bcpi", cam_K.float(), p_cam / z)
        u, v = uvw[..., 0], uvw[..., 1]                # 各 (B,NC,P)
        # 可见性：相机前方（z>0.5，防近裁剪面外点）且在画面内
        valid = (p_cam[..., 2] > 0.5) & (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)
        # grid_sample 归一化（align_corners=False 约定）
        gx = (u + 0.5) / w_img * 2 - 1
        gy = (v + 0.5) / h_img * 2 - 1
        grid = torch.stack([gx, gy], dim=-1)
        mask = valid.reshape(B, NC, N, -1).permute(0, 2, 1, 3).reshape(B, N, -1)
        return grid, mask
    def forward(self, patch_tokens: torch.Tensor, cam_K: torch.Tensor,
                cam_ego2cam: torch.Tensor,
                ) -> Dict[str, torch.Tensor]:
        """
        :param patch_tokens: (B,NC,960,d) 当前帧各相机 patch 特征（patch_fuse 后）
        :return: map_logits (B,4,100,100)、det_heatmap (B,2,100,100)、det_reg (B,6,100,100)
        """
        cfg = self.cfg
        B, NC, P_tok, d = patch_tokens.shape
        g = cfg.bev_grid
        N = g * g
        n_h = len(cfg.bev_ref_heights)
        n_s2 = cfg.bev_sub_samples ** 2
        P = N * n_h * n_s2
        gh = cfg.image_size[1] // 16
        gw = cfg.image_size[0] // 16
        grid, mask = self.project_points(cam_K, cam_ego2cam)
        # 关键：patch_tokens 内存布局是 [patch, 通道]，直接 reshape 会把通道与空间维
        # 交错混排。必须先 transpose 成 [通道, patch] 再 reshape —— 否则 BEV 采样
        # 取到空间错位的特征。
        feat = patch_tokens.transpose(2, 3).reshape(B * NC, d, gh, gw)  # (B*4,256,24,40)
        samp = F.grid_sample(
            feat, grid.reshape(B * NC, P, 1, 2),
            mode="bilinear", padding_mode="zeros", align_corners=False)
        samp = samp.reshape(B, NC, d, N, -1).permute(0, 3, 1, 4, 2)
        sampled = samp.reshape(B, N, -1, d)
        bev = self.bev_embed.expand(B, -1, -1)
        for layer in self.layers:
            bev = layer(bev, sampled, mask)
        bev = self.out_norm(bev)
        feat2d = bev.transpose(1, 2).reshape(B, d, g, g)
        up = self.trunk(feat2d)
        return {
            "map_logits": self.map_out(up),
            "det_heatmap": self.det_heatmap(up),
            "det_reg": self.det_reg(up),
        }


class _ForesightBlock(nn.Module):
    """时空可分离注意力块（借鉴 DINO-Foresight 的 separable attention）。

    每层 = 逐位置时序 attn（(B*P,T,d)）→ 逐帧空间 attn（(B*T,P,d)）→ FFN，
    全部 pre-LN 残差。复杂度 O(P·T² + T·P²)，远低于全时空 attention 的 O((T·P)²)。
    空间项让 t+1 的 mask query 能看到邻域位置的历史 → 运动目标跨 patch 的
    特征迁移可学（旧版纯时序逐位置建模在结构上做不到这一点）。
    """

    def __init__(self, d: int, heads: int, dropout: float = 0.1,
                 use_spatial: bool = True):
        super().__init__()
        self.use_spatial = use_spatial
        self.ln_t = nn.LayerNorm(d)
        self.attn_t = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        if use_spatial:
            self.ln_s = nn.LayerNorm(d)
            self.attn_s = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        self.ln_f = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d * 2, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,T,P,d) → (B,T,P,d)。"""
        B, T, P, d = x.shape
        # ---- 时序：同一 patch 位置跨帧 ----
        y = self.ln_t(x).permute(0, 2, 1, 3).reshape(B * P, T, d)
        y = self.attn_t(y, y, y, need_weights=False)[0]
        x = x + self.drop(y.reshape(B, P, T, d).permute(0, 2, 1, 3))
        # ---- 空间：同一帧跨 patch 位置 ----
        if self.use_spatial:
            y = self.ln_s(x).reshape(B * T, P, d)
            y = self.attn_s(y, y, y, need_weights=False)[0]
            x = x + self.drop(y.reshape(B, T, P, d))
        x = x + self.drop(self.ffn(self.ln_f(x)))
        return x


class ForesightPredictor(nn.Module):
    """未来特征预测器（stage 0，DINO-Foresight 化 + AD-E2E-JEPA "proj" 模式）。

    相对旧版的升级（均可在 Config 中开关回退，便于消融）：
      1) 时空可分离注意力（foresight_spatial，默认开）：见 _ForesightBlock；
      2) 预测目标三选一（foresight_target）：
         - "pca"：冻结目标分支多层 concat（1536 维）→ 离线 PCA → foresight_pca_dim；
         - "last"：仅末层 384 维（旧行为）；
         - "proj"：共享可学习 PatchProjector 的投影空间（foresight_proj_dim 维，
           patch 数同时 16×↓），预测器直接在投影空间运行（论文 Eq.8），
           注意力复杂度较 pca 模式降 ~16×，且预测结果可直接用于 rollout/规划；
      3) full-mask 查询 + 任意时距 Δt 查询：训练循环随机采 k∈[1, foresight_max_step]；
      4) 深度 2 → foresight_layers（默认 6）。
    dt/ego Δpose 条件保留（本仓库相对 DINO-Foresight 的驾驶场景特有项，
    对应论文的 action conditioning Eq.3/5：Δpose 即相对位姿动作）。
    """

    def __init__(self, cfg: Config, out_dim: Optional[int] = None):
        super().__init__()
        # proj 模式：预测器全程在投影空间（dp 维、60 patch）运行；
        # pca/last 模式：输入为 patch_fuse 后的 d_model 维（旧行为）
        d = cfg.foresight_proj_dim if cfg.foresight_target == "proj" else cfg.d_model
        self.cfg = cfg
        self.dim = d
        self.out_dim = out_dim or (
            cfg.foresight_proj_dim if cfg.foresight_target == "proj"
            else cfg.foresight_pca_dim if cfg.foresight_target == "pca"
            else 384)
        self.mask_q = nn.Parameter(torch.randn(1, 1, 1, d) * 0.02)
        self.dt_embed = nn.Linear(d, d)
        self.ego_motion_mlp = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([
            _ForesightBlock(d, cfg.foresight_heads, dropout=0.1,
                            use_spatial=cfg.foresight_spatial)
            for _ in range(cfg.foresight_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.pred = nn.Linear(d, self.out_dim)

    def forward(self, patch_hist: torch.Tensor, dt_hist: torch.Tensor,
                motion_hist: torch.Tensor, dt_q: torch.Tensor,
                motion_q: torch.Tensor) -> torch.Tensor:
        """
        :param patch_hist: (B,T,P,d) 历史帧 patch 特征（pca/last：在线分支 patch_fuse 后；
            proj：PatchProjector 后的投影特征）
        :param dt_hist: (B,T) 历史帧相对当前帧的 Δt（秒，≤0）
        :param motion_hist: (B,T,3) 历史帧相对当前帧的 Δpose
        :param dt_q: (B,1) 目标帧 Δt（>0）
        :param motion_q: (B,1,3) 目标帧 Δpose
        :return: (B,P,out_dim) 预测的目标帧特征（投影 / PCA / 末层原始特征空间）
        """
        B, T, P, d = patch_hist.shape
        pe = self.dt_embed(sinusoidal_dt(dt_hist, d))            # (B,T,d)
        mpe = self.ego_motion_mlp(motion_hist)                   # (B,T,d)
        x = patch_hist + (pe + mpe)[:, :, None]                  # (B,T,P,d)
        pe_q = self.dt_embed(sinusoidal_dt(dt_q, d)) \
            + self.ego_motion_mlp(motion_q)                      # (B,1,d)
        q = (self.mask_q + pe_q[:, :, None]).expand(B, 1, P, d)  # full-mask 查询
        x = torch.cat([x, q], dim=1)                             # (B,T+1,P,d)
        for blk in self.blocks:
            x = blk(x)
        return self.pred(self.out_norm(x[:, -1]))                # (B,P,out_dim)


def _grid_norm(xy: torch.Tensor, cfg: Config, ps: float,
               Hm: float, Wm: float) -> torch.Tensor:
    """ego 坐标 (...,2) → grid_sample 归一化网格 (...,2)（align_corners=False）。

    栅格约定（全链路统一）：行 = 前向 x、列 = 横向 y。grid_sample 的
    grid[..., 0] 对应 W 维（列 → y），grid[..., 1] 对应 H 维（行 → x）。
    align_corners=False 下，连续像素坐标 u=(xy-min)/ps 已含 cell 中心 0.5
    （像素 c 的中心 u=c+0.5），归一化 (u/W)*2-1 直接对齐，不能再加 +0.5；
    多加 0.5 会整体偏移半个像素（0.5m 网格上约 0.25m）。
    """
    gx = ((xy[..., 1] - cfg.pc_y_min) / ps) / Wm * 2 - 1         # y → W
    gy = ((xy[..., 0] - cfg.pc_x_min) / ps) / Hm * 2 - 1         # x → H
    return torch.stack([gx, gy], dim=-1)


def _bev_sample_grid(xy: torch.Tensor, cfg: Config) -> torch.Tensor:
    """BEV 语义地图栅格（bev_map_size @ bev_pixel_size/bev_map_stride）。"""
    return _grid_norm(xy, cfg, cfg.bev_pixel_size / cfg.bev_map_stride,
                      *cfg.bev_map_size)


def _det_sample_grid(xy: torch.Tensor, cfg: Config) -> torch.Tensor:
    """检测热力图栅格（det_map_size @ det_pixel_size，与 BEV 地图栅格解耦）。"""
    return _grid_norm(xy, cfg, cfg.det_pixel_size, *cfg.det_map_size)


@torch.no_grad()
def geo_rescore_selection(out: Dict[str, torch.Tensor], cfg: Config
                          ) -> Dict[str, torch.Tensor]:
    """推理时几何重打分（MGMP arXiv:2610.10646 思想：学习先验提议、显式几何裁决）。

    现状：最终轨迹 = argmax(学习 scorer)。scorer 可能选中在预测 BEV 上明明
    压线/撞障碍的 proposal（学习分与几何约束之间存在鸿沟）。本函数对 scorer
    top-m 的候选做显式校验并重选：

      geo = (w_s·softmax_m(scorer) + w_lane·车道合规 − w_col·碰撞代理
             − w_oob·越界 + w_prog·进度)

      * 车道合规：waypoint 处 map_logits softmax 的（lane+roadblock）概率均值
        （软代价——地图是网络预测有噪声，不做硬约束，与 MGMP 的精确几何不同）；
      * 碰撞代理：ego footprint（后轴中心 + 四角，含 col_margin 膨胀）处
        sigmoid(det_heatmap) 各类最大值，取 8 步 × 5 点的 max
        （单点高置信命中即否决该候选，MGMP 的硬碰撞 veto 的软版本；
        预测热力图是静态快照、不含未来运动，故仍是加权项而非严格门控）；
      * 越界：任一步 footprint 采样点落在栅格外的比例（按步取 max）；未知区域
        不再按“无碰撞/可行驶”处理，由 w_g_oob 单独惩罚（避免把未知区域当安全区）；
      * scorer 分：在 top-m 候选内做 softmax（相对偏好，量纲与其余项可比；
        若对全 64 条 softmax，峰值分布会独吞 ~1.0 压垮几何项）；
      * 进度：末端前向位移限幅 20m。

    全部经 grid_sample 双线性采样，可微、纯张量、无逐样本循环。
    训练时不调用（训练损失直接作用于 proposal_list / scores）。

    :param out: model.forward 的输出（需含 proposal_list / proposal_scores /
        map_logits / det_heatmap）
    :return: 原地更新并返回 out：覆盖 "trajectory"，新增 "geo_sel"（选中下标）、
        "geo_scores"（top-m 几何分）、"geo_oob"（top-m 越界比例）
    """
    props = out["proposal_list"][-1].float()           # (B,N,K,3)
    scores = out["proposal_scores"].float()            # (B,N)
    B, N, K = props.shape[:3]
    m = min(cfg.geo_rescore_m, N)
    top = scores.topk(m, dim=1).indices                # (B,m)
    cand = props.gather(1, top[:, :, None, None].expand(-1, -1, K, 3))  # (B,m,K,3)
    xy = cand[..., :2]
    grid = _bev_sample_grid(xy, cfg)[:, :, :, None, :]                 # BEV 地图栅格
    # 车道合规：预测地图 drivable 类（1=lane、2=roadblock）soft 概率；
    # 出图 waypoint 不计入（其概率由 padding=0 得到，会系统性低估）
    p_map = torch.softmax(out["map_logits"].float(), dim=1)
    drivable = (p_map[:, 1] + p_map[:, 2])[:, None]                    # (B,1,H,W)
    lane = F.grid_sample(drivable.expand(B, m, -1, -1).reshape(B * m, 1, *drivable.shape[-2:]),
                         grid.reshape(B * m, K, 1, 2),
                         mode="bilinear", padding_mode="zeros",
                         align_corners=False).reshape(B, m, K)         # (B,m,K)
    valid_lane = ((grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)).squeeze(-1)
    # 出图 waypoint 从 lane 均值中剔除（其 padding=0 概率会低估）；只保留
    # 落到地图内的 waypoint 求比例，避免“分母恒 8、分子先乘 0”导致偏低。
    # 全部出图时 lane=0，由下方 oob 单独严惩，避免同一越界被重复扣分。
    valid_f = valid_lane.float()
    lane = (lane * valid_f).sum(-1) / valid_f.sum(-1).clamp_min(1.0)  # (B,m)

    # 碰撞代理：ego footprint（后轴中心 + 四角，含 col_margin 膨胀）采样，
    # 8 步 × 5 点取 max——4.9m×2.3m 的车不是一个点，只采样中心会漏掉路沿/障碍。
    yaw = cand[..., 2]                                                # (B,m,K)
    c, s = torch.cos(yaw), torch.sin(yaw)
    hl = 0.5 * (cfg.ego_len + 2.0 * cfg.col_margin)
    hw = 0.5 * (cfg.ego_wid + 2.0 * cfg.col_margin)
    ox = torch.tensor([hl, hl, -hl, -hl, 0.0], device=props.device, dtype=props.dtype)
    oy = torch.tensor([hw, -hw, hw, -hw, 0.0], device=props.device, dtype=props.dtype)
    dx = ox[None, None, None, :] * c[..., None] - oy[None, None, None, :] * s[..., None]
    dy = ox[None, None, None, :] * s[..., None] + oy[None, None, None, :] * c[..., None]
    foot_xy = torch.stack([cand[..., 0, None] + dx, cand[..., 1, None] + dy], dim=-1)  # (B,m,K,P,2)
    grid_foot = _det_sample_grid(foot_xy, cfg)                        # (B,m,K,P,2)
    valid_foot = ((grid_foot[..., 0].abs() <= 1.0)
                  & (grid_foot[..., 1].abs() <= 1.0))                 # (B,m,K,P)
    heat = torch.sigmoid(out["det_heatmap"].float()).amax(dim=1, keepdim=True)  # (B,1,H,W)
    col_vals = F.grid_sample(heat.expand(B, m, -1, -1).reshape(B * m, 1, *heat.shape[-2:]),
                             grid_foot.reshape(B * m, K * 5, 1, 2),
                             mode="bilinear", padding_mode="zeros",
                             align_corners=False).reshape(B, m, K, 5)
    col_vals = torch.where(valid_foot, col_vals, torch.zeros_like(col_vals))
    col = col_vals.amax(dim=-1).amax(dim=-1)                          # (B,m)
    # 未知区域（出图）不再按“无碰撞”处理：取“任一步 footprint 出图比例”的最大值，
    # 避免只有末端一步越界时被 8 步平均稀释。
    oob = (1.0 - valid_foot.float().mean(dim=-1)).amax(dim=-1)        # (B,m)

    prog = (cand[:, :, -1, 0] / 20.0).clamp(0.0, 1.0)                  # (B,m)
    s_prob = torch.softmax(scores.gather(1, top), dim=-1)              # top-m 内 softmax
    geo = (cfg.w_g_scorer * s_prob + cfg.w_g_lane * lane
           - cfg.w_g_col * col - cfg.w_g_oob * oob
           + cfg.w_g_prog * prog)                                     # (B,m)
    sel_m = geo.argmax(dim=1)                                          # (B,)
    sel = top.gather(1, sel_m[:, None]).squeeze(1)                     # 原 N 维下标
    out["trajectory"] = props[torch.arange(B, device=props.device), sel]
    out["geo_sel"] = sel
    out["geo_scores"] = geo
    out["geo_oob"] = oob
    return out


class MyDriveModel(nn.Module):
    """DINOv3+LoRA 多帧编码 + 时序 planner + lite BEV 头。

    输入 batch 键（来自 OverfitDataset）：
      img_cur (B,4,3,H,W) / img_hist (B,3,3,H,W) / cam_K (B,4,3,3) /
      cam_ego2cam (B,4,4,4) / ego_status (B,4,11) / ego_motion (B,4,3) / frame_dt (B,4)
    输出：trajectory (B,8,3)、map_logits (B,4,100,100)、det_heatmap (B,2,100,100)、
          det_reg (B,6,100,100)
    """

    def __init__(self, cfg: Config, weights_path: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.encoder = DinoEncoder(cfg, weights_path)
        embed_dim = self.encoder.embed_dim             # 384
        self.gh, self.gw = self.encoder.gh, self.encoder.gw  # 24, 40
        # 每相机 16 个可学习 scene token（4 路相机；历史前视复用 cam0 的）。
        # std=0.02 初始化（过小会导致冻结期 token 恒等、解冻后坍缩）
        self.scene_embeds = nn.Parameter(
            torch.randn(1, 4, cfg.num_scene_tokens, embed_dim) * 0.02)  # (1,4,16,384)
        self.planner = ProposalPlanner(cfg)
        self.bev_head = LiteBEVHead(cfg)
        # ---- foresight projector（stage 0 "proj" 模式；其余模式不参与前向） ----
        # 恒创建：① 权重随 checkpoint 保存，stage 0 训出的 projector 可被下游
        # 阶段/迁移实验直接复用（论文 §3.4：预训练 projector 提升下游模仿学习）；
        # ② 加载旧 checkpoint 时 strict=False 下该模块随机初始化，无副作用。
        self.foresight_projector = PatchProjector(cfg)
        # ---- foresight PCA 目标（stage 0 "pca" 模式）：由 md_pretext prepare-pca
        # 离线生成、load_foresight_pca 载入。注册为 buffer → 随 checkpoint 保存/加载、
        # 随 .to() 迁移。恒等占位 + ready=0：未载入 PCA 就在 pca 模式下前向会显式报错。
        in_dim = self.encoder.embed_dim * 4                     # 4 层 concat = 1536
        self.register_buffer("foresight_pca_mean", torch.zeros(in_dim))
        self.register_buffer(
            "foresight_pca_comp",
            torch.eye(in_dim)[: cfg.foresight_pca_dim].clone())  # (pca_dim, in_dim) 占位
        self.register_buffer("foresight_pca_ready",
                             torch.zeros((), dtype=torch.uint8))

    def _encode(self, imgs: torch.Tensor, cams: List[int], need_patch: bool = True,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """imgs (B,M,3,H,W) → scene (B,M,S,d)、patch (B,M,gh*gw,d)（或 None）。

        B 与 M 合并成 B*M 张图一次过 DINO（DINO 是单图编码器）。
        """
        B, M = imgs.shape[:2]
        st = self.scene_embeds[:, cams]
        st = st.expand(B, -1, -1, -1).reshape(B * M, self.cfg.num_scene_tokens, -1)
        flat = imgs.reshape(B * M, *imgs.shape[2:])
        scene, patch = self.encoder(flat, st, need_patch=need_patch)
        return (scene.reshape(B, M, self.cfg.num_scene_tokens, -1),
                patch.reshape(B, M, self.gh * self.gw, -1) if need_patch else None)
    def forward(self, batch: Dict[str, torch.Tensor], with_planner: bool = True,
                with_bev: bool = True, dino_no_grad: bool = False,
                ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        # 三个开关对应 3 阶段日程（见 set_train_stage）：stage 1 只 planner、
        # stage 2 只 BEV（DINO no_grad 省显存）、stage 3 联合
        def _run_dino():
            # 当前帧 4 路（M=4）：scene + patch（BEV 需要 patch）
            scene_cur, patch_cur = self._encode(batch["img_cur"], cams=[0, 1, 2, 3],
                                                need_patch=with_bev)
            # 历史 3 帧前视（M=3）：只要 scene（不用 patch，省算力）
            scene_hist = None
            if with_planner:
                scene_hist, _ = self._encode(batch["img_hist"], cams=[0, 0, 0],
                                             need_patch=False)
            return scene_cur, patch_cur, scene_hist
        if dino_no_grad:
            with torch.no_grad():
                scene_cur, patch_cur, scene_hist = _run_dino()
        else:
            scene_cur, patch_cur, scene_hist = _run_dino()
        if with_planner:
            # 当前帧 4 路 scene tokens 全量喂 planner：(B,4,16,256) → (B,64,256)
            r_all = scene_cur.reshape(scene_cur.shape[0], -1, scene_cur.shape[-1])
            queue = scene_hist.reshape(scene_hist.shape[0], -1, scene_hist.shape[-1])
            out.update(self.planner(
                r_all, queue, batch["ego_status"], batch["ego_motion"], batch["frame_dt"]))
        if with_bev:
            out.update(self.bev_head(patch_cur, batch["cam_K"], batch["cam_ego2cam"]))
        # 推理/评估时的几何重打分（MGMP）：学习 scorer 提议 top-m，显式几何裁决。
        # 训练态跳过——训练损失直接作用于 proposal_list / scores，重选不影响梯度。
        if (with_planner and with_bev and self.cfg.geo_rescore and not self.training):
            out = geo_rescore_selection(out, self.cfg)
        return out

    def make_frozen_target_branch(self) -> None:
        """深拷贝整个编码器（含 LoRA）作为冻结目标分支（stage 0 开始时调用一次）。

        必须深拷贝而非只快照 scene_embeds：LoRA 位于 backbone 每个 block 内，
        只快照 scene_embeds 会让目标前向仍走可训 LoRA（目标随训练漂移）。
        存入 self.__dict__ 以绕过 nn.Module 子模块注册：否则会①重复写进每个
        checkpoint；②落入 param_groups 兜底分支，将来被 set_train_stage 误解冻。
        注意：__dict__ 中的模块不随 model.to(device) 移动，本调用已保证在 .to() 之后。
        """
        import copy
        self.__dict__["_tgt_encoder"] = copy.deepcopy(self.encoder)
        for p in self._tgt_encoder.parameters():
            p.requires_grad = False
        self._tgt_scene_embeds = self.scene_embeds.detach().clone()
    def load_foresight_pca(self, pca_path) -> None:
        """载入 prepare-pca 生成的 PCA 矩阵（mean + components）。"""
        d = torch.load(str(pca_path), map_location="cpu", weights_only=True)
        assert d["components"].shape == tuple(self.foresight_pca_comp.shape), \
            f"PCA 维度不符：{d['components'].shape} vs {tuple(self.foresight_pca_comp.shape)}"
        self.foresight_pca_mean.copy_(d["mean"].float())
        self.foresight_pca_comp.copy_(d["components"].float())
        self.foresight_pca_ready.fill_(1)
        print(f"[foresight] PCA 目标已载入（{d.get('n_samples', '?')} 样本，"
              f"explained_var={d.get('explained_var', float('nan')):.3f}）")

    def foresight_target_from(self, img: torch.Tensor) -> torch.Tensor:
        """冻结目标分支提目标帧特征：pca 模式 = 4 层 concat(1536)→PCA 投影；
        last 模式 = 仅末层 384 维（旧行为，消融用）。img: (B,3,H,W)。"""
        B = img.shape[0]
        st = self._tgt_scene_embeds[:, 0].expand(B, -1, -1)
        final, inters = self._tgt_encoder.model.forward_features(img, st)
        n_pre = self._tgt_encoder.n_prefix
        if self.cfg.foresight_target == "pca":
            if not bool(self.foresight_pca_ready):
                raise RuntimeError(
                    "foresight_target='pca' 但未载入 PCA 矩阵："
                    "先运行 python -m my_drive.train.md_pretext prepare-pca，"
                    "或将 cfg.foresight_target 改为 'last'")
            raw = torch.cat([t[:, n_pre:] for t in inters] + [final[:, n_pre:]],
                            dim=-1)                              # (B,P,1536)
            return (raw.float() - self.foresight_pca_mean) @ self.foresight_pca_comp.T
        return final[:, n_pre:]                                  # (B,P,384)

    def _foresight_target_patch(self, img: torch.Tensor) -> torch.Tensor:
        """冻结目标分支的 patch_fuse 特征 (B,P,d_model)（proj 模式目标侧输入）。

        目标分支整体冻结（deepcopy 于 stage 0 起点）→ 256 维目标特征不漂移；
        共享 projector 的可训练性由损失侧控制（pred loss 中 sg，SIGReg 保留梯度，
        论文 Eq.9/10）。
        """
        B = img.shape[0]
        st = self._tgt_scene_embeds[:, 0].expand(B, -1, -1)
        final, inters = self._tgt_encoder.model.forward_features(img, st)
        n_pre = self._tgt_encoder.n_prefix
        raw = torch.cat([t[:, n_pre:] for t in inters] + [final[:, n_pre:]],
                        dim=-1)                                  # (B,P,1536)
        return self._tgt_encoder.patch_fuse(raw)                 # (B,P,256)

    def _project_patches(self, patch: torch.Tensor) -> torch.Tensor:
        """patch_fuse 特征 (B,T,gh*gw,d) → 共享 projector 投影 (B,T,(gh//4)(gw//4),dp)。"""
        B, T = patch.shape[:2]
        z = self.foresight_projector(
            patch.reshape(B * T, *patch.shape[2:]), self.gh, self.gw)
        return z.reshape(B, T, *z.shape[1:])

    def forward_foresight(self, batch: Dict[str, torch.Tensor], predictor: nn.Module,
                          future_step: int = 1,
                          ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """返回 (预测目标帧特征, 目标特征, aux)。

        多步长（DINO-Foresight full-mask + 随机时距）：batch 为 8 帧窗口
        f0..f7 = t-3..t+4（前 4 帧历史，后 4 帧候选目标），目标 = f[3+future_step]，
        k∈[1, cfg.foresight_max_step] 由训练循环随机采样（0.5/1.0/1.5/2.0s）。

        在线分支：t-3..t 四帧 patch 特征（带梯度）；
        目标分支：目标帧走完全冻结的编码器副本（特征提取 no_grad），目标不漂移。

        "proj" 模式（AD-E2E-JEPA）：两侧 patch_fuse 特征经共享 projector 压缩后
        预测/监督（目标侧在损失中 sg，见 md_pretext）；aux 含 z_hist / z_tgt
        （保留 projector 梯度，供 SIGReg 使用，论文 Eq.10）。
        "pca"/"last" 模式：行为与旧版一致，aux 为空 dict。
        """
        imgs = batch["imgs"]                                     # (B,8,3,H,W)
        k = int(future_step)
        _, patch_in = self._encode(imgs[:, :4], cams=[0, 0, 0, 0], need_patch=True)
        if self.cfg.foresight_target == "proj":
            z_hist = self._project_patches(patch_in)             # (B,4,P',dp)
            with torch.no_grad():
                tgt_feat = self._foresight_target_patch(imgs[:, 3 + k])
            z_tgt = self.foresight_projector(tgt_feat, self.gh, self.gw)  # (B,P',dp)
            pred = predictor(z_hist,
                             batch["dt8"][:, :4], batch["motion8"][:, :4],
                             batch["dt8"][:, 3 + k: 4 + k],
                             batch["motion8"][:, 3 + k: 4 + k])
            return pred, z_tgt, {"z_hist": z_hist, "z_tgt": z_tgt}
        with torch.no_grad():
            tgt = self.foresight_target_from(imgs[:, 3 + k])
        pred = predictor(patch_in,
                         batch["dt8"][:, :4], batch["motion8"][:, :4],
                         batch["dt8"][:, 3 + k: 4 + k],
                         batch["motion8"][:, 3 + k: 4 + k])
        return pred, tgt, {}

    def forward_foresight_rollout(
            self, batch: Dict[str, torch.Tensor], predictor: nn.Module,
            num_steps: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """proj 模式可选：自回归 rollout（论文 Eq.12 / A.2.2，TBPTT 截断反传）。

        从初始上下文 z_{t-3..t} 出发逐步预测 f4..f[3+num_steps]：
        每步上下文丢弃最旧帧、追加最新预测；首步上下文保留梯度（复用 TF 链），
        后续步对上下文 sg（TBPTT，梯度不回穿更早的 rollout 步）。
        dt/motion 条件逐步用 _se2_rebase 重基到新的当前帧（论文 Eq.3 的相对位姿
        动作在移动原点下的等价变换）。

        :return: preds、z_tgts 各 num_steps 个 (B,P',dp)；z_tgts 保留 projector
                 梯度（供 SIGReg），pred 监督时在损失侧 sg 目标。
        """
        assert self.cfg.foresight_target == "proj", "rollout 仅在 proj 模式下可用"
        imgs, dt8, mo8 = batch["imgs"], batch["dt8"], batch["motion8"]
        _, patch_in = self._encode(imgs[:, :4], cams=[0, 0, 0, 0], need_patch=True)
        z_ctx = self._project_patches(patch_in)                  # (B,4,P',dp)
        preds: List[torch.Tensor] = []
        tgts: List[torch.Tensor] = []
        for j in range(1, num_steps + 1):
            cur = 3 + j - 1                                      # 当前帧下标
            win = slice(j - 1, j - 1 + 4)                        # 上下文窗口 j-1..j+2
            dt_hist = dt8[:, win] - dt8[:, cur:cur + 1]          # 重基到当前帧
            mo_hist = _se2_rebase(mo8[:, win], 3)                # 窗口末帧 = 当前帧
            dt_q = (dt8[:, 3 + j] - dt8[:, cur])[:, None]        # (B,1)
            mo_q = _se2_rebase(torch.stack([mo8[:, cur], mo8[:, 3 + j]], dim=1),
                               0)[:, 1:2]                        # (B,1,3)
            ctx_in = z_ctx if j == 1 else z_ctx.detach()         # TBPTT
            pred = predictor(ctx_in, dt_hist, mo_hist, dt_q, mo_q)
            with torch.no_grad():
                tgt_feat = self._foresight_target_patch(imgs[:, 3 + j])
            z_tgt = self.foresight_projector(tgt_feat, self.gh, self.gw)
            preds.append(pred)
            tgts.append(z_tgt)
            # 滑窗：丢最旧帧、追加最新预测（论文 Eq.22）
            z_ctx = torch.cat([z_ctx[:, 1:], pred[:, None]], dim=1)
        return preds, tgts

    def param_groups(self) -> Dict[str, List[nn.Parameter]]:
        """按模块名分组参数：dino（主干，恒冻结）/ lora / planner / bev / foresight。

        - encoder.* 中 LoRA 支路（linear_a_/linear_b_）与 neck/patch_fuse 归 lora；
          encoder 其余（ViT 权重）归 dino（恒冻结）；
        - scene_embeds 归 lora；planner.* 归 planner；bev_head.* 归 bev；
        - foresight_projector.* 归 foresight（仅 stage 0 "proj" 模式训练；
          stage 1-3 恒冻结，见 set_train_stage）。
        """
        groups: Dict[str, List[nn.Parameter]] = {
            "dino": [], "lora": [], "planner": [], "bev": [], "foresight": []}
        for name, p in self.named_parameters():
            if name.startswith("encoder."):
                if ("linear_a_" in name or "linear_b_" in name or ".neck." in name
                        or ".patch_fuse." in name):
                    groups["lora"].append(p)
                else:
                    groups["dino"].append(p)
            elif name.startswith("scene_embeds"):
                groups["lora"].append(p)
            elif name.startswith("planner."):
                groups["planner"].append(p)
            elif name.startswith("bev_head."):
                groups["bev"].append(p)
            elif name.startswith("foresight_projector."):
                groups["foresight"].append(p)
            else:
                groups["planner"].append(p)
        return groups
    def set_train_stage(self, stage: int) -> Dict[str, bool]:
        """3 阶段冻结日程（直接切 requires_grad，调用方负责新建 optimizer）：

        stage 1: 冻结 DINO；LoRA + Planner 可训（不前向 BEV）
        stage 2: 冻结 DINO + LoRA + Planner；BEV 可训（DINO 前向 no_grad，省显存）
        stage 3: 冻结 DINO；LoRA + Planner + BEV 可训（联合）

        foresight_projector 在 stage 1-3 恒冻结（不参与下游前向；
        如需按论文 §3.4 把预训练 projector 迁移进下游结构，另行接入）。
        """
        trainable = {
            1: {"lora": True, "planner": True, "bev": False},
            2: {"lora": False, "planner": False, "bev": True},
            3: {"lora": True, "planner": True, "bev": True},
        }[stage]
        groups = self.param_groups()
        for p in groups["dino"]:
            p.requires_grad = False
        for p in groups["foresight"]:
            p.requires_grad = False
        for gname, flag in trainable.items():
            for p in groups[gname]:
                p.requires_grad = flag
        flags = {
            "with_planner": stage != 2,
            "with_bev": stage != 1,
            "dino_no_grad": stage == 2,
        }
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[stage {stage}] 可训练参数 {n_train/1e6:.2f}M | "
              f"planner={flags['with_planner']} bev={flags['with_bev']} "
              f"dino_no_grad={flags['dino_no_grad']}")
        return flags
