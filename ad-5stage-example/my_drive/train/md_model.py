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

全局符号约定：
  B    batch；M 一次编码图像数（当前帧 M=4 / 历史 M=3 / pretext M=5）
  H,W  图像高宽 = 384, 640；S 每相机 scene token = 16
  E   DINOv3 特征维 = 384；d 模型维 = 256
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
      argmax(quality)             → 最终轨迹
    监督策略（见 md_loss）：WTA（每层只监督离 GT 最近的那条）+ 质量分数回归负 ADE。
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model                                # 256
        N = cfg.num_proposals                          # 64
        out_dim = NUM_FUTURE_POSES * 3                 # 24 = 8 步 × (x,y,heading)
        self.proposal_embed = nn.Embedding(N, d)
        self.ego_mlp = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))
        self.dt_embed = nn.Linear(d, d)
        self.ego_motion_mlp = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.init_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, out_dim))
        self.decoder = nn.ModuleList([
            _DecoderBlock(d, cfg.planner_heads, d * 2) for _ in range(cfg.decoder_layers)])
        self.refine_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, out_dim))
            for _ in range(cfg.decoder_layers)])
        self.pos_embed = nn.Sequential(nn.Linear(out_dim, d * 2), nn.ReLU(),
                                       nn.Linear(d * 2, d))
        self.scorer = nn.ModuleList([
            _DecoderBlock(d, cfg.planner_heads, d * 2) for _ in range(cfg.scorer_layers)])
        self.quality_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
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
        proposal_list = [self.init_head(tokens).reshape(B, N, NUM_FUTURE_POSES, 3)]
        # 逐层精修：tokens 过 decoder，每层回归一次 proposals（深监督）
        for block, head in zip(self.decoder, self.refine_heads):
            tokens = block(tokens, ctx)                              # (B,N,d)
            proposal_list.append(head(tokens).reshape(B, N, NUM_FUTURE_POSES, 3))
        final_props = proposal_list[-1]                              # (B,N,8,3)
        # scorer：proposal 嵌入（detach，scorer 梯度不回流生成器）→ 质量 logit
        embedded = self.pos_embed(final_props.detach().reshape(B, N, -1))  # (B,N,d)
        for block in self.scorer:
            embedded = block(embedded, ctx)
        scores = self.quality_head(embedded + ego_token).squeeze(-1)   # (B,N)
        # argmax 质量 logit 选轨迹
        sel = scores.argmax(dim=1)                                   # (B,)
        trajectory = final_props[torch.arange(B, device=scores.device), sel]  # (B,8,3)
        return {"trajectory": trajectory, "proposal_list": proposal_list,
                "proposal_scores": scores}


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
class ForesightPredictor(nn.Module):
    """逐 patch 位置的时序预测器（stage 0 用）：从 t-3..t 的 patch 特征预测
    t+1 帧同位置**原始 DINO patch 特征**（384 维，目标全程冻结 → 不漂移、无 EMA）。

    逐位置时序 attention（每位置 5 个 token）极轻量；空间上下文已由 DINO 自带。
    """

    def __init__(self, cfg: Config, out_dim: int = 384):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.mask_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.dt_embed = nn.Linear(d, d)
        self.ego_motion_mlp = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=cfg.planner_heads,
                                           dim_feedforward=d * 2, dropout=0.1,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.out_norm = nn.LayerNorm(d)
        self.pred = nn.Linear(d, out_dim)
    def forward(self, patch_hist: torch.Tensor, dt5: torch.Tensor,
                motion5: torch.Tensor) -> torch.Tensor:
        """
        :param patch_hist: (B,4,P=960,d) 输入帧 t-3..t 的 patch 特征（在线分支）
        :param dt5: (B,5) 各帧（含目标帧 t+1）相对 t 的 Δt（秒）
        :param motion5: (B,5,3) 各帧（含目标帧）相对 t 的 Δpose
        :return: (B,P,out_dim) 预测的 t+1 原始 patch 特征
        """
        B, nf, P, d = patch_hist.shape
        x = patch_hist.permute(0, 2, 1, 3).reshape(B * P, nf, d)
        pe = self.dt_embed(sinusoidal_dt(dt5[:, :4], d))
        mpe = self.ego_motion_mlp(motion5[:, :4])
        x = x + (pe + mpe)[:, None].expand(B, P, nf, d).reshape(B * P, nf, d)
        pe_q = self.dt_embed(sinusoidal_dt(dt5[:, 4:], d)) \
            + self.ego_motion_mlp(motion5[:, 4:])
        q = (self.mask_q + pe_q)[:, None].expand(B, P, 1, d).reshape(B * P, 1, d)
        out = self.out_norm(self.encoder(torch.cat([x, q], dim=1)))
        return self.pred(out[:, -1]).reshape(B, P, -1)
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
    def forward_foresight(self, batch: Dict[str, torch.Tensor], predictor: nn.Module,
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (预测 t+1 patch 特征 (B,P,384), 目标原始 patch 特征 (B,P,384))。

        在线分支：t-3..t 四帧 patch 特征（带梯度）；
        目标分支：t+1 帧走完全冻结的编码器副本（no_grad），目标不漂移。
        """
        imgs = batch["imgs"]
        B = imgs.shape[0]
        _, patch_in = self._encode(imgs[:, :4], cams=[0, 0, 0, 0], need_patch=True)
        with torch.no_grad():
            st = self._tgt_scene_embeds[:, 0].expand(B, -1, -1)
            final, _ = self._tgt_encoder.model.forward_features(imgs[:, 4], st)
            tgt = final[:, self._tgt_encoder.n_prefix:]
        pred = predictor(patch_in, batch["dt5"], batch["motion5"])
        return pred, tgt
    def param_groups(self) -> Dict[str, List[nn.Parameter]]:
        """按模块名分组参数：dino（主干，恒冻结）/ lora / planner / bev。

        - encoder.* 中 LoRA 支路（linear_a_/linear_b_）与 neck/patch_fuse 归 lora；
          encoder 其余（ViT 权重）归 dino（恒冻结）；
        - scene_embeds 归 lora；planner.* 归 planner；bev_head.* 归 bev。
        """
        groups: Dict[str, List[nn.Parameter]] = {"dino": [], "lora": [], "planner": [], "bev": []}
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
            else:
                groups["planner"].append(p)
        return groups
    def set_train_stage(self, stage: int) -> Dict[str, bool]:
        """3 阶段冻结日程（直接切 requires_grad，调用方负责新建 optimizer）：

        stage 1: 冻结 DINO；LoRA + Planner 可训（不前向 BEV）
        stage 2: 冻结 DINO + LoRA + Planner；BEV 可训（DINO 前向 no_grad，省显存）
        stage 3: 冻结 DINO；LoRA + Planner + BEV 可训（联合）
        """
        trainable = {
            1: {"lora": True, "planner": True, "bev": False},
            2: {"lora": False, "planner": False, "bev": True},
            3: {"lora": True, "planner": True, "bev": True},
        }[stage]
        groups = self.param_groups()
        for p in groups["dino"]:
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
