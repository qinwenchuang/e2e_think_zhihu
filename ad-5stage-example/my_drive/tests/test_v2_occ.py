"""V2 占据/BEV scorer 的轻量回归测试（不依赖 GPU/数据）。"""
import torch

from my_drive.md_common import Config
from my_drive.train.md_spatial import BEVProposalFeatures, ContinuousOccDecoder


def test_occ_decoder_shape():
    cfg = Config()
    dec = ContinuousOccDecoder(cfg)
    bev = torch.randn(2, cfg.bev_grid * cfg.bev_grid, cfg.d_model)
    q = torch.randn(2, 128, 3) * torch.tensor([15.0, 10.0, 2.0])
    out = dec(bev, q)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()


def test_proposal_features_shape():
    cfg = Config()
    feat = BEVProposalFeatures(cfg)
    B, N = 2, 4
    proposals = torch.zeros(B, N, 8, 3)
    proposals[..., 0] = torch.linspace(1, 20, 8)
    bev = {
        "map_logits": torch.randn(B, 4, 100, 100),
        "det_heatmap": torch.randn(B, 2, 100, 100),
        "bev_features": torch.randn(B, cfg.bev_grid * cfg.bev_grid, cfg.d_model),
    }
    out = feat(proposals, bev)
    assert out.shape == (B, N, feat.feature_dim)
    assert torch.isfinite(out).all()


def test_occ_config_defaults():
    cfg = Config()
    assert cfg.occ_enabled and cfg.occ_num_queries > 0
    assert cfg.w_occ > 0 and cfg.occ_fourier_bands >= 1
    assert cfg.planner_use_bev and cfg.planner_bev_detach
