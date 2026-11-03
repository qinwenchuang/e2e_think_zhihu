"""PA+BANE 单元测试：pytest -q my_drive/tests/test_md_map_ccl.py"""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from my_drive.md_common import Config
from my_drive.train.md_loss import compute_losses
from my_drive.train.md_map_ccl import bane_negative_mask, pixel_to_anchor_bane_loss
from my_drive.train.md_model import LiteBEVHead


def _classification_case(batch: int = 2, size: int = 12):
    gt = torch.zeros(batch, size, size, dtype=torch.long)
    gt[..., size // 3:] = 1
    gt[..., 2 * size // 3:] = 2
    gt[..., -2:, -2:] = 3
    logits = torch.randn(batch, 4, size, size)
    embedding = torch.randn(batch, 8, size, size, requires_grad=True)
    return embedding, logits, gt


class TestBaneSampling(unittest.TestCase):
    def test_selects_only_misclassified_boundary_subset(self):
        gt = torch.zeros(1, 10, 10, dtype=torch.long)
        gt[..., :5] = 1
        gt[..., 5:] = 2
        logits = torch.zeros(1, 4, 10, 10)
        logits[:, 1] = 1.0  # 全部预测为类 1，右半类 2 为误分类区域
        mask = bane_negative_mask(logits, gt, ratio=0.3)
        errors = (gt == 2)
        self.assertTrue(mask.any())
        self.assertLess(int(mask.sum()), int(errors.sum()))
        self.assertFalse(bool((mask & ~errors).any()))
        # 距离变换取样应优先误差区域边缘，而非全部内部像素。
        self.assertTrue(bool(mask[:, 5].any()) or bool(mask[:, 9].any()))

    def test_invalid_ratio_fails_fast(self):
        logits = torch.zeros(1, 4, 4, 4)
        gt = torch.zeros(1, 4, 4, dtype=torch.long)
        with self.assertRaises(ValueError):
            bane_negative_mask(logits, gt, ratio=0.0)


class TestPixelAnchorLoss(unittest.TestCase):
    def test_loss_backward_and_diagnostics(self):
        cfg = Config(map_ccl_enabled=True, map_ccl_pos_samples=32,
                     map_ccl_neg_samples=32)
        embedding, logits, gt = _classification_case()
        loss, info = pixel_to_anchor_bane_loss(embedding, logits, gt, cfg)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertGreaterEqual(float(info["map_bane_fraction"]), 0.0)
        self.assertLessEqual(float(info["map_bane_fraction"]), 1.0)
        self.assertGreater(float(info["map_pa_terms"]), 0.0)
        loss.backward()
        self.assertIsNotNone(embedding.grad)
        self.assertTrue(torch.isfinite(embedding.grad).all())

    def test_perfect_prediction_falls_back_to_regular_negatives(self):
        cfg = Config(map_ccl_pos_samples=16, map_ccl_neg_samples=16)
        embedding, _, gt = _classification_case(batch=1)
        logits = F.one_hot(gt, num_classes=4).permute(0, 3, 1, 2).float() * 20.0
        loss, info = pixel_to_anchor_bane_loss(embedding, logits, gt, cfg)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(info["map_bane_fraction"]), 0.0)
        self.assertGreater(float(info["map_pa_terms"]), 0.0)

    def test_single_class_returns_differentiable_zero(self):
        cfg = Config()
        embedding = torch.randn(1, 8, 4, 4, requires_grad=True)
        logits = torch.zeros(1, 4, 4, 4)
        gt = torch.zeros(1, 4, 4, dtype=torch.long)
        loss, info = pixel_to_anchor_bane_loss(embedding, logits, gt, cfg)
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(float(info["map_pa_terms"]), 0.0)
        loss.backward()
        self.assertIsNotNone(embedding.grad)


class TestModelAndLossIntegration(unittest.TestCase):
    def test_bev_head_ccl_is_training_only(self):
        cfg = Config(d_model=16, bev_heads=4, bev_layers=1, bev_grid=4,
                     image_size=(32, 32), bev_ref_heights=(0.0,),
                     bev_sub_samples=1, map_ccl_enabled=True, map_ccl_embed_dim=8)
        head = LiteBEVHead(cfg)
        patch = torch.randn(1, 1, 4, 16)
        cam_k = torch.eye(3).reshape(1, 1, 3, 3)
        cam_e2c = torch.eye(4).reshape(1, 1, 4, 4)
        head.eval()
        with torch.no_grad():
            eval_out = head(patch, cam_k, cam_e2c)
        self.assertNotIn("map_ccl_embedding", eval_out)
        head.train()
        train_out = head(patch, cam_k, cam_e2c)
        self.assertEqual(train_out["map_ccl_embedding"].shape, (1, 8, 8, 8))
        gt = torch.randint(0, 4, (1, 8, 8))
        ccl, _ = pixel_to_anchor_bane_loss(
            train_out["map_ccl_embedding"], train_out["map_logits"], gt, cfg)
        total = train_out["map_logits"].float().mean() * 0.0 + ccl
        total.backward()
        self.assertIsNotNone(head.map_ccl_proj.weight.grad)

    def test_compute_losses_adds_ccl_only_with_training_embedding(self):
        cfg = Config(map_ccl_enabled=True, w_map_ccl=0.1,
                     map_ccl_pos_samples=16, map_ccl_neg_samples=16)
        B, H, W = 1, 8, 8
        gt = torch.randint(0, 4, (B, H, W))
        pred = {
            "map_logits": torch.randn(B, 4, H, W),
            "det_heatmap": torch.randn(B, 2, H, W),
            "det_reg": torch.randn(B, 6, H, W),
        }
        batch = {
            "traj": torch.zeros(B, 8, 3), "bev_map": gt,
            "det_heatmap": torch.zeros(B, 2, H, W),
            "det_reg": torch.zeros(B, 6, H, W),
            "det_reg_mask": torch.zeros(B, H, W),
        }
        base = compute_losses(pred, batch, cfg, with_planner=False, with_bev=True)
        self.assertNotIn("map_pa_bane", base)
        self.assertIn("map_boundary_miou", base)
        self.assertIn("map_iou_lane", base)
        pred_ccl = dict(pred)
        pred_ccl["map_ccl_embedding"] = torch.randn(B, 8, H, W)
        augmented = compute_losses(
            pred_ccl, batch, cfg, with_planner=False, with_bev=True)
        self.assertIn("map_pa_bane", augmented)
        torch.testing.assert_close(
            augmented["loss"] - base["loss"], 0.1 * augmented["map_pa_bane"],
            rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
