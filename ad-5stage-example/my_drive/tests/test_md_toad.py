"""TOAD 的无数据/无 DINO 单元测试：pytest my_drive/tests/test_md_toad.py"""
from __future__ import annotations

import unittest

import torch

from my_drive.md_common import Config
from my_drive.train.md_model import ProposalPlanner, geo_rescore_selection
from my_drive.eval.md_toad import bicycle_inverse, bicycle_rollout, toad_search


class _DistanceScorer:
    """轻量 scorer 替身：越接近目标轨迹分越高，接口与 ProposalPlanner 一致。"""

    def __init__(self, target: torch.Tensor):
        self.target = target
        self.calls = 0

    def score_trajectories(self, trajectories, ctx, ego_token):
        self.calls += 1
        return -(trajectories[..., :2] - self.target[:, None, :, :2]).square().mean((-1, -2))


def _ego_inputs(batch: int = 2, speed: float = 2.0):
    ego_status = torch.zeros(batch, 4, 11)
    ego_status[:, -1, 3] = speed
    ego_motion = torch.zeros(batch, 4, 3)
    frame_dt = torch.tensor([-1.5, -1.0, -0.5, 0.0]).repeat(batch, 1)
    return ego_status, ego_motion, frame_dt


def _search_input(target_yaw: float = 0.12):
    torch.manual_seed(7)
    B, N, K = 2, 4, 8
    v0 = torch.full((B,), 2.0)
    target_controls = torch.zeros(B, K, 2)
    target_controls[..., 1] = target_yaw
    target = bicycle_rollout(target_controls, v0)
    yaw_rates = torch.tensor([0.0, 0.03, -0.03, 0.06])
    controls = torch.zeros(B, N, K, 2)
    controls[..., 1] = yaw_rates[None, :, None]
    props = bicycle_rollout(controls, v0)
    scorer = _DistanceScorer(target)
    scores = scorer.score_trajectories(props, None, None)
    out = {
        "proposal_list": [props], "proposal_scores": scores,
        "planner_context": None, "ego_token": None,
    }
    return out, scorer, props, target


class TestBicycleModel(unittest.TestCase):
    def test_control_trajectory_roundtrip(self):
        torch.manual_seed(3)
        controls = torch.randn(2, 5, 8, 2) * torch.tensor([0.2, 0.04])
        v0 = torch.tensor([-1.5, 3.0])  # 同时覆盖倒车（负纵向速度）
        trajectory = bicycle_rollout(controls, v0)
        recovered = bicycle_inverse(trajectory, v0)
        self.assertEqual(trajectory.shape, (2, 5, 8, 3))
        self.assertEqual(recovered.shape, controls.shape)
        torch.testing.assert_close(recovered, controls, rtol=1e-4, atol=1e-5)
        rerolled = bicycle_rollout(recovered, v0)
        torch.testing.assert_close(rerolled, trajectory, rtol=1e-4, atol=1e-5)

    def test_unbatched_mean_shape(self):
        controls = torch.zeros(2, 8, 2)
        trajectory = bicycle_rollout(controls, torch.tensor([2.0, 2.0]))
        self.assertEqual(trajectory.shape, (2, 8, 3))
        self.assertTrue(torch.isfinite(trajectory).all())


class TestToadSearch(unittest.TestCase):
    def test_search_improves_model_objective_and_is_deterministic(self):
        cfg = Config(toad_iterations=8, toad_samples=128, toad_elites=16,
                     toad_std_floor_accel=0.05, toad_std_floor_yaw_rate=0.05,
                     toad_anchor_weight=0.01, toad_comfort_weight=0.0, toad_seed=11)
        ego = _ego_inputs()
        out1, scorer1, props, _ = _search_input()
        base_score1 = out1["proposal_scores"].max(dim=1).values
        toad_search(out1, scorer1, *ego, cfg)
        out2, scorer2, _, _ = _search_input()
        toad_search(out2, scorer2, *ego, cfg)

        torch.testing.assert_close(out1["toad_trajectory"], out2["toad_trajectory"])
        torch.testing.assert_close(out1["toad_score"], out2["toad_score"])
        self.assertTrue(torch.isfinite(out1["toad_trajectory"]).all())
        self.assertTrue((out1["toad_objective_gain"] >= 0).all())
        self.assertTrue((out1["toad_model_score_gain"] > 0).all())
        # 初始 proposals 评分 1 次 + CEM 每轮 1 次 + 最终均值评分 1 次。
        self.assertEqual(scorer1.calls, cfg.toad_iterations + 2)
        # 原 proposals/scores 不被搜索改写。
        self.assertIs(out1["proposal_list"][0], props)
        self.assertEqual(out1["proposal_list"][0].shape[1], 4)
        self.assertTrue((out1["toad_score"] >= base_score1).all())

    def test_zero_iteration_falls_back_to_base(self):
        cfg = Config(toad_iterations=0, toad_comfort_weight=0.0)
        out, scorer, props, _ = _search_input()
        toad_search(out, scorer, *_ego_inputs(), cfg)
        base = props[torch.arange(2), out["proposal_scores"].argmax(dim=1)]
        torch.testing.assert_close(out["toad_trajectory"], base)
        self.assertEqual(int(out["toad_accept"].sum()), 0)
        torch.testing.assert_close(out["toad_objective_gain"], torch.zeros(2))

    def test_arbitrary_trajectory_scorer_interface(self):
        cfg = Config(d_model=16, planner_heads=4, num_proposals=4,
                     decoder_layers=1, scorer_layers=1)
        planner = ProposalPlanner(cfg)
        trajectories = torch.randn(2, 7, 8, 3, requires_grad=True)
        ctx, ego = torch.randn(2, 5, 16), torch.randn(2, 1, 16)
        scores = planner.score_trajectories(trajectories, ctx, ego)
        self.assertEqual(scores.shape, (2, 7))
        self.assertTrue(torch.isfinite(scores).all())
        scores.sum().backward()
        self.assertIsNone(trajectories.grad)  # 保留 scorer detach 语义

    def test_search_with_real_planner_scorer(self):
        cfg = Config(d_model=16, planner_heads=4, num_proposals=4,
                     decoder_layers=1, scorer_layers=1,
                     toad_iterations=2, toad_samples=16, toad_elites=4)
        torch.manual_seed(19)
        planner = ProposalPlanner(cfg).eval()
        v0 = torch.tensor([2.0, 3.0])
        controls = torch.randn(2, 4, 8, 2) * torch.tensor([0.1, 0.02])
        props = bicycle_rollout(controls, v0)
        ctx, ego = torch.randn(2, 5, 16), torch.randn(2, 1, 16)
        out = {"proposal_list": [props], "planner_context": ctx, "ego_token": ego}
        with torch.no_grad():
            out["proposal_scores"] = planner.score_trajectories(props, ctx, ego)
        toad_search(out, planner, *_ego_inputs(), cfg)
        self.assertEqual(out["toad_trajectory"].shape, (2, 8, 3))
        self.assertEqual(out["toad_score"].shape, (2,))
        self.assertTrue(torch.isfinite(out["toad_trajectory"]).all())
        self.assertTrue((out["toad_objective_gain"] >= 0).all())


class TestGeoCompatibility(unittest.TestCase):
    def test_toad_is_extra_candidate_without_mutating_proposals(self):
        cfg = Config(geo_rescore_m=8)
        B, N, K = 1, 4, 8
        props = torch.zeros(B, N, K, 3)
        scores = torch.zeros(B, N)
        toad = torch.zeros(B, K, 3)
        toad[0, :, 0] = torch.arange(1, K + 1).float()
        out = {
            "proposal_list": [props], "proposal_scores": scores,
            "toad_trajectory": toad, "toad_score": torch.tensor([10.0]),
            # 全图可行驶、无障碍，隔离验证候选拼接与索引。
            "map_logits": torch.tensor([0.0, 10.0, 0.0, 0.0]).reshape(1, 4, 1, 1).expand(1, 4, 100, 100),
            "det_heatmap": torch.full((1, 2, 100, 100), -20.0),
        }
        result = geo_rescore_selection(out, cfg)
        self.assertEqual(int(result["geo_sel"].item()), N)
        self.assertEqual(float(result["geo_toad_selected"].item()), 1.0)
        self.assertEqual(result["proposal_list"][0].shape, (B, N, K, 3))
        self.assertEqual(result["proposal_scores"].shape, (B, N))
        torch.testing.assert_close(result["trajectory"], toad)


if __name__ == "__main__":
    unittest.main()
