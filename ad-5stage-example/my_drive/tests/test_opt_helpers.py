"""my_drive 工程优化回归测试（不依赖训练数据/GPU）。"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from nuplan.common.maps.abstract_map import SemanticMapLayer
from shapely.geometry import MultiPolygon, Polygon

from my_drive.md_common import Config
from my_drive.data.md_bev import build_map_label
from my_drive.data.md_data import OverfitDataset
from my_drive.train.md_grpo import _decompose, _step_occupancies, compute_rewards


def test_token_filter_uses_one_set():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        idx = [{"token": f"t{i % 500}", "split": "train"} for i in range(2000)]
        (root / "index.json").write_text(json.dumps(idx))
        ds = OverfitDataset(root, Config(), tokens=[f"t{i}" for i in range(500)])
        assert len(ds) == 2000


def test_grpo_reuses_occupancy_maps():
    cfg = Config()
    rng = np.random.RandomState(0)
    fb = rng.randn(1, 8, 4, 7).astype(np.float32)
    fb[..., 3:5] = 1.0
    fv = np.ones((1, 8, 4), np.uint8)
    gt = np.zeros((1, 8, 3), np.float32)
    gt[..., 0] = np.arange(1, 9) * 2
    traj = np.zeros((1, 2, 8, 3), np.float32)
    traj[..., 0] = np.arange(1, 9) * 2
    bm = np.ones((1, *cfg.bev_map_size), np.int64)
    step = _step_occupancies(fb[0], fv[0], cfg)
    r0 = compute_rewards(traj, fb, fv, bm, gt, cfg,
                         cfg.ego_len, cfg.ego_wid, cfg.col_margin)
    r1 = compute_rewards(traj, fb, fv, bm, gt, cfg,
                         cfg.ego_len, cfg.ego_wid, cfg.col_margin, occs=[step])
    assert np.allclose(r0, r1)
    d0 = _decompose(traj, fb, fv, bm, gt, cfg,
                    cfg.ego_len, cfg.ego_wid, cfg.col_margin)
    d1 = _decompose(traj, fb, fv, bm, gt, cfg,
                    cfg.ego_len, cfg.ego_wid, cfg.col_margin, occs=step)
    assert d0 == d1


def test_map_label_single_query_and_multipolygon():
    cfg = Config()
    square = Polygon([(0, 0), (6, 0), (6, 6), (0, 6)])
    multi = MultiPolygon([
        Polygon([(-20, -6), (-14, -6), (-14, 0), (-20, 0)]),
        Polygon([(20, 6), (26, 6), (26, 12), (20, 12)]),
    ])

    class Obj:
        def __init__(self, polygon):
            self.polygon = polygon

    class API:
        def __init__(self):
            self.calls = 0

        def get_proximal_map_objects(self, point, radius, layers):
            self.calls += 1
            out = {layer: [] for layer in layers}
            out[SemanticMapLayer.ROADBLOCK] = [Obj(square)]
            out[SemanticMapLayer.LANE] = [Obj(multi)]
            return out

    api = API()
    label = build_map_label(api, np.array([0.0, 0.0, 0.0]), cfg)
    assert api.calls == 1
    assert (label > 0).any()
