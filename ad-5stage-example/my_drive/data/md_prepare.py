"""md_prepare：构建自包含有标签数据集（图像 + 标签缓存到本地）。

用法：
    python -m my_drive.data.md_prepare --train 10000 --val 2000 --test 1500 \
        --cache-root my_drive/workspace/cache

只读外部数据（navsim_logs pickle / sensor_blobs / nuplan 地图），
输出到 --cache-root。--image-width/height 只影响标签口径记录，图像原样复制
（Dataset 加载时才 resize）。
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
from ..md_common import WORK_ROOT, Config
from .md_data import prepare_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=10000, help="训练帧数")
    ap.add_argument("--val", type=int, default=2000, help="验证帧数")
    ap.add_argument("--test", type=int, default=1500, help="测试帧数")
    ap.add_argument("--max-logs", type=int, default=2000, help="最多扫描的日志数")
    ap.add_argument("--log-names", type=str, nargs="*", default=None, help="指定日志")
    ap.add_argument("--image-width", type=int, default=640)
    ap.add_argument("--image-height", type=int, default=384)
    ap.add_argument("--cache-root", type=str, default=str(WORK_ROOT / "cache"))
    args = ap.parse_args()
    cfg = Config(image_size=(args.image_width, args.image_height))
    t0 = time.time()
    index = prepare_dataset(cfg,
                            split_counts={"train": args.train, "val": args.val,
                                          "test": args.test},
                            cache_root=Path(args.cache_root),
                            max_logs=args.max_logs, log_names=args.log_names)
    dt = time.time() - t0
    print(f"完成：{len(index)} 样本，耗时 {dt:.1f}s → {args.cache_root}")
if __name__ == "__main__":
    main()
