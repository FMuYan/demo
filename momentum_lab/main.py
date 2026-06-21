"""
main.py  —  动量守恒实验仿真教具入口
《探究碰撞中的不变量》 · 致远队

用法：
    python main.py              # 启动 pygame 仿真（默认弹性碰撞）
    python main.py --export     # 仅导出图表/CSV，不打开窗口
    python main.py --e 0.5      # 指定恢复系数 (0.0 / 0.5 / 1.0)
"""

from __future__ import annotations

import argparse
from momentum_lab.model.block import Block
from momentum_lab.export import export_chart


def parse_args():
    parser = argparse.ArgumentParser(description="动量守恒实验")
    parser.add_argument(
        "--e",
        type=float,
        default=1.0,
        help="恢复系数 e (0=完全非弹性, 1=完全弹性, 默认 1.0)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="导出图表和 CSV 后退出（不启动 pygame 窗口）",
    )
    parser.add_argument(
        "--out", type=str, default="outputs", help="导出目录（默认 ./outputs）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    block_a = Block(x=0.5, m=1.0, v=4.0)
    block_b = Block(x=3.5, m=3.0, v=0.0)

    e = max(0.0, min(1.0, args.e))

    if args.export:
        paths = export_chart(block_a, block_b, e=e, output_dir=args.out)
        print(f"已导出 {len(paths)} 文件：")
        for p in paths:
            print(f"  {p}")
    else:
        from momentum_lab.ui import Scene

        scene = Scene(block_a, block_b, e=e, fps=60)
        scene.run()


if __name__ == "__main__":
    main()
