"""
export/chart.py  —  碰撞过程数据导出
使用 matplotlib + pandas 生成图表和 CSV
"""

from __future__ import annotations

# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import matplotlib.patches as mpatches
from pathlib import Path

from momentum_lab.model.block import (
    Block,
    D,
)


def simulate_collision(
    block_a: Block,
    block_b: Block,
    e: float = 1.0,
    duration: float = 6.0,
    dt: float = 0.01,
) -> pd.DataFrame:
    """
    对碰撞过程进行时序模拟，返回每一时刻的状态 DataFrame。

    列：t, xa, va, pa, eka, xb, vb, pb, ekb, p_total, ek_total, collided
    """
    a = Block(block_a.x, block_a.m, block_a.v)
    b = Block(block_b.x, block_b.m, block_b.v)
    collision = D(a, b, e=e)

    records = []
    collided = False
    PIXELS_PER_M = 100
    BLOCK_W_A = max(20, min(120, int(a.m * 30))) / PIXELS_PER_M

    for step in range(int(duration / dt)):
        t = step * dt

        # 碰撞检测
        if not collided and (a.x + BLOCK_W_A) >= b.x and a.v > b.v:
            collision.block_1 = a
            collision.block_2 = b
            new_a, new_b = collision.collide()
            new_a.x = b.x - BLOCK_W_A
            a, b = new_a, new_b
            collided = True

        collision.block_1 = a
        collision.block_2 = b
        records.append(
            {
                "t": round(t, 4),
                "xa": round(a.x, 6),
                "va": round(a.v, 6),
                "pa": round(a.moment, 6),
                "eka": round(a.k_energy, 6),
                "xb": round(b.x, 6),
                "vb": round(b.v, 6),
                "pb": round(b.moment, 6),
                "ekb": round(b.k_energy, 6),
                "p_total": round(
                    collision.total_momentum,
                    6,
                ),
                "ek_total": round(
                    collision.total_k_energy,
                    6,
                ),
                "collided": collided,
            }
        )

        a.x += a.v * dt
        b.x += b.v * dt

    return pd.DataFrame(records)


def export_chart(
    block_a: Block,
    block_b: Block,
    e: float = 1.0,
    output_dir: str | Path = ".",
    duration: float = 6.0,
) -> list[Path]:
    """
    导出碰撞分析图表（PNG）和原始数据（CSV）到 output_dir。
    返回生成的文件路径列表。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = simulate_collision(
        block_a,
        block_b,
        e=e,
        duration=duration,
    )
    e_label = D(
        block_1=block_a,
        block_2=block_b,
        e=e,
    ).kind.label

    # ── 合并图：速度、动量、动能 ─────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    fig.suptitle(
        f"{e_label} ($e={e}$)",
        fontsize=14,
        fontproperties=_my_font(),
    )

    t_col = df["t"]
    collision_t = df.loc[df["collided"], "t"].min() if df["collided"].any() else None

    # -- 子图1：速度-时间 --
    ax1 = axes[0]
    ax1.plot(t_col, df["va"], color="#4682C8", label=f"$v_A (m={block_a.m}kg)$")
    ax1.plot(t_col, df["vb"], color="#DC503C", label=f"$v_B (m={block_b.m}kg)$")
    if collision_t:
        ax1.axvline(
            collision_t, color="gray", linestyle="--", linewidth=1, label="碰撞时刻"
        )
    ax1.set_ylabel("速度 ($m/s$)", fontproperties=_my_font())
    ax1.legend(prop=_my_font())
    ax1.grid(True, alpha=0.3)

    # -- 子图2：动量-时间 --
    ax2 = axes[1]
    ax2.plot(t_col, df["pa"], color="#4682C8", linestyle="-", label="$p_A$")
    ax2.plot(t_col, df["pb"], color="#DC503C", linestyle="-", label="$p_B$")
    ax2.plot(
        t_col,
        df["p_total"],
        color="#2A2A2A",
        linestyle="--",
        linewidth=2,
        label="总动量 $p$",
    )
    if collision_t:
        ax2.axvline(collision_t, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("动量 ($kg·m/s$)", fontproperties=_my_font())
    ax2.legend(prop=_my_font())
    ax2.grid(True, alpha=0.3)

    # -- 子图3：动能-时间 --
    ax3 = axes[2]
    ax3.plot(t_col, df["eka"], color="#4682C8", label="$E_{k_A}$")
    ax3.plot(t_col, df["ekb"], color="#DC503C", label="$E_{k_B}$")
    ax3.plot(
        t_col,
        df["ek_total"],
        color="#3CA050",
        linestyle="--",
        linewidth=2,
        label="总动能 $E_k$",
    )
    if collision_t:
        ax3.axvline(
            collision_t, color="gray", linestyle="--", linewidth=1, label="碰撞时刻"
        )
    ax3.set_xlabel("时间 ($s$)", fontproperties=_my_font())
    ax3.set_ylabel("动能 ($J$)", fontproperties=_my_font())
    ax3.legend(prop=_my_font())
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    pic_path = output_dir / "combined_chart.png"
    fig.savefig(pic_path, dpi=150)
    plt.close(fig)

    # ── CSV 原始数据 ─────────────────────────────────────────
    data_path = output_dir / "collision_data.csv"
    df.to_csv(
        data_path,
        index=False,
        encoding="utf-8-sig",
    )

    return [pic_path, data_path]


def _my_font():
    """matplotlib FontProperties（优先 Maple Mono NF CN）"""
    from matplotlib.font_manager import (
        FontProperties,
    )

    for name in ["Maple Mono NF CN", "Fira Math"]:
        try:
            fp = FontProperties(family=name)
            return fp
        except Exception:
            continue
    return FontProperties()
