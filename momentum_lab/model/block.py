from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    FLOAT_LIKE = float | np.double


class CollisionType(Enum):
    """
    碰撞类型——仅表达物理语义，不绑定具体 e 值。
    e 在 [0, 1] 连续取值，由 from_e() 按区间映射。
    """

    ELASTIC = "elastic"  # e ≈ 1.0：完全弹性，动能守恒
    PARTIAL = "partial"  # 0 < e < 1：非完全弹性，动能损失
    INELASTIC = "inelastic"  # e ≈ 0.0：完全非弹性，碰后共速

    @classmethod
    def from_e(cls, e: float) -> CollisionType:
        if np.isclose(e, 1.0, atol=0.01):
            return cls.ELASTIC
        if np.isclose(e, 0.0, atol=0.01):
            return cls.INELASTIC
        return cls.PARTIAL

    @property
    def label(self) -> str:
        return {
            CollisionType.ELASTIC: "完全弹性碰撞",
            CollisionType.PARTIAL: "非完全弹性碰撞",
            CollisionType.INELASTIC: "完全非弹性碰撞",
        }[self]


@dataclass
class Block:
    x: float
    m: float
    v: float

    @property
    def k_energy(self) -> float:
        return 0.5 * self.m * self.v**2

    @property
    def moment(self) -> float:
        return self.m * self.v

    def __repr__(self):
        return (
            f"Block(x={self.x}, m={self.m}, v={self.v}, "
            f"k_energy={self.k_energy}, moment={self.moment})"
        )


@dataclass
class D:
    """两体正碰模型

    block_1: 物块（左/先动）
    block_2: 物块（右/被碰）
    e:       恢复系数 0 ≤ e ≤ 1
    kind:    由 e 自动推断的 CollisionType
    """

    block_1: Block
    block_2: Block
    e: float = 1.0
    kind: CollisionType = field(init=False)

    def __post_init__(self):
        self.kind = CollisionType.from_e(self.e)

    # ── 工厂方法 ────────────────────────────────────────────
    @classmethod
    def elastic(cls, b1: Block, b2: Block) -> D:
        return cls(b1, b2, e=1.0)

    @classmethod
    def inelastic(cls, b1: Block, b2: Block) -> D:
        return cls(b1, b2, e=0.0)

    @classmethod
    def partially_elastic(cls, b1: Block, b2: Block, e: float = 0.5) -> D:
        """e 可取 (0, 1) 内任意值"""
        return cls(b1, b2, e=float(np.clip(e, 0.0, 1.0)))

    # ── 碰撞计算 ────────────────────────────────────────────
    def collide(self) -> tuple[Block, Block]:
        """
        执行一次碰撞，返回碰后新 Block（不修改原对象）。

        一维正碰公式（动量守恒 + 恢复系数定义）：
            v1' = [(m1 - e·m2)v1 + (1+e)m2·v2] / (m1+m2)
            v2' = [(m2 - e·m1)v2 + (1+e)m1·v1] / (m1+m2)
        """
        m1, v1 = self.block_1.m, self.block_1.v
        m2, v2 = self.block_2.m, self.block_2.v
        e = self.e
        M = m1 + m2
        v1n = ((m1 - e * m2) * v1 + (1 + e) * m2 * v2) / M
        v2n = ((m2 - e * m1) * v2 + (1 + e) * m1 * v1) / M
        return Block(self.block_1.x, m1, v1n), Block(self.block_2.x, m2, v2n)

    # ── 系统量 ──────────────────────────────────────────────
    @property
    def total_momentum(self) -> float:
        return self.block_1.moment + self.block_2.moment

    @property
    def total_k_energy(self) -> float:
        return self.block_1.k_energy + self.block_2.k_energy
