from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    FLOAT_LIKE = float | np.double
    INT_LIKE = int | np.int8


class CollisionType(Enum):
    """碰撞类型，由恢复系数 e 决定"""

    ELASTIC = 1.0  # 完全弹性碰撞：动能守恒
    PARTIALLY_ELASTIC = 0.5  # 非完全弹性碰撞：动能有损失
    INELASTIC = 0.0  # 完全非弹性碰撞：碰后共速

    @classmethod
    def from_e(cls, e: float) -> CollisionType:
        """根据恢复系数推断碰撞类型（精确匹配预设值，否则归为非完全弹性）"""
        for member in cls:
            if np.isclose(e, member.value):
                return member
        return cls.PARTIALLY_ELASTIC

    @property
    def label(self) -> str:
        return {
            CollisionType.ELASTIC: "完全弹性碰撞",
            CollisionType.PARTIALLY_ELASTIC: "非完全弹性碰撞",
            CollisionType.INELASTIC: "完全非弹性碰撞",
        }[self]


# 考虑一维正碰
@dataclass
class Block:
    x: FLOAT_LIKE
    m: FLOAT_LIKE
    v: FLOAT_LIKE

    @property
    def k_energy(self) -> FLOAT_LIKE:
        return 0.5 * self.m * self.v**2

    @property
    def moment(self) -> FLOAT_LIKE:
        return self.m * self.v

    # XXX
    def __repr__(self):
        return f"Block({self.x = }, {self.m = }, {self.v = }, {self.k_energy = }, {self.moment = })"


@dataclass
class D:
    """两体正碰模型

    block_1: 物块对象（左/先动）
    block_2: 物块对象（右/被碰）
    e:       恢复系数（0~1），决定碰撞类型
    kind:    由 e 自动推断的 CollisionType
    """

    block_1: Block
    block_2: Block
    e: FLOAT_LIKE = 1.0
    kind: CollisionType = field(init=False)

    def __post_init__(self):
        self.kind = CollisionType.from_e(self.e)

    @classmethod
    def elastic(cls, block_1: Block, block_2: Block) -> D:
        """完全弹性碰撞"""
        return cls(block_1, block_2, e=CollisionType.ELASTIC.value)

    @classmethod
    def inelastic(cls, block_1: Block, block_2: Block) -> D:
        """完全非弹性碰撞"""
        return cls(block_1, block_2, e=CollisionType.INELASTIC.value)

    @classmethod
    def partially_elastic(cls, block_1: Block, block_2: Block, e: float = 0.5) -> D:
        """非完全弹性碰撞，默认 e=0.5"""
        assert 0.0 < e < 1.0, "非完全弹性碰撞要求 0 < e < 1"
        return cls(block_1, block_2, e=e)

    def collide(self) -> tuple[Block, Block]:
        """
        执行一次碰撞，返回碰后的新 Block 对象（不修改原物块）。

        公式推导（一维正碰，动量守恒 + 恢复系数定义）：
            m1*v1 + m2*v2 = m1*v1' + m2*v2'
            e = (v2' - v1') / (v1 - v2)

        解得：
            v1' = (m1 - e*m2)*v1 + (1+e)*m2*v2
                  ────────────────────────────────
                             m1 + m2

            v2' = (m2 - e*m1)*v2 + (1+e)*m1*v1
                  ────────────────────────────────
                             m1 + m2
        """
        m1, m2 = self.block_1.m, self.block_2.m
        v1, v2 = self.block_1.v, self.block_2.v
        e = self.e

        v1_new = ((m1 - e * m2) * v1 + (1 + e) * m2 * v2) / (m1 + m2)
        v2_new = ((m2 - e * m1) * v2 + (1 + e) * m1 * v1) / (m1 + m2)

        return Block(self.block_1.x, m1, v1_new), Block(self.block_2.x, m2, v2_new)

    @property
    def total_momentum(self) -> float:
        """系统总动量"""
        return self.block_1.moment + self.block_2.moment

    @property
    def total_k_energy(self) -> float:
        """系统总动能"""
        return self.block_1.k_energy + self.block_2.k_energy

    def is_momentum_conserved(
        self, after: tuple[Block, Block], tol: float = 1e-9
    ) -> bool:
        """验证碰撞前后动量守恒（数值精度检验）"""
        p_after = after[0].moment + after[1].moment
        return bool(np.isclose(self.total_momentum, p_after, atol=tol))
