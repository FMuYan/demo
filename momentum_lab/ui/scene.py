"""
ui/scene.py  —  pygame 可视化场景（带交互输入面板）
功能要点：
- 面板输入框宽度与显示格式已固定（避免长小数）
- 物块参数按行左-中-右布局（标签 | 输入框 | 单位）
- 修改面板参数（质量/速度/e）后立即更新系统初始动量与动能显示
- 记录运行历史（逐帧），导出逻辑：
    · 按下 S → 设置 checkpoint，等待下次碰撞后自动截取并导出真实历史片段
    · 再次按 S（pending 中）→ 取消等待
    · 若当前无法碰撞（相对速度方向不满足），立即以预测模拟导出
- 导出完成会在面板显示临时提示
- 保留卫语句、match、C(hex) 颜色用法等先前重构的样式
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import pygame
import pandas as pd
import matplotlib.pyplot as plt
from momentum_lab.export import export_chart
from momentum_lab.model.block import Block, D
from momentum_lab.ui.const import *
from momentum_lab.ui.surface import (
    C,
    sim_x,
    u_block_w,
    u_block_h,
    u_load_fonts,
    u_panel_heading,
    u_draw_velocity_arrow,
)
from momentum_lab.ui.display import *


class Scene:
    E_PRESETS = [1.0, 0.5, 0.0]

    def __init__(self, block_a: Block, block_b: Block, e: float = 1.0, fps: int = 60):
        self._init_a = Block(block_a.x, block_a.m, block_a.v)
        self._init_b = Block(block_b.x, block_b.m, block_b.v)
        self._e_idx = self.E_PRESETS.index(e) if e in self.E_PRESETS else 0

        self.block_a = Block(block_a.x, block_a.m, block_a.v)
        self.block_b = Block(block_b.x, block_b.m, block_b.v)
        self.collision = D(self.block_a, self.block_b, e=self.E_PRESETS[self._e_idx])
        self.fps = fps

        self.paused = False
        self.first_collision_recorded = False
        self.collision_count = 0

        self.initial_p = self.collision.total_momentum
        self.initial_ek = self.collision.total_k_energy

        self.p_before: float | None = None
        self.p_after: float | None = None
        self.ek_before: float | None = None
        self.ek_after: float | None = None

        self._flash = 0

        # 输入缓冲与状态（使用统一格式化）
        self.input_buffers: Dict[str, str] = {
            "ma": self._format_field_value("ma", block_a.m),
            "va": self._format_field_value("va", block_a.v),
            "mb": self._format_field_value("mb", block_b.m),
            "vb": self._format_field_value("vb", block_b.v),
            "e": self._format_field_value("e", self.E_PRESETS[self._e_idx]),
        }
        self.active_input: str | None = None

        # 绘制用 box rect 与 点击用 hit rect（包含单位区域）
        self.box_rects: Dict[str, pygame.Rect] = {}
        self.hit_rects: Dict[str, pygame.Rect] = {}

        self._paused_before_edit = False

        # 光标闪烁
        self._cursor_acc = 0.0
        self._cursor_visible = True
        self._cursor_blink_interval = 0.5

        # 导出提示（短暂显示用户可见信息）
        self._export_msg: str = ""
        self._export_msg_timer: float = 0.0  # seconds

        # 导出 pending：按下 S 后等待下次碰撞自动触发导出
        self._export_pending: bool = False
        self._export_pending_since: float = 0.0  # 按下 S 时的模拟时刻
        self._export_pending_hist_len: int = 0  # 按下 S 时的历史帧数（用于截取片段）

        # 运行时间与历史记录（每帧记录用于导出真实历史）
        self._time = 0.0
        self._history: list[dict] = []

    # 统一格式化面板字段显示
    def _format_field_value(self, field: str, val: float) -> str:
        """
        Format values for panel display to avoid long float expansions.
        - mass (ma, mb): 1 decimal place
        - velocity (va, vb): 2 decimal places
        - e: 2 decimal places
        """
        try:
            if field in ("ma", "mb"):
                return f"{val:.1f}"
            if field in ("va", "vb"):
                return f"{val:.2f}"
            if field == "e":
                return f"{val:.2f}"
        except Exception:
            pass
        return str(val)

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((W, H), pygame.NOFRAME)
        pygame.display.set_caption("动量守恒演示")
        clock = pygame.time.Clock()
        fonts = u_load_fonts()

        self._compute_panel_layout(fonts)

        while True:
            dt = clock.tick(self.fps) / 1000.0
            self._update_cursor(dt)

            # 更新导出提示计时器
            if self._export_msg_timer > 0:
                self._export_msg_timer -= dt
                if self._export_msg_timer <= 0:
                    self._export_msg = ""
                    self._export_msg_timer = 0.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                # 鼠标点击检测（hit_rects 包含单位区）
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    hit = None
                    for field, hit_rect in self.hit_rects.items():
                        if hit_rect.collidepoint(mx, my):
                            hit = field
                            break
                    if hit is not None:
                        # 进入编辑（卫语句，减少嵌套）
                        self.active_input = hit
                        self.input_buffers[hit] = ""
                        self._paused_before_edit = self.paused
                        self.paused = True
                    else:
                        # 点击面板外：若处于编辑状态则取消编辑并恢复暂停状态
                        if self.active_input is not None:
                            field = self.active_input
                            self.input_buffers[field] = self._current_value_str(field)
                            self.active_input = None
                            self.paused = self._paused_before_edit

                # 键盘事件
                if event.type == pygame.KEYDOWN:
                    # 编辑模式优先处理（卫语句）
                    if self.active_input is not None:
                        field = self.active_input

                        # 允许编辑时按 S 导出：先提交/恢复，再导出
                        if event.key == pygame.K_s:
                            if not self.input_buffers.get(field):
                                self.input_buffers[field] = self._current_value_str(
                                    field
                                )
                                self.active_input = None
                                self.paused = self._paused_before_edit
                            else:
                                self._commit_active_input()
                            self._export_charts()
                            continue

                        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            self._commit_active_input()
                        elif event.key == pygame.K_ESCAPE:
                            # 取消编辑：恢复缓冲为当前实际值并恢复暂停状态
                            self.input_buffers[field] = self._current_value_str(field)
                            self.active_input = None
                            self.paused = self._paused_before_edit
                        elif event.key == pygame.K_BACKSPACE:
                            if self.input_buffers.get(field):
                                self.input_buffers[field] = self.input_buffers[field][
                                    :-1
                                ]
                        else:
                            ch = event.unicode
                            if ch and (ch.isdigit() or ch in ".-"):
                                self.input_buffers[field] += ch
                        continue

                    # 非编辑模式的全局键：使用 match 简化分支
                    match event.key:
                        case pygame.K_q | pygame.K_ESCAPE:
                            pygame.quit()
                            sys.exit()
                        case pygame.K_SPACE:
                            self.paused = not self.paused
                        case pygame.K_r:
                            self._reset()
                        case pygame.K_e:
                            self._cycle_e()
                        case pygame.K_s:
                            # 导出当前仿真图表与 CSV
                            self._export_charts()
                        case _:
                            pass  # 未处理的键

            # always update physics when not paused, then record current state
            if not self.paused:
                self._update(dt)
                if self._flash > 0:
                    self._flash -= 1

            # advance time and record history every frame (包含暂停时的快照也有用)
            self._time += dt
            self._record_history()

            self._draw(screen, fonts)
            pygame.display.flip()

    def _update_cursor(self, dt: float):
        self._cursor_acc += dt
        if self._cursor_acc >= self._cursor_blink_interval:
            self._cursor_acc -= self._cursor_blink_interval
            self._cursor_visible = not self._cursor_visible

    def _update(self, dt: float):
        a, b = self.block_a, self.block_b
        wa = u_block_w(a)
        ax_right = sim_x(a.x) + wa
        bx_left = sim_x(b.x)

        if ax_right >= bx_left and a.v > b.v:
            self.collision.block_1 = a
            self.collision.block_2 = b
            if not self.first_collision_recorded:
                self.p_before = self.collision.total_momentum
                self.ek_before = self.collision.total_k_energy
            na, nb = self.collision.collide()
            na.x = b.x - wa / PIXELS_PER_M
            nb.x = b.x
            self.block_a, self.block_b = na, nb
            self.collision.block_1 = na
            self.collision.block_2 = nb
            if not self.first_collision_recorded:
                self.p_after = self.collision.total_momentum
                self.ek_after = self.collision.total_k_energy
                self.first_collision_recorded = True
            self.collision_count += 1
            self._flash = 18

            # ★ 若有导出 pending，碰撞发生后立即截取历史并导出
            if self._export_pending:
                self._export_pending = False
                self._do_export_after_collision()

        self.block_a.x += self.block_a.v * dt
        self.block_b.x += self.block_b.v * dt

        for blk in (self.block_a, self.block_b):
            w = u_block_w(blk)
            sx = sim_x(blk.x)
            if sx < SIM_PAD:
                blk.x = 0.0
                blk.v = abs(blk.v)
            right_limit = SIM_W - SIM_PAD - w
            if sx > right_limit:
                blk.x = (right_limit - ORIGIN_X) / PIXELS_PER_M
                blk.v = -abs(blk.v)

    def _reset(self):
        self.block_a = Block(self._init_a.x, self._init_a.m, self._init_a.v)
        self.block_b = Block(self._init_b.x, self._init_b.m, self._init_b.v)
        self.collision.block_1 = self.block_a
        self.collision.block_2 = self.block_b
        self.first_collision_recorded = False
        self.collision_count = 0
        self.p_before = None
        self.p_after = None
        self.ek_before = None
        self.ek_after = None
        self._flash = 0

        # 输入缓冲重置为格式化后的初始显示
        self.input_buffers = {
            "ma": self._format_field_value("ma", self.block_a.m),
            "va": self._format_field_value("va", self.block_a.v),
            "mb": self._format_field_value("mb", self.block_b.m),
            "vb": self._format_field_value("vb", self.block_b.v),
            "e": self._format_field_value("e", self.collision.e),
        }
        self.active_input = None
        self._paused_before_edit = False

        self.initial_p = self.collision.total_momentum
        self.initial_ek = self.collision.total_k_energy

        # 重置时取消任何等待中的导出，并清空运行历史
        self._export_pending = False
        self._time = 0.0
        self._history.clear()

    def _cycle_e(self):
        self._e_idx = (self._e_idx + 1) % len(self.E_PRESETS)
        new_e = self.E_PRESETS[self._e_idx]
        self.collision = D(self.block_a, self.block_b, e=new_e)
        # 更新显示为格式化后的 e
        self.input_buffers["e"] = self._format_field_value("e", self.collision.e)

        # 切换碰撞系数时清除历史并重置碰撞标记（因为物理行为语义已变）
        self._export_pending = False
        self._time = 0.0
        self._history.clear()
        self.first_collision_recorded = False
        self.collision_count = 0
        self.p_before = None
        self.p_after = None
        self.ek_before = None
        self.ek_after = None

    def _predict_collision_time(self) -> float | None:
        """
        估算从当前状态起第一次接触的时间（秒）。
        - 返回 None 表示无法预测（通常是相对速度不足以接触）；
        - 返回 0.0 表示已经接触（立即碰撞）。
        计算方法（一维直线近似）：
            dist_to_contact = b.x - a.x - block_width_m
            t = dist_to_contact / (a.v - b.v)
        block_width_m 由 _block_w(a) / PIXELS_PER_M 估算（与绘制一致）。
        """
        a = self.block_a
        b = self.block_b

        # 只有当左块速度大于右块速度才可能相遇
        if a.v <= b.v:
            return None

        # 估算物块在世界坐标中的宽度（与 _block_w/绘制使用一致）
        # _block_w 返回像素宽度；除以 PIXELS_PER_M 得到米
        try:
            block_w_m = u_block_w(a) / PIXELS_PER_M
        except Exception:
            block_w_m = 0.5  # 兜底的估计值

        dist = b.x - a.x - block_w_m
        # 如果距离已经小于等于0，说明已经接触
        if dist <= 0:
            return 0.0
        rel_v = a.v - b.v
        if rel_v <= 0:
            return None
        t = dist / rel_v
        return t if t >= 0 else None

    def _record_history(self):
        """把当前时刻的状态追加到 self._history。保留全部记录，按需可加上长度上限。"""
        a = self.block_a
        b = self.block_b
        try:
            self.collision.block_1 = a
            self.collision.block_2 = b
            p_total = self.collision.total_momentum
            ek_total = self.collision.total_k_energy
        except Exception:
            p_total = a.moment + b.moment
            ek_total = a.k_energy + b.k_energy

        entry = {
            "t": round(self._time, 6),
            "xa": round(a.x, 6),
            "va": round(a.v, 6),
            "pa": round(a.moment, 6),
            "eka": round(a.k_energy, 6),
            "xb": round(b.x, 6),
            "vb": round(b.v, 6),
            "pb": round(b.moment, 6),
            "ekb": round(b.k_energy, 6),
            "p_total": round(p_total, 6),
            "ek_total": round(ek_total, 6),
            "collided": bool(self.first_collision_recorded),
        }
        self._history.append(entry)

    def _plot_from_df(self, df: pd.DataFrame, pic_path: Path, title: str | None = None):
        """根据历史 DataFrame 生成速度/动量/动能三合一图并保存为 pic_path。"""
        t_col = df["t"]
        collision_t = (
            df.loc[df["collided"], "t"].min()
            if ("collided" in df.columns and df["collided"].any())
            else None
        )

        fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
        if title:
            fig.suptitle(title, fontsize=14)

        # 速度
        ax1 = axes[0]
        ax1.plot(t_col, df["va"], color="#4682C8", label=f"v_A (m={self.block_a.m}kg)")
        ax1.plot(t_col, df["vb"], color="#DC503C", label=f"v_B (m={self.block_b.m}kg)")
        if collision_t is not None:
            ax1.axvline(
                collision_t, color="gray", linestyle="--", linewidth=1, label="碰撞时刻"
            )
        ax1.set_ylabel("速度 (m/s)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 动量
        ax2 = axes[1]
        ax2.plot(t_col, df["pa"], color="#4682C8", label="p_A")
        ax2.plot(t_col, df["pb"], color="#DC503C", label="p_B")
        ax2.plot(
            t_col,
            df["p_total"],
            color="#2A2A2A",
            linestyle="--",
            linewidth=2,
            label="总动量 p",
        )
        if collision_t is not None:
            ax2.axvline(collision_t, color="gray", linestyle="--", linewidth=1)
        ax2.set_ylabel("动量 (kg·m/s)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 动能
        ax3 = axes[2]
        ax3.plot(t_col, df["eka"], color="#4682C8", label="E_k_A")
        ax3.plot(t_col, df["ekb"], color="#DC503C", label="E_k_B")
        ax3.plot(
            t_col,
            df["ek_total"],
            color="#3CA050",
            linestyle="--",
            linewidth=2,
            label="总动能 E_k",
        )
        if collision_t is not None:
            ax3.axvline(
                collision_t, color="gray", linestyle="--", linewidth=1, label="碰撞时刻"
            )
        ax3.set_xlabel("时间 (s)")
        ax3.set_ylabel("动能 (J)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(pic_path, dpi=150)
        plt.close(fig)

    def _export_charts(self):
        """按下 S 键时调用。
        行为：
        - 若当前没有 pending：设置 checkpoint，等待下次碰撞后自动导出真实历史片段。
          特例：若当前物理状态不可能发生碰撞（a.v <= b.v），
          则立即以预测模拟方式导出（与原有逻辑一致）。
        - 若已有 pending（再次按 S）：取消等待。
        """
        # 再次按 S → 取消 pending
        if self._export_pending:
            self._export_pending = False
            self._export_msg = "已取消导出等待"
            self._export_msg_timer = 2.0
            print("[export] 用户取消了等待中的导出")
            return

        # 检查是否有可能发生碰撞；若不可能则立即做预测模拟导出
        pred = self._predict_collision_time()
        if pred is None:
            # 无法碰撞：立即以模拟预测导出
            self._do_export_simulated()
            return

        # 设置 checkpoint，等待下次碰撞
        self._export_pending = True
        self._export_pending_since = self._time
        self._export_pending_hist_len = len(self._history)
        self._export_msg = "⏳ 等待碰撞后自动导出… (再按 S 取消)"
        self._export_msg_timer = 999.0  # 持续显示直到碰撞或取消
        print(
            f"[export] checkpoint 已设置 t={self._time:.3f}s，预计碰撞约 {pred:.2f}s 后"
        )

    def _do_export_after_collision(self):
        """碰撞发生时（由 _update 调用），截取 checkpoint 之后的历史并导出 CSV + PNG。"""
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        outdir = Path("outputs") / f"export_{ts}"
        outdir.mkdir(parents=True, exist_ok=True)

        try:
            # 截取 checkpoint 之后的帧
            slice_start = self._export_pending_hist_len
            df = pd.DataFrame(self._history[slice_start:])

            if df.empty:
                # 极少数情况：checkpoint 恰好在碰撞帧，退回全量
                df = pd.DataFrame(self._history)

            csv_path = outdir / "collision_history.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.6f")

            pic_path = outdir / "combined_chart.png"
            self._plot_from_df(
                df,
                pic_path,
                title=(
                    f"碰撞记录  e={self.collision.e}  "
                    f"mA={self.block_a.m} kg  mB={self.block_b.m} kg"
                ),
            )

            self._export_msg = f"✅ 已导出（{outdir.name}，{len(df)} 帧）"
            self._export_msg_timer = 4.0
            print(f"[export] 碰撞后导出完成 → {outdir}  rows={len(df)}")

        except Exception as exc:
            self._export_msg = f"导出失败: {exc}"
            self._export_msg_timer = 6.0
            print(f"[export] 导出失败: {exc}")

    def _do_export_simulated(self):
        """无法碰撞时，以预测模拟时长调用 export_chart 立即导出。"""
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        outdir = Path("outputs") / f"export_{ts}"
        outdir.mkdir(parents=True, exist_ok=True)

        try:
            pred = self._predict_collision_time()
            tail = 2.0
            default_duration = 6.0
            duration = (
                default_duration if pred is None else max(default_duration, pred + tail)
            )

            paths = export_chart(
                self.block_a,
                self.block_b,
                e=self.collision.e,
                output_dir=outdir,
                duration=duration,
            )

            self._export_msg = (
                f"已导出模拟 {len(paths)} 文件（{outdir.name}，{duration:.1f}s）"
            )
            self._export_msg_timer = 3.0
            print(
                f"[export] 模拟导出 {len(paths)} 文件 → {outdir} (duration={duration:.2f}s)"
            )
            for p in paths:
                print("  ", p)

        except Exception as exc:
            self._export_msg = f"导出失败: {exc}"
            self._export_msg_timer = 6.0
            print(f"[export] 导出失败: {exc}")

    def _draw(self, screen, fonts):
        screen.fill(C(BG))
        pygame.draw.rect(screen, C(BG), (0, 0, SIM_W, H))

        self._draw_track(screen, fonts)
        self._draw_blocks(screen, fonts)
        self._draw_panel(screen, fonts)
        self._draw_hintbar(screen, fonts)

        if self.paused:
            self._draw_paused_overlay(screen, fonts)

    def _draw_track(self, screen, fonts):
        pygame.draw.line(
            screen,
            C(FLOOR_COL),
            (SIM_PAD - 10, FLOOR_Y),
            (SIM_W - SIM_PAD + 10, FLOOR_Y),
            2,
        )
        for m in range(0, 9):
            gx = sim_x(m)
            pygame.draw.line(screen, C(FLOOR_COL), (gx, FLOOR_Y), (gx, FLOOR_Y + 8), 1)
            lbl = fonts["small"].render(str(m), True, C(TEXT_SEC_STRONG))
            screen.blit(lbl, (gx - lbl.get_width() // 2, FLOOR_Y + 11))
        unit = fonts["small"].render("m", True, C(TEXT_SEC_STRONG))
        screen.blit(unit, (SIM_W - SIM_PAD + 14, FLOOR_Y + 11))

    def _draw_blocks(self, screen, fonts):
        for blk, col_hex, col_dim_hex, label in (
            (self.block_a, COL_A, COL_A_DIM, "A"),
            (self.block_b, COL_B, COL_B_DIM, "B"),
        ):
            col_color = C(col_hex)
            col_dim_color = C(col_dim_hex)

            w = u_block_w(blk)
            h = u_block_h(blk)
            sx = sim_x(blk.x)
            sy = FLOOR_Y - h

            # 碰撞闪光：高亮边框（用 RGBA）
            if self._flash > 0:
                alpha = min(255, self._flash * 14)
                glow = pygame.Surface((w + 6, h + 6), pygame.SRCALPHA)
                tmp = C(col_hex)
                tmp.a = alpha
                glow.fill(tmp)
                screen.blit(glow, (sx - 3, sy - 3))

            # 物块主体（渐变感：上亮下暗）
            body = pygame.Surface((w, h), pygame.SRCALPHA)
            for row in range(h):
                t = row / h
                r = int(col_color.r * (1 - t * 0.35) + col_dim_color.r * t * 0.35)
                g = int(col_color.g * (1 - t * 0.35) + col_dim_color.g * t * 0.35)
                b_ = int(col_color.b * (1 - t * 0.35) + col_dim_color.b * t * 0.35)
                pygame.draw.line(body, (r, g, b_), (0, row), (w, row))
            screen.blit(body, (sx, sy))

            lbl = fonts["h2"].render(label, True, C(MAIN))
            mlbl = fonts["small"].render(f"{blk.m} kg", True, C(MAIN))
            screen.blit(lbl, (sx + w // 2 - lbl.get_width() // 2, sy + 10))
            screen.blit(mlbl, (sx + w // 2 - mlbl.get_width() // 2, sy + 32))

            u_draw_velocity_arrow(screen, blk, sx, sy, w, C(col_hex))

            vtext = fonts["small"].render(f"{blk.v:.2f} m/s", True, C(col_hex))
            screen.blit(vtext, (sx + w // 2 - vtext.get_width() // 2, FLOOR_Y - h - 40))

    def _draw_panel(self, screen, fonts):
        px = SIM_W
        pygame.draw.rect(screen, C(PANEL_BG), (px, 0, PANEL_W, H))
        pygame.draw.line(screen, C(DIVIDER), (px, 0), (px, H), 1)

        pad = 20
        cy = 24

        cy = u_panel_heading(screen, fonts, px + pad, cy, "实验参数", C(TEXT_HEAD))
        cy += 8

        # 显示导出提示文本（若有）
        if self._export_msg_timer > 0 and self._export_msg:
            msg_surf = fonts["small"].render(self._export_msg, True, C(AMBER))
            screen.blit(msg_surf, (px + pad, cy))
            cy += msg_surf.get_height() + 6

        content_w = PANEL_W - pad * 2
        col_gap = 20
        col_total_w = (content_w - col_gap) // 2

        # compute widths for label / input / unit arrangement
        # label_max covers "质量" and "速度"
        label_w_candidates = [
            fonts["small"].render("质量", True, C(TEXT_SEC_STRONG)).get_width(),
            fonts["small"].render("速度", True, C(TEXT_SEC_STRONG)).get_width(),
        ]
        label_max_w = max(label_w_candidates) + 8  # padding after label

        unit_examples = ["kg", "m/s", f"({self.collision.kind.label})"]
        unit_width = max(
            (
                fonts["small"].render(u, True, C(TEXT_SEC_STRONG)).get_width()
                for u in unit_examples
            )
        )
        unit_width = max(unit_width, 36)

        # restore box width calculation to previous logic (not dependent on label)
        box_w = max(MIN_BOX_W, col_total_w - unit_width - UNIT_PADDING)
        # but cap label width so that label + box + unit fit into content_w
        cap_label_w = min(
            label_max_w, content_w - box_w - unit_width - UNIT_PADDING - 8
        )
        left_x = px + pad

        # ---------------- e 行（保持原样：右侧输入） ----------------
        cy += ROW_H
        e_label_s = fonts["small"].render("碰撞系数", True, C(TEXT_SEC_STRONG))
        screen.blit(
            e_label_s, (left_x - 10, cy + (ROW_H - e_label_s.get_height()) // 2)
        )

        e_box_x = px + pad + content_w - box_w - unit_width - UNIT_PADDING
        e_box_rect = pygame.Rect(e_box_x, cy, box_w, ROW_H)
        e_hit_rect = pygame.Rect(
            e_box_x, cy, box_w + unit_width + UNIT_PADDING + HIT_EXTRA, ROW_H
        )
        self.box_rects["e"] = e_box_rect
        self.hit_rects["e"] = e_hit_rect

        pygame.draw.rect(screen, C(BOTTON_COL), e_box_rect, border_radius=6)
        pygame.draw.rect(screen, C(DIVIDER), e_box_rect, 1, border_radius=6)
        if self.active_input == "e":
            pygame.draw.rect(screen, C("#2878DC"), e_box_rect, 2, border_radius=6)

        if self.active_input == "e":
            disp = self.input_buffers.get("e", "")
            if self._cursor_visible:
                disp = disp + "|"
        else:
            disp = self._current_value_str("e")
        clip = screen.get_clip()
        screen.set_clip(e_box_rect.inflate(-4, 0))
        ds = fonts["small"].render(disp, True, C(TEXT_PRI))
        screen.blit(
            ds,
            (
                e_box_rect.right - BOX_INNER_PADDING - ds.get_width(),
                e_box_rect.y + (ROW_H - ds.get_height()) // 2,
            ),
        )
        screen.set_clip(clip)

        kind_label_s = fonts["small"].render(
            f"({self.collision.kind.label})", True, C(TEXT_SEC)
        )
        kind_x = min(
            px + PANEL_W - pad - kind_label_s.get_width(),
            e_box_rect.right + UNIT_PADDING,
        )
        screen.blit(
            kind_label_s,
            (kind_x, e_box_rect.y + (ROW_H - kind_label_s.get_height()) // 2),
        )

        cy += ROW_H + ROW_GAP + 6

        # ---------------- 物块 A: 两行（每行：label | box | unit） ----------------
        heading_a = fonts["h2"].render("物块 A", True, C(COL_A))
        screen.blit(heading_a, (left_x, cy))
        cy += heading_a.get_height() + HEADING_FIELD_GAP

        # 质量 行
        ma_label_s = fonts["small"].render("质量", True, C(TEXT_SEC_STRONG))
        label_x = left_x
        screen.blit(ma_label_s, (label_x, cy + (ROW_H - ma_label_s.get_height()) // 2))
        ma_box_x = left_x + cap_label_w
        ma_box = pygame.Rect(ma_box_x, cy, box_w, ROW_H)
        self.box_rects["ma"] = ma_box
        self.hit_rects["ma"] = pygame.Rect(
            ma_box.x, ma_box.y, box_w + unit_width + UNIT_PADDING + HIT_EXTRA, ROW_H
        )
        pygame.draw.rect(screen, C(BOTTON_COL), ma_box, border_radius=6)
        pygame.draw.rect(screen, C(DIVIDER), ma_box, 1, border_radius=6)
        if self.active_input == "ma":
            pygame.draw.rect(screen, C("#2878DC"), ma_box, 2, border_radius=6)
        # value
        if self.active_input == "ma":
            mdisp = self.input_buffers.get("ma", "") + (
                "|" if self._cursor_visible else ""
            )
        else:
            mdisp = self._current_value_str("ma")
        clip = screen.get_clip()
        screen.set_clip(ma_box.inflate(-4, 0))
        mds = fonts["small"].render(mdisp, True, C(COL_A))
        screen.blit(
            mds,
            (
                ma_box.right - BOX_INNER_PADDING - mds.get_width(),
                ma_box.y + (ROW_H - mds.get_height()) // 2,
            ),
        )
        screen.set_clip(clip)
        # unit
        unit_s = fonts["small"].render("kg", True, C(TEXT_SEC))
        screen.blit(
            unit_s,
            (
                ma_box.right + UNIT_PADDING,
                ma_box.y + (ROW_H - unit_s.get_height()) // 2,
            ),
        )

        cy += ROW_H + ROW_GAP

        # 速度 行
        va_label_s = fonts["small"].render("速度", True, C(TEXT_SEC_STRONG))
        screen.blit(va_label_s, (label_x, cy + (ROW_H - va_label_s.get_height()) // 2))
        va_box_x = left_x + cap_label_w
        va_box = pygame.Rect(va_box_x, cy, box_w, ROW_H)
        self.box_rects["va"] = va_box
        self.hit_rects["va"] = pygame.Rect(
            va_box.x, va_box.y, box_w + unit_width + UNIT_PADDING + HIT_EXTRA, ROW_H
        )
        pygame.draw.rect(screen, C(BOTTON_COL), va_box, border_radius=6)
        pygame.draw.rect(screen, C(DIVIDER), va_box, 1, border_radius=6)
        if self.active_input == "va":
            pygame.draw.rect(screen, C("#2878DC"), va_box, 2, border_radius=6)
        if self.active_input == "va":
            vdisp = self.input_buffers.get("va", "") + (
                "|" if self._cursor_visible else ""
            )
        else:
            vdisp = self._current_value_str("va")
        clip = screen.get_clip()
        screen.set_clip(va_box.inflate(-4, 0))
        vds = fonts["small"].render(vdisp, True, C(COL_A))
        screen.blit(
            vds,
            (
                va_box.right - BOX_INNER_PADDING - vds.get_width(),
                va_box.y + (ROW_H - vds.get_height()) // 2,
            ),
        )
        screen.set_clip(clip)
        unit_vs = fonts["small"].render("m/s", True, C(TEXT_SEC))
        screen.blit(
            unit_vs,
            (
                va_box.right + UNIT_PADDING,
                va_box.y + (ROW_H - unit_vs.get_height()) // 2,
            ),
        )

        cy += ROW_H + ROW_GAP + 6

        # ---------------- 物块 B: 两行（每行：label | box | unit） ----------------
        heading_b = fonts["h2"].render("物块 B", True, C(COL_B))
        screen.blit(heading_b, (left_x, cy))
        cy += heading_b.get_height() + HEADING_FIELD_GAP

        # 质量 行
        mb_label_s = fonts["small"].render("质量", True, C(TEXT_SEC_STRONG))
        screen.blit(mb_label_s, (label_x, cy + (ROW_H - mb_label_s.get_height()) // 2))
        mb_box_x = left_x + cap_label_w
        mb_box = pygame.Rect(mb_box_x, cy, box_w, ROW_H)
        self.box_rects["mb"] = mb_box
        self.hit_rects["mb"] = pygame.Rect(
            mb_box.x, mb_box.y, box_w + unit_width + UNIT_PADDING + HIT_EXTRA, ROW_H
        )
        pygame.draw.rect(screen, C(BOTTON_COL), mb_box, border_radius=6)
        pygame.draw.rect(screen, C(DIVIDER), mb_box, 1, border_radius=6)
        if self.active_input == "mb":
            pygame.draw.rect(screen, C("#2878DC"), mb_box, 2, border_radius=6)
        if self.active_input == "mb":
            mbdisp = self.input_buffers.get("mb", "") + (
                "|" if self._cursor_visible else ""
            )
        else:
            mbdisp = self._current_value_str("mb")
        clip = screen.get_clip()
        screen.set_clip(mb_box.inflate(-4, 0))
        mbds = fonts["small"].render(mbdisp, True, C(COL_B))
        screen.blit(
            mbds,
            (
                mb_box.right - BOX_INNER_PADDING - mbds.get_width(),
                mb_box.y + (ROW_H - mbds.get_height()) // 2,
            ),
        )
        screen.set_clip(clip)
        unit_mb = fonts["small"].render("kg", True, C(TEXT_SEC))
        screen.blit(
            unit_mb,
            (
                mb_box.right + UNIT_PADDING,
                mb_box.y + (ROW_H - unit_mb.get_height()) // 2,
            ),
        )

        cy += ROW_H + ROW_GAP

        # 速度 行
        vb_label_s = fonts["small"].render("速度", True, C(TEXT_SEC_STRONG))
        screen.blit(vb_label_s, (label_x, cy + (ROW_H - vb_label_s.get_height()) // 2))
        vb_box_x = left_x + cap_label_w
        vb_box = pygame.Rect(vb_box_x, cy, box_w, ROW_H)
        self.box_rects["vb"] = vb_box
        self.hit_rects["vb"] = pygame.Rect(
            vb_box.x, vb_box.y, box_w + unit_width + UNIT_PADDING + HIT_EXTRA, ROW_H
        )
        pygame.draw.rect(screen, C(BOTTON_COL), vb_box, border_radius=6)
        pygame.draw.rect(screen, C(DIVIDER), vb_box, 1, border_radius=6)
        if self.active_input == "vb":
            pygame.draw.rect(screen, C("#2878DC"), vb_box, 2, border_radius=6)
        if self.active_input == "vb":
            vbdisp = self.input_buffers.get("vb", "") + (
                "|" if self._cursor_visible else ""
            )
        else:
            vbdisp = self._current_value_str("vb")
        clip = screen.get_clip()
        screen.set_clip(vb_box.inflate(-4, 0))
        vbds = fonts["small"].render(vbdisp, True, C(COL_B))
        screen.blit(
            vbds,
            (
                vb_box.right - BOX_INNER_PADDING - vbds.get_width(),
                vb_box.y + (ROW_H - vbds.get_height()) // 2,
            ),
        )
        screen.set_clip(clip)
        unit_vb = fonts["small"].render("m/s", True, C(TEXT_SEC))
        screen.blit(
            unit_vb,
            (
                vb_box.right + UNIT_PADDING,
                vb_box.y + (ROW_H - unit_vb.get_height()) // 2,
            ),
        )

        # ensure cy moves to below the last input box so totals area is visible
        cy = max(cy + ROW_H + ROW_GAP + 6, vb_box.bottom + 6)

        # 分割线
        pygame.draw.line(screen, C(DIVIDER), (px + pad, cy), (px + PANEL_W - pad, cy))
        cy += 18

        # 动能 / 动量 — 简洁并列显示，动态计算列宽
        title_s = fonts["h2"].render("系统总能", True, C(TEXT_HEAD))
        screen.blit(title_s, (left_x, cy))
        cy += title_s.get_height() + 12

        label_texts = ["总动能", "总动量"]
        label_w = (
            max(
                fonts["small"].render(t, True, C(TEXT_SEC_STRONG)).get_width()
                for t in label_texts
            )
            + 8
        )
        remaining_w = content_w - label_w
        col_w = (remaining_w - VAL_COL_GAP) // 2
        col1_x = left_x + label_w
        col2_x = col1_x + col_w + VAL_COL_GAP

        hdr_init = fonts["small"].render("初始", True, C(TEXT_SEC_STRONG))
        hdr_now = fonts["small"].render("当前", True, C(TEXT_SEC_STRONG))
        screen.blit(hdr_init, (col1_x + (col_w - hdr_init.get_width()) // 2, cy))
        screen.blit(hdr_now, (col2_x + (col_w - hdr_now.get_width()) // 2, cy))
        cy += hdr_init.get_height() + 10

        ek_label = fonts["small"].render("总动能", True, C(TEXT_SEC_STRONG))
        ek_init_s = fonts["small"].render(f"{self.initial_ek:.2f} J", True, C(TEXT_PRI))
        ek_now_s = fonts["small"].render(
            f"{self.collision.total_k_energy:.2f} J", True, C(TEXT_PRI)
        )
        screen.blit(ek_label, (left_x, cy))
        screen.blit(ek_init_s, (col1_x + (col_w - ek_init_s.get_width()) // 2, cy))
        screen.blit(ek_now_s, (col2_x + (col_w - ek_now_s.get_width()) // 2, cy))
        cy += ek_init_s.get_height() + 14

        p_label = fonts["small"].render("总动量", True, C(TEXT_SEC_STRONG))
        p_init_s = fonts["small"].render(
            f"{self.initial_p:.2f} kg·m/s", True, C(TEXT_PRI)
        )
        p_now_s = fonts["small"].render(
            f"{self.collision.total_momentum:.2f} kg·m/s", True, C(TEXT_PRI)
        )
        screen.blit(p_label, (left_x, cy))
        screen.blit(p_init_s, (col1_x + (col_w - p_init_s.get_width()) // 2, cy))
        screen.blit(p_now_s, (col2_x + (col_w - p_now_s.get_width()) // 2, cy))
        cy += p_init_s.get_height() + 14

    def _draw_hintbar(self, screen, fonts):
        by = H - HINT_H
        pygame.draw.rect(screen, C(BUTTON_COL), (0, by, W, HINT_H))
        pygame.draw.line(screen, C(DIVIDER), (0, by), (W, by))

        keys = [
            ("SPACE", "暂停/继续"),
            ("R", "重置"),
            ("E", "切换碰撞类型"),
            ("S", "导出图表/CSV"),
            ("Q", "退出"),
        ]
        cx = 20
        for k, desc in keys:
            kb = fonts["small"].render(k, True, C(TEXT_HEAD))
            kw = kb.get_width() + 14
            kh = kb.get_height() + 6
            ky = by + (HINT_H - kh) // 2
            pygame.draw.rect(screen, C(BOTTON_COL), (cx, ky, kw, kh), border_radius=4)
            pygame.draw.rect(
                screen, C("#3C4155"), (cx, ky, kw, kh), width=1, border_radius=4
            )
            screen.blit(kb, (cx + 7, ky + 3))
            cx += kw + 6
            db = fonts["small"].render(desc, True, C(TEXT_SEC_STRONG))
            screen.blit(db, (cx, by + (HINT_H - db.get_height()) // 2))
            cx += db.get_width() + 28

    def _draw_paused_overlay(self, screen, fonts):
        overlay = pygame.Surface((SIM_W, SIM_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 80))
        screen.blit(overlay, (0, 0))
        s = fonts["h1"].render("已暂停", True, C(AMBER))
        screen.blit(
            s, (SIM_W // 2 - s.get_width() // 2, FLOOR_Y // 2 - s.get_height() // 2)
        )

    def _compute_panel_layout(self, fonts):
        px = SIM_W
        pad = 20
        content_w = PANEL_W - pad * 2
        col_gap = 20
        col_total_w = (content_w - col_gap) // 2

        # label_max covers "质量" and "速度"
        label_w_candidates = [
            fonts["small"].render("质量", True, C(TEXT_SEC_STRONG)).get_width(),
            fonts["small"].render("速度", True, C(TEXT_SEC_STRONG)).get_width(),
        ]
        label_max_w = max(label_w_candidates) + 8  # padding after label

        unit_examples = ["kg", "m/s", f"({self.collision.kind.label})"]
        unit_width = max(
            (
                fonts["small"].render(u, True, C(TEXT_SEC_STRONG)).get_width()
                for u in unit_examples
            )
        )
        unit_width = max(unit_width, 36)

        # keep original box width logic (per-column)
        box_w = max(MIN_BOX_W, col_total_w - unit_width - UNIT_PADDING)

        # cap label width so it doesn't push box off content area
        cap_label_w = min(
            label_max_w, content_w - box_w - unit_width - UNIT_PADDING - 8
        )
        left_x = px + pad

        cy = 24
        cy += fonts["h2"].get_height() + 6

        # e box (same position)
        e_box_x = px + pad + content_w - box_w - unit_width - UNIT_PADDING
        self.box_rects["e"] = pygame.Rect(e_box_x, cy, box_w, ROW_H)
        self.hit_rects["e"] = pygame.Rect(
            e_box_x, cy, box_w + unit_width + UNIT_PADDING + HIT_EXTRA, ROW_H
        )

        cy += ROW_H + ROW_GAP + 6
        cy += fonts["h2"].get_height() + 6

        # 物块 A
        self.box_rects["ma"] = pygame.Rect(left_x + cap_label_w, cy, box_w, ROW_H)
        self.hit_rects["ma"] = pygame.Rect(
            left_x + cap_label_w,
            cy,
            box_w + unit_width + UNIT_PADDING + HIT_EXTRA,
            ROW_H,
        )

        cy += ROW_H + ROW_GAP
        self.box_rects["va"] = pygame.Rect(left_x + cap_label_w, cy, box_w, ROW_H)
        self.hit_rects["va"] = pygame.Rect(
            left_x + cap_label_w,
            cy,
            box_w + unit_width + UNIT_PADDING + HIT_EXTRA,
            ROW_H,
        )

        cy += ROW_H + ROW_GAP + 6
        cy += fonts["h2"].get_height() + 6

        # 物块 B
        self.box_rects["mb"] = pygame.Rect(left_x + cap_label_w, cy, box_w, ROW_H)
        self.hit_rects["mb"] = pygame.Rect(
            left_x + cap_label_w,
            cy,
            box_w + unit_width + UNIT_PADDING + HIT_EXTRA,
            ROW_H,
        )

        cy += ROW_H + ROW_GAP
        self.box_rects["vb"] = pygame.Rect(left_x + cap_label_w, cy, box_w, ROW_H)
        self.hit_rects["vb"] = pygame.Rect(
            left_x + cap_label_w,
            cy,
            box_w + unit_width + UNIT_PADDING + HIT_EXTRA,
            ROW_H,
        )

    def _current_value_str(self, field_name: str) -> str:
        match field_name:
            case "ma":
                return self._format_field_value("ma", self.block_a.m)
            case "va":
                return self._format_field_value("va", self.block_a.v)
            case "mb":
                return self._format_field_value("mb", self.block_b.m)
            case "vb":
                return self._format_field_value("vb", self.block_b.v)
            case "e":
                return self._format_field_value("e", self.collision.e)
            case _:
                return ""

    def _commit_active_input(self):
        # 卫语句：没有 active_input 则直接返回
        if not self.active_input:
            return

        field = self.active_input
        buf = self.input_buffers.get(field, "")

        try:
            val = float(buf)
        except Exception:
            # 恢复为当前值并退出
            self.input_buffers[field] = self._current_value_str(field)
            self.active_input = None
            self.paused = self._paused_before_edit
            return

        # 使用 match 处理字段逻辑，结构更清晰
        match field:
            case "ma":
                self.block_a.m = max(0.0001, val)
            case "va":
                self.block_a.v = val
            case "mb":
                self.block_b.m = max(0.0001, val)
            case "vb":
                self.block_b.v = val
            case "e":
                val = max(0.0, min(1.0, val))
                self.collision = D(self.block_a, self.block_b, e=val)
                # 用户通过编辑更改 e 时也应清除历史与碰撞记录
                self._time = 0.0
                self._history.clear()
                self.first_collision_recorded = False
                self.collision_count = 0
                self.p_before = None
                self.p_after = None
                self.ek_before = None
                self.ek_after = None
            case _:
                pass

        # 更新面板缓冲为当前实际值并取消编辑 (使用统一格式)
        self.input_buffers[field] = self._current_value_str(field)
        self.active_input = None
        self.paused = self._paused_before_edit

        # ★ 参数变更后重新计算初始动量/动能（面板"初始"列立即同步）
        self.collision.block_1 = self.block_a
        self.collision.block_2 = self.block_b
        self.initial_p = self.collision.total_momentum
        self.initial_ek = self.collision.total_k_energy

        # 清空历史与碰撞记录，以新参数为新起点（e 字段已在 match 内处理，此处统一覆盖）
        self._export_pending = False
        self._time = 0.0
        self._history.clear()
        self.first_collision_recorded = False
        self.collision_count = 0
        self.p_before = None
        self.p_after = None
        self.ek_before = None
        self.ek_after = None
