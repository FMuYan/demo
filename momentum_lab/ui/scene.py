"""
ui/scene.py  —  pygame 可视化场景（带交互输入面板）
布局：左侧仿真轨道区 | 右侧信息面板（固定 320px）

改动说明：
- 点击输入时自动暂停，编辑完成或取消后恢复之前的暂停状态。
- 切换恢复系数 e 时保留当前物块质量/速度（不再重置），除非按 R 重置。
- 输入框显示单位（非编辑时），编辑时显示纯数字缓冲。
- 恢复右侧面板中的动量/动能显示，支持碰撞前后对比。
"""

from __future__ import annotations

import pygame
import sys
from momentum_lab.model.block import Block, D

# ── 调色板 ──────────────────────────────────────────────────
MAIN = "#DEDEDE"
BG = MAIN  # 深色背景
TRACK_BG = MAIN  # 轨道区背景
PANEL_BG = "#E8E8E8"  # 面板背景
DIVIDER = "#2F2F2F"  # 分割线
FLOOR_COL = "#5C5C5C"  # 地面
BUTTON_COL = "#C7C7C7"
TEXT_PRI = "#1A1C2E"  # 主文字
TEXT_SEC = "#454547"  # 次要文字
TEXT_HEAD = "#1A1C2E"
BOTTON_COL = "#FFFFFF"


COL_A = (77, 166, 255)  # 物块 A —— 亮蓝
COL_B = (255, 107, 90)  # 物块 B —— 珊瑚红
COL_A_DIM = (40, 90, 150)
COL_B_DIM = (150, 55, 45)
GREEN = (72, 199, 116)
AMBER = (255, 190, 60)
RED = (255, 80, 80)


# ── 窗口 & 布局 ─────────────────────────────────────────────
W, H = 1120, 620
PANEL_W = 320
SIM_W = W - PANEL_W  # 800px 仿真区宽度

FLOOR_Y = H - 160  # 地面 y（留底部空间给 hint bar）
# BLOCK_H = 68
HINT_H = 44  # 底部按键提示栏高度
Y_FONTS = "Maple Mono NF CN"  # 字体

# 仿真坐标系：仿真区左右各留 60px 边距
SIM_PAD = 60
TRACK_W = SIM_W - SIM_PAD * 2  # 实际轨道像素宽
PIXELS_PER_M = TRACK_W / 8.0  # 8m 的轨道
ORIGIN_X = SIM_PAD  # 仿真 x=0 对应屏幕像素（相对仿真区）


def sim_x(x: float) -> int:
    """仿真坐标 → 屏幕 x（绝对）"""
    return int(ORIGIN_X + x * PIXELS_PER_M)


def _block_w(blk: Block) -> int:
    """质量 → 宽度，区间 [40, 110] px"""
    return max(40, min(110, int(blk.m * 22 + 28)))


def _block_h(blk: Block) -> int:
    """质量 → 宽度，区间 [40, 110] px"""
    return max(40, min(110, int(blk.m * 22 + 28)))


# ── 字体加载 ──────────────────────────────────────
def _load_fonts():
    def F(size, is_bold=False):
        return pygame.font.SysFont(Y_FONTS, size, bold=is_bold)

    return {
        "h1": F(20, is_bold=True),
        "h2": F(16, is_bold=True),
        "body": F(15),
        "small": F(13),
        "hint": F(13),
        "mono": F(14),
    }


class Scene:
    """
    主场景

    键位：
        SPACE   暂停 / 继续
        R       重置
        E       切换碰撞类型（保留当前质量/速度）
        Q/ESC   退出

    交互输入：在右侧面板点击参数项进入编辑，输入数字后按回车确认，Esc 取消。
    点击输入时会自动暂停，编辑结束后恢复到编辑前的暂停状态。
    """

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

        self.p_before: float | None = None
        self.p_after: float | None = None
        self.ek_before: float | None = None
        self.ek_after: float | None = None

        # 碰撞闪光效果计时（帧数）
        self._flash = 0

        # 输入面板状态
        # 字段标识：ma, va, mb, vb, e
        self.input_buffers: dict[str, str] = {
            "ma": f"{self.block_a.m}",
            "va": f"{self.block_a.v}",
            "mb": f"{self.block_b.m}",
            "vb": f"{self.block_b.v}",
            "e": f"{self.collision.e}",
        }
        self.active_input: str | None = None
        # 在运行时由 _compute_panel_layout 填充（字段 -> pygame.Rect）
        self.input_rects: dict[str, pygame.Rect] = {}

        # 编辑时的暂停状态保存
        self._paused_before_edit = False

    # ── 主循环 ───────────────────────────────────────────────
    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((W, H), pygame.NOFRAME)
        # screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("动量守恒演示")
        clock = pygame.time.Clock()
        fonts = _load_fonts()

        # 预计算一次面板布局，用于点击命中检测
        self._compute_panel_layout(fonts)

        while True:
            dt = clock.tick(self.fps) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    # 点击面板输入框，选中并自动暂停
                    for field, rect in self.input_rects.items():
                        if rect.collidepoint(mx, my):
                            self.active_input = field
                            # buffer 初始化为当前字符串表示
                            self.input_buffers[field] = self._current_value_str(field)
                            # 记住之前的暂停状态并暂停仿真
                            self._paused_before_edit = self.paused
                            self.paused = True
                            break
                    else:
                        # 如果当前处于编辑状态但点击面板外，取消编辑并恢复暂停状态
                        if self.active_input is not None:
                            self.active_input = None
                            self.paused = self._paused_before_edit

                if event.type == pygame.KEYDOWN:
                    # 如果正在输入，优先处理编辑事件
                    if self.active_input is not None:
                        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            # 确认并应用
                            self._commit_active_input()
                        elif event.key == pygame.K_ESCAPE:
                            # 取消编辑，恢复暂停状态
                            self.active_input = None
                            self.paused = self._paused_before_edit
                        elif event.key == pygame.K_BACKSPACE:
                            self.input_buffers[self.active_input] = self.input_buffers[self.active_input][:-1]
                        else:
                            ch = event.unicode
                            if ch and (ch.isdigit() or ch in ".-"):
                                self.input_buffers[self.active_input] += ch
                        # 不再进行其他键位响应
                        continue

                    # 非输入模式的键位响应
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self._reset()
                    elif event.key == pygame.K_e:
                        self._cycle_e()

            if not self.paused:
                self._update(dt)
                if self._flash > 0:
                    self._flash -= 1

            self._draw(screen, fonts)
            pygame.display.flip()

    # ── 物理 ─────────────────────────────────────────────────
    def _update(self, dt: float):
        a, b = self.block_a, self.block_b
        wa = _block_w(a)
        ax_right = sim_x(a.x) + wa
        bx_left = sim_x(b.x)

        if ax_right >= bx_left and a.v > b.v:
            self.collision.block_1 = a
            self.collision.block_2 = b
            if not self.first_collision_recorded:
                self.p_before = self.collision.total_momentum
                self.ek_before = self.collision.total_k_energy
            na, nb = self.collision.collide()
            # 防止碰撞后重叠
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

        self.block_a.x += self.block_a.v * dt
        self.block_b.x += self.block_b.v * dt

        # 左右边界反弹
        for blk in (self.block_a, self.block_b):
            w = _block_w(blk)
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

        # 重置输入缓冲
        self.input_buffers = {
            "ma": f"{self.block_a.m}",
            "va": f"{self.block_a.v}",
            "mb": f"{self.block_b.m}",
            "vb": f"{self.block_b.v}",
            "e": f"{self.collision.e}",
        }
        self.active_input = None
        self._paused_before_edit = False

    def _cycle_e(self):
        # 切换恢复系数，但保留当前质量/速度/位置状态（不重置）
        self._e_idx = (self._e_idx + 1) % len(self.E_PRESETS)
        new_e = self.E_PRESETS[self._e_idx]
        # 重新构建 collision 以更新 kind，但使用当前 block 对象
        self.collision = D(self.block_a, self.block_b, e=new_e)

    # ── 渲染 ─────────────────────────────────────────────────
    def _draw(self, screen: pygame.Surface, fonts: dict):
        screen.fill(BG)

        # 仿真区底色
        pygame.draw.rect(screen, TRACK_BG, (0, 0, SIM_W, H))

        # self._draw_grid(screen)
        self._draw_track(screen, fonts)
        self._draw_blocks(screen, fonts)
        self._draw_panel(screen, fonts)
        self._draw_hintbar(screen, fonts)

        if self.paused:
            self._draw_paused_overlay(screen, fonts)

    # def _draw_grid(self, screen):
    #     """轻量网格线"""
    #     for m in range(0, 9):
    #         gx = sim_x(m)
    #         pygame.draw.line(screen, FLOOR_GRID, (gx, 60), (gx, FLOOR_Y), 1)

    def _draw_track(self, screen, fonts):
        """轨道：地面 + 刻度 + 标尺标签"""
        # 地面
        pygame.draw.line(
            screen,
            FLOOR_COL,
            (SIM_PAD - 10, FLOOR_Y),
            (SIM_W - SIM_PAD + 10, FLOOR_Y),
            2,
        )
        # 刻度 & 标签
        for m in range(0, 9):
            gx = sim_x(m)
            pygame.draw.line(screen, FLOOR_COL, (gx, FLOOR_Y), (gx, FLOOR_Y + 8), 1)
            lbl = fonts["small"].render(str(m), True, TEXT_SEC)
            screen.blit(lbl, (gx - lbl.get_width() // 2, FLOOR_Y + 11))
        # 单位
        unit = fonts["small"].render("m", True, TEXT_SEC)
        screen.blit(unit, (SIM_W - SIM_PAD + 14, FLOOR_Y + 11))

    def _draw_blocks(self, screen, fonts):
        for blk, col, col_dim, label in (
            (self.block_a, COL_A, COL_A_DIM, "A"),
            (self.block_b, COL_B, COL_B_DIM, "B"),
        ):
            w = _block_w(blk)
            h = _block_h(blk)
            sx = sim_x(blk.x)
            sy = FLOOR_Y - h

            # 碰撞闪光：高亮边框
            if self._flash > 0:
                # glow_rect = pygame.Rect(sx - 3, sy - 3, w + 6, h + 6)
                alpha = min(255, self._flash * 14)
                glow = pygame.Surface((w + 6, h + 6), pygame.SRCALPHA)
                glow.fill((*col, alpha))
                screen.blit(glow, (sx - 3, sy - 3))

            # 物块主体（渐变感：上亮下暗）
            body = pygame.Surface((w, h), pygame.SRCALPHA)
            for row in range(h):
                t = row / h
                r = int(col[0] * (1 - t * 0.35) + col_dim[0] * t * 0.35)
                g = int(col[1] * (1 - t * 0.35) + col_dim[1] * t * 0.35)
                b_ = int(col[2] * (1 - t * 0.35) + col_dim[2] * t * 0.35)
                pygame.draw.line(body, (r, g, b_), (0, row), (w, row))
            screen.blit(body, (sx, sy))

            # 圆角遮罩（用 border_radius rect）
            # overlay = pygame.Surface((w, h), pygame.SRCALPHA)
            # pygame.draw.rect(overlay, (0, 0, 0, 0), (0, 0, w, h), border_radius=8)
            # pygame.draw.rect(
            #     screen, col, pygame.Rect(sx, sy, w, h), width=2, border_radius=8
            # )

            # 物块标签（字母 + 质量）
            # BLOCK_LABEL_COL
            lbl = fonts["h2"].render(label, True, MAIN)
            mlbl = fonts["small"].render(f"{blk.m} kg", True, MAIN)
            screen.blit(lbl, (sx + w // 2 - lbl.get_width() // 2, sy + 10))
            screen.blit(mlbl, (sx + w // 2 - mlbl.get_width() // 2, sy + 32))

            # 速度箭头
            _draw_velocity_arrow(screen, blk, sx, sy, w, col)

            # 速度数值（物块上方）
            vtext = fonts["small"].render(f"{blk.v:.2f} m/s", True, col)
            screen.blit(vtext, (sx + w // 2 - vtext.get_width() // 2, FLOOR_Y - h - 40))

    def _draw_panel(self, screen, fonts):
        """右侧信息面板（包含输入框与动量/动能展示）"""
        px = SIM_W
        pygame.draw.rect(screen, PANEL_BG, (px, 0, PANEL_W, H))
        pygame.draw.line(screen, DIVIDER, (px, 0), (px, H), 1)

        cy = 24
        pad = 20

        # ── 标题 ──────────────────────────────────────────
        cy = _panel_heading(screen, fonts, px + pad, cy, "参数（点击以编辑）", TEXT_HEAD)
        cy += 4

        # 物块 A 输入
        cy = _panel_heading(screen, fonts, px + pad, cy, "物块 A", COL_A)
        cy += 6
        cy = self._draw_input_row(screen, fonts, px + pad, cy, PANEL_W - pad * 2, "质量 (kg)", "ma", val_col=COL_A)
        cy = self._draw_input_row(screen, fonts, px + pad, cy, PANEL_W - pad * 2, "速度 (m/s)", "va", val_col=COL_A)
        cy += 6

        # 物块 B 输入
        cy = _panel_heading(screen, fonts, px + pad, cy, "物块 B", COL_B)
        cy += 6
        cy = self._draw_input_row(screen, fonts, px + pad, cy, PANEL_W - pad * 2, "质量 (kg)", "mb", val_col=COL_B)
        cy = self._draw_input_row(screen, fonts, px + pad, cy, PANEL_W - pad * 2, "速度 (m/s)", "vb", val_col=COL_B)
        cy += 10

        # 恢复系数输入
        cy = _panel_heading(screen, fonts, px + pad, cy, "碰撞参数", TEXT_HEAD)
        cy += 6
        cy = self._draw_input_row(screen, fonts, px + pad, cy, PANEL_W - pad * 2, "恢复系数 e", "e")

        cy += 12
        # ── 碰撞验证（动量 / 动能）──────────────────────────────────────
        cy = _panel_heading(screen, fonts, px + pad, cy, "动能 / 动量", TEXT_HEAD)
        cy += 8

        self.collision.block_1 = self.block_a
        self.collision.block_2 = self.block_b

        if (
            self.p_before is not None
            and self.p_after is not None
            and self.ek_before is not None
            and self.ek_after is not None
        ):
            cy = _kv_row(
                screen,
                fonts,
                px + pad,
                cy,
                PANEL_W - pad * 2,
                "碰撞后 总动量 p",
                f"{self.p_after:.4f} kg·m/s",
            )
            cy = _kv_row(
                screen,
                fonts,
                px + pad,
                cy,
                PANEL_W - pad * 2,
                "碰撞后 总动能 Ek",
                f"{self.ek_after:.4f} J",
            )
            cy += 6
            # 显示碰撞前后的比较（简单标色）
            ok = abs(self.p_after - self.p_before) < 1e-6 if (self.p_before is not None and self.p_after is not None) else False
            col = GREEN if ok else AMBER
            status = "动量守恒（近似）" if ok else "动量变化（超出容差）"
            s = fonts["small"].render(status, True, col)
            screen.blit(s, (px + pad, cy))
            cy += s.get_height() + 2
        else:
            # 显示当前值
            p_now = self.collision.total_momentum
            ek_now = self.collision.total_k_energy
            cy = _kv_row(
                screen,
                fonts,
                px + pad,
                cy,
                PANEL_W - pad * 2,
                "当前 总动量 p",
                f"{p_now:.4f} kg·m/s",
            )
            cy = _kv_row(
                screen,
                fonts,
                px + pad,
                cy,
                PANEL_W - pad * 2,
                "当前 总动能 Ek",
                f"{ek_now:.4f} J",
            )

        # 说明提示
        tip = fonts["small"].render("回车确认，Esc 取消。点击数值区域开始编辑。", True, TEXT_SEC)
        screen.blit(tip, (px + pad, H - 80))

    def _draw_hintbar(self, screen, fonts):
        """底部按键说明栏"""
        by = H - HINT_H
        pygame.draw.rect(screen, BUTTON_COL, (0, by, W, HINT_H))
        pygame.draw.line(screen, DIVIDER, (0, by), (W, by))

        keys = [
            ("SPACE", "暂停/继续"),
            ("R", "重置"),
            ("E", "切换碰撞类型"),
            ("Q", "退出"),
        ]
        cx = 20
        for k, desc in keys:
            # 按键 badge
            kb = fonts["small"].render(k, True, TEXT_HEAD)
            kw = kb.get_width() + 14
            kh = kb.get_height() + 6
            ky = by + (HINT_H - kh) // 2
            pygame.draw.rect(screen, BOTTON_COL, (cx, ky, kw, kh), border_radius=4)
            pygame.draw.rect(
                screen, (60, 65, 85), (cx, ky, kw, kh), width=1, border_radius=4
            )
            screen.blit(kb, (cx + 7, ky + 3))
            cx += kw + 6

            db = fonts["small"].render(desc, True, TEXT_SEC)
            screen.blit(db, (cx, by + (HINT_H - db.get_height()) // 2))
            cx += db.get_width() + 28

    def _draw_paused_overlay(self, screen, fonts):
        overlay = pygame.Surface((SIM_W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 80))
        screen.blit(overlay, (0, 0))
        s = fonts["h1"].render("已暂停", True, AMBER)
        screen.blit(
            s, (SIM_W // 2 - s.get_width() // 2, FLOOR_Y // 2 - s.get_height() // 2)
        )

    # ── 面板辅助函数（输入框实现） ────────────────────────────────
    def _compute_panel_layout(self, fonts):
        """基于字体计算输入框的位置 rect（用于点击检测）。"""
        px = SIM_W
        pad = 20
        cy = 24
        cy += fonts["h2"].get_height() + 2 + 4  # 标题高度

        # 跳过到物块 A 标题
        cy += fonts["h2"].get_height() + 6

        box_h = fonts["small"].get_height() + 8
        box_w = 110
        right_x = px + (PANEL_W - pad) - box_w

        # ma
        self.input_rects["ma"] = pygame.Rect(right_x, cy + 0, box_w, box_h)
        # va
        self.input_rects["va"] = pygame.Rect(right_x, cy + box_h + 6, box_w, box_h)
        # move cy forward past A
        cy = cy + box_h * 2 + 6 + 6 + fonts["h2"].get_height()

        # B title handled, place mb, vb
        # mb
        self.input_rects["mb"] = pygame.Rect(right_x, cy + 0, box_w, box_h)
        # vb
        self.input_rects["vb"] = pygame.Rect(right_x, cy + box_h + 6, box_w, box_h)

        # e field near lower area
        self.input_rects["e"] = pygame.Rect(right_x, H - 140, box_w, box_h)

    def _unit_for_field(self, field_name: str) -> str:
        return {"ma": " kg", "va": " m/s", "mb": " kg", "vb": " m/s", "e": ""}.get(field_name, "")

    def _draw_input_row(self, screen, fonts, x, y, w, key, field_name, val_col=None):
        """绘制一行带输入框的键值对，点击可编辑。"""
        ks = fonts["small"].render(key, True, TEXT_SEC)
        screen.blit(ks, (x, y))

        # 输入框位置（每次按布局计算，以便动态显示）
        box_w = 110
        box_h = ks.get_height() + 8
        box_x = x + w - box_w
        box_y = y
        rect = pygame.Rect(box_x, box_y, box_w, box_h)
        self.input_rects[field_name] = rect

        # 背景
        pygame.draw.rect(screen, BOTTON_COL, rect, border_radius=6)
        pygame.draw.rect(screen, DIVIDER, rect, width=1, border_radius=6)

        # 如果是活动输入框，画高亮边框
        if self.active_input == field_name:
            pygame.draw.rect(screen, (40, 120, 220), rect, width=2, border_radius=6)

        # 显示当前缓冲或实时数值；非编辑状态在数值后显示单位
        if self.active_input == field_name:
            display_text = self.input_buffers.get(field_name, self._current_value_str(field_name))
        else:
            raw = self.input_buffers.get(field_name, self._current_value_str(field_name))
            # 有时 buffer 中会包含 the numeric string; ensure we show current actual value if different
            display_text = raw + self._unit_for_field(field_name)

        txt_surf = fonts["small"].render(display_text, True, val_col or TEXT_PRI)
        screen.blit(txt_surf, (box_x + 8, box_y + (box_h - txt_surf.get_height()) // 2))

        return y + box_h + 6

    def _current_value_str(self, field_name: str) -> str:
        if field_name == "ma":
            return f"{self.block_a.m}"
        if field_name == "va":
            return f"{self.block_a.v}"
        if field_name == "mb":
            return f"{self.block_b.m}"
        if field_name == "vb":
            return f"{self.block_b.v}"
        if field_name == "e":
            return f"{self.collision.e}"
        return ""

    def _commit_active_input(self):
        """尝试解析并应用 active_input 的值，失败则恢复为旧值。编辑结束后恢复暂停状态。"""
        if not self.active_input:
            return
        buf = self.input_buffers.get(self.active_input, "")
        try:
            val = float(buf)
        except Exception:
            # 恢复为当前值
            self.input_buffers[self.active_input] = self._current_value_str(self.active_input)
            self.active_input = None
            # 恢复暂停状态
            self.paused = self._paused_before_edit
            return

        # 根据字段应用到物理属性
        if self.active_input == "ma":
            self.block_a.m = max(0.0001, val)
        elif self.active_input == "va":
            self.block_a.v = val
        elif self.active_input == "mb":
            self.block_b.m = max(0.0001, val)
        elif self.active_input == "vb":
            self.block_b.v = val
        elif self.active_input == "e":
            # 限制到 [0,1]
            val = max(0.0, min(1.0, val))
            # 重新构建 collision 对象以更新 kind（保留当前 block 状态）
            self.collision = D(self.block_a, self.block_b, e=val)

        # 更新面板缓冲并取消编辑
        self.input_buffers[self.active_input] = self._current_value_str(self.active_input)
        self.active_input = None
        # 恢复暂停状态
        self.paused = self._paused_before_edit

    # ── 仿真区辅助 ───────────────────────────────────────────────


def _draw_velocity_arrow(screen, blk, sx, sy, w, col):
    """速度箭头，画在物块上方"""
    if abs(blk.v) < 0.05:
        return
    cx = sx + w // 2
    ay = sy - 14
    sign = 1 if blk.v > 0 else -1
    length = min(90, max(16, int(abs(blk.v) * 18)))
    tip = cx + sign * length
    pygame.draw.line(screen, col, (cx, ay), (tip, ay), 2)
    pygame.draw.polygon(
        screen,
        col,
        [
            (tip, ay),
            (tip - sign * 10, ay - 5),
            (tip - sign * 10, ay + 5),
        ],
    )


# ── 面板辅助函数 ─────────────────────────────────────────────


def _panel_heading(screen, fonts, x, y, text, color):
    s = fonts["h2"].render(text, True, color)
    screen.blit(s, (x, y))
    return y + s.get_height() + 2


def _kv_row(screen, fonts, x, y, w, key, val, val_col=None):
    ks = fonts["small"].render(key, True, TEXT_SEC)
    vs = fonts["small"].render(val, True, val_col or TEXT_PRI)
    screen.blit(ks, (x, y))
    screen.blit(vs, (x + w - vs.get_width(), y))
    return y + ks.get_height() + 5
