"""
ui/scene.py  —  pygame 可视化场景（优化面板布局与样式）
改动要点：
- 每次绘制时动态计算输入框布局以支持不同字体/尺寸
- 在右侧面板增加阴影与标题栏，E（恢复系数）放在标题栏右侧
- 统一面板内间距、分割线与字体层级，数值右对齐，单位在数值右侧靠外显示
- 修复 Esc 取消编辑时恢复缓冲的顺序问题
- 小的视觉优化：更明显的分割线、按钮提示区对齐、更一致的颜色
"""

from __future__ import annotations

import pygame
import sys
from momentum_lab.model.block import Block, D

# ── 调色板 ──────────────────────────────────────────────────
MAIN = "#F6F6F8"
BG = MAIN
TRACK_BG = "#FFFFFF"
PANEL_BG = "#F0F2F4"
DIVIDER = "#D0D6DB"
FLOOR_COL = "#5C5C5C"
BUTTON_COL = "#FFFFFF"
TEXT_PRI = "#1A1C2E"
TEXT_SEC = "#6B6F76"
TEXT_HEAD = "#0E1114"

COL_A = (77, 166, 255)
COL_B = (255, 107, 90)
GREEN = (72, 199, 116)
AMBER = (255, 190, 60)

# ── 窗口 & 布局 ─────────────────────────────────────────────
W, H = 1120, 620
PANEL_W = 320
SIM_W = W - PANEL_W
FLOOR_Y = H - 160
HINT_H = 44
Y_FONTS = None  # use system default
SIM_PAD = 60
TRACK_W = SIM_W - SIM_PAD * 2
PIXELS_PER_M = TRACK_W / 8.0
ORIGIN_X = SIM_PAD


def sim_x(x: float) -> int:
    return int(ORIGIN_X + x * PIXELS_PER_M)


def _block_w(blk: Block) -> int:
    return max(40, min(110, int(blk.m * 22 + 28)))


def _block_h(blk: Block) -> int:
    return max(40, min(110, int(blk.m * 22 + 28)))


# ── 字体加载（更合理的层级）
def _load_fonts():
    def F(size, is_bold=False):
        return pygame.font.SysFont(Y_FONTS, size, bold=is_bold)

    return {
        "title": F(18, is_bold=True),
        "h2": F(15, is_bold=True),
        "body": F(14),
        "small": F(13),
        "hint": F(12),
        "mono": F(13),
    }


class Scene:
    """
    主场景 — 优化后的右侧实验面板
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

        # initial metrics (set on reset/init)
        self.initial_p = self.collision.total_momentum
        self.initial_ek = self.collision.total_k_energy

        self.p_before: float | None = None
        self.p_after: float | None = None
        self.ek_before: float | None = None
        self.ek_after: float | None = None

        self._flash = 0

        # input state
        self.input_buffers: dict[str, str] = {
            "ma": f"{self.block_a.m}",
            "va": f"{self.block_a.v}",
            "mb": f"{self.block_b.m}",
            "vb": f"{self.block_b.v}",
            "e": f"{self.collision.e}",
        }
        self.active_input: str | None = None
        self.input_rects: dict[str, pygame.Rect] = {}
        self._paused_before_edit = False

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("动量守恒演示")
        clock = pygame.time.Clock()
        fonts = _load_fonts()

        while True:
            dt = clock.tick(self.fps) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    for field, rect in self.input_rects.items():
                        if rect.collidepoint(mx, my):
                            self.active_input = field
                            # 清空缓冲以便快速输入
                            self.input_buffers[field] = ""
                            self._paused_before_edit = self.paused
                            self.paused = True
                            break
                    else:
                        if self.active_input is not None:
                            # cancel edit when clicking outside
                            self.active_input = None
                            self.paused = self._paused_before_edit

                if event.type == pygame.KEYDOWN:
                    if self.active_input is not None:
                        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            self._commit_active_input()
                        elif event.key == pygame.K_ESCAPE:
                            # restore buffer for that field, then clear active
                            old = self.active_input
                            if old:
                                self.input_buffers[old] = self._current_value_str(old)
                            self.active_input = None
                            self.paused = self._paused_before_edit
                        elif event.key == pygame.K_BACKSPACE:
                            if self.input_buffers.get(self.active_input):
                                self.input_buffers[self.active_input] = self.input_buffers[self.active_input][:-1]
                        else:
                            ch = event.unicode
                            if ch and (ch.isdigit() or ch in ".-"):
                                self.input_buffers[self.active_input] += ch
                        continue

                    # non-edit keys
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

        self.input_buffers = {
            "ma": f"{self.block_a.m}",
            "va": f"{self.block_a.v}",
            "mb": f"{self.block_b.m}",
            "vb": f"{self.block_b.v}",
            "e": f"{self.collision.e}",
        }
        self.active_input = None
        self._paused_before_edit = False

        self.initial_p = self.collision.total_momentum
        self.initial_ek = self.collision.total_k_energy

    def _cycle_e(self):
        self._e_idx = (self._e_idx + 1) % len(self.E_PRESETS)
        new_e = self.E_PRESETS[self._e_idx]
        self.collision = D(self.block_a, self.block_b, e=new_e)

    def _draw(self, screen: pygame.Surface, fonts: dict):
        screen.fill(BG)
        pygame.draw.rect(screen, TRACK_BG, (0, 0, SIM_W, H))
        self._draw_track(screen, fonts)
        self._draw_blocks(screen, fonts)
        self._draw_panel(screen, fonts)
        self._draw_hintbar(screen, fonts)
        if self.paused:
            self._draw_paused_overlay(screen, fonts)

    def _draw_track(self, screen, fonts):
        pygame.draw.line(screen, FLOOR_COL, (SIM_PAD - 10, FLOOR_Y), (SIM_W - SIM_PAD + 10, FLOOR_Y), 2)
        for m in range(0, 9):
            gx = sim_x(m)
            pygame.draw.line(screen, FLOOR_COL, (gx, FLOOR_Y), (gx, FLOOR_Y + 8), 1)
            lbl = fonts["small"].render(str(m), True, TEXT_SEC)
            screen.blit(lbl, (gx - lbl.get_width() // 2, FLOOR_Y + 11))
        unit = fonts["small"].render("m", True, TEXT_SEC)
        screen.blit(unit, (SIM_W - SIM_PAD + 14, FLOOR_Y + 11))

    def _draw_blocks(self, screen, fonts):
        for blk, col, col_dim, label in ((self.block_a, COL_A, (40, 90, 150), "A"), (self.block_b, COL_B, (150, 55, 45), "B")):
            w = _block_w(blk)
            h = _block_h(blk)
            sx = sim_x(blk.x)
            sy = FLOOR_Y - h
            if self._flash > 0:
                alpha = min(255, self._flash * 14)
                glow = pygame.Surface((w + 6, h + 6), pygame.SRCALPHA)
                glow.fill((*col, alpha))
                screen.blit(glow, (sx - 3, sy - 3))
            body = pygame.Surface((w, h), pygame.SRCALPHA)
            for row in range(h):
                t = row / h
                r = int(col[0] * (1 - t * 0.35) + col_dim[0] * t * 0.35)
                g = int(col[1] * (1 - t * 0.35) + col_dim[1] * t * 0.35)
                b_ = int(col[2] * (1 - t * 0.35) + col_dim[2] * t * 0.35)
                pygame.draw.line(body, (r, g, b_), (0, row), (w, row))
            screen.blit(body, (sx, sy))
            lbl = fonts["h2"].render(label, True, MAIN)
            mlbl = fonts["small"].render(f"{blk.m} kg", True, MAIN)
            screen.blit(lbl, (sx + w // 2 - lbl.get_width() // 2, sy + 10))
            screen.blit(mlbl, (sx + w // 2 - mlbl.get_width() // 2, sy + 32))
            _draw_velocity_arrow(screen, blk, sx, sy, w, col)
            vtext = fonts["small"].render(f"{blk.v:.2f} m/s", True, col)
            screen.blit(vtext, (sx + w // 2 - vtext.get_width() // 2, FLOOR_Y - h - 40))

    def _draw_panel(self, screen, fonts):
        px = SIM_W
        # panel shadow
        shadow = pygame.Surface((PANEL_W, H), pygame.SRCALPHA)
        shadow.fill((0, 0, 0, 18))
        screen.blit(shadow, (px + 6, 6))
        # panel background
        pygame.draw.rect(screen, PANEL_BG, (px, 0, PANEL_W, H))
        pygame.draw.line(screen, DIVIDER, (px, 0), (px, H), 1)

        # compute layout dynamically
        self._compute_panel_layout(fonts)

        cy = 18
        pad = 16

        # title bar with E on right
        title_h = 36
        title_rect = pygame.Rect(px + pad, cy, PANEL_W - pad * 2, title_h)
        pygame.draw.rect(screen, (230, 235, 240), title_rect, border_radius=6)
        title_s = fonts["title"].render("动量守恒实验", True, TEXT_HEAD)
        screen.blit(title_s, (title_rect.x + 10, title_rect.y + (title_h - title_s.get_height()) // 2))
        # E small box on right
        e_box = self.input_rects.get("e")
        if e_box:
            # draw small label and box
            e_label = fonts["small"].render("e", True, TEXT_SEC)
            screen.blit(e_label, (e_box.x - 22, title_rect.y + (title_h - e_label.get_height()) // 2))
            pygame.draw.rect(screen, BUTTON_COL, e_box, border_radius=6)
            pygame.draw.rect(screen, DIVIDER, e_box, width=1, border_radius=6)
            # render value (buffer if active)
            ev = self.input_buffers.get("e", self._current_value_str("e")) if self.active_input != "e" else self.input_buffers.get("e", "")
            ev_s = fonts["small"].render(ev, True, TEXT_PRI)
            screen.blit(ev_s, (e_box.x + 8, e_box.y + (e_box.h - ev_s.get_height()) // 2))

        cy += title_h + 12

        # sections: A and B
        # A
        cy = _panel_heading(screen, fonts, px + pad, cy, "物块 A", COL_A)
        cy += 6
        cy = self._draw_input_row(screen, fonts, px + pad, cy, PANEL_W - pad * 2, "质量", "ma", val_col=COL_A)
        cy = self._draw_input_row(screen, fonts, px + pad, cy, PANEL_W - pad * 2, "速度", "va", val_col=COL_A)
        # separator
        pygame.draw.line(screen, DIVIDER, (px + pad, cy + 6), (px + PANEL_W - pad, cy + 6))
        cy += 14

        # B
        cy = _panel_heading(screen, fonts, px + pad, cy, "物块 B", COL_B)
        cy += 6
        cy = self._draw_input_row(screen, fonts, px + pad, cy, PANEL_W - pad * 2, "质量", "mb", val_col=COL_B)
        cy = self._draw_input_row(screen, fonts, px + pad, cy, PANEL_W - pad * 2, "速度", "vb", val_col=COL_B)
        cy += 6

        # momentum / energy
        cy = _panel_heading(screen, fonts, px + pad, cy, "初始 / 当前", TEXT_HEAD)
        cy += 8
        # initial
        init_p = self.initial_p
        init_ek = self.initial_ek
        cy = _kv_row(screen, fonts, px + pad, cy, PANEL_W - pad * 2, "初始 总动量 p", f"{init_p:.4f} kg·m/s")
        cy = _kv_row(screen, fonts, px + pad, cy, PANEL_W - pad * 2, "初始 总动能 Ek", f"{init_ek:.4f} J")
        cy += 6
        # current
        self.collision.block_1 = self.block_a
        self.collision.block_2 = self.block_b
        p_now = self.collision.total_momentum
        ek_now = self.collision.total_k_energy
        cy = _kv_row(screen, fonts, px + pad, cy, PANEL_W - pad * 2, "当前 总动量 p", f"{p_now:.4f} kg·m/s")
        cy = _kv_row(screen, fonts, px + pad, cy, PANEL_W - pad * 2, "当前 总动能 Ek", f"{ek_now:.4f} J")

        # hint
        tip = fonts["hint"].render("回车确认，Esc 取消。点击数值区域开始编辑。", True, TEXT_SEC)
        screen.blit(tip, (px + pad, H - 56))

    def _compute_panel_layout(self, fonts):
        px = SIM_W
        pad = 16
        cy = 18
        # title height
        title_h = 36
        cy += title_h + 12
        box_h = fonts["small"].get_height() + 8
        box_w = 86
        right_x = px + (PANEL_W - pad) - box_w
        # e box in title area
        self.input_rects["e"] = pygame.Rect(right_x, 18 + (36 - box_h) // 2, box_w, box_h)
        # blocks
        cy += fonts["h2"].get_height() + 6
        self.input_rects["ma"] = pygame.Rect(right_x, cy + 0, box_w, box_h)
        self.input_rects["va"] = pygame.Rect(right_x, cy + box_h + 6, box_w, box_h)
        cy = cy + box_h * 2 + 6 + 14 + fonts["h2"].get_height()
        self.input_rects["mb"] = pygame.Rect(right_x, cy + 0, box_w, box_h)
        self.input_rects["vb"] = pygame.Rect(right_x, cy + box_h + 6, box_w, box_h)

    def _unit_for_field(self, field_name: str) -> str:
        return {"ma": "kg", "va": "m/s", "mb": "kg", "vb": "m/s", "e": ""}.get(field_name, "")

    def _draw_input_row(self, screen, fonts, x, y, w, key, field_name, val_col=None):
        ks = fonts["small"].render(key, True, TEXT_SEC)
        screen.blit(ks, (x, y))
        box_w = 86
        box_h = ks.get_height() + 8
        box_x = x + w - box_w
        box_y = y
        rect = pygame.Rect(box_x, box_y, box_w, box_h)
        self.input_rects[field_name] = rect
        pygame.draw.rect(screen, BUTTON_COL, rect, border_radius=6)
        pygame.draw.rect(screen, DIVIDER, rect, width=1, border_radius=6)
        if self.active_input == field_name:
            pygame.draw.rect(screen, (60, 120, 220), rect, width=2, border_radius=6)
        if self.active_input == field_name:
            display_text = self.input_buffers.get(field_name, "")
        else:
            display_text = self._current_value_str(field_name)
        # render right-aligned inside box
        txt_surf = fonts["small"].render(display_text, True, val_col or TEXT_PRI)
        screen.blit(txt_surf, (rect.right - 8 - txt_surf.get_width(), box_y + (box_h - txt_surf.get_height()) // 2))
        # unit to the right outside box
        unit = self._unit_for_field(field_name)
        if unit:
            unit_s = fonts["small"].render(unit, True, TEXT_SEC)
            screen.blit(unit_s, (rect.right + 8, box_y + (box_h - unit_s.get_height()) // 2))
        return y + box_h + 6

    def _current_value_str(self, field_name: str) -> str:
        if field_name == "ma":
            return f"{self.block_a.m:.3f}"
        if field_name == "va":
            return f"{self.block_a.v:.3f}"
        if field_name == "mb":
            return f"{self.block_b.m:.3f}"
        if field_name == "vb":
            return f"{self.block_b.v:.3f}"
        if field_name == "e":
            return f"{self.collision.e:.2f}"
        return ""

    def _commit_active_input(self):
        if not self.active_input:
            return
        buf = self.input_buffers.get(self.active_input, "")
        try:
            val = float(buf)
        except Exception:
            # restore and exit
            self.input_buffers[self.active_input] = self._current_value_str(self.active_input)
            self.active_input = None
            self.paused = self._paused_before_edit
            return
        if self.active_input == "ma":
            self.block_a.m = max(0.0001, val)
        elif self.active_input == "va":
            self.block_a.v = val
        elif self.active_input == "mb":
            self.block_b.m = max(0.0001, val)
        elif self.active_input == "vb":
            self.block_b.v = val
        elif self.active_input == "e":
            val = max(0.0, min(1.0, val))
            self.collision = D(self.block_a, self.block_b, e=val)
        # update buffer to current formatted
        self.input_buffers[self.active_input] = self._current_value_str(self.active_input)
        self.active_input = None
        self.paused = self._paused_before_edit

    def _draw_hintbar(self, screen, fonts):
        by = H - HINT_H
        pygame.draw.rect(screen, BUTTON_COL, (0, by, W, HINT_H))
        pygame.draw.line(screen, DIVIDER, (0, by), (W, by))
        keys = [("SPACE", "暂停/继续"), ("R", "重置"), ("E", "切换碰撞类型"), ("Q", "退出")]
        cx = 20
        for k, desc in keys:
            kb = fonts["small"].render(k, True, TEXT_HEAD)
            kw = kb.get_width() + 14
            kh = kb.get_height() + 6
            ky = by + (HINT_H - kh) // 2
            pygame.draw.rect(screen, BUTTON_COL, (cx, ky, kw, kh), border_radius=4)
            pygame.draw.rect(screen, DIVIDER, (cx, ky, kw, kh), width=1, border_radius=4)
            screen.blit(kb, (cx + 7, ky + 3))
            cx += kw + 6
            db = fonts["small"].render(desc, True, TEXT_SEC)
            screen.blit(db, (cx, by + (HINT_H - db.get_height()) // 2))
            cx += db.get_width() + 28

    def _draw_paused_overlay(self, screen, fonts):
        overlay = pygame.Surface((SIM_W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 60))
        screen.blit(overlay, (0, 0))
        s = fonts["title"].render("已暂停", True, AMBER)
        screen.blit(s, (SIM_W // 2 - s.get_width() // 2, FLOOR_Y // 2 - s.get_height() // 2))


def _draw_velocity_arrow(screen, blk, sx, sy, w, col):
    if abs(blk.v) < 0.05:
        return
    cx = sx + w // 2
    ay = sy - 14
    sign = 1 if blk.v > 0 else -1
    length = min(90, max(16, int(abs(blk.v) * 18)))
    tip = cx + sign * length
    pygame.draw.line(screen, col, (cx, ay), (tip, ay), 2)
    pygame.draw.polygon(screen, col, [(tip, ay), (tip - sign * 10, ay - 5), (tip - sign * 10, ay + 5)])


def _panel_heading(screen, fonts, x, y, text, color):
    s = fonts["h2"].render(text, True, color)
    screen.blit(s, (x, y))
    return y + s.get_height() + 2


def _kv_row(screen, fonts, x, y, w, key, val, val_col=None):
    ks = fonts["small"].render(key, True, TEXT_SEC)
    vs = fonts["small"].render(val, True, val_col or TEXT_PRI)
    screen.blit(ks, (x, y))
    screen.blit(vs, (x + w - vs.get_width(), y))
    return y + ks.get_height() + 6
