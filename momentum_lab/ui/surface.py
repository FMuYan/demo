import pygame as pg
from momentum_lab.model import Block
from momentum_lab.ui.const import *


def C(hex_str: str) -> pg.Color:
    return pg.Color(hex_str)


def sim_x(x: float) -> int:
    return int(ORIGIN_X + x * PIXELS_PER_M)


def u_block_w(blk: Block) -> int:
    return max(40, min(110, int(blk.m * 22 + 28)))


def u_block_h(blk: Block) -> int:
    return max(40, min(110, int(blk.m * 22 + 28)))


def u_load_fonts():
    def font(size, is_bold=False):
        return pg.font.SysFont(Y_FONTS, size, bold=is_bold)

    return {
        "h1": font(21, is_bold=True),
        "h2": font(19, is_bold=True),
        "body": font(16),
        "small": font(14),
        "hint": font(14),
        "mono": font(15),
    }


def u_panel_heading(screen, fonts, x, y, text, color):
    s = fonts["h2"].render(text, True, color)
    screen.blit(s, (x, y))
    return y + s.get_height() + 2


def u_draw_velocity_arrow(screen, blk, sx, sy, w, col_color):
    if abs(blk.v) < 0.05:
        return
    cx = sx + w // 2
    ay = sy - 14
    sign = 1 if blk.v > 0 else -1
    length = min(90, max(16, int(abs(blk.v) * 18)))
    tip = cx + sign * length
    pg.draw.line(screen, col_color, (cx, ay), (tip, ay), 2)
    pg.draw.polygon(
        screen,
        col_color,
        [(tip, ay), (tip - sign * 10, ay - 5), (tip - sign * 10, ay + 5)],
    )
