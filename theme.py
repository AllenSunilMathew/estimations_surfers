"""Shared visual theme — the light-blue "sky card" look from the preloader.

Everything that more than one screen needs lives here so the preloader, main
menu, HUD and overlays all look like one game:

    font(...)            rounded / heavy font (or your own font.ttf)
    build_card_bg(...)   off-white page + shadowed blue gradient card
    text(...)            white text with a soft blue drop shadow
    panel(...)           rounded blue panel, white border, soft shadow
    header_band(...)     darker title strip across the top of a panel
    coin(...), star(...) the gold coin from the preloader
    runner_icon(...)     little stick-runner icon
    Bubbles              soft drifting white bubbles for a lively background
"""

import math
import os
import random

import numpy as np
import pygame

from constants import (
    PAGE_BG, SKY_A, SKY_B, PANEL_FILL, PANEL_DARK, EDGE_SOFT,
    INK, INK_SHADOW, WHITE,
)


# ────────────────────────────────────────────────────────────────────
# fonts
# ────────────────────────────────────────────────────────────────────
_FONT_PATH = "unset"


def _resolve_font_path():
    """Custom font: drop a font.ttf / font.otf next to the code (or in assets/) to override.
    Otherwise use the first rounded / heavy system font we can find."""
    global _FONT_PATH
    if _FONT_PATH != "unset":
        return _FONT_PATH
    _FONT_PATH = None
    base = os.path.dirname(os.path.abspath(__file__))
    for d in (base, os.path.join(base, "assets")):
        for n in ("font.ttf", "font.otf"):
            p = os.path.join(d, n)
            if os.path.isfile(p):
                _FONT_PATH = p
                return p
    for n in ("arialroundedmtbold", "lilitaone", "fredoka", "baloo2", "nunitoblack", "nunito",
              "poppinsbold", "segoeuiblack", "arialblack", "verdana"):
        p = pygame.font.match_font(n)
        if p:
            _FONT_PATH = p
            break
    return _FONT_PATH


def font(name, size, bold=False, italic=False):
    """Project-wide font factory (the `name` argument is kept for compatibility)."""
    path = _resolve_font_path()
    try:
        if path:
            f = pygame.font.Font(path, size)
            f.set_italic(italic)
            return f
    except Exception:
        pass
    try:
        return pygame.font.SysFont("Arial", size, bold=True, italic=italic)
    except Exception:
        return pygame.font.SysFont(None, size, bold=True, italic=italic)


# ────────────────────────────────────────────────────────────────────
# background card
# ────────────────────────────────────────────────────────────────────
def build_card_bg(w, h, margin=22, radius=34):
    """Off-white page with a soft shadowed card holding a radial blue gradient."""
    bg = pygame.Surface((w, h))
    bg.fill(PAGE_BG)
    rect = pygame.Rect(margin, margin, w - 2 * margin, h - 2 * margin)
    sh = pygame.Surface((w, h), pygame.SRCALPHA)
    for i in range(16, 0, -1):
        pygame.draw.rect(sh, (40, 70, 100, 5), rect.inflate(i * 2, i * 2).move(0, 12),
                         border_radius=radius + i)
    bg.blit(sh, (0, 0))

    cw, ch = rect.size
    ys, xs = np.mgrid[0:ch, 0:cw]
    d = np.clip(np.sqrt(((xs - cw / 2) / (cw * 0.62)) ** 2 + ((ys - ch / 2) / (ch * 0.75)) ** 2), 0, 1)[..., None]
    c0, c1 = np.array(SKY_A), np.array(SKY_B)
    arr = (c0 + (c1 - c0) * d).astype(np.uint8)
    card = pygame.Surface((cw, ch), pygame.SRCALPHA)
    card.blit(pygame.surfarray.make_surface(arr.transpose(1, 0, 2)), (0, 0))
    mask = pygame.Surface((cw, ch), pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, cw, ch), border_radius=radius)
    card.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    bg.blit(card, rect.topleft)
    return bg


# ────────────────────────────────────────────────────────────────────
# text / panels
# ────────────────────────────────────────────────────────────────────
def text(surf, fnt, value, color, pos, shadow=INK_SHADOW, off=2, anchor="topleft"):
    """Draw text (optionally with a drop shadow). `anchor` is any pygame Rect attribute
    ('topleft', 'center', 'midtop', 'topright', ...). Returns the text rect."""
    img = fnt.render(str(value), True, color)
    r = img.get_rect(**{anchor: pos})
    if shadow:
        surf.blit(fnt.render(str(value), True, shadow), (r.x + off, r.y + off))
    surf.blit(img, r.topleft)
    return r


def panel(surf, rect, radius=22, fill=PANEL_FILL):
    """Rounded blue panel with a soft drop shadow and a white border."""
    pad = 40
    sh = pygame.Surface((rect.w + 2 * pad, rect.h + 2 * pad), pygame.SRCALPHA)
    base = pygame.Rect(pad, pad + 8, rect.w, rect.h)
    for i in range(14, 0, -1):
        pygame.draw.rect(sh, (20, 80, 120, 5), base.inflate(i * 2, i * 2), border_radius=radius + i)
    surf.blit(sh, (rect.x - pad, rect.y - pad))
    pygame.draw.rect(surf, fill, rect, border_radius=radius)
    pygame.draw.rect(surf, WHITE, rect, 3, border_radius=radius)
    pygame.draw.rect(surf, EDGE_SOFT, rect.inflate(-9, -9), 1, border_radius=max(2, radius - 4))


def header_band(surf, rect, height=44, radius=18):
    band = pygame.Rect(rect.x + 4, rect.y + 4, rect.w - 8, height)
    pygame.draw.rect(surf, PANEL_DARK, band, border_top_left_radius=radius,
                     border_top_right_radius=radius)
    return band


def soft_shadow_ellipse(surf, rect, alpha=60):
    sh = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.ellipse(sh, (30, 100, 140, alpha), sh.get_rect())
    surf.blit(sh, rect.topleft)


# ────────────────────────────────────────────────────────────────────
# artwork
# ────────────────────────────────────────────────────────────────────
def star(cx, cy, ro, ri, sx=1.0):
    pts = []
    for i in range(10):
        r = ro if i % 2 == 0 else ri
        a = math.radians(-90 + i * 36)
        pts.append((cx + math.cos(a) * r * sx, cy + math.sin(a) * r))
    return pts


def coin(surf, cx, cy, r, squash=1.0):
    rx = max(2, int(r * squash))
    pygame.draw.ellipse(surf, (185, 100, 12), (cx - rx - 3, cy - r - 3, 2 * rx + 6, 2 * r + 6))
    pygame.draw.ellipse(surf, (255, 198, 40), (cx - rx, cy - r, 2 * rx, 2 * r))
    if r > 7:
        pygame.draw.ellipse(surf, (255, 230, 120), (cx - rx + 4, cy - r + 4, 2 * rx - 8, 2 * r - 8), 2)
        pygame.draw.polygon(surf, (255, 140, 20), star(cx, cy, r * 0.55, r * 0.24, squash))


def runner_icon(surf, cx, cy, s=1.0, color=WHITE, outline=INK):
    """Little running stick-figure: coloured limbs with a teal outline."""
    def P(x, y):
        return (cx + x * s, cy + y * s)
    limbs = [
        [P(3, -16), P(17, -8), P(27, -19)],
        [P(3, -16), P(-10, -10), P(-19, -1)],
        [P(-4, 6), P(10, 14), P(8, 31)],
        [P(-4, 6), P(-16, 19), P(-31, 15)],
        [P(5, -22), P(-4, 6)],
    ]
    w = max(2, int(6 * s))
    for pts in limbs:
        pygame.draw.lines(surf, outline, False, pts, w + 4)
    hx, hy = int(cx + 8 * s), int(cy - 31 * s)
    pygame.draw.circle(surf, outline, (hx, hy), int(10 * s) + 2)
    for pts in limbs:
        pygame.draw.lines(surf, color, False, pts, w)
    pygame.draw.circle(surf, color, (hx, hy), int(8 * s))


# ────────────────────────────────────────────────────────────────────
# drifting bubbles
# ────────────────────────────────────────────────────────────────────
class Bubbles:
    """Soft translucent white bubbles that slowly float up the card."""

    def __init__(self, w, h, n=36, seed=5, area=None):
        rng = random.Random(seed)
        self.area = area or pygame.Rect(0, 0, w, h)
        a = self.area
        self._cache = {}
        self.items = [[rng.uniform(a.left, a.right), rng.uniform(a.top, a.bottom),
                       rng.uniform(10, 30), rng.choice((4, 6, 9, 13)),
                       rng.choice((50, 80, 110)), rng.uniform(0, 6.28)] for _ in range(n)]

    def _sprite(self, r, a):
        key = (r, a)
        if key not in self._cache:
            s = pygame.Surface((2 * r + 4, 2 * r + 4), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 255, 255, a // 3), (r + 2, r + 2), r)
            pygame.draw.circle(s, (255, 255, 255, a), (r + 2, r + 2), r, 2)
            self._cache[key] = s
        return self._cache[key]

    def update(self, dt):
        a = self.area
        for it in self.items:
            it[1] -= it[2] * dt
            if it[1] < a.top - 20:
                it[1] = a.bottom + 20

    def draw(self, surf, t):
        for x, y, _, r, alpha, ph in self.items:
            spr = self._sprite(r, alpha)
            surf.blit(spr, (int(x + math.sin(t * 0.8 + ph) * 8) - r - 2, int(y) - r - 2))
