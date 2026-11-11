"""Preloader splash screen and main menu — light-blue "sky card" theme.

Public API (unchanged, used by main.py):
    Preloader(screen): update(dt) -> bool, draw(), .done
    MainMenu(screen):  update(dt), draw(cam_frame=None)

Shared look (fonts, card, panels, coins) lives in theme.py so the in-game HUD
matches these two screens.
"""

import math

import cv2
import numpy as np
import pygame

from constants import *
from theme import (font as _font, title_font as _title_font, build_card_bg, text as _text, panel as _panel,
                   header_band as _header_band, coin as _coin, Bubbles,
                   soft_shadow_ellipse)


# ── chunky "block" letters (drawn with strokes, so they look the same on every PC) ──
_GLYPH_W = {"L": .60, "O": .78, "A": .90, "D": .74, "I": 0.0, "N": .76, "G": .78}


def _arc(cx, cy, rx, ry, a0, a1, n=14):
    return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + ry * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def _glyph_paths(ch):
    w = _GLYPH_W[ch]
    if ch == "L":
        return [[(0, 0), (0, 1), (w, 1)]]
    if ch == "I":
        return [[(0, 0), (0, 1)]]
    if ch == "N":
        return [[(0, 1), (0, 0), (w, 1), (w, 0)]]
    if ch == "A":
        return [[(0, 1), (w / 2, 0), (w, 1)], [(w * .2, .66), (w * .8, .66)]]
    if ch == "O":
        r = .3
        pts = [(r, 0)] + _arc(w - r, r, r, r, -90, 0) + _arc(w - r, 1 - r, r, r, 0, 90) \
            + _arc(r, 1 - r, r, r, 90, 180) + _arc(r, r, r, r, 180, 270)
        return [pts + [pts[0]]]
    if ch == "D":
        r = .34
        return [[(0, 0), (w - r, 0)] + _arc(w - r, r, r, r, -90, 0) + _arc(w - r, 1 - r, r, r, 0, 90) + [(0, 1), (0, 0)]]
    if ch == "G":
        return [_arc(w / 2, .5, w / 2, .5, -38, -360, 30) + [(w * .5, .5)]]
    raise KeyError(ch)


def _block_text(text, h, color, thick=0.25, gap=0.13, ss=3):
    """Return an anti-aliased SRCALPHA surface with `text` in chunky rounded block letters."""
    t = thick * h
    adv = [_GLYPH_W[c] * h + t for c in text]
    width = int(sum(adv) + gap * h * (len(text) - 1) + t)
    height = int(h + t)
    big = pygame.Surface((width * ss, height * ss), pygame.SRCALPHA)
    x = t / 2
    for c, a in zip(text, adv):
        for path in _glyph_paths(c):
            pts = [((x + px * h) * ss, (t / 2 + py * h) * ss) for px, py in path]
            for p, q in zip(pts, pts[1:]):
                pygame.draw.line(big, color, p, q, int(t * ss))
            for p in pts:
                pygame.draw.circle(big, color, (int(p[0]), int(p[1])), int(t * ss / 2))
        x += a + gap * h
    return pygame.transform.smoothscale(big, (width, height))



# ────────────────────────────────────────────────────────────────────
class Preloader:
    """Light-blue 'LOADING' splash: big white block letters, a little runner sliding
    across them as the percentage climbs (inspired by the reference design)."""

    TIPS = [
        "Stand in front of your camera with your full body visible!",
        "Good lighting helps the AI track your pose.",
        "Exaggerate your gestures for better detection.",
        "Stay still for a moment - the game calibrates your height.",
    ]

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.W, self.H = WINDOW_WIDTH, WINDOW_HEIGHT
        self.progress = 0.0
        self.done = False
        self._alpha = 255
        self._fading = False
        self.t = 0.0

        self.f_brand = _title_font(26)
        self.f_sub = _font("Arial", 16)
        self.f_pct = _title_font(22)
        self.f_tip = _font("Arial", 16)

        self._bg = self._build_bg()
        self._text = _block_text("LOADING", 92, (255, 255, 255))
        self._text_shadow = _block_text("LOADING", 92, (30, 110, 150, 70))
        self.tx = (self.W - self._text.get_width()) // 2
        self.ty = self.H // 2 - 95

    # ── static background: soft shadowed card with a radial blue gradient ──
    def _build_bg(self):
        return build_card_bg(self.W, self.H, margin=22, radius=34)

    def update(self, dt) -> bool:
        self.t += dt
        if self._fading:
            self._alpha -= dt * 500
            if self._alpha <= 0:
                self._alpha = 0
                self.done = True
            return self.done
        self.progress += dt * 0.40
        if self.progress >= 1.0:
            self.progress = 1.0
            self._fading = True
        return False

    # ── the little runner, lying down and swimming over the letters ──
    def _kid(self, kx, ky, s=1.25):
        scr = self.screen
        P = lambda dx, dy: (int(kx + dx * s), int(ky + dy * s))
        R = lambda r: max(1, int(r * s))
        hoodie, hoodie_d = (214, 50, 52), (160, 32, 40)
        jeans, shoe, skin, hair = (52, 98, 172), (246, 246, 246), (255, 198, 152), (92, 54, 34)
        sw = math.sin(self.t * 9)

        for k, sign in enumerate((1, -1)):                      # legs (kicking)
            hip = P(-22, 4 + k * 6)
            foot = P(-64, 4 + k * 6 + sign * sw * 7)
            pygame.draw.line(scr, jeans, hip, foot, R(14))
            pygame.draw.circle(scr, jeans, foot, R(7))
            pygame.draw.ellipse(scr, shoe, (foot[0] - R(12), foot[1] - R(6), R(15), R(12)))
        for k, sign in enumerate((-1, 1)):                      # arms stretched forward
            sh = P(16, -6 + k * 8)
            hand = P(54, -12 + k * 12 + sign * sw * 5)
            pygame.draw.line(scr, hoodie, sh, hand, R(10))
            pygame.draw.circle(scr, skin, hand, R(6))
        body = pygame.Rect(0, 0, R(58), R(32))
        body.center = P(-2, 0)
        pygame.draw.rect(scr, hoodie_d, body.move(0, 3), border_radius=R(14))
        pygame.draw.rect(scr, hoodie, body, border_radius=R(14))
        pygame.draw.rect(scr, hoodie_d, (body.x, body.y + R(4), R(8), body.h - R(8)), border_radius=R(4))
        pygame.draw.circle(scr, hoodie_d, P(22, -9), R(10))      # hood
        pygame.draw.circle(scr, skin, P(40, -2), R(14))          # head
        pygame.draw.circle(scr, hair, P(35, -7), R(12))
        pygame.draw.circle(scr, skin, P(41, -1), R(11))
        pygame.draw.circle(scr, hair, P(36, -9), R(9))
        pygame.draw.circle(scr, (40, 40, 50), P(46, -2), R(2))

    def draw(self):
        s = self.screen
        s.blit(self._bg, (0, 0))
        cx = self.W // 2
        p = self.progress

        # brand line
        for text, font, y, col in (("TREASURE RUNNER", self.f_brand, 68, WHITE),
                                   ("AI Pose Controlled Game", self.f_sub, 104, (225, 245, 252))):
            sh = font.render(text, True, (40, 120, 160))
            im = font.render(text, True, col)
            s.blit(sh, (cx - im.get_width() // 2 + 2, y + 2))
            s.blit(im, (cx - im.get_width() // 2, y))

        tx, ty = self.tx, self.ty
        tw, th = self._text.get_size()

        # soft floor shadow under the word
        floor_y = ty + th + 34
        for i in range(7):
            a = 14 - i * 2
            sh = pygame.Surface((tw + 40 - i * 18, 26 - i * 3), pygame.SRCALPHA)
            pygame.draw.ellipse(sh, (30, 100, 140, max(a, 3)), sh.get_rect())
            s.blit(sh, (cx - sh.get_width() // 2, floor_y - sh.get_height() // 2))

        s.blit(self._text_shadow, (tx, ty + 10))
        s.blit(self._text, (tx, ty))

        # runner slides along the word with the progress
        kx = tx + 40 + p * (tw - 80)
        ky = ty + th * 0.80 + math.sin(self.t * 5) * 3
        sh = pygame.Surface((170, 26), pygame.SRCALPHA)
        pygame.draw.ellipse(sh, (30, 100, 140, 60), sh.get_rect())
        s.blit(sh, (kx - 85, floor_y - 12))
        self._kid(kx, ky)

        # percent + thin bar
        pct = self.f_pct.render(f"{int(p * 100)} %", True, WHITE)
        py_ = floor_y + 52
        s.blit(pct, (cx - pct.get_width() // 2, py_))
        bar = pygame.Rect(cx - 130, py_ + 38, 260, 6)
        trk = pygame.Surface(bar.size, pygame.SRCALPHA)
        pygame.draw.rect(trk, (255, 255, 255, 80), (0, 0, *bar.size), border_radius=3)
        s.blit(trk, bar.topleft)
        if p > 0.01:
            pygame.draw.rect(s, WHITE, (bar.x, bar.y, int(bar.w * p), bar.h), border_radius=3)

        tip = self.TIPS[min(len(self.TIPS) - 1, int(p * len(self.TIPS)))]
        tp = self.f_tip.render(tip, True, (232, 246, 252))
        s.blit(tp, (cx - tp.get_width() // 2, self.H - 90))

        if self._fading:
            fo = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            fo.fill((0, 0, 0, 255 - int(max(self._alpha, 0))))
            s.blit(fo, (0, 0))



# ────────────────────────────────────────────────────────────────────
class MainMenu:
    """Title screen in the preloader's sky-card style: camera preview,
    controls guide and a start prompt."""

    ROWS = [("up", "Arms Up", "Jump", GREEN_DARK),
            ("crouch", "Sit / Crouch Down", "Slide", ORANGE_DARK),
            ("left", "/  Shape  (L hand down, R up)", "Move Left", BLUE_DARK),
            ("right", "\\  Shape  (L hand up, R down)", "Move Right", BLUE_DARK)]

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.W, self.H = WINDOW_WIDTH, WINDOW_HEIGHT
        self.t = 0.0

        self.f_title = _title_font(56)
        self.f_sub = _font("Arial", 22)
        self.f_panel = _title_font(22)
        self.f_gesture = _font("Arial", 18)
        self.f_action = _title_font(18)
        self.f_prompt = _title_font(22)
        self.f_small = _font("Arial", 15)
        self.f_tiny = _title_font(13)

        self.cam_rect = pygame.Rect(80, 205, 500, 370)
        self.ctl_rect = pygame.Rect(620, 205, 580, 370)
        self.cam_inner = pygame.Rect(self.cam_rect.x + 12, self.cam_rect.y + 56,
                                     self.cam_rect.w - 24, self.cam_rect.h - 68)
        # Clickable camera switch button docked at bottom-right inside camera preview
        self.cam_switch_rect = pygame.Rect(self.cam_inner.right - 150, self.cam_inner.bottom - 46, 140, 38)
        card = pygame.Rect(22, 22, self.W - 44, self.H - 44)
        self._bubbles = Bubbles(self.W, self.H, 34, seed=9, area=card)
        self._bg = build_card_bg(self.W, self.H, margin=22, radius=34)
        self._ui = self._build_ui()
        self._mask = pygame.Surface(self.cam_inner.size, pygame.SRCALPHA)
        self._mask.fill((*PANEL_FILL, 255))
        pygame.draw.rect(self._mask, (0, 0, 0, 0), (0, 0, *self.cam_inner.size), border_radius=12)

    # ── static layer (title + panels) ───────────────────────────────
    def _build_ui(self):
        surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        cx = self.W // 2

        # title: white block text, deep-blue extrusion, soft blue shadow (like the preloader)
        title = "TREASURE  RUNNER"
        for dx, dy, col in ((5, 6, (30, 100, 140)), (3, 4, (40, 120, 160)), (0, 0, WHITE)):
            im = self.f_title.render(title, True, col)
            surf.blit(im, (cx - im.get_width() // 2 + dx, 44 + dy))
        _text(surf, self.f_sub, "AI Pose Controlled Game", SOFT_WHITE, (cx, 132), anchor="midtop")

        _panel(surf, self.cam_rect)
        _panel(surf, self.ctl_rect)
        for rect, title in ((self.cam_rect, "Camera Preview"), (self.ctl_rect, "Controls")):
            _header_band(surf, rect, 44)
            _text(surf, self.f_panel, title, WHITE, (rect.x + 24, rect.y + 12))

        ry = self.ctl_rect.y + 62
        for pose, gesture, action, col in self.ROWS:
            row = pygame.Rect(self.ctl_rect.x + 18, ry, self.ctl_rect.w - 36, 66)
            pygame.draw.rect(surf, ROW_FILL, row, border_radius=14)
            pygame.draw.rect(surf, EDGE_SOFT, row, 2, border_radius=14)
            self._pose_icon(surf, row.x + 40, row.centery, pose, WHITE)
            _text(surf, self.f_gesture, gesture, WHITE, (row.x + 84, row.centery), anchor="midleft")
            a = self.f_action.render(action, True, col)
            pill = pygame.Rect(row.right - a.get_width() - 36, row.centery - 17, a.get_width() + 24, 34)
            pygame.draw.rect(surf, WHITE, pill, border_radius=17)
            surf.blit(a, (pill.x + 12, pill.centery - a.get_height() // 2))
            ry += 76
        return surf

    @staticmethod
    def _pose_icon(surf, cx, cy, pose, col):
        w = 4
        if pose == "crouch":
            pygame.draw.circle(surf, col, (cx, cy - 6), 6)
            pygame.draw.line(surf, col, (cx, cy), (cx, cy + 10), w)
            pygame.draw.lines(surf, col, False, [(cx - 13, cy + 8), (cx, cy + 2), (cx + 13, cy + 8)], w)
            pygame.draw.lines(surf, col, False, [(cx - 10, cy + 24), (cx - 9, cy + 14), (cx, cy + 10),
                                                 (cx + 9, cy + 14), (cx + 10, cy + 24)], w)
            return
        pygame.draw.circle(surf, col, (cx, cy - 20), 6)
        pygame.draw.line(surf, col, (cx, cy - 13), (cx, cy + 8), w)
        if pose == "up":
            pygame.draw.lines(surf, col, False, [(cx - 14, cy - 28), (cx - 3, cy - 10),
                                                 (cx + 3, cy - 10), (cx + 14, cy - 28)], w)
        elif pose == "left":
            pygame.draw.line(surf, col, (cx - 16, cy + 4), (cx + 16, cy - 20), w)
        else:
            pygame.draw.line(surf, col, (cx - 16, cy - 20), (cx + 16, cy + 4), w)
        pygame.draw.lines(surf, col, False, [(cx - 8, cy + 22), (cx, cy + 8), (cx + 8, cy + 22)], w)

    # ── update / draw ───────────────────────────────────────────────
    def update(self, dt):
        self.t += dt
        self._bubbles.update(dt)

    def _frame_surface(self, frame, size):
        tw, th = size
        h, w = frame.shape[:2]
        s = max(tw / w, th / h)
        nw, nh = int(w * s + 0.5), int(h * s + 0.5)
        rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (nw, nh))
        x0, y0 = (nw - tw) // 2, (nh - th) // 2
        rgb = np.ascontiguousarray(rgb[y0:y0 + th, x0:x0 + tw])
        return pygame.image.frombuffer(rgb.tobytes(), (tw, th), "RGB")

    def draw(self, cam_frame=None, is_fullscreen=False, mouse_pos=(0, 0), cam_name="WEBCAM"):
        s = self.screen
        s.blit(self._bg, (0, 0))
        self._bubbles.draw(s, self.t)
        s.blit(self._ui, (0, 0))

        # Symmetrical spinning coins beside the title
        sq = 0.2 + 0.8 * abs(math.cos(self.t * 3))
        for x in (170, self.W - 170):
            _coin(s, x, 92, 30, sq)

        # camera preview
        ci = self.cam_inner
        live = cam_frame is not None
        pygame.draw.rect(s, (16, 62, 94), ci, border_radius=12)
        if live:
            s.blit(self._frame_surface(cam_frame, ci.size), ci.topleft)
        else:
            _text(s, self.f_gesture, "NO CAMERA DETECTED", (255, 190, 190), ci.center, anchor="center")
        s.blit(self._mask, ci.topleft)
        pygame.draw.rect(s, WHITE, ci, 2, border_radius=12)

        # Live status dot in camera panel header
        dot = (70, 235, 110) if live else (255, 110, 110)
        pygame.draw.circle(s, dot, (self.cam_rect.right - 100, self.cam_rect.y + 24), 6)
        _text(s, self.f_small, "LIVE" if live else "OFFLINE", WHITE,
              (self.cam_rect.right - 88, self.cam_rect.y + 16))

        # ── Interactive Camera Switch Button inside preview (Bottom-Right) ──
        cs = self.cam_switch_rect
        is_hover_cs = cs.collidepoint(mouse_pos)
        cps = pygame.Surface(cs.size, pygame.SRCALPHA)
        cs_bg = (52, 150, 196, 235) if is_hover_cs else (*PANEL_DARK, 225)
        pygame.draw.rect(cps, cs_bg, (0, 0, *cs.size), border_radius=14)
        border_c = GOLD_BRIGHT if is_hover_cs else WHITE
        pygame.draw.rect(cps, border_c, (0, 0, *cs.size), 2, border_radius=14)
        s.blit(cps, cs.topleft)
        cam_lbl = self.f_tiny.render(f"📷 {cam_name} [C]", True, GOLD_BRIGHT if is_hover_cs else WHITE)
        s.blit(cam_lbl, (cs.centerx - cam_lbl.get_width() // 2, cs.centery - cam_lbl.get_height() // 2))

        # start prompt (pulsing white pill)
        pulse = abs(math.sin(self.t * 2.2))
        label = self.f_prompt.render("Raise Both Hands  or  Press SPACE  to Start!", True, INK)
        pw, ph = label.get_width() + 80, 56
        pill = pygame.Rect(self.W // 2 - pw // 2, 598, pw, ph)
        glow = pygame.Surface((pw + 40, ph + 40), pygame.SRCALPHA)
        pygame.draw.rect(glow, (255, 255, 255, int(30 + 80 * pulse)), (0, 0, pw + 40, ph + 40), border_radius=34)
        s.blit(glow, (pill.x - 20, pill.y - 20))
        soft_shadow_ellipse(s, pygame.Rect(pill.x + 10, pill.bottom - 6, pw - 20, 16), 50)
        pygame.draw.rect(s, WHITE, pill, border_radius=28)
        pygame.draw.rect(s, (int(255 - 40 * pulse), int(200 - 10 * pulse), int(40 + 20 * pulse)),
                         pill, 3, border_radius=28)
        s.blit(label, (pill.centerx - label.get_width() // 2, pill.centery - label.get_height() // 2))

        _text(s, self.f_small, "Press ESC to Quit", SOFT_WHITE, (self.W // 2, 668), anchor="midtop")