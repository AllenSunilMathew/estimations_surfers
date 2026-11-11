"""Sky-card "AI Pose Controlled Game" renderer for Treasure Runner.

Enhanced 3D renderer with proper perspective ground, depth fog, parallax
buildings, speed lines, and richer obstacle / player art.

Public API used by main.py / GameEngine is unchanged:
    Renderer(screen), tick(dt), draw_bg(), draw_hud(...), draw_pose_panel(...),
    draw_game_panel(engine), draw_game_over(...), draw_calibrating(...)
"""

import math
import random

import cv2
import numpy as np
import pygame

from constants import *
from theme import (font as _theme_font, title_font as _theme_title_font, build_card_bg, text as _draw_text, panel as _panel,
                   header_band as _header_band, coin as _coin, star as _star,
                   runner_icon as _runner_icon_art, soft_shadow_ellipse)


class _Particle:
    """Small coin-collection sparkle."""
    __slots__ = ("x", "y", "vx", "vy", "life", "size", "color")

    def __init__(self, x, y, colors=None):
        self.x = float(x)
        self.y = float(y)
        self.vx = random.uniform(-100, 100)
        self.vy = random.uniform(-180, -40)
        self.life = 1.0
        self.size = random.randint(2, 5)
        self.color = random.choice(colors or [GOLD, (255, 230, 90), WHITE])

    def tick(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 260 * dt
        self.life -= dt * 1.8
        return self.life > 0


class _SpeedLine:
    """Diagonal streak for speed-feel."""
    __slots__ = ("x", "y", "length", "speed", "alpha")

    def __init__(self, w, h):
        self.x = random.randint(0, w)
        self.y = random.randint(0, h)
        self.length = random.randint(30, 80)
        self.speed = random.uniform(600, 1200)
        self.alpha = random.randint(30, 90)


class Renderer:
    N_TIES = 28
    CARD_MARGIN = 14
    SIDE = 28

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.W, self.H = WINDOW_WIDTH, WINDOW_HEIGHT

        pygame.font.init()
        self.f_title = _theme_title_font(38)
        self.f_sub = self._font("Arial", 18, True)
        self.f_panel = _theme_title_font(22)
        self.f_hdr = self._font("Arial", 17, True)
        self.f_score_lbl = _theme_title_font(16)
        self.f_score = _theme_title_font(34)
        self.f_large = _theme_title_font(28)
        self.f_medium = self._font("Arial", 20, True)
        self.f_gesture = self._font("Arial", 20, True)
        self.f_small = self._font("Arial", 15, True)
        self.f_tiny = self._font("Arial", 13, True)
        self.f_go = _theme_title_font(52)

        # ── layout ──────────────────────────────────────────────────
        margin, gap = self.SIDE, 14
        panel_y = HUD_HEIGHT + 14
        panel_h = self.H - panel_y - 24
        total_w = self.W - 2 * margin - gap
        pose_w = int(total_w * 0.44)
        game_w = total_w - pose_w

        self.pose_rect = pygame.Rect(margin, panel_y, pose_w, panel_h)
        self.game_rect = pygame.Rect(margin + pose_w + gap, panel_y, game_w, panel_h)

        # Top Bar badges (Right side of header)
        # Speed pill: x=880 to 1030 (w=150)
        # Score pill: x=1045 to 1250 (w=205)
        self.speed_rect = pygame.Rect(self.W - margin - 370, 22, 150, 72)
        self.score_rect = pygame.Rect(self.W - margin - 205, 22, 205, 72)

        pr = self.pose_rect
        self.cam_rect = pygame.Rect(pr.x + 10, pr.y + 54, pr.w - 20, pr.h - 64)
        cr = self.cam_rect
        # Clickable camera switch button docked at bottom-right inside camera panel
        self.cam_switch_rect = pygame.Rect(cr.right - 170, cr.bottom - 60, 158, 50)
        self.scene_rect = self.game_rect.inflate(-18, -18)

        # 3D scene geometry
        S = self.scene_rect
        self.vp_lx = S.w / 2
        self.vp_ly = int(S.h * 0.36)          # vanishing point higher for more 3D
        self.road_w_top = 40
        self.road_w_bot = int(S.w * 0.68)
        self.vp_x = S.x + self.vp_lx
        self.vp_y = S.y + self.vp_ly
        self.road_bottom = S.bottom

        self.t = 0.0
        self._dt = 0.0
        self.track_phase = 0.0
        self.particles = []

        # speed lines
        self._speed_lines = [_SpeedLine(S.w, S.h) for _ in range(15)]

        # cached layers
        self._chrome = self._build_chrome()
        self._back = self._build_back()
        self._front = self._build_front()
        self._scene_mask = self._corner_mask(S.size, 16)
        self._cam_mask = self._corner_mask(self.cam_rect.size, 12)
        self._game_header = self._build_game_header()

        # fog surface (reused each frame)
        self._fog_surf = pygame.Surface(S.size, pygame.SRCALPHA)

    # ────────────────────────────────────────────────────────────────
    # helpers
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    def _font(name, size, bold=False, italic=False):
        return _theme_font(name, size, bold, italic)

    def _text(self, font, value, color, pos, shadow=True, surf=None):
        surf = surf or self.screen
        r = _draw_text(surf, font, value, color, pos, shadow=INK_SHADOW if shadow else None)
        return r

    @staticmethod
    def _corner_mask(size, radius):
        m = pygame.Surface(size, pygame.SRCALPHA)
        m.fill((*PANEL_FILL, 255))
        pygame.draw.rect(m, (0, 0, 0, 0), (0, 0, *size), border_radius=radius)
        return m

    def _world2screen(self, lane: float, z: float):
        z = max(0.0, min(1.45, z))
        p = z ** 1.5
        y = self.vp_y + (self.road_bottom - self.vp_y) * p
        width = self.road_w_top + (self.road_w_bot - self.road_w_top) * p
        x = self.vp_x - width / 2 + width * (lane + 0.5) / 3
        scale = 0.15 + 0.85 * p        # scale factor for objects
        return int(x), int(y), scale

    def tick(self, dt):
        self.t += dt
        self._dt = dt
        self.particles = [p for p in self.particles if p.tick(dt)]

    # ────────────────────────────────────────────────────────────────
    # icons (white with teal details, to sit on the blue panels)
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    def _camera_icon(surf, x, y):
        pygame.draw.rect(surf, WHITE, (x, y + 4, 34, 24), border_radius=7)
        pygame.draw.polygon(surf, WHITE, [(x + 34, y + 12), (x + 46, y + 5), (x + 46, y + 27), (x + 34, y + 20)])
        pygame.draw.circle(surf, INK, (x + 17, y + 16), 8)
        pygame.draw.circle(surf, SKY_A, (x + 17, y + 16), 4)
        pygame.draw.rect(surf, WHITE, (x + 6, y, 10, 6), border_radius=2)

    @staticmethod
    def _gamepad_icon(surf, x, y):
        pygame.draw.rect(surf, WHITE, (x, y + 2, 48, 28), border_radius=13)
        pygame.draw.rect(surf, INK, (x + 9, y + 14, 14, 4))
        pygame.draw.rect(surf, INK, (x + 14, y + 9, 4, 14))
        pygame.draw.circle(surf, HOODIE_RED, (x + 34, y + 11), 3)
        pygame.draw.circle(surf, GOLD_BRIGHT, (x + 40, y + 18), 3)

    @staticmethod
    def _trophy(surf, cx, cy):
        g, gd = GOLD_BRIGHT, (200, 130, 20)
        pygame.draw.circle(surf, gd, (cx - 24, cy - 12), 9, 4)
        pygame.draw.circle(surf, gd, (cx + 24, cy - 12), 9, 4)
        pygame.draw.polygon(surf, g, [(cx - 20, cy - 26), (cx + 20, cy - 26), (cx + 15, cy + 2),
                                      (cx + 7, cy + 11), (cx - 7, cy + 11), (cx - 15, cy + 2)])
        pygame.draw.rect(surf, gd, (cx - 5, cy + 10, 10, 11))
        pygame.draw.rect(surf, g, (cx - 15, cy + 20, 30, 8), border_radius=3)
        pygame.draw.polygon(surf, (255, 240, 170), _star(cx, cy - 11, 9, 4))

    # ────────────────────────────────────────────────────────────────
    # cached chrome (card background, header, panel frames)
    # ────────────────────────────────────────────────────────────────
    def _build_chrome(self):
        W, H = self.W, self.H
        surf = build_card_bg(W, H, margin=self.CARD_MARGIN, radius=30)
        _runner_icon_art(surf, 90, 60, 1.1)
        _coin(surf, 155, 56, 18)
        _coin(surf, 196, 70, 12)

        cx = 550
        for dx, dy, col in ((4, 5, (30, 100, 140)), (2, 3, INK_SHADOW), (0, 0, WHITE)):
            im = self.f_title.render("TREASURE  RUNNER", True, col)
            surf.blit(im, (cx - im.get_width() // 2 + dx, 20 + dy))
        sub = self.f_sub.render("Move  \u2022  Play  \u2022  Have Fun!", True, SOFT_WHITE)
        shs = self.f_sub.render("Move  \u2022  Play  \u2022  Have Fun!", True, INK_SHADOW)
        surf.blit(shs, (cx - sub.get_width() // 2 + 2, 64 + 2))
        surf.blit(sub, (cx - sub.get_width() // 2, 64))

        # Speed pill background frame
        _panel(surf, self.speed_rect, radius=22, fill=PANEL_DARK)
        # Score pill background frame
        pill = self.score_rect
        _panel(surf, pill, radius=22, fill=PANEL_DARK)
        self._trophy(surf, pill.x + 36, pill.centery - 2)

        _panel(surf, self.pose_rect)
        _panel(surf, self.game_rect)

        pr = self.pose_rect
        _header_band(surf, pr, 46)
        self._camera_icon(surf, pr.x + 18, pr.y + 11)
        self._text(self.f_panel, "Your Pose", WHITE, (pr.x + 68, pr.y + 10), True, surf)
        return surf

    def _build_game_header(self):
        w, h = 360, 48
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.polygon(surf, (*PANEL_DARK, 238), [(0, 0), (w, 0), (w - 60, h), (0, h)])
        self._gamepad_icon(surf, 14, 8)
        self._text(self.f_panel, "Game Screen", WHITE, (74, 9), True, surf)
        return surf

    # ────────────────────────────────────────────────────────────────
    # cached scene layers — enhanced 3D background
    # ────────────────────────────────────────────────────────────────
    def _proj(self, lat, p):
        """Ground point at lateral offset `lat` (±1 = rail edge) and depth p."""
        half = (self.road_w_top + (self.road_w_bot - self.road_w_top) * p) / 2
        gy = self.vp_ly + (self.road_bot_l - self.vp_ly) * p
        return self.vp_lx + lat * half, gy

    @property
    def road_bot_l(self):
        return self.scene_rect.h

    @staticmethod
    def _leaf(surf, base, ang, length, col):
        a = math.radians(ang)
        dx, dy = math.cos(a), math.sin(a)
        px, py = -dy, dx
        left, right = [], []
        for i in range(11):
            t = i / 10
            cx = base[0] + dx * length * t
            cy = base[1] + dy * length * t + length * 0.45 * t * t
            wd = math.sin(math.pi * min(t * 1.1, 1.0)) * length * 0.13 + 1
            left.append((cx + px * wd, cy + py * wd))
            right.append((cx - px * wd, cy - py * wd))
        pygame.draw.polygon(surf, col, left + right[::-1])

    def _palm(self, surf, base, top, size):
        steps = 16
        prev = base
        for i in range(1, steps + 1):
            t = i / steps
            x = base[0] + (top[0] - base[0]) * t + math.sin(t * math.pi) * 10 * size
            y = base[1] + (top[1] - base[1]) * t
            wdt = max(3, int(13 * size * (1 - 0.35 * t)))
            pygame.draw.line(surf, (122, 78, 46), prev, (x, y), wdt)
            if i % 2 == 0:
                pygame.draw.line(surf, (92, 58, 34), (x - wdt / 2, y), (x + wdt / 2, y), 2)
            prev = (x, y)
        tip = prev
        for k, ang in enumerate((-175, -140, -105, -75, -40, -5, 28, 152)):
            col = (36, 150, 72) if k % 2 == 0 else (26, 118, 56)
            self._leaf(surf, tip, ang, 88 * size, col)
        pygame.draw.circle(surf, (90, 60, 30), (int(tip[0]), int(tip[1])), max(3, int(6 * size)))

    def _build_back(self):
        w, h = self.scene_rect.size
        surf = pygame.Surface((w, h))
        hor = self.vp_ly
        rng = random.Random(11)

        # sky gradient — deeper blue at top, lighter at horizon
        top, mid, bot = (28, 88, 195), (58, 138, 235), (178, 220, 250)
        for y in range(h):
            q = min(1.0, y / max(1, hor + 30))
            if q < 0.5:
                t = q * 2
                c = tuple(int(top[i] + (mid[i] - top[i]) * t) for i in range(3))
            else:
                t = (q - 0.5) * 2
                c = tuple(int(mid[i] + (bot[i] - mid[i]) * t) for i in range(3))
            pygame.draw.line(surf, c, (0, y), (w, y))

        # sun glow
        sun_x, sun_y = int(w * 0.75), int(hor * 0.35)
        for r in range(60, 0, -2):
            a = int(15 * (r / 60))
            gs = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(gs, (255, 240, 180, a), (r, r), r)
            surf.blit(gs, (sun_x - r, sun_y - r))
        pygame.draw.circle(surf, (255, 248, 220), (sun_x, sun_y), 18)

        # clouds with depth layers
        for cx, cy, s, alpha in ((80, 55, 0.8, 200), (220, 35, 1.1, 220),
                                  (400, 65, 0.9, 190), (w - 100, 45, 1.0, 210)):
            cs = pygame.Surface((int(100 * s), int(50 * s)), pygame.SRCALPHA)
            for ox, oy, r in ((20, 18, 16), (40, 12, 22), (64, 18, 16), (42, 24, 18)):
                pygame.draw.circle(cs, (255, 255, 255, alpha),
                                   (int(ox * s), int(oy * s)), int(r * s))
            surf.blit(cs, (int(cx), int(cy)))

        # buildings (far → near, both sides) — enhanced with more 3D depth
        pal = [(236, 120, 150), (250, 172, 120), (110, 196, 200), (172, 132, 232),
               (242, 204, 122), (112, 162, 236), (232, 142, 192), (130, 200, 160)]
        base_y = hor + 4
        for side in (-1, 1):
            for k in range(10):
                f = k / 9
                d_in = 30 + f * (w / 2 - 15)
                bw = int(36 + f * 130)
                bh = int(50 + f * 300)
                x0 = int(self.vp_lx + side * d_in) if side > 0 else int(self.vp_lx - d_in - bw)
                col = pal[(k * 2 + (0 if side < 0 else 3)) % len(pal)]
                # main face
                pygame.draw.rect(surf, col, (x0, base_y - bh, bw, bh))
                # darker side face for 3D depth
                dark = tuple(int(c * 0.72) for c in col)
                side_w = max(4, bw // 5)
                if side < 0:
                    pygame.draw.rect(surf, dark, (x0 + bw - side_w, base_y - bh, side_w, bh))
                else:
                    pygame.draw.rect(surf, dark, (x0, base_y - bh, side_w, bh))
                # roof/top highlight
                pygame.draw.rect(surf, tuple(int(c * 0.60) for c in col),
                                 (x0 - 3, base_y - bh - 8, bw + 6, 10))
                # windows
                wstep = max(10, int(11 + f * 6))
                for wx in range(x0 + 6, x0 + bw - 8, wstep):
                    for wy in range(base_y - bh + 14, base_y - 12, wstep + 5):
                        c = (255, 236, 170) if rng.random() < 0.75 else (90, 110, 160)
                        pygame.draw.rect(surf, c,
                                         (wx, wy, max(3, wstep // 2 - 1), max(4, wstep // 2 + 1)))
                # billboards on some buildings
                if k in (3, 5, 7):
                    pygame.draw.rect(surf, (60, 200, 210),
                                     (x0 + 6, base_y - bh + 24, bw - 12, 34), border_radius=4)
                    pygame.draw.rect(surf, WHITE,
                                     (x0 + 6, base_y - bh + 24, bw - 12, 34), 2, border_radius=4)

        # distant station gate / tunnel — more detailed 3D arch
        cx = int(self.vp_lx)
        # back wall
        pygame.draw.rect(surf, (100, 114, 152), (cx - 100, hor - 70, 200, 76))
        # top ledge
        pygame.draw.rect(surf, (140, 154, 190), (cx - 110, hor - 82, 220, 14))
        # arch opening (dark)
        pygame.draw.rect(surf, (20, 28, 70), (cx - 62, hor - 52, 124, 58),
                         border_top_left_radius=62, border_top_right_radius=62)
        # inner arch (slightly lighter = depth)
        pygame.draw.rect(surf, (50, 74, 160), (cx - 40, hor - 38, 80, 44),
                         border_top_left_radius=40, border_top_right_radius=40)
        # arch pillars
        for px in (-66, 58):
            pygame.draw.rect(surf, (120, 134, 170), (cx + px, hor - 58, 8, 64))
        # signal light
        pygame.draw.rect(surf, (255, 170, 60), (cx - 5, hor - 16, 10, 12), border_radius=2)
        # lamp posts on tunnel entrance
        for lx in (-84, -70, 70, 84):
            pygame.draw.circle(surf, (255, 225, 140), (cx + lx, hor - 56), 3)

        # side ground with perspective gravel texture
        wt, wb = self.road_w_top, self.road_w_bot
        # Left ground
        pygame.draw.polygon(surf, (118, 96, 100),
                            [(0, hor), (cx - wt / 2, hor), (cx - wb / 2, h), (0, h)])
        # Right ground
        pygame.draw.polygon(surf, (128, 104, 104),
                            [(cx + wt / 2, hor), (w, hor), (w, h), (cx + wb / 2, h)])

        # gravel dots for ground texture
        for _ in range(200):
            gx, gy = rng.randint(0, w), rng.randint(hor + 5, h)
            depth = (gy - hor) / max(1, h - hor)
            if depth > 0.1:
                c = rng.randint(90, 140)
                sz = max(1, int(1 + depth * 3))
                pygame.draw.circle(surf, (c, c - 10, c - 20), (gx, gy), sz)

        # palms with enhanced depth
        for fx, by, tx, ty, sz in ((0.18, 0.48, 0.20, 0.08, 1.0), (0.82, 0.52, 0.80, 0.04, 1.15),
                                   (0.32, 0.42, 0.33, 0.28, 0.55), (0.70, 0.42, 0.69, 0.27, 0.55)):
            self._palm(surf, (w * fx, h * by), (w * tx, h * ty), sz)
        return surf

    def _build_front(self):
        w, h = self.scene_rect.size
        surf = pygame.Surface((w, h), pygame.SRCALPHA)

        def hpx(p):
            return 20 + 250 * p

        for side in (-1, 1):
            lat = side * 1.18

            def pt(frac, p):
                x, gy = self._proj(lat, p)
                return x, gy - frac * hpx(p)

            def quad(f0, f1, p0, p1, col):
                pygame.draw.polygon(surf, col, [pt(f0, p0), pt(f1, p0), pt(f1, p1), pt(f0, p1)])

            p_far, p_near = 0.02, 1.7
            quad(0.0, 1.0, p_far, p_near, (36, 100, 214))        # body
            quad(0.94, 1.0, p_far, p_near, (232, 238, 250))      # roof
            quad(0.0, 0.07, p_far, p_near, (28, 34, 62))         # skirt
            quad(0.11, 0.21, p_far, p_near, (255, 150, 40))      # orange stripe
            quad(0.21, 0.25, p_far, p_near, (235, 240, 250))     # white stripe

            ps, p = [], 1.7
            while p > 0.06:
                ps.append(p)
                p = 1.0 / (1.0 / p + 0.62)
            for a, b in zip(ps, ps[1:]):
                span = a - b
                pa, pb = b + span * 0.16, a - span * 0.16
                quad(0.42, 0.86, pa, pb, (236, 242, 250))
                ia, ib = pa + (pb - pa) * 0.06, pb - (pb - pa) * 0.06
                quad(0.45, 0.83, ia, ib, (104, 196, 240))
        return surf

    # ────────────────────────────────────────────────────────────────
    # HUD / panels
    # ────────────────────────────────────────────────────────────────
    def draw_bg(self):
        self.screen.blit(self._chrome, (0, 0))

    def draw_hud(self, score, fps, gesture, confidence, speed=5.0):
        # ── Speed Pill (Clean dedicated badge) ──────────────────────
        sx, sy = self.speed_rect.x, self.speed_rect.y
        # Lightning icon
        pygame.draw.polygon(self.screen, GOLD_BRIGHT, [
            (sx + 24, sy + 18), (sx + 14, sy + 38), (sx + 22, sy + 38),
            (sx + 16, sy + 54), (sx + 32, sy + 32), (sx + 24, sy + 32)
        ])
        self._text(self.f_tiny, "SPEED", SOFT_WHITE, (sx + 40, sy + 12), True)
        mult = speed / 5.0
        spd_text = f"{mult:.1f}x"
        if mult >= 1.8:
            spd_text += " ⚡"
        spd_col = GOLD_BRIGHT if mult >= 1.5 else (140, 240, 255)
        self._text(self.f_panel, spd_text, spd_col, (sx + 40, sy + 30), True)

        # ── Score Pill (Clean dedicated badge) ──────────────────────
        px, py = self.score_rect.x, self.score_rect.y
        self._text(self.f_tiny, "SCORE", SOFT_WHITE, (px + 72, py + 12), True)
        self._text(self.f_large, f"{score:,}", GOLD_BRIGHT, (px + 72, py + 30), True)

    def _frame_surface(self, frame, size):
        tw, th = size
        h, w = frame.shape[:2]
        s = max(tw / w, th / h)
        nw, nh = int(w * s + 0.5), int(h * s + 0.5)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (nw, nh))
        x0, y0 = (nw - tw) // 2, (nh - th) // 2
        rgb = np.ascontiguousarray(rgb[y0:y0 + th, x0:x0 + tw])
        return pygame.image.frombuffer(rgb.tobytes(), (tw, th), "RGB")

    def draw_pose_panel(self, frame, gesture, confidence, fps, cam_name="WEBCAM",
                        is_switching=False, status_msg="", mouse_pos=(0, 0)):
        r, cr = self.pose_rect, self.cam_rect

        # ── Header: Live FPS | Confidence (clean right-alignment) ───
        fps_txt = self.f_hdr.render(f"FPS: {int(fps)}", True, (120, 255, 160))
        lbl = self.f_hdr.render("Confidence: ", True, WHITE)
        val = self.f_hdr.render(f"{int(confidence * 100)}%", True, GOLD_BRIGHT)
        total = 12 + 8 + fps_txt.get_width() + 20 + lbl.get_width() + val.get_width()
        x = r.right - 18 - total
        cy = r.y + 23
        pygame.draw.circle(self.screen, (90, 240, 130), (x + 6, cy), 5)
        self.screen.blit(fps_txt, (x + 16, cy - fps_txt.get_height() // 2))
        dx = x + 16 + fps_txt.get_width() + 10
        pygame.draw.line(self.screen, EDGE_SOFT, (dx, cy - 12), (dx, cy + 12), 2)
        self.screen.blit(lbl, (dx + 10, cy - lbl.get_height() // 2))
        self.screen.blit(val, (dx + 10 + lbl.get_width(), cy - val.get_height() // 2))

        # ── Camera Feed Box ─────────────────────────────────────────
        pygame.draw.rect(self.screen, (16, 62, 94), cr, border_radius=12)
        if frame is not None and not is_switching:
            self.screen.blit(self._frame_surface(frame, cr.size), cr.topleft)
        else:
            msg = status_msg if status_msg else ("SWITCHING CAMERA..." if is_switching else "WAITING FOR CAMERA")
            t = self.f_medium.render(msg, True, (255, 220, 140) if is_switching else (255, 190, 190))
            self.screen.blit(t, (cr.centerx - t.get_width() // 2, cr.centery - 10))
        self.screen.blit(self._cam_mask, cr.topleft)
        pygame.draw.rect(self.screen, WHITE, cr, 2, border_radius=12)

        # ── Bottom Row in Camera Feed: Gesture (Left) + Camera Swap (Right) ──
        # 1. Gesture Pill (Left: w=300, h=48)
        pill = pygame.Rect(cr.x + 10, cr.bottom - 56, 305, 48)
        ps = pygame.Surface(pill.size, pygame.SRCALPHA)
        pygame.draw.rect(ps, (*PANEL_DARK, 230), (0, 0, *pill.size), border_radius=16)
        pygame.draw.rect(ps, (255, 255, 255, 255), (0, 0, *pill.size), 2, border_radius=16)
        self.screen.blit(ps, pill.topleft)
        _runner_icon_art(self.screen, pill.x + 30, pill.centery + 1, 0.48)
        lab = self.f_gesture.render("Gesture:", True, WHITE)
        self.screen.blit(lab, (pill.x + 64, pill.centery - lab.get_height() // 2))
        gcol = {Gesture.NEUTRAL: SOFT_WHITE}.get(gesture, GOLD_BRIGHT)
        val = self.f_gesture.render(gesture, True, gcol)
        self.screen.blit(val, (pill.x + 64 + lab.get_width() + 10, pill.centery - val.get_height() // 2))

        # 2. Camera Swap Button (Right: w=155, h=48, aligned on same row)
        cs = self.cam_switch_rect
        is_hover = cs.collidepoint(mouse_pos)
        cps = pygame.Surface(cs.size, pygame.SRCALPHA)
        bg = (52, 150, 196, 235) if is_hover else (*PANEL_DARK, 230)
        pygame.draw.rect(cps, bg, (0, 0, *cs.size), border_radius=16)
        border_c = GOLD_BRIGHT if is_hover else WHITE
        pygame.draw.rect(cps, border_c, (0, 0, *cs.size), 2 if not is_hover else 3, border_radius=16)
        self.screen.blit(cps, cs.topleft)

        swap_lbl = self.f_tiny.render("📷 Swap [C]", True, (190, 235, 250))
        name_lbl = self.f_small.render(cam_name, True, GOLD_BRIGHT if is_hover else WHITE)
        self.screen.blit(swap_lbl, (cs.centerx - swap_lbl.get_width() // 2, cs.y + 6))
        self.screen.blit(name_lbl, (cs.centerx - name_lbl.get_width() // 2, cs.y + 24))

    # ────────────────────────────────────────────────────────────────
    # game scene — enhanced 3D
    # ────────────────────────────────────────────────────────────────
    def draw_game_panel(self, engine):
        S = self.scene_rect
        if not engine.game_over:
            self.track_phase = (self.track_phase + self._dt * engine.speed * 0.14) % (1.0 / self.N_TIES)

        prev_clip = self.screen.get_clip()
        self.screen.set_clip(S)
        self.screen.blit(self._back, S.topleft)
        self._draw_track(engine.speed)
        self._draw_speed_lines(engine.speed, S)
        self.screen.blit(self._front, S.topleft)

        # draw all objects sorted by z (far to near) for proper 3D ordering
        all_objects = []
        for coin in engine.coins:
            if not coin.collected:
                all_objects.append(('coin', coin.z, coin))
        for obstacle in engine.obstacles:
            all_objects.append(('obs', obstacle.z, obstacle))
        all_objects.append(('player', 0.92, engine.player))
        if engine.police_z <= 1.35:
            all_objects.append(('police', engine.police_z, engine))
        all_objects.sort(key=lambda x: x[1])

        for kind, z, obj in all_objects:
            if kind == 'coin':
                self._draw_coin(obj)
            elif kind == 'obs':
                self._draw_obstacle(obj)
            elif kind == 'player':
                self._draw_player(obj, engine.is_stumbling)
            elif kind == 'police':
                self._draw_police_and_dog(obj)

        # particles
        for lane, z in engine.coin_collect_events:
            sx, sy, _ = self._world2screen(lane, z)
            for _ in range(12):
                self.particles.append(_Particle(sx, sy - 20))
        for lane, z, kind in engine.hit_events:
            sx, sy, _ = self._world2screen(lane, z)
            cols = [(50, 56, 90), (96, 108, 170), WHITE, (255, 170, 50)] if kind == "bird" \
                else [(205, 48, 52), WHITE, (255, 200, 40)]
            for _ in range(18):
                self.particles.append(_Particle(sx, sy - 40, cols))
        for p in self.particles:
            pygame.draw.circle(self.screen, p.color, (int(p.x), int(p.y)), max(1, int(p.size * p.life)))

        # depth fog overlay — objects far away fade into atmosphere
        self._draw_depth_fog(S)

        self._draw_lives(S, engine)
        if engine.flash_timer > 0:
            fl = pygame.Surface(S.size, pygame.SRCALPHA)
            fl.fill((255, 45, 45, int(engine.flash_timer / 0.30 * 80)))
            self.screen.blit(fl, S.topleft)

        self.screen.blit(self._game_header, S.topleft)
        self.screen.blit(self._scene_mask, S.topleft)
        self.screen.set_clip(prev_clip)
        pygame.draw.rect(self.screen, WHITE, S, 2, border_radius=16)

        # decorative pause button (top-right)
        c = (self.game_rect.right - 48, self.game_rect.y + 50)
        pygame.draw.circle(self.screen, (20, 90, 130), (c[0], c[1] + 3), 38)
        pygame.draw.circle(self.screen, WHITE, c, 38)
        pygame.draw.circle(self.screen, PANEL_DARK, c, 33)
        pygame.draw.rect(self.screen, WHITE, (c[0] - 12, c[1] - 14, 8, 28), border_radius=2)
        pygame.draw.rect(self.screen, WHITE, (c[0] + 4, c[1] - 14, 8, 28), border_radius=2)

    def _draw_depth_fog(self, S):
        """Subtle atmospheric fog near the vanishing point for 3D depth."""
        fog = self._fog_surf
        fog.fill((0, 0, 0, 0))
        hor = self.vp_ly
        fog_h = int(hor * 0.8)
        # gradient from transparent at bottom to hazy at horizon
        for y in range(fog_h):
            a = int(45 * (1.0 - y / fog_h) ** 2)
            pygame.draw.line(fog, (178, 220, 250, a), (0, hor - fog_h + y), (S.w, hor - fog_h + y))
        self.screen.blit(fog, S.topleft)

    def _draw_speed_lines(self, speed, S):
        """Diagonal motion streaks that intensify at higher speed."""
        if speed < 6:
            return
        intensity = min(1.0, (speed - 6) / 8)
        for sl in self._speed_lines:
            sl.y += sl.speed * self._dt
            if sl.y > S.h + sl.length:
                sl.y = -sl.length
                sl.x = random.randint(0, S.w)
            a = int(sl.alpha * intensity)
            if a > 5:
                x1 = S.x + sl.x
                y1 = S.y + int(sl.y)
                x2 = x1 - int(sl.length * 0.3)
                y2 = y1 - sl.length
                ls = pygame.Surface((abs(x1 - x2) + 4, sl.length + 4), pygame.SRCALPHA)
                pygame.draw.line(ls, (255, 255, 255, a), (ls.get_width() - 2, ls.get_height() - 2), (2, 2), 2)
                self.screen.blit(ls, (min(x1, x2), min(y1, y2)))

    def _draw_track(self, speed=5.0):
        S = self.scene_rect
        vx, vy, bot = self.vp_x, self.vp_y, S.bottom
        wt, wb = self.road_w_top, self.road_w_bot

        # main road surface — slightly warmer gray for 3D feel
        pygame.draw.polygon(self.screen, (160, 100, 68),
                            [(vx - wt / 2, vy), (vx + wt / 2, vy), (vx + wb / 2, bot), (vx - wb / 2, bot)])
        # road highlight strip along center for 3D curvature
        pygame.draw.polygon(self.screen, (175, 112, 78),
                            [(vx - wt * 0.15, vy), (vx + wt * 0.15, vy),
                             (vx + wb * 0.15, bot), (vx - wb * 0.15, bot)])

        # cross ties (railroad sleepers) with 3D perspective thickness
        n = self.N_TIES
        for i in range(n):
            z = (i / n + self.track_phase) % 1.0
            p = z ** 1.5
            y = vy + (bot - vy) * p
            width = wt + (wb - wt) * p
            th = max(2, int(2 + 18 * p))
            # shadow
            pygame.draw.rect(self.screen, (90, 52, 32),
                             (int(vx - width / 2), int(y + 2), int(width), th))
            # tie
            pygame.draw.rect(self.screen, (112, 64, 42),
                             (int(vx - width / 2), int(y), int(width), th))
            # highlight on top edge
            pygame.draw.rect(self.screen, (150, 92, 62),
                             (int(vx - width / 2), int(y), int(width), max(1, th // 4)))

        # lane rails with perspective
        for lane in range(3):
            for off in (-0.27, 0.27):
                f = (lane + 0.5 + off) / 3
                xt = vx - wt / 2 + wt * f
                xb = vx - wb / 2 + wb * f
                # rail shadow
                pygame.draw.line(self.screen, (80, 60, 50), (xt, vy), (xb + 5, bot), 3)
                # rail highlight
                pygame.draw.polygon(self.screen, (204, 211, 226),
                                    [(xt - 1, vy), (xt + 1, vy), (xb + 4, bot), (xb - 4, bot)])
                pygame.draw.line(self.screen, (120, 130, 150), (xt - 1, vy), (xb - 4, bot), 1)

    def _draw_obstacle(self, obs):
        if obs.z < 0.05 or obs.z > 1.45:
            return
        if obs.type == "bird":
            self._draw_bird(obs)
            return
        if obs.type == "train":
            self._draw_train(obs)
            return
        x, y, scale = self._world2screen(obs.lane, obs.z)

        # ground shadow for 3D depth
        sh_w = max(8, int(80 * scale))
        sh_h = max(3, int(12 * scale))
        sh_surf = pygame.Surface((sh_w, sh_h), pygame.SRCALPHA)
        pygame.draw.ellipse(sh_surf, (20, 14, 30, int(100 * scale)), sh_surf.get_rect())
        self.screen.blit(sh_surf, (x - sh_w // 2, y - sh_h // 2 + 2))

        if obs.type == "barrier":
            w, h = max(10, int(110 * scale)), max(8, int(68 * scale))
            pw = max(3, int(14 * scale))
            lamp = max(3, int(10 * scale))
            for px in (x - w // 2, x + w // 2 - pw):
                # post shadow
                pygame.draw.rect(self.screen, (160, 36, 40), (px + 2, y - h + 2, pw, h), border_radius=2)
                # post
                pygame.draw.rect(self.screen, (205, 48, 52), (px, y - h, pw, h), border_radius=2)
                # lamp
                pygame.draw.circle(self.screen, (60, 66, 82), (px + pw // 2, y - h), lamp + 1)
                pygame.draw.circle(self.screen, (160, 165, 178), (px + pw // 2, y - h), lamp)
                # lamp glow
                glow = pygame.Surface((lamp * 4, lamp * 4), pygame.SRCALPHA)
                pygame.draw.circle(glow, (255, 200, 100, 30), (lamp * 2, lamp * 2), lamp * 2)
                self.screen.blit(glow, (px + pw // 2 - lamp * 2, y - h - lamp * 2))
                # base
                pygame.draw.rect(self.screen, (60, 66, 82), (px - 2, y - 3, pw + 4, 4))

            plank = pygame.Rect(x - w // 2, y - int(h * 0.82), w, max(4, int(h * 0.36)))
            pygame.draw.rect(self.screen, (248, 248, 250), plank)
            sw = max(3, int(14 * scale))
            for sx in range(plank.x, plank.right, sw * 2):
                pygame.draw.rect(self.screen, (212, 50, 54),
                                 (sx, plank.y, min(sw, plank.right - sx), plank.h))
            pygame.draw.rect(self.screen, (80, 40, 40), plank, 1)
        else:
            # low bar with enhanced 3D
            w = max(10, int(120 * scale))
            bh = max(4, int(20 * scale))
            bar_y = y - int(65 * scale)
            pw = max(3, int(9 * scale))
            for px in (x - w // 2, x + w // 2 - pw):
                # post shadow
                pygame.draw.rect(self.screen, (42, 50, 70), (px + 2, bar_y + 2, pw, y - bar_y))
                # post
                pygame.draw.rect(self.screen, (62, 70, 90), (px, bar_y, pw, y - bar_y))
            # bar shadow
            bar_sh = pygame.Rect(x - w // 2 + 3, bar_y + 3, w, bh)
            pygame.draw.rect(self.screen, (180, 140, 20), bar_sh)
            # bar
            bar = pygame.Rect(x - w // 2, bar_y, w, bh)
            pygame.draw.rect(self.screen, (255, 200, 40), bar)
            sw = max(3, int(12 * scale))
            for sx in range(bar.x, bar.right, sw * 2):
                pygame.draw.polygon(self.screen, (40, 40, 50),
                                    [(sx, bar.bottom), (sx + sw, bar.bottom),
                                     (min(sx + sw + bh, bar.right), bar.y), (min(sx + bh, bar.right), bar.y)])
            pygame.draw.rect(self.screen, (60, 50, 30), bar, 1)

    def _draw_bird(self, obs):
        """Front-view bird flapping towards the player. Duck (slide) to pass under it."""
        sway = math.sin(self.t * 3 + obs.phase) * 0.18
        x, y, s = self._world2screen(obs.lane + sway, obs.z)
        fly_y = y - int(86 * s)

        # dynamic shadow on ground
        soft_shadow_ellipse(self.screen, pygame.Rect(x - int(30 * s), y - int(5 * s),
                                                     max(8, int(60 * s)), max(3, int(12 * s))), 70)
        flap = math.sin(self.t * 16 + obs.phase)
        body_c, wing_c, wing_d = (50, 56, 90), (88, 100, 160), (36, 42, 72)
        for side in (-1, 1):
            root = (x + side * 10 * s, fly_y - 4 * s)
            tip = (x + side * 62 * s, fly_y - 10 * s - flap * 30 * s)
            back = (x + side * 40 * s, fly_y + 14 * s - flap * 12 * s)
            low = (x + side * 8 * s, fly_y + 12 * s)
            pygame.draw.polygon(self.screen, wing_c, [root, tip, back, low])
            pygame.draw.polygon(self.screen, wing_d, [root, tip, back, low], max(1, int(2 * s)))
        pygame.draw.ellipse(self.screen, body_c,
                            (x - 16 * s, fly_y - 17 * s, 32 * s, 38 * s))
        pygame.draw.ellipse(self.screen, (170, 180, 215),
                            (x - 9 * s, fly_y - 4 * s, 18 * s, 22 * s))
        hy = fly_y - 22 * s
        pygame.draw.circle(self.screen, body_c, (int(x), int(hy)), max(3, int(13 * s)))
        pygame.draw.polygon(self.screen, (255, 160, 40),
                            [(x - 5 * s, hy + 3 * s), (x + 5 * s, hy + 3 * s), (x, hy + 14 * s)])
        for ex in (-5, 5):
            pygame.draw.circle(self.screen, WHITE, (int(x + ex * s), int(hy - 2 * s)), max(2, int(4.5 * s)))
            pygame.draw.circle(self.screen, BLACK, (int(x + ex * s), int(hy - 1 * s)), max(1, int(2 * s)))

    def _draw_train(self, obs):
        """3D Subway Surfers style oncoming subway train with perspective roof, sides, and headlights."""
        if obs.z < 0.05 or obs.z > 1.45:
            return

        lane = obs.lane
        z_front = obs.z
        z_back = max(0.01, obs.z - 0.28)

        fx, fy, fs = self._world2screen(lane, z_front)
        bx, by, bs = self._world2screen(lane, z_back)

        # Dimensions
        fw = max(18, int(130 * fs))
        fh = max(22, int(145 * fs))

        bw = max(10, int(130 * bs))
        bh = max(12, int(145 * bs))

        # Ground shadow beneath the train
        sh_w = max(20, int(140 * fs))
        sh_h = max(6, int(22 * fs))
        sh_surf = pygame.Surface((sh_w, sh_h), pygame.SRCALPHA)
        pygame.draw.ellipse(sh_surf, (20, 14, 30, int(120 * min(1.0, fs))), sh_surf.get_rect())
        self.screen.blit(sh_surf, (fx - sh_w // 2, fy - sh_h // 2 + 2))

        # Headlight projection cone onto track (before drawing train body)
        if fs > 0.18:
            light_len = int(120 * fs)
            light_w = int(140 * fs)
            cone_surf = pygame.Surface((light_w * 2, light_len), pygame.SRCALPHA)
            beam_pts = [(light_w, 0), (0, light_len), (light_w * 2, light_len)]
            pygame.draw.polygon(cone_surf, (255, 250, 180, int(45 * min(1.0, fs))), beam_pts)
            self.screen.blit(cone_surf, (fx - light_w, fy))

        # Front face corners: top-left, top-right, bot-right, bot-left
        ftl = (fx - fw // 2, fy - fh)
        ftr = (fx + fw // 2, fy - fh)
        fbr = (fx + fw // 2, fy)
        fbl = (fx - fw // 2, fy)

        # Back face corners
        btl = (bx - bw // 2, by - bh)
        btr = (bx + bw // 2, by - bh)
        bbr = (bx + bw // 2, by)
        bbl = (bx - bw // 2, by)

        # 1. Roof (metallic curved perspective)
        roof_col = (175, 185, 202)
        roof_edge = (120, 130, 145)
        pygame.draw.polygon(self.screen, roof_col, [ftl, ftr, btr, btl])
        pygame.draw.polygon(self.screen, roof_edge, [ftl, ftr, btr, btl], max(1, int(2 * fs)))

        # Roof ventilation ridges
        if fs > 0.35:
            for r_step in (0.32, 0.64):
                rx1 = ftl[0] + (btl[0] - ftl[0]) * r_step
                ry1 = ftl[1] + (btl[1] - ftl[1]) * r_step
                rx2 = ftr[0] + (btr[0] - ftr[0]) * r_step
                ry2 = ftr[1] + (btr[1] - ftr[1]) * r_step
                pygame.draw.line(self.screen, (140, 150, 165), (rx1, ry1), (rx2, ry2), max(1, int(2 * fs)))

        # 2. Side Walls (depending on camera perspective)
        if fx > self.vp_x - 12:
            # Left wall visible
            side_col = (38, 68, 120)
            pygame.draw.polygon(self.screen, side_col, [ftl, btl, bbl, fbl])
            s_top_f = (fbl[0], fbl[1] - int(fh * 0.40))
            s_bot_f = (fbl[0], fbl[1] - int(fh * 0.30))
            s_top_b = (bbl[0], bbl[1] - int(bh * 0.40))
            s_bot_b = (bbl[0], bbl[1] - int(bh * 0.30))
            pygame.draw.polygon(self.screen, (220, 48, 52), [s_top_f, s_top_b, s_bot_b, s_bot_f])

        if fx < self.vp_x + 12:
            # Right wall visible
            side_col = (30, 54, 98)
            pygame.draw.polygon(self.screen, side_col, [ftr, btr, bbr, fbr])
            s_top_f = (fbr[0], fbr[1] - int(fh * 0.40))
            s_bot_f = (fbr[0], fbr[1] - int(fh * 0.30))
            s_top_b = (bbr[0], bbr[1] - int(bh * 0.40))
            s_bot_b = (bbr[0], bbr[1] - int(bh * 0.30))
            pygame.draw.polygon(self.screen, (220, 48, 52), [s_top_f, s_top_b, s_bot_b, s_bot_f])

        # 3. Front Face of Train
        front_body = (48, 86, 148)
        front_dark = (32, 58, 102)
        front_rect = pygame.Rect(fx - fw // 2, fy - fh, fw, fh)
        r_top = max(4, int(14 * fs))
        pygame.draw.rect(self.screen, front_dark, front_rect, border_top_left_radius=r_top, border_top_right_radius=r_top)
        inner_rect = front_rect.inflate(-max(2, int(4 * fs)), -max(2, int(4 * fs)))
        pygame.draw.rect(self.screen, front_body, inner_rect, border_top_left_radius=r_top, border_top_right_radius=r_top)

        # Windshield
        win_w = max(10, int(fw * 0.76))
        win_h = max(6, int(fh * 0.34))
        win_x = fx - win_w // 2
        win_y = fy - fh + max(4, int(16 * fs))
        win_rect = pygame.Rect(win_x, win_y, win_w, win_h)
        pygame.draw.rect(self.screen, (18, 32, 56), win_rect, border_radius=max(2, int(4 * fs)))
        if fs > 0.3:
            pygame.draw.line(self.screen, (100, 180, 240), (win_x + 3, win_y + win_h - 4), (win_x + win_w - 6, win_y + 4), max(1, int(2 * fs)))

        # Destination Sign
        sign_w = max(8, int(fw * 0.50))
        sign_h = max(3, int(10 * fs))
        sign_rect = pygame.Rect(fx - sign_w // 2, win_y - sign_h - max(1, int(2 * fs)), sign_w, sign_h)
        pygame.draw.rect(self.screen, (20, 20, 20), sign_rect, border_radius=2)
        pygame.draw.rect(self.screen, (255, 170, 30), sign_rect.inflate(-2, -2), border_radius=1)

        # Front Red Accent Stripe
        stripe_h = max(2, int(10 * fs))
        stripe_y = win_rect.bottom + max(2, int(6 * fs))
        pygame.draw.rect(self.screen, (220, 48, 52), (fx - fw // 2 + 2, stripe_y, fw - 4, stripe_h))

        # Twin Headlights
        hl_r = max(2, int(9 * fs))
        hl_y = stripe_y + stripe_h + max(3, int(8 * fs))
        for side_sign in (-1, 1):
            hl_x = fx + side_sign * int(fw * 0.30)
            pygame.draw.circle(self.screen, (200, 210, 220), (hl_x, hl_y), hl_r + 2)
            pygame.draw.circle(self.screen, (255, 255, 220), (hl_x, hl_y), hl_r)
            pygame.draw.circle(self.screen, WHITE, (hl_x - 1, hl_y - 1), max(1, hl_r // 2))

        # Hazard Bumper (Cowcatcher) at bottom
        bump_h = max(4, int(18 * fs))
        bump_y = fy - bump_h
        bump_rect = pygame.Rect(fx - fw // 2 - max(1, int(2 * fs)), bump_y, fw + max(2, int(4 * fs)), bump_h)
        pygame.draw.rect(self.screen, (255, 200, 30), bump_rect, border_radius=2)
        stripe_w = max(2, int(8 * fs))
        for sx in range(bump_rect.x, bump_rect.right, stripe_w * 2):
            pts = [(sx, bump_rect.bottom),
                   (min(bump_rect.right, sx + stripe_w), bump_rect.bottom),
                   (min(bump_rect.right, sx + stripe_w + bump_h), bump_rect.top),
                   (min(bump_rect.right, sx + bump_h), bump_rect.top)]
            if len(pts) >= 3:
                pygame.draw.polygon(self.screen, (30, 30, 35), pts)
        pygame.draw.rect(self.screen, (50, 40, 20), bump_rect, 1, border_radius=2)

    def _draw_coin(self, coin):
        if coin.z < 0.06 or coin.z > 1.1:
            return
        x, y, scale = self._world2screen(coin.lane, coin.z)
        y -= int(36 * scale)
        r = max(4, int(30 * scale))

        # coin glow for 3D pop
        if r > 6:
            glow = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow, (255, 200, 40, 25), (r * 2, r * 2), r * 2)
            self.screen.blit(glow, (x - r * 2, y - r * 2))

        sq = 0.45 + 0.55 * abs(math.cos(math.radians(coin.rotation)))
        rx = max(2, int(r * sq))
        # shadow
        pygame.draw.ellipse(self.screen, (185, 100, 12), (x - rx - 3, y - r - 3, 2 * rx + 6, 2 * r + 6))
        # body
        pygame.draw.ellipse(self.screen, (255, 198, 40), (x - rx, y - r, 2 * rx, 2 * r))
        if r > 7:
            pygame.draw.ellipse(self.screen, (255, 230, 120),
                                (x - rx + 4, y - r + 4, 2 * rx - 8, 2 * r - 8), 2)
            pygame.draw.polygon(self.screen, (255, 140, 20), _star(x, y, r * 0.55, r * 0.24, sq))
            # specular highlight
            if rx > 6:
                pygame.draw.circle(self.screen, (255, 250, 200), (x - rx // 3, y - r // 3), max(2, r // 5))

    def _draw_player(self, player, is_stumbling=False):
        S = self.scene_rect
        lane = player.visual_lane()
        x, y, _ = self._world2screen(lane, 0.92)
        yo = int(player.y_offset)
        yf = y + yo
        run = math.sin(self.t * 14)

        # Stumble wobble and sweat
        if is_stumbling:
            x += int(math.sin(self.t * 36) * 5)

        # colour palette
        hoodie      = (214, 50, 52)
        hoodie_d    = (160, 32, 40)
        hoodie_hi   = (235, 80, 75)     # highlight
        hoodie_edge = (120, 22, 28)     # dark edge/outline
        jeans       = (52, 98, 172)
        jeans_d     = (36, 72, 136)     # darker denim
        jeans_hi    = (72, 120, 200)
        shoe        = (246, 246, 246)
        shoe_sole   = (60, 60, 70)
        hair        = (92, 54, 34)
        hair_hi     = (120, 72, 46)
        skin        = (255, 198, 152)
        skin_d      = (220, 160, 120)

        scr = self.screen

        # ── stumble sweat droplets ──────────────────────────────────
        if is_stumbling:
            for s_i, (dx_s, dy_s) in enumerate([(-18, -100), (16, -104), (-22, -82)]):
                sw_y = yf + dy_s + int(math.sin(self.t * 16 + s_i) * 5)
                sw_x = x + dx_s
                pygame.draw.circle(scr, (160, 225, 255), (sw_x, sw_y), 3)

        # ── motion trail (ghosting) when running fast ───────────────
        if not player.is_sliding and not player.is_jumping and not is_stumbling:
            trail = pygame.Surface((60, 120), pygame.SRCALPHA)
            pygame.draw.rect(trail, (*hoodie, 25), (10, 10, 38, 50), border_radius=13)
            pygame.draw.rect(trail, (*jeans, 20), (10, 58, 38, 30), border_radius=8)
            scr.blit(trail, (x - 30 - 6, yf - 100))

        # ── ground shadow — 3D ellipse that scales with jump ───────
        sh = pygame.Surface((100, 28), pygame.SRCALPHA)
        sw = max(25, 70 + yo // 4)
        shadow_alpha = max(30, 130 + yo)
        pygame.draw.ellipse(sh, (20, 14, 30, shadow_alpha), (50 - sw // 2, 7, sw, 16))
        scr.blit(sh, (x - 50, y + 1))

        # ═══════════════════════════════════════════════════════════
        # SLIDING POSE
        # ═══════════════════════════════════════════════════════════
        if player.is_sliding:
            # body (flat)
            pygame.draw.rect(scr, hoodie_edge, (x - 30, yf - 33, 60, 32), border_radius=13)
            pygame.draw.rect(scr, hoodie, (x - 28, yf - 35, 56, 30), border_radius=12)
            # highlight strip
            pygame.draw.rect(scr, hoodie_hi, (x - 26, yf - 35, 52, 8), border_radius=6)
            # belt
            pygame.draw.rect(scr, hoodie_d, (x - 28, yf - 12, 56, 8), border_radius=4)
            # hood
            pygame.draw.circle(scr, hoodie_edge, (x, yf - 42), 14)
            pygame.draw.circle(scr, hoodie_d, (x, yf - 41), 13)
            # face peeking out
            pygame.draw.circle(scr, skin, (x + 6, yf - 38), 8)
            pygame.draw.circle(scr, hair, (x - 2, yf - 44), 10)
            # eye
            pygame.draw.circle(scr, (40, 40, 50), (x + 9, yf - 38), 2)

            # legs tucked
            for fx in (-20, 6):
                pygame.draw.rect(scr, jeans_d, (x + fx + 1, yf - 5, 14, 9), border_radius=3)
                pygame.draw.rect(scr, jeans, (x + fx, yf - 6, 14, 8), border_radius=3)
                # shoes with sole
                pygame.draw.ellipse(scr, shoe_sole, (x + fx - 3, yf + 1, 20, 10))
                pygame.draw.ellipse(scr, shoe, (x + fx - 2, yf - 1, 18, 8))

            # slide sparks / dust
            for i in range(4):
                dx = -32 - i * 14 + random.randint(-4, 4)
                dy = random.randint(-10, 4)
                a = max(15, 90 - i * 22)
                sz = max(2, 6 - i)
                ds = pygame.Surface((sz * 2, sz * 2), pygame.SRCALPHA)
                pygame.draw.circle(ds, (210, 190, 170, a), (sz, sz), sz)
                scr.blit(ds, (x + dx, yf - 8 + dy))
            return

        # ═══════════════════════════════════════════════════════════
        # RUNNING / JUMPING POSE  (full 3D character)
        # ═══════════════════════════════════════════════════════════

        # ── LEGS ────────────────────────────────────────────────────
        hip = yf - 36
        if player.is_jumping:
            feet = [(x - 14, yf - 10), (x + 10, yf - 2)]
        else:
            feet = [(x - 9 + int(run * 10), yf - 4 - max(0, int(-run * 6))),
                    (x + 9 - int(run * 10), yf - 4 - max(0, int(run * 6)))]

        for i, (fx_, fy_) in enumerate(feet):
            hip_x = x + (fx_ - x) * 0.3
            # leg outline
            pygame.draw.line(scr, jeans_d, (hip_x + 1, hip + 1), (fx_ + 1, fy_ + 1), 14)
            # leg main
            pygame.draw.line(scr, jeans, (hip_x, hip), (fx_, fy_), 12)
            # knee highlight
            knee_x = (hip_x + fx_) / 2
            knee_y = (hip + fy_) / 2
            pygame.draw.circle(scr, jeans_hi, (int(knee_x - 2), int(knee_y)), 5)
            # shoe with sole and highlight
            pygame.draw.ellipse(scr, shoe_sole, (fx_ - 10, fy_, 22, 11))
            pygame.draw.ellipse(scr, shoe, (fx_ - 9, fy_ - 2, 20, 10))
            pygame.draw.ellipse(scr, (255, 255, 255), (fx_ - 5, fy_ - 2, 10, 5))   # toe highlight
            # shoe stripe
            pygame.draw.line(scr, hoodie, (fx_ - 6, fy_ + 2), (fx_ + 6, fy_ + 2), 2)

        # ── BODY (torso) ───────────────────────────────────────────
        # outline
        pygame.draw.rect(scr, hoodie_edge, (x - 21, yf - 81, 42, 54), border_radius=14)
        # main hoodie body
        pygame.draw.rect(scr, hoodie, (x - 19, yf - 82, 38, 50), border_radius=13)
        # 3D shading — darker right side
        pygame.draw.rect(scr, hoodie_d, (x + 6, yf - 80, 13, 48), border_radius=8)
        # highlight on left shoulder
        hl = pygame.Surface((16, 20), pygame.SRCALPHA)
        pygame.draw.rect(hl, (*hoodie_hi, 100), (0, 0, 16, 20), border_radius=8)
        scr.blit(hl, (x - 18, yf - 80))
        # center zip / seam
        pygame.draw.line(scr, hoodie_d, (x, yf - 72), (x, yf - 44), 2)
        # belt area
        pygame.draw.rect(scr, hoodie_edge, (x - 19, yf - 40, 38, 9), border_radius=4)
        # pocket detail
        pygame.draw.rect(scr, hoodie_d, (x - 14, yf - 56, 12, 10), border_radius=3)
        pygame.draw.rect(scr, hoodie_d, (x + 2, yf - 56, 12, 10), border_radius=3)

        # ── ARMS ───────────────────────────────────────────────────
        if player.is_jumping:
            arms = [((x - 17, yf - 72), (x - 34, yf - 100)),
                    ((x + 17, yf - 72), (x + 34, yf - 100))]
        else:
            sw_ = int(run * 9)
            arms = [((x - 17, yf - 72), (x - 27, yf - 48 + sw_)),
                    ((x + 17, yf - 72), (x + 27, yf - 48 - sw_))]

        for i, (a, b) in enumerate(arms):
            # arm outline
            pygame.draw.line(scr, hoodie_edge, (a[0] + 1, a[1] + 1), (b[0] + 1, b[1] + 1), 12)
            # arm main
            pygame.draw.line(scr, hoodie, a, b, 10)
            # arm highlight
            mid_ax = (a[0] + b[0]) / 2
            mid_ay = (a[1] + b[1]) / 2
            pygame.draw.circle(scr, hoodie_hi, (int(mid_ax - 2), int(mid_ay)), 4)
            # hand
            pygame.draw.circle(scr, skin_d, (b[0] + 1, b[1] + 1), 7)     # shadow
            pygame.draw.circle(scr, skin, b, 6)
            # fingers hint
            if player.is_jumping:
                for fx in (-3, 0, 3):
                    pygame.draw.circle(scr, skin, (b[0] + fx, b[1] - 5), 2)

        # ── HEAD ───────────────────────────────────────────────────
        # hood
        pygame.draw.circle(scr, hoodie_edge, (x, yf - 82), 15)      # outline
        pygame.draw.circle(scr, hoodie_d, (x, yf - 82), 14)
        # ears (peeking out)
        pygame.draw.circle(scr, skin_d, (x - 14, yf - 97), 5)
        pygame.draw.circle(scr, skin, (x - 13, yf - 97), 4)
        pygame.draw.circle(scr, skin_d, (x + 14, yf - 97), 5)
        pygame.draw.circle(scr, skin, (x + 13, yf - 97), 4)
        # hair
        pygame.draw.circle(scr, hair, (x, yf - 98), 14)
        # hair highlight
        pygame.draw.circle(scr, hair_hi, (x - 5, yf - 104), 6)
        # hair texture arc
        pygame.draw.arc(scr, (60, 34, 20), (x - 14, yf - 112, 28, 28), 0.4, math.pi - 0.4, 3)
        # face
        pygame.draw.circle(scr, skin, (x, yf - 93), 10)
        # face shadow on lower cheek
        pygame.draw.circle(scr, skin_d, (x + 3, yf - 90), 6)
        # eyes
        pygame.draw.circle(scr, WHITE, (x - 4, yf - 95), 3)
        pygame.draw.circle(scr, WHITE, (x + 4, yf - 95), 3)
        pygame.draw.circle(scr, (40, 40, 50), (x - 3, yf - 95), 2)
        pygame.draw.circle(scr, (40, 40, 50), (x + 5, yf - 95), 2)
        # eyebrow
        pygame.draw.line(scr, (60, 34, 20), (x - 7, yf - 99), (x - 1, yf - 98), 2)
        pygame.draw.line(scr, (60, 34, 20), (x + 2, yf - 98), (x + 8, yf - 99), 2)
        # mouth
        if player.is_jumping:
            # open mouth (excited)
            pygame.draw.circle(scr, (180, 80, 80), (x, yf - 90), 3)
        else:
            # slight smile
            pygame.draw.arc(scr, (180, 90, 70), (x - 4, yf - 93, 8, 8), 3.5, 6.0, 2)

        # ── JUMP wind effect ───────────────────────────────────────
        if player.is_jumping and yo < -30:
            for i in range(3):
                wy = yf + 10 + i * 18
                ww = 20 + i * 8
                ws = pygame.Surface((ww, 4), pygame.SRCALPHA)
                a = max(15, 60 - abs(yo) // 3 - i * 15)
                pygame.draw.line(ws, (200, 220, 255, a), (0, 2), (ww, 2), 2)
                scr.blit(ws, (x - ww // 2, wy))

    def _draw_police_and_dog(self, engine):
        """Draw Police Inspector and Dog pursuing right behind the runner."""
        if engine.police_z > 1.35:
            return

        pz = engine.police_z
        plane = engine.police_lane

        # ── 1. POLICE INSPECTOR ──────────────────────────────────────
        px, py, ps = self._world2screen(plane - 0.08, pz)
        run_cycle = math.sin(self.t * 15)

        # Ground shadow
        psh_w = max(14, int(65 * ps))
        psh_h = max(4, int(16 * ps))
        psh = pygame.Surface((psh_w, psh_h), pygame.SRCALPHA)
        pygame.draw.ellipse(psh, (20, 14, 30, int(110 * min(1.0, ps))), (0, 0, psh_w, psh_h))
        self.screen.blit(psh, (px - psh_w // 2, py - psh_h // 2))

        # Colors for Inspector
        c_navy = (22, 42, 85)
        c_navy_dark = (14, 28, 58)
        c_gold = (255, 215, 60)
        c_skin = (245, 190, 150)
        c_skin_d = (215, 155, 120)
        c_boots = (25, 25, 30)

        # LEGS (running animation)
        hip_y = py - int(34 * ps)
        leg_w = max(3, int(11 * ps))

        # Left leg & Right leg
        l_foot_y = py - max(0, int(run_cycle * 12 * ps))
        l_foot_x = px - int(12 * ps) + int(run_cycle * 14 * ps)
        r_foot_y = py - max(0, int(-run_cycle * 12 * ps))
        r_foot_x = px + int(12 * ps) - int(run_cycle * 14 * ps)

        # Draw trouser legs
        pygame.draw.line(self.screen, c_navy_dark, (px - int(6 * ps), hip_y), (l_foot_x, l_foot_y), leg_w)
        pygame.draw.line(self.screen, c_navy, (px + int(6 * ps), hip_y), (r_foot_x, r_foot_y), leg_w)

        # Boots
        boot_w = max(4, int(16 * ps))
        boot_h = max(3, int(9 * ps))
        pygame.draw.ellipse(self.screen, c_boots, (l_foot_x - boot_w // 2, l_foot_y - boot_h // 2, boot_w, boot_h))
        pygame.draw.ellipse(self.screen, c_boots, (r_foot_x - boot_w // 2, r_foot_y - boot_h // 2, boot_w, boot_h))

        # TORSO (Inspector's Jacket)
        torso_w = max(12, int(42 * ps))
        torso_h = max(14, int(48 * ps))
        torso_y = hip_y - torso_h + int(4 * ps)
        torso_rect = pygame.Rect(px - torso_w // 2, torso_y, torso_w, torso_h)
        pygame.draw.rect(self.screen, c_navy_dark, torso_rect, border_radius=max(2, int(8 * ps)))
        pygame.draw.rect(self.screen, c_navy, torso_rect.inflate(-max(1, int(2 * ps)), -max(1, int(2 * ps))), border_radius=max(2, int(8 * ps)))

        # Gold buttons down coat
        if ps > 0.35:
            for b_idx in range(3):
                by = torso_y + int((12 + b_idx * 12) * ps)
                pygame.draw.circle(self.screen, c_gold, (px, by), max(1, int(2.5 * ps)))

        # Police Belt
        belt_h = max(3, int(8 * ps))
        belt_y = torso_rect.bottom - belt_h - int(2 * ps)
        pygame.draw.rect(self.screen, (20, 20, 22), (px - torso_w // 2, belt_y, torso_w, belt_h))
        pygame.draw.rect(self.screen, c_gold, (px - int(5 * ps), belt_y - 1, int(10 * ps), belt_h + 2), border_radius=1)

        # ARMS
        arm_w = max(2, int(8 * ps))
        l_hand = (px - int(22 * ps), torso_y + int(24 * ps) + int(run_cycle * 10 * ps))
        r_hand = (px + int(24 * ps), torso_y + int(10 * ps) - int(run_cycle * 8 * ps))

        # Left arm
        pygame.draw.line(self.screen, c_navy, (px - int(16 * ps), torso_y + int(8 * ps)), l_hand, arm_w)
        pygame.draw.circle(self.screen, c_skin, l_hand, max(2, int(4 * ps)))

        # Right arm
        pygame.draw.line(self.screen, c_navy, (px + int(16 * ps), torso_y + int(8 * ps)), r_hand, arm_w)
        pygame.draw.circle(self.screen, c_skin, r_hand, max(2, int(4 * ps)))

        # Police Baton in right hand
        baton_len = int(28 * ps)
        baton_end = (r_hand[0] + int(baton_len * 0.7), r_hand[1] - int(baton_len * 0.7))
        pygame.draw.line(self.screen, (35, 30, 25), r_hand, baton_end, max(2, int(4 * ps)))
        pygame.draw.circle(self.screen, (60, 50, 45), baton_end, max(1, int(2.5 * ps)))

        # HEAD & CAP
        head_r = max(4, int(14 * ps))
        head_y = torso_y - head_r + int(2 * ps)
        pygame.draw.circle(self.screen, c_skin, (px, head_y), head_r)

        # Mustache & Features
        if ps > 0.35:
            pygame.draw.circle(self.screen, (30, 30, 40), (px - int(4 * ps), head_y - int(2 * ps)), max(1, int(2 * ps)))
            pygame.draw.circle(self.screen, (30, 30, 40), (px + int(4 * ps), head_y - int(2 * ps)), max(1, int(2 * ps)))
            pygame.draw.line(self.screen, (60, 40, 25), (px - int(7 * ps), head_y - int(6 * ps)), (px - int(2 * ps), head_y - int(4 * ps)), max(1, int(2 * ps)))
            pygame.draw.line(self.screen, (60, 40, 25), (px + int(2 * ps), head_y - int(4 * ps)), (px + int(7 * ps), head_y - int(6 * ps)), max(1, int(2 * ps)))
            m_pts = [(px - int(8 * ps), head_y + int(5 * ps)),
                     (px, head_y + int(2 * ps)),
                     (px + int(8 * ps), head_y + int(5 * ps)),
                     (px, head_y + int(8 * ps))]
            pygame.draw.polygon(self.screen, (70, 45, 30), m_pts)

        # Police Cap
        cap_w = max(10, int(36 * ps))
        cap_h = max(4, int(14 * ps))
        cap_y = head_y - head_r - int(2 * ps)
        cap_rect = pygame.Rect(px - cap_w // 2, cap_y, cap_w, cap_h)
        pygame.draw.ellipse(self.screen, c_navy, cap_rect)
        visor_rect = pygame.Rect(px - cap_w // 2 - int(2 * ps), cap_y + cap_h - int(4 * ps), cap_w + int(4 * ps), max(3, int(7 * ps)))
        pygame.draw.ellipse(self.screen, (20, 20, 25), visor_rect)
        pygame.draw.polygon(self.screen, c_gold, [(px, cap_y + 1), (px + int(4 * ps), cap_y + int(6 * ps)), (px, cap_y + int(10 * ps)), (px - int(4 * ps), cap_y + int(6 * ps))])

        # ── 2. POLICE PITBULL / DOG ──────────────────────────────────
        dx, dy, ds = self._world2screen(plane + 0.16, pz + 0.015)
        drun = math.sin(self.t * 18 + 0.6)

        dsh_w = max(12, int(46 * ds))
        dsh_h = max(4, int(14 * ds))
        dsh = pygame.Surface((dsh_w, dsh_h), pygame.SRCALPHA)
        pygame.draw.ellipse(dsh, (20, 14, 30, int(110 * min(1.0, ds))), (0, 0, dsh_w, dsh_h))
        self.screen.blit(dsh, (dx - dsh_w // 2, dy - dsh_h // 2))

        c_dog = (185, 128, 75)
        c_dog_dark = (145, 95, 50)
        c_dog_belly = (220, 175, 130)
        c_collar = (220, 40, 45)

        db_w = max(10, int(36 * ds))
        db_h = max(8, int(24 * ds))
        db_y = dy - db_h - max(3, int(8 * ds))

        d_leg_w = max(2, int(5 * ds))

        fl_y = dy - max(0, int(drun * 8 * ds))
        fl_x = dx + int(10 * ds) + int(drun * 8 * ds)
        pygame.draw.line(self.screen, c_dog_dark, (dx + int(8 * ds), db_y + db_h - int(4 * ds)), (fl_x, fl_y), d_leg_w)
        pygame.draw.circle(self.screen, c_dog_dark, (fl_x, fl_y), max(1, int(2.5 * ds)))

        bl_y = dy - max(0, int(-drun * 8 * ds))
        bl_x = dx - int(10 * ds) - int(drun * 8 * ds)
        pygame.draw.line(self.screen, c_dog, (dx - int(8 * ds), db_y + db_h - int(4 * ds)), (bl_x, bl_y), d_leg_w)
        pygame.draw.circle(self.screen, c_dog, (bl_x, bl_y), max(1, int(2.5 * ds)))

        body_rect = pygame.Rect(dx - db_w // 2, db_y, db_w, db_h)
        pygame.draw.ellipse(self.screen, c_dog, body_rect)
        pygame.draw.ellipse(self.screen, c_dog_belly, (dx - db_w // 2 + 2, db_y + int(db_h * 0.4), db_w - 4, int(db_h * 0.5)))

        dh_r = max(4, int(13 * ds))
        dh_x = dx + int(14 * ds)
        dh_y = db_y + int(2 * ds)
        pygame.draw.circle(self.screen, c_dog, (dh_x, dh_y), dh_r)

        collar_rect = pygame.Rect(dh_x - int(7 * ds), dh_y + int(4 * ds), int(10 * ds), max(3, int(6 * ds)))
        pygame.draw.rect(self.screen, c_collar, collar_rect, border_radius=1)
        if ds > 0.35:
            for s_idx in range(3):
                pygame.draw.circle(self.screen, WHITE, (collar_rect.x + int((2 + s_idx * 3) * ds), collar_rect.centery), max(1, int(1.5 * ds)))

        snout_w = max(4, int(11 * ds))
        snout_h = max(3, int(9 * ds))
        pygame.draw.ellipse(self.screen, c_dog_dark, (dh_x + int(3 * ds), dh_y - int(2 * ds), snout_w, snout_h))
        pygame.draw.circle(self.screen, (25, 20, 20), (dh_x + int(12 * ds), dh_y), max(1, int(2 * ds)))
        pygame.draw.circle(self.screen, (25, 20, 20), (dh_x + int(4 * ds), dh_y - int(5 * ds)), max(1, int(2 * ds)))
        ear_pts = [(dh_x - int(2 * ds), dh_y - dh_r),
                   (dh_x + int(5 * ds), dh_y - dh_r - int(5 * ds)),
                   (dh_x + int(8 * ds), dh_y - dh_r + int(3 * ds))]
        pygame.draw.polygon(self.screen, c_dog_dark, ear_pts)

        tongue_w = max(2, int(6 * ds))
        tongue_h = max(3, int(9 * ds))
        pygame.draw.ellipse(self.screen, (255, 120, 150), (dh_x + int(7 * ds), dh_y + int(4 * ds), tongue_w, tongue_h))

        tail_ang = math.sin(self.t * 22) * 0.4
        t_base = (dx - db_w // 2, db_y + int(4 * ds))
        t_tip = (t_base[0] - int(14 * ds), t_base[1] - int(10 * ds) + int(tail_ang * 12 * ds))
        pygame.draw.line(self.screen, c_dog, t_base, t_tip, max(2, int(4 * ds)))

    def _draw_lives(self, S, engine):
        if engine.police_active:
            pulse = abs(math.sin(self.t * 6))
            badge_w = 200
            bg = pygame.Surface((badge_w, 36), pygame.SRCALPHA)
            bg_col = (int(190 + 50 * pulse), int(35 * (1 - pulse)), int(45 * (1 - pulse)), 230)
            pygame.draw.rect(bg, bg_col, (0, 0, badge_w, 36), border_radius=14)
            pygame.draw.rect(bg, (255, 230, 80), (0, 0, badge_w, 36), 2, border_radius=14)
            self.screen.blit(bg, (S.x + 10, S.bottom - 46))
            txt = self.f_small.render(f"🚨 PURSUIT: {max(0.0, engine.police_timer):.1f}s", True, WHITE)
            self.screen.blit(txt, (S.x + 10 + (badge_w - txt.get_width()) // 2, S.bottom - 38))
        else:
            badge_w = 140
            bg = pygame.Surface((badge_w, 34), pygame.SRCALPHA)
            pygame.draw.rect(bg, (*PANEL_DARK, 215), (0, 0, badge_w, 34), border_radius=14)
            pygame.draw.rect(bg, (255, 255, 255, 255), (0, 0, badge_w, 34), 2, border_radius=14)
            self.screen.blit(bg, (S.x + 10, S.bottom - 44))
            label = self.f_tiny.render("LIFE", True, WHITE)
            self.screen.blit(label, (S.x + 20, S.bottom - 33))
            cx = S.x + 85
            cy = S.bottom - 30
            color = (242, 65, 75) if engine.lives > 0 else (150, 190, 210)
            pygame.draw.circle(self.screen, color, (cx - 4, cy), 6)
            pygame.draw.circle(self.screen, color, (cx + 4, cy), 6)
            pygame.draw.polygon(self.screen, color, [(cx - 10, cy + 2), (cx, cy + 12), (cx + 10, cy + 2)])

    # ────────────────────────────────────────────────────────────────
    # overlays
    # ────────────────────────────────────────────────────────────────
    def draw_game_over(self, score, high_score, cause_of_death=""):
        overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        overlay.fill((225, 243, 250, 120))
        self.screen.blit(overlay, (0, 0))
        rect = pygame.Rect((self.W - 600) // 2, (self.H - 350) // 2, 600, 350)
        _panel(self.screen, rect, 28)
        cx = rect.centerx
        for dx, dy, col in ((4, 5, (130, 28, 36)), (0, 0, WHITE)):
            title = self.f_go.render("RUN OVER!", True, col)
            self.screen.blit(title, (cx - title.get_width() // 2 + dx, rect.y + 22 + dy))

        # Cause of death banner
        if cause_of_death:
            cod_lbl = self.f_small.render(cause_of_death, True, (255, 230, 100))
            cod_w = cod_lbl.get_width() + 32
            cod_rect = pygame.Rect(cx - cod_w // 2, rect.y + 80, cod_w, 30)
            pygame.draw.rect(self.screen, (170, 32, 42), cod_rect, border_radius=15)
            pygame.draw.rect(self.screen, WHITE, cod_rect, 2, border_radius=15)
            self.screen.blit(cod_lbl, (cx - cod_lbl.get_width() // 2, cod_rect.centery - cod_lbl.get_height() // 2))

        _draw_text(self.screen, self.f_large, f"Score: {score:,}", GOLD_BRIGHT,
                   (cx, rect.y + 124), anchor="midtop")
        _draw_text(self.screen, self.f_medium, f"Best: {high_score:,}", WHITE,
                   (cx, rect.y + 172), anchor="midtop")

        pulse = abs(math.sin(self.t * 3))
        label = self.f_small.render("RAISE BOTH HANDS OR PRESS SPACE TO RESTART", True, INK)
        pill = pygame.Rect(0, 0, label.get_width() + 44, 44)
        pill.center = (cx, rect.y + 250)
        pygame.draw.rect(self.screen, WHITE, pill, border_radius=22)
        pygame.draw.rect(self.screen, (int(255 - 40 * pulse), int(200 - 10 * pulse), int(40 + 20 * pulse)),
                         pill, 3, border_radius=22)
        self.screen.blit(label, (pill.centerx - label.get_width() // 2,
                                 pill.centery - label.get_height() // 2))
        _draw_text(self.screen, self.f_tiny, "ESC  \u2022  QUIT", SOFT_WHITE,
                   (cx, rect.y + 296), anchor="midtop")

    def draw_calibrating(self, progress: float):
        cr = self.cam_rect
        width = cr.w - 80
        cx, cy = cr.centerx, cr.y + 62
        bgp = pygame.Surface((width + 40, 74), pygame.SRCALPHA)
        pygame.draw.rect(bgp, (*PANEL_DARK, 225), (0, 0, width + 40, 74), border_radius=16)
        pygame.draw.rect(bgp, (255, 255, 255, 255), (0, 0, width + 40, 74), 2, border_radius=16)
        self.screen.blit(bgp, (cx - (width + 40) // 2, cy - 36))
        txt = self.f_medium.render("GET READY! STAND STILL", True, WHITE)
        self.screen.blit(txt, (cx - txt.get_width() // 2, cy - 28))
        bar = pygame.Rect(cx - width // 2, cy + 8, width, 14)
        trk = pygame.Surface(bar.size, pygame.SRCALPHA)
        pygame.draw.rect(trk, (255, 255, 255, 80), (0, 0, *bar.size), border_radius=7)
        self.screen.blit(trk, bar.topleft)
        fill = pygame.Rect(bar.x, bar.y, int(width * max(0, min(1, progress))), bar.h)
        if fill.w:
            pygame.draw.rect(self.screen, WHITE, fill, border_radius=7)
