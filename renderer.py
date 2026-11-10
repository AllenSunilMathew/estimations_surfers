"""Sky-card "AI Pose Controlled Game" renderer for Treasure Runner.

Uses the same light-blue card look as the preloader / main menu (see theme.py):
white chunky text with soft blue shadows, blue glass panels with white borders,
gold coins and the red-hoodie runner. Everything is drawn with Pygame
primitives (no image assets). Static chrome and scenery are cached once, so
per-frame cost is small.

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
from theme import (font as _theme_font, build_card_bg, text as _draw_text, panel as _panel,
                   header_band as _header_band, coin as _coin, star as _star,
                   runner_icon as _runner_icon_art, soft_shadow_ellipse)


class _Particle:
    """Small coin-collection sparkle."""
    __slots__ = ("x", "y", "vx", "vy", "life", "size", "color")

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.vx = random.uniform(-100, 100)
        self.vy = random.uniform(-180, -40)
        self.life = 1.0
        self.size = random.randint(2, 5)
        self.color = random.choice([GOLD, (255, 230, 90), WHITE])

    def tick(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 260 * dt
        self.life -= dt * 1.8
        return self.life > 0


class Renderer:
    N_TIES = 22
    CARD_MARGIN = 14          # off-white page border around the blue card
    SIDE = 28                 # panel margin from the window edge

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.W, self.H = WINDOW_WIDTH, WINDOW_HEIGHT

        pygame.font.init()
        self.f_title = self._font("Arial", 40, True, True)
        self.f_sub = self._font("Arial", 22, True, True)
        self.f_panel = self._font("Arial", 26, True, True)
        self.f_hdr = self._font("Arial", 19, True, True)
        self.f_score_lbl = self._font("Arial", 20, True, True)
        self.f_score = self._font("Arial", 40, True)
        self.f_large = self._font("Arial", 30, True)
        self.f_medium = self._font("Arial", 21, True)
        self.f_gesture = self._font("Arial", 24, True, True)
        self.f_small = self._font("Arial", 16, True)
        self.f_tiny = self._font("Arial", 13, True)
        self.f_go = self._font("Arial", 56, True, True)

        # ── layout ──────────────────────────────────────────────────
        margin, gap = self.SIDE, 14
        panel_y = HUD_HEIGHT + 16
        panel_h = self.H - panel_y - 36
        total_w = self.W - 2 * margin - gap
        pose_w = int(total_w * POSE_PANEL_RATIO)
        game_w = total_w - pose_w
        self.pose_rect = pygame.Rect(margin, panel_y, pose_w, panel_h)
        self.game_rect = pygame.Rect(margin + pose_w + gap, panel_y, game_w, panel_h)
        self.score_rect = pygame.Rect(self.W - margin - 250, 26, 250, 76)
        pr = self.pose_rect
        self.cam_rect = pygame.Rect(pr.x + 10, pr.y + 58, pr.w - 20, pr.h - 68)
        self.scene_rect = self.game_rect.inflate(-20, -20)

        # scene geometry (local = relative to scene_rect)
        S = self.scene_rect
        self.vp_lx = S.w / 2
        self.vp_ly = int(S.h * 0.38)
        self.road_w_top = 54
        self.road_w_bot = int(S.w * 0.62)
        # screen-space versions
        self.vp_x = S.x + self.vp_lx
        self.vp_y = S.y + self.vp_ly
        self.road_bottom = S.bottom

        self.t = 0.0
        self._dt = 0.0
        self.track_phase = 0.0
        self.particles = []

        # cached layers
        self._chrome = self._build_chrome()
        self._back = self._build_back()
        self._front = self._build_front()
        self._scene_mask = self._corner_mask(S.size, 16)
        self._cam_mask = self._corner_mask(self.cam_rect.size, 12)
        self._game_header = self._build_game_header()

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
        z = max(0.0, min(1.25, z))
        p = z ** 1.5
        y = self.vp_y + (self.road_bottom - self.vp_y) * p
        width = self.road_w_top + (self.road_w_bot - self.road_w_top) * p
        x = self.vp_x - width / 2 + width * (lane + 0.5) / 3
        return int(x), int(y), p

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

        # header: runner + coins on the left, title centred, score pill on the right
        _runner_icon_art(surf, 96, 66, 1.2)
        _coin(surf, 168, 62, 20)
        _coin(surf, 214, 78, 12)

        cx = W // 2
        for dx, dy, col in ((4, 5, (30, 100, 140)), (2, 3, INK_SHADOW), (0, 0, WHITE)):
            im = self.f_title.render("TREASURE  RUNNER", True, col)
            surf.blit(im, (cx - im.get_width() // 2 + dx, 24 + dy))
        sub = self.f_sub.render("Move  \u2022  Play  \u2022  Have Fun!", True, SOFT_WHITE)
        shs = self.f_sub.render("Move  \u2022  Play  \u2022  Have Fun!", True, INK_SHADOW)
        surf.blit(shs, (cx - sub.get_width() // 2 + 2, 70 + 2))
        surf.blit(sub, (cx - sub.get_width() // 2, 70))

        # score pill frame
        pill = self.score_rect
        _panel(surf, pill, radius=24, fill=PANEL_DARK)
        self._trophy(surf, pill.x + 52, pill.centery - 2)

        # panels
        _panel(surf, self.pose_rect)
        _panel(surf, self.game_rect)

        # pose panel header
        pr = self.pose_rect
        _header_band(surf, pr, 48)
        self._camera_icon(surf, pr.x + 20, pr.y + 12)
        self._text(self.f_panel, "Your Pose", WHITE, (pr.x + 78, pr.y + 11), True, surf)
        return surf

    def _build_game_header(self):
        w, h = 500, 56
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.polygon(surf, (*PANEL_DARK, 238), [(0, 0), (w, 0), (w - 70, h), (0, h)])
        self._gamepad_icon(surf, 14, 10)
        self._text(self.f_panel, "Game Screen", WHITE, (78, 11), True, surf)
        return surf

    # ────────────────────────────────────────────────────────────────
    # cached scene layers
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

        # sky
        top, bot = (58, 138, 235), (178, 220, 250)
        for y in range(h):
            q = min(1.0, y / max(1, hor + 30))
            pygame.draw.line(surf, tuple(int(top[i] + (bot[i] - top[i]) * q) for i in range(3)), (0, y), (w, y))
        for cx, cy, s in ((120, 70, 1.0), (330, 50, 0.8), (500, 95, 1.1)):
            for ox, oy, r in ((0, 0, 16), (20, -9, 22), (44, 0, 16), (22, 6, 18)):
                pygame.draw.circle(surf, (255, 255, 255), (int(cx + ox * s), int(cy + oy * s)), int(r * s))

        # buildings (far → near, both sides)
        pal = [(236, 120, 150), (250, 172, 120), (110, 196, 200), (172, 132, 232),
               (242, 204, 122), (112, 162, 236), (232, 142, 192)]
        base_y = hor + 4
        for side in (-1, 1):
            for k in range(8):
                f = k / 7
                d_in = 34 + f * (w / 2 - 20)
                bw = int(42 + f * 120)
                bh = int(62 + f * 270)
                x0 = int(self.vp_lx + side * d_in) if side > 0 else int(self.vp_lx - d_in - bw)
                col = pal[(k * 2 + (0 if side < 0 else 3)) % len(pal)]
                pygame.draw.rect(surf, col, (x0, base_y - bh, bw, bh))
                dark = tuple(int(c * 0.78) for c in col)
                stripe = max(6, bw // 6)
                sx = x0 + bw - stripe if side < 0 else x0
                pygame.draw.rect(surf, dark, (sx, base_y - bh, stripe, bh))
                pygame.draw.rect(surf, tuple(int(c * 0.65) for c in col), (x0 - 3, base_y - bh - 6, bw + 6, 8))
                wstep = max(10, int(11 + f * 6))
                for wx in range(x0 + 6, x0 + bw - 8, wstep):
                    for wy in range(base_y - bh + 14, base_y - 12, wstep + 5):
                        c = (255, 236, 170) if rng.random() < 0.75 else (90, 110, 160)
                        pygame.draw.rect(surf, c, (wx, wy, max(3, wstep // 2 - 1), max(4, wstep // 2 + 1)))
                if k in (3, 5):  # billboard
                    pygame.draw.rect(surf, (60, 200, 210), (x0 + 6, base_y - bh + 24, bw - 12, 34), border_radius=4)
                    pygame.draw.rect(surf, WHITE, (x0 + 6, base_y - bh + 24, bw - 12, 34), 2, border_radius=4)

        # distant station gate / tunnel
        cx = int(self.vp_lx)
        pygame.draw.rect(surf, (124, 138, 176), (cx - 96, hor - 64, 192, 70))
        pygame.draw.rect(surf, (158, 172, 208), (cx - 104, hor - 76, 208, 14))
        pygame.draw.rect(surf, (30, 42, 108), (cx - 58, hor - 48, 116, 54),
                         border_top_left_radius=58, border_top_right_radius=58)
        pygame.draw.rect(surf, (70, 104, 210), (cx - 36, hor - 34, 72, 40),
                         border_top_left_radius=36, border_top_right_radius=36)
        pygame.draw.rect(surf, (255, 170, 60), (cx - 5, hor - 14, 10, 12), border_radius=2)
        for lx in (-80, -66, 66, 80):
            pygame.draw.circle(surf, (255, 225, 140), (cx + lx, hor - 52), 3)

        # side ground
        wt, wb = self.road_w_top, self.road_w_bot
        pygame.draw.polygon(surf, (128, 104, 108), [(0, hor), (cx - wt / 2, hor), (cx - wb / 2, h), (0, h)])
        pygame.draw.polygon(surf, (138, 112, 110), [(cx + wt / 2, hor), (w, hor), (w, h), (cx + wb / 2, h)])

        # palms
        for fx, by, tx, ty, sz in ((0.22, 0.46, 0.24, 0.10, 1.0), (0.80, 0.50, 0.77, 0.05, 1.15),
                                   (0.34, 0.40, 0.35, 0.30, 0.55), (0.68, 0.40, 0.67, 0.29, 0.55)):
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

    def draw_hud(self, score, fps, gesture, confidence):
        px, py = self.score_rect.x, self.score_rect.y
        self._text(self.f_score_lbl, "Score", SOFT_WHITE, (px + 100, py + 7), True)
        self._text(self.f_score, f"{score:,}", GOLD_BRIGHT, (px + 100, py + 28), True)

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

    def draw_pose_panel(self, frame, gesture, confidence, fps):
        r, cr = self.pose_rect, self.cam_rect

        # live FPS | Confidence (right-aligned in header)
        fps_txt = self.f_hdr.render(f"FPS: {int(fps)}", True, (120, 255, 160))
        lbl = self.f_hdr.render("Confidence: ", True, WHITE)
        val = self.f_hdr.render(f"{int(confidence * 100)}%", True, GOLD_BRIGHT)
        total = 12 + 8 + fps_txt.get_width() + 36 + lbl.get_width() + val.get_width()
        x = r.right - 24 - total
        cy = r.y + 28
        pygame.draw.circle(self.screen, (90, 240, 130), (x + 6, cy), 6)
        self.screen.blit(fps_txt, (x + 20, cy - fps_txt.get_height() // 2))
        dx = x + 20 + fps_txt.get_width() + 18
        pygame.draw.line(self.screen, EDGE_SOFT, (dx, cy - 14), (dx, cy + 14), 2)
        self.screen.blit(lbl, (dx + 18, cy - lbl.get_height() // 2))
        self.screen.blit(val, (dx + 18 + lbl.get_width(), cy - val.get_height() // 2))

        # camera feed
        pygame.draw.rect(self.screen, (16, 62, 94), cr, border_radius=12)
        if frame is not None:
            self.screen.blit(self._frame_surface(frame, cr.size), cr.topleft)
        else:
            t = self.f_medium.render("WAITING FOR CAMERA", True, (255, 190, 190))
            self.screen.blit(t, (cr.centerx - t.get_width() // 2, cr.centery - 10))
        self.screen.blit(self._cam_mask, cr.topleft)
        pygame.draw.rect(self.screen, WHITE, cr, 2, border_radius=12)

        # gesture pill
        pill = pygame.Rect(cr.x + 14, cr.bottom - 68, 340, 54)
        ps = pygame.Surface(pill.size, pygame.SRCALPHA)
        pygame.draw.rect(ps, (*PANEL_DARK, 228), (0, 0, *pill.size), border_radius=18)
        pygame.draw.rect(ps, (255, 255, 255, 255), (0, 0, *pill.size), 2, border_radius=18)
        self.screen.blit(ps, pill.topleft)
        _runner_icon_art(self.screen, pill.x + 36, pill.centery + 2, 0.55)
        lab = self.f_gesture.render("Gesture:", True, WHITE)
        self.screen.blit(lab, (pill.x + 76, pill.centery - lab.get_height() // 2))
        gcol = {Gesture.NEUTRAL: SOFT_WHITE}.get(gesture, GOLD_BRIGHT)
        val = self.f_gesture.render(gesture, True, gcol)
        self.screen.blit(val, (pill.x + 76 + lab.get_width() + 12, pill.centery - val.get_height() // 2))

    # ────────────────────────────────────────────────────────────────
    # game scene
    # ────────────────────────────────────────────────────────────────
    def draw_game_panel(self, engine):
        S = self.scene_rect
        if not engine.game_over:
            self.track_phase = (self.track_phase + self._dt * engine.speed * 0.14) % (1.0 / self.N_TIES)

        prev_clip = self.screen.get_clip()
        self.screen.set_clip(S)
        self.screen.blit(self._back, S.topleft)
        self._draw_track()
        self.screen.blit(self._front, S.topleft)

        for coin in sorted(engine.coins, key=lambda c: c.z):
            if not coin.collected:
                self._draw_coin(coin)
        for obstacle in sorted(engine.obstacles, key=lambda o: o.z):
            self._draw_obstacle(obstacle)
        self._draw_player(engine.player)

        for lane, z in engine.coin_collect_events:
            sx, sy, _ = self._world2screen(lane, z)
            for _ in range(12):
                self.particles.append(_Particle(sx, sy - 20))
        for p in self.particles:
            pygame.draw.circle(self.screen, p.color, (int(p.x), int(p.y)), max(1, int(p.size * p.life)))

        self._draw_lives(S, engine.lives)
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

    def _draw_track(self):
        S = self.scene_rect
        vx, vy, bot = self.vp_x, self.vp_y, S.bottom
        wt, wb = self.road_w_top, self.road_w_bot
        pygame.draw.polygon(self.screen, (170, 104, 72),
                            [(vx - wt / 2, vy), (vx + wt / 2, vy), (vx + wb / 2, bot), (vx - wb / 2, bot)])

        n = self.N_TIES
        for i in range(n):
            z = (i / n + self.track_phase) % 1.0
            p = z ** 1.5
            y = vy + (bot - vy) * p
            width = wt + (wb - wt) * p
            th = max(2, int(2 + 15 * p))
            pygame.draw.rect(self.screen, (112, 64, 42), (int(vx - width / 2), int(y), int(width), th))
            pygame.draw.rect(self.screen, (150, 92, 62), (int(vx - width / 2), int(y), int(width), max(1, th // 4)))

        for lane in range(3):
            for off in (-0.27, 0.27):
                f = (lane + 0.5 + off) / 3
                xt = vx - wt / 2 + wt * f
                xb = vx - wb / 2 + wb * f
                pygame.draw.polygon(self.screen, (204, 211, 226),
                                    [(xt - 1, vy), (xt + 1, vy), (xb + 4, bot), (xb - 4, bot)])
                pygame.draw.line(self.screen, (120, 130, 150), (xt - 1, vy), (xb - 4, bot), 1)

    def _draw_obstacle(self, obs):
        if obs.z < 0.06 or obs.z > 1.1:
            return
        x, y, scale = self._world2screen(obs.lane, obs.z)
        if obs.type == "barrier":
            w, h = max(10, int(104 * scale)), max(8, int(64 * scale))
            pw = max(3, int(13 * scale))
            lamp = max(3, int(9 * scale))
            for px in (x - w // 2, x + w // 2 - pw):
                pygame.draw.rect(self.screen, (205, 48, 52), (px, y - h, pw, h), border_radius=2)
                pygame.draw.circle(self.screen, (60, 66, 82), (px + pw // 2, y - h), lamp + 1)
                pygame.draw.circle(self.screen, (160, 165, 178), (px + pw // 2, y - h), lamp)
                pygame.draw.rect(self.screen, (60, 66, 82), (px - 2, y - 3, pw + 4, 4))
            plank = pygame.Rect(x - w // 2, y - int(h * 0.82), w, max(4, int(h * 0.36)))
            pygame.draw.rect(self.screen, (248, 248, 250), plank)
            sw = max(3, int(14 * scale))
            for sx in range(plank.x, plank.right, sw * 2):
                pygame.draw.rect(self.screen, (212, 50, 54), (sx, plank.y, min(sw, plank.right - sx), plank.h))
            pygame.draw.rect(self.screen, (80, 40, 40), plank, 1)
        else:
            w = max(10, int(112 * scale))
            bh = max(4, int(18 * scale))
            bar_y = y - int(62 * scale)
            pw = max(3, int(8 * scale))
            for px in (x - w // 2, x + w // 2 - pw):
                pygame.draw.rect(self.screen, (62, 70, 90), (px, bar_y, pw, y - bar_y))
            bar = pygame.Rect(x - w // 2, bar_y, w, bh)
            pygame.draw.rect(self.screen, (255, 200, 40), bar)
            sw = max(3, int(12 * scale))
            for sx in range(bar.x, bar.right, sw * 2):
                pygame.draw.polygon(self.screen, (40, 40, 50),
                                    [(sx, bar.bottom), (sx + sw, bar.bottom),
                                     (min(sx + sw + bh, bar.right), bar.y), (min(sx + bh, bar.right), bar.y)])
            pygame.draw.rect(self.screen, (60, 50, 30), bar, 1)

    def _draw_coin(self, coin):
        if coin.z < 0.06 or coin.z > 1.1:
            return
        x, y, scale = self._world2screen(coin.lane, coin.z)
        y -= int(34 * scale)
        r = max(4, int(27 * scale))
        sq = 0.45 + 0.55 * abs(math.cos(math.radians(coin.rotation)))
        rx = max(2, int(r * sq))
        pygame.draw.ellipse(self.screen, (185, 100, 12), (x - rx - 3, y - r - 3, 2 * rx + 6, 2 * r + 6))
        pygame.draw.ellipse(self.screen, (255, 198, 40), (x - rx, y - r, 2 * rx, 2 * r))
        if r > 7:
            pygame.draw.ellipse(self.screen, (255, 230, 120),
                                (x - rx + 4, y - r + 4, 2 * rx - 8, 2 * r - 8), 2)
            pygame.draw.polygon(self.screen, (255, 140, 20), _star(x, y, r * 0.55, r * 0.24, sq))

    def _draw_player(self, player):
        S = self.scene_rect
        lane = player.visual_lane()
        x, y, _ = self._world2screen(lane, 0.92)
        yo = int(player.y_offset)
        yf = y + yo
        run = math.sin(self.t * 14)
        hoodie, hoodie_d = (214, 50, 52), (160, 32, 40)
        jeans, shoe = (52, 98, 172), (246, 246, 246)
        hair, skin = (92, 54, 34), (255, 198, 152)

        sh = pygame.Surface((70, 18), pygame.SRCALPHA)
        sw = max(30, 60 + yo // 3)
        pygame.draw.ellipse(sh, (20, 14, 30, 110), (35 - sw // 2, 3, sw, 12))
        self.screen.blit(sh, (x - 35, y + 4))

        if player.is_sliding:
            pygame.draw.rect(self.screen, hoodie, (x - 28, yf - 34, 56, 30), border_radius=12)
            pygame.draw.rect(self.screen, hoodie_d, (x - 28, yf - 12, 56, 8), border_radius=4)
            pygame.draw.circle(self.screen, hair, (x, yf - 40), 13)
            pygame.draw.circle(self.screen, hoodie_d, (x, yf - 30), 10)
            for fx in (-20, 6):
                pygame.draw.rect(self.screen, jeans, (x + fx, yf - 6, 14, 8), border_radius=3)
                pygame.draw.ellipse(self.screen, shoe, (x + fx - 2, yf, 18, 8))
            return

        hip = yf - 36
        if player.is_jumping:
            feet = [(x - 14, yf - 10), (x + 10, yf - 2)]
        else:
            feet = [(x - 9 + int(run * 10), yf - 4 - max(0, int(-run * 6))),
                    (x + 9 - int(run * 10), yf - 4 - max(0, int(run * 6)))]
        for fx_, fy_ in feet:
            pygame.draw.line(self.screen, jeans, (x + (fx_ - x) * 0.3, hip), (fx_, fy_), 12)
            pygame.draw.ellipse(self.screen, shoe, (fx_ - 9, fy_ - 2, 20, 10))

        pygame.draw.rect(self.screen, hoodie_d, (x - 19, yf - 80, 38, 52), border_radius=13)
        pygame.draw.rect(self.screen, hoodie, (x - 19, yf - 82, 38, 50), border_radius=13)
        pygame.draw.rect(self.screen, hoodie_d, (x - 19, yf - 40, 38, 9), border_radius=4)
        pygame.draw.line(self.screen, hoodie_d, (x, yf - 70), (x, yf - 44), 2)

        if player.is_jumping:
            arms = [((x - 17, yf - 72), (x - 34, yf - 100)), ((x + 17, yf - 72), (x + 34, yf - 100))]
        else:
            sw_ = int(run * 9)
            arms = [((x - 17, yf - 72), (x - 27, yf - 48 + sw_)), ((x + 17, yf - 72), (x + 27, yf - 48 - sw_))]
        for a, b in arms:
            pygame.draw.line(self.screen, hoodie, a, b, 10)
            pygame.draw.circle(self.screen, skin, b, 6)

        pygame.draw.circle(self.screen, hoodie_d, (x, yf - 82), 14)
        pygame.draw.circle(self.screen, skin, (x - 13, yf - 97), 4)
        pygame.draw.circle(self.screen, skin, (x + 13, yf - 97), 4)
        pygame.draw.circle(self.screen, hair, (x, yf - 98), 14)
        pygame.draw.arc(self.screen, (60, 34, 20), (x - 14, yf - 112, 28, 28), 0.4, math.pi - 0.4, 3)

    def _draw_lives(self, S, lives):
        bg = pygame.Surface((150, 34), pygame.SRCALPHA)
        pygame.draw.rect(bg, (*PANEL_DARK, 215), (0, 0, 150, 34), border_radius=14)
        pygame.draw.rect(bg, (255, 255, 255, 255), (0, 0, 150, 34), 2, border_radius=14)
        self.screen.blit(bg, (S.x + 10, S.bottom - 44))
        label = self.f_tiny.render("LIVES", True, WHITE)
        self.screen.blit(label, (S.x + 20, S.bottom - 33))
        for i in range(INITIAL_LIVES):
            cx = S.x + 82 + i * 24
            cy = S.bottom - 30
            color = (242, 65, 75) if i < lives else (150, 190, 210)
            pygame.draw.circle(self.screen, color, (cx - 4, cy), 5)
            pygame.draw.circle(self.screen, color, (cx + 4, cy), 5)
            pygame.draw.polygon(self.screen, color, [(cx - 9, cy + 2), (cx, cy + 11), (cx + 9, cy + 2)])

    # ────────────────────────────────────────────────────────────────
    # overlays
    # ────────────────────────────────────────────────────────────────
    def draw_game_over(self, score, high_score):
        overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        overlay.fill((225, 243, 250, 120))
        self.screen.blit(overlay, (0, 0))
        rect = pygame.Rect((self.W - 580) // 2, (self.H - 330) // 2, 580, 330)
        _panel(self.screen, rect, 28)
        cx = rect.centerx
        for dx, dy, col in ((4, 5, (130, 28, 36)), (0, 0, WHITE)):
            title = self.f_go.render("RUN OVER!", True, col)
            self.screen.blit(title, (cx - title.get_width() // 2 + dx, rect.y + 26 + dy))
        _draw_text(self.screen, self.f_large, f"Score: {score:,}", GOLD_BRIGHT,
                   (cx, rect.y + 120), anchor="midtop")
        _draw_text(self.screen, self.f_medium, f"Best: {high_score:,}", WHITE,
                   (cx, rect.y + 168), anchor="midtop")

        pulse = abs(math.sin(self.t * 3))
        label = self.f_small.render("RAISE BOTH HANDS OR PRESS SPACE TO RESTART", True, INK)
        pill = pygame.Rect(0, 0, label.get_width() + 44, 44)
        pill.center = (cx, rect.y + 246)
        pygame.draw.rect(self.screen, WHITE, pill, border_radius=22)
        pygame.draw.rect(self.screen, (int(255 - 40 * pulse), int(200 - 10 * pulse), int(40 + 20 * pulse)),
                         pill, 3, border_radius=22)
        self.screen.blit(label, (pill.centerx - label.get_width() // 2,
                                 pill.centery - label.get_height() // 2))
        _draw_text(self.screen, self.f_tiny, "ESC  \u2022  QUIT", SOFT_WHITE,
                   (cx, rect.y + 290), anchor="midtop")

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