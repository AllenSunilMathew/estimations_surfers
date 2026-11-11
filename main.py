"""Treasure Runner — AI Pose Controlled Game.

Main entry point: initialises Pygame, OpenCV camera (threaded), MediaPipe,
and audio, then runs the state-machine game loop.

Features:
- Native Windows window management Maximize / Fullscreen button enabled
- High-quality 1280x720 canvas dynamically scaled to any window resolution
- Threaded non-blocking CameraManager (instant C-key or click swap without hang/lag)
- Progressive speed scaling as score increases
- Audio manager for music and SFX
"""

import sys
import time

import pygame

from constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT, FPS_TARGET, GAME_TITLE, PAGE_BG,
    Gesture, State, CALIBRATION_FRAMES,
)
from camera_manager import CameraManager
from pose_detector import PoseDetector
from game_engine import GameEngine
from renderer import Renderer
from preloader import Preloader, MainMenu
from audio_manager import AudioManager


def enable_window_maximize():
    """Ensure the Windows OS title bar Maximize / Fullscreen button is enabled."""
    if sys.platform == "win32":
        try:
            import ctypes
            info = pygame.display.get_wm_info()
            hwnd = info.get("window")
            if hwnd:
                GWL_STYLE = -16
                WS_MAXIMIZEBOX = 0x00010000
                WS_THICKFRAME = 0x00040000
                style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
                ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, style | WS_MAXIMIZEBOX | WS_THICKFRAME)
                SWP_FRAMECHANGED = 0x0020
                SWP_NOMOVE = 0x0002
                SWP_NOSIZE = 0x0001
                SWP_NOZORDER = 0x0004
                ctypes.windll.user32.SetWindowPos(
                    hwnd, 0, 0, 0, 0, 0,
                    SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED
                )
                print("[Display] Windows title bar Maximize button enabled.")
        except Exception as e:
            print(f"[Display] Note: could not adjust Win32 window style: {e}")


def toggle_win_maximize():
    """Toggle maximize / restore state via Win32 API or Pygame."""
    if sys.platform == "win32":
        try:
            import ctypes
            info = pygame.display.get_wm_info()
            hwnd = info.get("window")
            if hwnd:
                # IsZoomed checks if window is currently maximized
                is_max = bool(ctypes.windll.user32.IsZoomed(hwnd))
                SW_MAXIMIZE = 3
                SW_RESTORE = 9
                ctypes.windll.user32.ShowWindow(hwnd, SW_RESTORE if is_max else SW_MAXIMIZE)
                return
        except Exception:
            pass


def main():
    # ── init pygame & resizable window (enables OS Maximize button) ─
    pygame.init()
    pygame.display.set_caption(GAME_TITLE)

    # Creating with pygame.RESIZABLE activates the Windows Maximize [□] button
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    enable_window_maximize()

    clock = pygame.time.Clock()

    # Virtual 1280x720 canvas: all rendering draws here with pixel perfection,
    # then scales to any window size / resolution (including maximized window)
    canvas = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))

    # ── audio ───────────────────────────────────────────────────────
    audio = AudioManager()

    # ── threaded camera manager (never freezes the main loop) ────────
    cam_manager = CameraManager()

    # ── game modules ────────────────────────────────────────────────
    detector = PoseDetector()
    engine = GameEngine()
    renderer = Renderer(canvas)
    preloader = Preloader(canvas)
    menu = MainMenu(canvas)

    # ── state ───────────────────────────────────────────────────────
    state = State.PRELOADER
    high_score = 0
    fps = 0
    gesture = Gesture.NEUTRAL
    confidence = 0.0
    cam_frame = None
    prev_t = time.time()
    music_playing = False

    def trigger_camera_switch():
        cam_manager.switch_next()

    # ── main game loop ──────────────────────────────────────────────
    running = True

    while running:
        now = time.time()
        dt = min(now - prev_t, 0.05)
        prev_t = now
        fps = int(clock.get_fps())

        # Map mouse coordinates from physical window back to 1280x720 canvas
        cur_w, cur_h = screen.get_size()
        scale = min(cur_w / WINDOW_WIDTH, cur_h / WINDOW_HEIGHT)
        rw = int(WINDOW_WIDTH * scale)
        rh = int(WINDOW_HEIGHT * scale)
        rx = (cur_w - rw) // 2
        ry = (cur_h - rh) // 2

        raw_mx, raw_my = pygame.mouse.get_pos()
        if scale > 0:
            vx = int((raw_mx - rx) / scale)
            vy = int((raw_my - ry) / scale)
            mouse_pos = (max(0, min(WINDOW_WIDTH - 1, vx)), max(0, min(WINDOW_HEIGHT - 1, vy)))
        else:
            mouse_pos = (raw_mx, raw_my)

        # ── events ──────────────────────────────────────────────────
        for ev in pygame.event.get():

            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.VIDEORESIZE:
                # Window was resized or maximized via Windows title bar button
                screen = pygame.display.set_mode((ev.w, ev.h), pygame.RESIZABLE)

            elif getattr(pygame, "WINDOWRESIZED", -1) != -1 and ev.type in (
                pygame.WINDOWRESIZED,
                getattr(pygame, "WINDOWSIZECHANGED", -1),
                getattr(pygame, "WINDOWMAXIMIZED", -1),
                getattr(pygame, "WINDOWRESTORED", -1)
            ):
                screen = pygame.display.get_surface()

            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                # Check clicks on Camera Swap button
                if state == State.MENU:
                    if menu.cam_switch_rect.collidepoint(mouse_pos):
                        trigger_camera_switch()

                elif state in (State.PLAYING, State.GAME_OVER):
                    if renderer.cam_switch_rect.collidepoint(mouse_pos):
                        trigger_camera_switch()

            elif ev.type == pygame.KEYDOWN:

                if ev.key == pygame.K_ESCAPE:
                    if state == State.PLAYING:
                        state = State.MENU
                        engine.reset()
                        detector.reset_calibration()
                        audio.stop_music()
                        music_playing = False
                    else:
                        running = False

                elif ev.key == pygame.K_F11:
                    toggle_win_maximize()

                elif ev.key == pygame.K_c:
                    trigger_camera_switch()

                elif ev.key == pygame.K_SPACE:
                    if state == State.MENU:
                        state = State.PLAYING
                        engine.reset()
                        detector.reset_calibration()
                        audio.play_music()
                        music_playing = True

                    elif state == State.GAME_OVER:
                        high_score = max(high_score, engine.score)
                        state = State.PLAYING
                        engine.reset()
                        detector.reset_calibration()
                        audio.play_music()
                        music_playing = True

        # ── threaded camera & pose estimation ───────────────────────
        raw_frame = cam_manager.get_latest_frame()
        if raw_frame is not None:
            gesture, confidence, annotated = detector.detect(raw_frame)
            cam_frame = annotated
        elif cam_manager.is_switching:
            cam_frame = None

        cam_name = cam_manager.get_camera_name()

        # ── state machine ───────────────────────────────────────────
        if state == State.PRELOADER:

            preloader.update(dt)
            preloader.draw()

            if preloader.done:
                state = State.MENU

        elif state == State.MENU:

            menu.update(dt)
            menu.draw(cam_frame, mouse_pos=mouse_pos, cam_name=cam_name)

            if gesture == Gesture.JUMP:
                state = State.PLAYING
                engine.reset()
                detector.reset_calibration()
                audio.play_music()
                music_playing = True

        elif state == State.PLAYING:

            prev_lives = engine.lives

            # Game update with progressive speed ramp
            engine.update(gesture, dt, detector.lane_target)
            renderer.tick(dt)

            # Hit / Game-over sound handling
            if engine.game_over:
                engine.flash_timer = min(engine.flash_timer, 0.10)
                high_score = max(high_score, engine.score)
                state = State.GAME_OVER
                audio.play_game_over()
                music_playing = False
            elif engine.lives < prev_lives or any(evt[2] == "stumble" for evt in engine.hit_events):
                audio.play_hit()

            # Render scene
            renderer.draw_bg()
            renderer.draw_hud(engine.score, fps, gesture, confidence, speed=engine.speed)
            renderer.draw_pose_panel(cam_frame, gesture, confidence, fps,
                                     cam_name=cam_name,
                                     is_switching=cam_manager.is_switching,
                                     status_msg=cam_manager.status_msg,
                                     mouse_pos=mouse_pos)
            renderer.draw_game_panel(engine)

            if detector.calibration_count < CALIBRATION_FRAMES:
                renderer.draw_calibrating(
                    detector.calibration_count / CALIBRATION_FRAMES
                )

        elif state == State.GAME_OVER:

            renderer.tick(dt)
            renderer.draw_bg()
            renderer.draw_hud(engine.score, fps, gesture, confidence, speed=engine.speed)
            renderer.draw_pose_panel(cam_frame, gesture, confidence, fps,
                                     cam_name=cam_name,
                                     is_switching=cam_manager.is_switching,
                                     status_msg=cam_manager.status_msg,
                                     mouse_pos=mouse_pos)
            renderer.draw_game_panel(engine)
            renderer.draw_game_over(engine.score, high_score, cause_of_death=engine.cause_of_death)

            if gesture == Gesture.JUMP:
                high_score = max(high_score, engine.score)
                state = State.PLAYING
                engine.reset()
                detector.reset_calibration()
                audio.play_music()
                music_playing = True

        # ── scale canvas to display (maintains perfect aspect ratio) ─
        if (cur_w, cur_h) == (WINDOW_WIDTH, WINDOW_HEIGHT):
            screen.blit(canvas, (0, 0))
        else:
            screen.fill(PAGE_BG)
            scaled = pygame.transform.smoothscale(canvas, (rw, rh))
            screen.blit(scaled, (rx, ry))

        pygame.display.flip()
        clock.tick(FPS_TARGET)

    # ── cleanup ─────────────────────────────────────────────────────
    audio.stop_music()
    cam_manager.release()
    detector.release()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
