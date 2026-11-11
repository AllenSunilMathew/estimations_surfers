"""Treasure Runner — AI Pose Controlled Game.

Main entry point: initialises Pygame, OpenCV camera, and MediaPipe,
then runs the state-machine game loop.
"""

import sys
import time

import cv2
import pygame

from constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT, FPS_TARGET, GAME_TITLE,
    CAMERA_WIDTH, CAMERA_HEIGHT,
    Gesture, State, CALIBRATION_FRAMES,
)
from pose_detector import PoseDetector
from game_engine import GameEngine
from renderer import Renderer
from preloader import Preloader, MainMenu


def main():
    # ── init ────────────────────────────────────────────────────────
    pygame.init()
    pygame.display.set_caption(GAME_TITLE)
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    # camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cam_ok = cap.isOpened()

    if not cam_ok:
        print("[WARN] Could not open webcam – game runs with placeholder feed.")

    # modules
    detector = PoseDetector()
    engine = GameEngine()
    renderer = Renderer(screen)
    preloader = Preloader(screen)
    menu = MainMenu(screen)

    # state
    state = State.PRELOADER
    high_score = 0
    fps = 0
    gesture = Gesture.NEUTRAL
    confidence = 0.0
    cam_frame = None
    prev_t = time.time()

    # ── loop ────────────────────────────────────────────────────────
    running = True

    while running:
        now = time.time()
        dt = min(now - prev_t, 0.05)
        prev_t = now
        fps = int(clock.get_fps())

        # ── events ──────────────────────────────────────────────────
        for ev in pygame.event.get():

            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.KEYDOWN:

                if ev.key == pygame.K_ESCAPE:
                    if state == State.PLAYING:
                        state = State.MENU
                        engine.reset()
                        detector.reset_calibration()
                    else:
                        running = False

                elif ev.key == pygame.K_SPACE:
                    if state == State.MENU:
                        state = State.PLAYING
                        engine.reset()
                        detector.reset_calibration()

                    elif state == State.GAME_OVER:
                        high_score = max(high_score, engine.score)
                        state = State.PLAYING
                        engine.reset()
                        detector.reset_calibration()

        # ── camera + pose ───────────────────────────────────────────
        if cam_ok:
            ret, raw = cap.read()

            if ret:
                raw = cv2.flip(raw, 1)
                gesture, confidence, annotated = detector.detect(raw)
                cam_frame = annotated
            else:
                cam_frame = None
        else:
            cam_frame = None

        # ── state machine ───────────────────────────────────────────
        if state == State.PRELOADER:

            preloader.update(dt)
            preloader.draw()

            if preloader.done:
                state = State.MENU

        elif state == State.MENU:

            menu.update(dt)
            menu.draw(cam_frame)

            if gesture == Gesture.JUMP:
                state = State.PLAYING
                engine.reset()
                detector.reset_calibration()

        elif state == State.PLAYING:

            engine.update(gesture, dt, detector.lane_target)
            renderer.tick(dt)

            renderer.draw_bg()
            renderer.draw_hud(engine.score, fps, gesture, confidence)
            renderer.draw_pose_panel(cam_frame, gesture, confidence, fps)
            renderer.draw_game_panel(engine)

            if detector.calibration_count < CALIBRATION_FRAMES:
                renderer.draw_calibrating(
                    detector.calibration_count / CALIBRATION_FRAMES
                )

            if engine.game_over:
                high_score = max(high_score, engine.score)
                state = State.GAME_OVER

        elif state == State.GAME_OVER:

            renderer.tick(dt)
            renderer.draw_bg()
            renderer.draw_hud(engine.score, fps, gesture, confidence)
            renderer.draw_pose_panel(cam_frame, gesture, confidence, fps)
            renderer.draw_game_panel(engine)
            renderer.draw_game_over(engine.score, high_score)

            if gesture == Gesture.JUMP:
                high_score = max(high_score, engine.score)
                state = State.PLAYING
                engine.reset()
                detector.reset_calibration()

        # ── flip ────────────────────────────────────────────────────
        pygame.display.flip()
        clock.tick(FPS_TARGET)

    # ── cleanup ─────────────────────────────────────────────────────
    cap.release()
    detector.release()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()