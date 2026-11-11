"""Threaded Camera Manager for Treasure Runner.

Runs camera frame capture in a dedicated background worker thread so that
`cap.read()` latency never blocks the main Pygame event loop, preventing
lag, frame drops, and Windows 'Not Responding' freezes when switching
between Camera 0 and Camera 1.
"""

import os
import threading
import time

import cv2

from constants import CAMERA_WIDTH, CAMERA_HEIGHT


class CameraManager:
    """Non-blocking, threaded camera capture and switching manager."""

    def __init__(self, preferred_idx=None):
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.is_switching = False
        self.status_msg = "Detecting cameras..."

        # Probe available cameras (0 and 1)
        self.available_cams = self._probe_cameras()
        if not self.available_cams:
            # Fallback list if probe had strict driver permissions
            self.available_cams = [0, 1]

        # Determine starting camera
        if preferred_idx is not None and preferred_idx in self.available_cams:
            self.current_idx = preferred_idx
        elif self.available_cams:
            self.current_idx = self.available_cams[0]
        else:
            self.current_idx = 0

        self.cap = None
        self._open_cam_sync(self.current_idx)

        # Background capture thread
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()

    # ── camera probing ──────────────────────────────────────────────

    @staticmethod
    def _probe_cameras():
        """Probe indices 0 and 1 safely without printing driver noise."""
        available = []
        old_stderr = None
        devnull = None
        try:
            old_stderr = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
        except Exception:
            pass

        try:
            for idx in (0, 1):
                # Try DSHOW first on Windows
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    available.append(idx)
                    cap.release()
        finally:
            if old_stderr is not None and devnull is not None:
                try:
                    os.dup2(old_stderr, 2)
                    os.close(devnull)
                    os.close(old_stderr)
                except Exception:
                    pass

        return available

    # ── internal camera open ────────────────────────────────────────

    def _open_cam_sync(self, idx):
        """Open camera synchronously (called only inside worker/init)."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        # On Windows, try cv2.CAP_DSHOW first to bypass MSMF freeze
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            try:
                # Minimize internal frame buffer latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            self.cap = cap
            self.current_idx = idx
            cam_name = "Default Camera (0)" if idx == 0 else f"Webcam ({idx})"
            self.status_msg = f"{cam_name} Connected"
            print(f"[CameraManager] Successfully opened {cam_name}")
            return True
        else:
            cap.release()
            self.cap = None
            cam_name = f"Camera {idx}"
            self.status_msg = f"{cam_name} Not Found"
            print(f"[CameraManager] Failed to open {cam_name}")
            return False

    # ── public API (called from main game loop) ─────────────────────

    def switch_next(self):
        """Asynchronously switch to the other camera index without freezing Pygame."""
        if self.is_switching:
            return  # already in progress

        candidates = self.available_cams if len(self.available_cams) > 1 else [0, 1]
        cur_pos = candidates.index(self.current_idx) if self.current_idx in candidates else 0
        next_idx = candidates[(cur_pos + 1) % len(candidates)]

        self.switch_to(next_idx)

    def switch_to(self, target_idx):
        """Switch to a specific camera index in a background thread."""
        if self.is_switching or target_idx == self.current_idx:
            return

        def _switch_task():
            self.is_switching = True
            cam_name = "Default Cam (0)" if target_idx == 0 else f"Webcam ({target_idx})"
            self.status_msg = f"Connecting to {cam_name}..."
            print(f"[CameraManager] Switching to {cam_name}...")

            with self.lock:
                # Clear previous frame so caller knows camera is reconnecting
                self.frame = None
                self._open_cam_sync(target_idx)

            self.is_switching = False

        threading.Thread(target=_switch_task, daemon=True).start()

    def get_latest_frame(self):
        """Non-blocking: return the latest mirrored BGR frame or None (0ms latency)."""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def get_camera_name(self):
        if self.current_idx == 0:
            return "DEFAULT CAM"
        elif self.current_idx == 1:
            return "WEBCAM"
        return f"CAM {self.current_idx}"

    def release(self):
        """Stop capture thread and release hardware resources cleanly."""
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    # ── background worker ───────────────────────────────────────────

    def _capture_worker(self):
        """Continuously grab frames from camera at up to 60 FPS."""
        while self.running:
            if self.is_switching or self.cap is None:
                time.sleep(0.015)
                continue

            try:
                ret, raw = self.cap.read()
                if ret and raw is not None:
                    # Mirror horizontally for natural video mirror effect
                    mirrored = cv2.flip(raw, 1)
                    with self.lock:
                        self.frame = mirrored
                else:
                    time.sleep(0.01)
            except Exception:
                time.sleep(0.02)

            time.sleep(0.005)
