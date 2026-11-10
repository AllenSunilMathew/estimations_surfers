"""Pose detection and gesture recognition using MediaPipe."""

from collections import Counter

import cv2
import mediapipe as mp
import numpy as np

from constants import (
    Gesture,
    GESTURE_ARM_THRESHOLD,
    GESTURE_DIAGONAL_THRESHOLD,
    GESTURE_CROUCH_THRESHOLD,
    CALIBRATION_FRAMES,
)


class PoseDetector:
    """Wraps MediaPipe Pose to detect body gestures from webcam frames."""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )
        # Calibration – learn standing nose‑Y over the first N frames
        self.calibration_count = 0
        self.calibrated_nose_y = None
        self._nose_y_samples: list[float] = []

        # Gesture smoothing (majority‑vote over recent frames)
        self._gesture_history: list[str] = []
        self._history_size = 5

    # ── public API ──────────────────────────────────────────────────

    def detect(self, frame):
        """Return (gesture, confidence, annotated_frame)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        rgb.flags.writeable = True

        gesture = Gesture.NEUTRAL
        confidence = 0.0

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            confidence = self._avg_confidence(lm)

            # Calibration phase
            if self.calibration_count < CALIBRATION_FRAMES:
                nose_y = lm[self.mp_pose.PoseLandmark.NOSE.value].y
                self._nose_y_samples.append(nose_y)
                self.calibration_count += 1
                if self.calibration_count == CALIBRATION_FRAMES:
                    self.calibrated_nose_y = float(np.mean(self._nose_y_samples))
            else:
                gesture = self._classify(lm)

            self._draw_skeleton(frame, results.pose_landmarks)

        return gesture, confidence, frame

    def reset_calibration(self):
        """Reset calibration so the next detect() starts fresh."""
        self.calibration_count = 0
        self.calibrated_nose_y = None
        self._nose_y_samples.clear()
        self._gesture_history.clear()

    def release(self):
        self.pose.close()

    # ── internal ────────────────────────────────────────────────────

    def _avg_confidence(self, landmarks):
        key = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
        ]
        return sum(landmarks[k.value].visibility for k in key) / len(key)

    def _classify(self, lm):
        P = self.mp_pose.PoseLandmark

        lw_y = lm[P.LEFT_WRIST.value].y
        rw_y = lm[P.RIGHT_WRIST.value].y
        ls_y = lm[P.LEFT_SHOULDER.value].y
        rs_y = lm[P.RIGHT_SHOULDER.value].y
        le_y = lm[P.LEFT_ELBOW.value].y
        re_y = lm[P.RIGHT_ELBOW.value].y
        nose_y = lm[P.NOSE.value].y

        # 1. JUMP — both wrists clearly above their respective shoulders
        if (lw_y < ls_y - GESTURE_ARM_THRESHOLD and
                rw_y < rs_y - GESTURE_ARM_THRESHOLD):
            return self._smooth(Gesture.JUMP)

        # 2. SLIDE — nose drops significantly from calibrated standing
        if self.calibrated_nose_y is not None:
            drop = nose_y - self.calibrated_nose_y
            if drop > GESTURE_CROUCH_THRESHOLD:
                return self._smooth(Gesture.SLIDE)

        # 3. LEFT  — "/" shape: left hand down, right hand up
        #    In image coords (Y↓): left‑wrist larger Y, right‑wrist smaller Y
        left_down = lw_y - rw_y          # positive → left is lower
        if left_down > GESTURE_DIAGONAL_THRESHOLD:
            # extra check: elbows follow the same trend
            if lw_y > le_y and rw_y < re_y:
                return self._smooth(Gesture.LEFT)

        # 4. RIGHT — "\" shape: left hand up, right hand down
        right_down = rw_y - lw_y
        if right_down > GESTURE_DIAGONAL_THRESHOLD:
            if rw_y > re_y and lw_y < le_y:
                return self._smooth(Gesture.RIGHT)

        return self._smooth(Gesture.NEUTRAL)

    def _smooth(self, gesture):
        """Majority‑vote filter to remove jitter."""
        self._gesture_history.append(gesture)
        if len(self._gesture_history) > self._history_size:
            self._gesture_history.pop(0)
        return Counter(self._gesture_history).most_common(1)[0][0]

    # ── skeleton drawing ────────────────────────────────────────────

    _CONNECTIONS = None  # built lazily

    def _build_connections(self):
        P = self.mp_pose.PoseLandmark
        self._CONNECTIONS = [
            # Torso
            (P.LEFT_SHOULDER, P.RIGHT_SHOULDER, (0, 255, 255)),
            (P.LEFT_SHOULDER, P.LEFT_HIP, (0, 255, 200)),
            (P.RIGHT_SHOULDER, P.RIGHT_HIP, (0, 255, 200)),
            (P.LEFT_HIP, P.RIGHT_HIP, (0, 255, 255)),
            # Left arm
            (P.LEFT_SHOULDER, P.LEFT_ELBOW, (255, 200, 0)),
            (P.LEFT_ELBOW, P.LEFT_WRIST, (255, 150, 0)),
            # Right arm
            (P.RIGHT_SHOULDER, P.RIGHT_ELBOW, (255, 200, 0)),
            (P.RIGHT_ELBOW, P.RIGHT_WRIST, (255, 150, 0)),
            # Left leg
            (P.LEFT_HIP, P.LEFT_KNEE, (0, 200, 255)),
            (P.LEFT_KNEE, P.LEFT_ANKLE, (0, 150, 255)),
            # Right leg
            (P.RIGHT_HIP, P.RIGHT_KNEE, (0, 200, 255)),
            (P.RIGHT_KNEE, P.RIGHT_ANKLE, (0, 150, 255)),
        ]

    _KEY_JOINTS = None

    def _build_key_joints(self):
        P = self.mp_pose.PoseLandmark
        self._KEY_JOINTS = [
            P.NOSE,
            P.LEFT_SHOULDER, P.RIGHT_SHOULDER,
            P.LEFT_ELBOW, P.RIGHT_ELBOW,
            P.LEFT_WRIST, P.RIGHT_WRIST,
            P.LEFT_HIP, P.RIGHT_HIP,
            P.LEFT_KNEE, P.RIGHT_KNEE,
            P.LEFT_ANKLE, P.RIGHT_ANKLE,
        ]

    def _draw_skeleton(self, frame, pose_landmarks):
        if self._CONNECTIONS is None:
            self._build_connections()
        if self._KEY_JOINTS is None:
            self._build_key_joints()

        h, w = frame.shape[:2]
        lm = pose_landmarks.landmark

        # Lines
        for start, end, color in self._CONNECTIONS:
            p1, p2 = lm[start.value], lm[end.value]
            if p1.visibility > 0.5 and p2.visibility > 0.5:
                x1, y1 = int(p1.x * w), int(p1.y * h)
                x2, y2 = int(p2.x * w), int(p2.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), color, 5)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Joints
        for joint in self._KEY_JOINTS:
            pt = lm[joint.value]
            if pt.visibility > 0.5:
                x, y = int(pt.x * w), int(pt.y * h)
                cv2.circle(frame, (x, y), 8, (255, 0, 255), -1)
                cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
