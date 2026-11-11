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
    LANE_MOVE_THRESHOLD,
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

        # Lane control – body position left / right of the calibrated centre
        self.calibrated_x = None
        self._x_samples: list[float] = []
        self._offset = 0.0
        self.lane_target = None      # 0 left · 1 centre · 2 right · None = unknown

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
                self._x_samples.append(self._body_x(lm))
                self.calibration_count += 1
                if self.calibration_count == CALIBRATION_FRAMES:
                    self.calibrated_nose_y = float(np.mean(self._nose_y_samples))
                    self.calibrated_x = float(np.mean(self._x_samples))
                    self.lane_target = 1
            else:
                gesture = self._classify(lm)
                self._update_lane(lm)
                # show lane movement in the "Gesture:" label
                if gesture == Gesture.NEUTRAL:
                    if self.lane_target == 0:
                        gesture = Gesture.LEFT
                    elif self.lane_target == 2:
                        gesture = Gesture.RIGHT

            self._draw_skeleton(frame, results.pose_landmarks)

        self._draw_lane_guides(frame)

        return gesture, confidence, frame

    def reset_calibration(self):
        """Reset calibration so the next detect() starts fresh."""
        self.calibration_count = 0
        self.calibrated_nose_y = None
        self._nose_y_samples.clear()
        self.calibrated_x = None
        self._x_samples.clear()
        self._offset = 0.0
        self.lane_target = None
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

    def _body_x(self, lm):
        """Horizontal centre of the shoulders (0-1, image is already mirrored)."""
        P = self.mp_pose.PoseLandmark
        return (lm[P.LEFT_SHOULDER.value].x + lm[P.RIGHT_SHOULDER.value].x) / 2

    def _update_lane(self, lm):
        """Step left / right of the calibrated centre -> lane 0 / 2 (with hysteresis)."""
        if self.calibrated_x is None:
            return
        off = self._body_x(lm) - self.calibrated_x
        self._offset = 0.5 * self._offset + 0.5 * off   # light smoothing
        t = LANE_MOVE_THRESHOLD
        lane = self.lane_target if self.lane_target is not None else 1
        if lane == 1:
            if self._offset < -t:
                lane = 0
            elif self._offset > t:
                lane = 2
        elif lane == 0 and self._offset > -t * 0.5:
            lane = 1
        elif lane == 2 and self._offset < t * 0.5:
            lane = 1
        self.lane_target = lane

    def _draw_lane_guides(self, frame):
        """Two vertical lines on the camera feed: step past them to change lane."""
        if self.calibrated_x is None:
            return
        h, w = frame.shape[:2]
        for sign in (-1, 1):
            x = int((self.calibrated_x + sign * LANE_MOVE_THRESHOLD) * w)
            for y in range(0, h, 24):
                cv2.line(frame, (x, y), (x, min(h, y + 12)), (255, 255, 255), 2)
        cx = int(self.calibrated_x * w)
        cv2.line(frame, (cx, h - 30), (cx, h), (0, 255, 255), 3)

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

        # Left / right are handled by body position (see _update_lane).

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