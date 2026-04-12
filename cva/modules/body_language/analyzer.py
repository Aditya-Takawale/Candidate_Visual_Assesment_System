"""
Module 2 — Body Language & Gaze
- MediaPipe Holistic: 33-point pose (posture/slouch)
- Gaze approximation from MediaPipe FaceMesh iris/eye landmarks
- Fidget detection via frame-difference optical flow
- Emotion stub (HSEmotion integration point)
- Baseline calibration: first N frames define personal posture baseline
"""

from __future__ import annotations
import numpy as np
import cv2
import time
from collections import deque
from typing import Optional, Deque

from cva.config.settings import (
    GAZE_OFF_CAMERA_THRESHOLD_SEC,
    SLOUCH_ANGLE_THRESHOLD_DEG,
    FIDGET_MOTION_THRESHOLD,
    FIDGET_WINDOW_SEC,
    NEGATIVE_EMOTION_THRESHOLD_SEC,
    BASELINE_CALIBRATION_FRAMES,
    EMA_ALPHA,
)
from cva.common.models import FrameFeatures, RedFlag, RedFlagSeverity
from cva.common.logger import get_logger

logger = get_logger(__name__)

NEGATIVE_EMOTIONS = {"disinterested", "sad", "angry", "fear", "disgust"}


def _angle_between(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute angle in degrees between two 2D points relative to vertical axis."""
    delta = p2 - p1
    angle = float(np.degrees(np.arctan2(abs(delta[0]), abs(delta[1]))))
    return angle


class BodyLanguageAnalyzer:
    """
    Runs MediaPipe Holistic per frame.
    Computes posture angle, gaze direction, and fidget score.
    Maintains temporal state for red-flag triggering.
    """

    def __init__(self):
        self._holistic = None
        self._face_mesh = None
        self._mock = False
        self._load()

        self._posture_baseline: Optional[float] = None
        self._baseline_samples: list = []
        self._calibrated = False

        self._gaze_off_start: Optional[float] = None
        self._gaze_off_seconds: float = 0.0

        self._negative_emotion_start: Optional[float] = None
        self._negative_emotion_seconds: float = 0.0

        self._prev_gray: Optional[np.ndarray] = None
        self._fidget_window: Deque[float] = deque(maxlen=int(FIDGET_WINDOW_SEC * 3))

        self._smoothed_posture_angle: float = 0.0
        self._smoothed_gaze_score: float = 1.0

    def _load(self) -> None:
        try:
            import mediapipe as mp
            self._mp_holistic = mp.solutions.holistic
            self._mp_face_mesh = mp.solutions.face_mesh
            self._holistic = self._mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._face_mesh_model = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
            logger.info("MediaPipe Holistic + FaceMesh loaded.")
        except ImportError:
            logger.warning("MediaPipe not installed — using mock body language analyzer.")
            self._mock = True

    def process_frame(self, frame: np.ndarray, features: FrameFeatures) -> FrameFeatures:
        now = time.time()

        if self._mock:
            features.gaze_on_camera = True
            features.posture_angle_deg = 5.0
            features.fidget_score = 0.05
            features.emotion = "neutral"
            return features

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        holistic_result = self._holistic.process(rgb)
        face_result = self._face_mesh_model.process(rgb)

        features = self._compute_posture(holistic_result, h, w, features, now)
        features = self._compute_gaze(face_result, w, features, now)
        features = self._compute_fidget(frame, features)

        return features

    def _compute_posture(self, result, h: int, w: int, features: FrameFeatures, now: float) -> FrameFeatures:
        if not result.pose_landmarks:
            return features

        lm = result.pose_landmarks.landmark

        try:
            left_shoulder = np.array([lm[11].x * w, lm[11].y * h])
            right_shoulder = np.array([lm[12].x * w, lm[12].y * h])
            left_hip = np.array([lm[23].x * w, lm[23].y * h])
            right_hip = np.array([lm[24].x * w, lm[24].y * h])

            mid_shoulder = (left_shoulder + right_shoulder) / 2
            mid_hip = (left_hip + right_hip) / 2

            angle = _angle_between(mid_hip, mid_shoulder)

            if not self._calibrated:
                self._baseline_samples.append(angle)
                if len(self._baseline_samples) >= BASELINE_CALIBRATION_FRAMES:
                    self._posture_baseline = float(np.median(self._baseline_samples))
                    self._calibrated = True
                    logger.info(f"Posture baseline calibrated: {self._posture_baseline:.1f}°")

            self._smoothed_posture_angle = (
                EMA_ALPHA * angle + (1 - EMA_ALPHA) * self._smoothed_posture_angle
            )

            baseline = self._posture_baseline if self._calibrated else 0.0
            deviation = abs(self._smoothed_posture_angle - baseline)
            features.posture_angle_deg = deviation
            features.posture_slouch = deviation > SLOUCH_ANGLE_THRESHOLD_DEG
        except Exception as e:
            logger.debug(f"Posture compute error: {e}")

        return features

    def _compute_gaze(self, result, w: int, features: FrameFeatures, now: float) -> FrameFeatures:
        """
        Gaze approximation using iris landmarks (MediaPipe FaceMesh refined).
        If iris center deviates significantly from eye center → looking away.
        """
        if not result.multi_face_landmarks:
            features.gaze_on_camera = False
            self._update_gaze_timer(False, now, features)
            return features

        try:
            lm = result.multi_face_landmarks[0].landmark
            LEFT_IRIS = 468
            RIGHT_IRIS = 473
            LEFT_EYE_INNER = 133
            LEFT_EYE_OUTER = 33
            RIGHT_EYE_INNER = 362
            RIGHT_EYE_OUTER = 263

            left_iris_x = lm[LEFT_IRIS].x
            right_iris_x = lm[RIGHT_IRIS].x
            left_eye_center = (lm[LEFT_EYE_INNER].x + lm[LEFT_EYE_OUTER].x) / 2
            right_eye_center = (lm[RIGHT_EYE_INNER].x + lm[RIGHT_EYE_OUTER].x) / 2

            left_ratio = abs(left_iris_x - left_eye_center)
            right_ratio = abs(right_iris_x - right_eye_center)
            avg_deviation = (left_ratio + right_ratio) / 2

            gaze_on = avg_deviation < 0.07
            gaze_score = max(0.0, 1.0 - avg_deviation / 0.15)
            self._smoothed_gaze_score = (
                EMA_ALPHA * gaze_score + (1 - EMA_ALPHA) * self._smoothed_gaze_score
            )
            features.gaze_on_camera = gaze_on
            self._update_gaze_timer(gaze_on, now, features)
        except Exception as e:
            logger.debug(f"Gaze compute error: {e}")
            features.gaze_on_camera = True

        return features

    def _update_gaze_timer(self, gaze_on: bool, now: float, features: FrameFeatures) -> None:
        if not gaze_on:
            if self._gaze_off_start is None:
                self._gaze_off_start = now
            self._gaze_off_seconds = now - self._gaze_off_start
        else:
            self._gaze_off_start = None
            self._gaze_off_seconds = 0.0
        features.gaze_off_seconds = self._gaze_off_seconds

    def _compute_fidget(self, frame: np.ndarray, features: FrameFeatures) -> FrameFeatures:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is not None:
            diff = cv2.absdiff(self._prev_gray, gray)
            motion = float(diff.mean()) / 255.0
            self._fidget_window.append(motion)
            features.fidget_score = float(np.mean(self._fidget_window))
        self._prev_gray = gray
        return features

    def get_red_flags(self, features: FrameFeatures) -> list:
        flags = []
        if features.gaze_off_seconds > GAZE_OFF_CAMERA_THRESHOLD_SEC:
            flags.append(RedFlag(
                module="body_language",
                reason=f"Gaze off-camera for {features.gaze_off_seconds:.1f}s",
                severity=RedFlagSeverity.MEDIUM,
                confidence=0.9,
            ))
        if features.posture_slouch:
            flags.append(RedFlag(
                module="body_language",
                reason=f"Slouch detected: {features.posture_angle_deg:.1f}° deviation from baseline",
                severity=RedFlagSeverity.LOW,
                confidence=0.8,
            ))
        if features.fidget_score > FIDGET_MOTION_THRESHOLD:
            flags.append(RedFlag(
                module="body_language",
                reason=f"Excessive movement detected (score={features.fidget_score:.2f})",
                severity=RedFlagSeverity.LOW,
                confidence=0.75,
            ))
        return flags
