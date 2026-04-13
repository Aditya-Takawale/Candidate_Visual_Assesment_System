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
    Runs MediaPipe FaceMesh per frame (single model for gaze + posture + fidget).
    Computes posture angle from head tilt, gaze direction from iris, fidget from frame diff.
    """

    def __init__(self):
        self._face_mesh_model = None
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
            self._mp_face_mesh = mp.solutions.face_mesh
            # Single FaceMesh model replaces Holistic + FaceMesh (saves ~200ms/frame)
            self._face_mesh_model = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,  # iris model adds ~200ms — not needed
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._holistic = None  # Not used anymore
            logger.info("MediaPipe FaceMesh loaded (unified model).")
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

        # Downscale to 320x240 for faster MediaPipe inference
        small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        h, w = small.shape[:2]

        # Single FaceMesh pass for gaze + posture (replaces Holistic + FaceMesh)
        face_result = self._face_mesh_model.process(rgb)

        features = self._compute_posture_from_face(face_result, h, w, features, now)
        features = self._compute_gaze(face_result, w, features, now)
        features = self._compute_fidget_fast(small, features)

        return features

    def _compute_posture_from_face(self, result, h: int, w: int, features: FrameFeatures, now: float) -> FrameFeatures:
        """Estimate head tilt / slouch from face landmarks (no Holistic needed)."""
        if not result.multi_face_landmarks:
            return features

        try:
            lm = result.multi_face_landmarks[0].landmark
            # Use nose tip (1), chin (152), forehead (10) to estimate head tilt
            nose = np.array([lm[1].x * w, lm[1].y * h])
            chin = np.array([lm[152].x * w, lm[152].y * h])
            forehead = np.array([lm[10].x * w, lm[10].y * h])

            # Head tilt = angle of forehead-chin line from vertical
            vec = forehead - chin
            angle = abs(np.degrees(np.arctan2(vec[0], -vec[1])))

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
        Gaze approximation using nose direction relative to face center.
        No iris landmarks needed (refine_landmarks=False for speed).
        If nose tip deviates significantly from face midpoint → looking away.
        """
        if not result.multi_face_landmarks:
            features.gaze_on_camera = False
            self._update_gaze_timer(False, now, features)
            return features

        try:
            lm = result.multi_face_landmarks[0].landmark
            # Use nose tip (1) vs face center (left ear 234, right ear 454)
            nose_x = lm[1].x
            left_face = lm[234].x   # left cheek
            right_face = lm[454].x  # right cheek
            face_center_x = (left_face + right_face) / 2
            face_width = abs(right_face - left_face)

            # Nose deviation as fraction of face width
            if face_width > 0.01:
                deviation = abs(nose_x - face_center_x) / face_width
            else:
                deviation = 0.0

            # Also check vertical: forehead (10) vs chin (152) tilt
            nose_y = lm[1].y
            forehead_y = lm[10].y
            chin_y = lm[152].y
            face_height = abs(chin_y - forehead_y)
            face_center_y = (forehead_y + chin_y) / 2
            v_deviation = abs(nose_y - face_center_y) / max(face_height, 0.01)

            total_deviation = (deviation + v_deviation * 0.5) / 1.5
            gaze_on = total_deviation < 0.15
            gaze_score = max(0.0, 1.0 - total_deviation / 0.3)
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

    def _compute_fidget_fast(self, small_frame: np.ndarray, features: FrameFeatures) -> FrameFeatures:
        """Fast fidget detection using frame differencing (already downscaled)."""
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
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
