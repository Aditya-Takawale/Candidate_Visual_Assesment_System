"""
Module 3 — First Impression & Session Management
- Punctuality tracking (join timestamp vs scheduled time)
- Smile detection via MediaPipe FaceMesh lip landmarks
- Voice Activity Detection via webrtcvad
- Speech energy (RMS) and speaking rate via librosa
- Speech pause detection
- Filler word detection stub
Active only during first FIRST_IMPRESSION_DURATION seconds of session.
"""

from __future__ import annotations
import numpy as np
import time
import threading
from typing import Optional

from cva.config.settings import (
    PUNCTUALITY_GRACE_PERIOD_SEC,
    SMILE_LANDMARK_THRESHOLD,
    SPEECH_RMS_THRESHOLD,
    SPEECH_PAUSE_MAX_SEC,
    FILLER_WORDS,
    FIRST_IMPRESSION_DURATION,
    EMA_ALPHA,
)
from cva.common.models import FrameFeatures, RedFlag, RedFlagSeverity
from cva.common.logger import get_logger

logger = get_logger(__name__)


class SmileDetector:
    """Detects smile using MediaPipe FaceMesh lip corner landmarks."""

    LIP_CORNER_LEFT = 61
    LIP_CORNER_RIGHT = 291
    UPPER_LIP = 13
    LOWER_LIP = 14

    def __init__(self):
        self._face_mesh = None
        self._mock = False
        self._load()
        self._smoothed_smile: float = 0.0

    def _load(self) -> None:
        try:
            import mediapipe as mp
            self._mp = mp.solutions.face_mesh
            self._face_mesh = self._mp.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
            )
            logger.info("SmileDetector FaceMesh loaded.")
        except ImportError:
            logger.warning("MediaPipe not installed — smile detection in mock mode.")
            self._mock = True

    def detect(self, frame: np.ndarray, features: FrameFeatures) -> FrameFeatures:
        if self._mock:
            features.smile_detected = False
            return features

        import cv2
        # Downscale aggressively — smile only needs lip landmarks
        small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            features.smile_detected = False
            return features

        try:
            lm = result.multi_face_landmarks[0].landmark
            left = np.array([lm[self.LIP_CORNER_LEFT].x, lm[self.LIP_CORNER_LEFT].y])
            right = np.array([lm[self.LIP_CORNER_RIGHT].x, lm[self.LIP_CORNER_RIGHT].y])
            upper = np.array([lm[self.UPPER_LIP].x, lm[self.UPPER_LIP].y])
            lower = np.array([lm[self.LOWER_LIP].x, lm[self.LOWER_LIP].y])

            mouth_width = float(np.linalg.norm(right - left))
            mouth_height = float(np.linalg.norm(lower - upper))
            ratio = mouth_height / (mouth_width + 1e-6)

            self._smoothed_smile = EMA_ALPHA * ratio + (1 - EMA_ALPHA) * self._smoothed_smile
            features.smile_detected = self._smoothed_smile < SMILE_LANDMARK_THRESHOLD
        except Exception as e:
            logger.debug(f"Smile detection error: {e}")

        return features


class SpeechAnalyzer:
    """
    Real-time audio analysis using sounddevice + librosa.
    Measures RMS energy, detects pauses.
    Runs in a background thread.
    """

    SAMPLE_RATE = 16000
    CHUNK_DURATION = 0.5

    def __init__(self):
        self._rms: float = 0.0
        self._speech_active: bool = False
        self._last_speech_time: float = time.time()
        self._pause_duration: float = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._mock = False
        self._load()

    def _load(self) -> None:
        try:
            import sounddevice  # noqa: F401
            import librosa       # noqa: F401
            logger.info("SpeechAnalyzer (sounddevice + librosa) ready.")
        except ImportError:
            logger.warning("sounddevice/librosa not installed — speech analysis in mock mode.")
            self._mock = True

    def start(self) -> None:
        if self._mock:
            return
        self._running = True
        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _audio_loop(self) -> None:
        import sounddevice as sd
        import librosa

        chunk_size = int(self.SAMPLE_RATE * self.CHUNK_DURATION)
        while self._running:
            try:
                audio = sd.rec(chunk_size, samplerate=self.SAMPLE_RATE, channels=1, dtype="float32")
                sd.wait()
                audio = audio.flatten()
                # Guard against NaN/Inf from bad audio devices
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                audio = np.clip(audio, -1.0, 1.0)
                rms = float(np.sqrt(max(0.0, np.mean(audio.astype(np.float64) ** 2))))
                if not np.isfinite(rms):
                    rms = 0.0
                self._rms = EMA_ALPHA * rms + (1 - EMA_ALPHA) * self._rms
                self._speech_active = rms > SPEECH_RMS_THRESHOLD

                now = time.time()
                if self._speech_active:
                    self._last_speech_time = now
                    self._pause_duration = 0.0
                else:
                    self._pause_duration = now - self._last_speech_time
            except Exception as e:
                logger.debug(f"Audio loop error: {e}")
                time.sleep(0.5)

    @property
    def rms(self) -> float:
        return self._rms

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    @property
    def pause_duration(self) -> float:
        return self._pause_duration


class FirstImpressionAnalyzer:
    """
    Orchestrates all first-impression signals.
    Active only during the first FIRST_IMPRESSION_DURATION seconds.
    """

    def __init__(self, scheduled_start_time: Optional[float] = None):
        self._session_start: float = time.time()
        self._scheduled_start: float = scheduled_start_time or self._session_start
        self._smile_detector = SmileDetector()
        self._speech_analyzer = SpeechAnalyzer()
        self._active = True
        self._punctuality_flag: Optional[RedFlag] = None
        self._speech_analyzer.start()
        self._check_punctuality()

    def _check_punctuality(self) -> None:
        delay = self._session_start - self._scheduled_start
        if delay > PUNCTUALITY_GRACE_PERIOD_SEC:
            self._punctuality_flag = RedFlag(
                module="first_impression",
                reason=f"Candidate joined {delay:.0f}s late",
                severity=RedFlagSeverity.MEDIUM,
                confidence=1.0,
            )
            logger.info(f"Punctuality red flag: {delay:.0f}s late.")

    def is_active(self) -> bool:
        elapsed = time.time() - self._session_start
        self._active = elapsed <= FIRST_IMPRESSION_DURATION
        return self._active

    def process_frame(self, frame: np.ndarray, features: FrameFeatures) -> FrameFeatures:
        if not self.is_active():
            return features

        features = self._smile_detector.detect(frame, features)
        features.speech_rms = self._speech_analyzer.rms
        features.speech_active = self._speech_analyzer.speech_active
        return features

    def get_red_flags(self, features: FrameFeatures) -> list:
        flags = []
        if self._punctuality_flag:
            flags.append(self._punctuality_flag)
            self._punctuality_flag = None

        if not features.smile_detected and self.is_active():
            flags.append(RedFlag(
                module="first_impression",
                reason="No smile detected during greeting window",
                severity=RedFlagSeverity.LOW,
                confidence=0.7,
            ))

        if features.speech_rms < SPEECH_RMS_THRESHOLD and features.speech_active is False:
            flags.append(RedFlag(
                module="first_impression",
                reason=f"Low speech energy (RMS={features.speech_rms:.4f})",
                severity=RedFlagSeverity.LOW,
                confidence=0.75,
            ))

        if self._speech_analyzer.pause_duration > SPEECH_PAUSE_MAX_SEC:
            flags.append(RedFlag(
                module="first_impression",
                reason=f"Long speech pause ({self._speech_analyzer.pause_duration:.1f}s)",
                severity=RedFlagSeverity.LOW,
                confidence=0.7,
            ))

        return flags

    def stop(self) -> None:
        self._speech_analyzer.stop()
