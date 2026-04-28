"""
Multi-Frame Aggregator
- Maintains rolling buffer of FrameFeatures
- Applies EMA smoothing across all signals
- Enforces temporal rules (e.g., bad posture > 5s → flag)
- Holds scoring until WARMUP_FRAMES are collected
"""

from __future__ import annotations
import time
from collections import deque
from typing import Deque, List

from cva.config.settings import (
    EMA_ALPHA,
    FRAME_BUFFER_SIZE,
    WARMUP_FRAMES,
    GAZE_OFF_CAMERA_THRESHOLD_SEC,
    SLOUCH_ANGLE_THRESHOLD_DEG,
    FIDGET_MOTION_THRESHOLD,
    SMILE_LANDMARK_THRESHOLD,
    SPEECH_RMS_THRESHOLD,
    IDENTITY_COSINE_THRESHOLD,
    GROOMING_CONFIDENCE_THRESHOLD,
)
from cva.common.models import FrameFeatures, AggregatedFeatures, RedFlag, RedFlagSeverity
from cva.common.logger import get_logger

logger = get_logger(__name__)


class MultiFrameAggregator:
    """
    Aggregates per-frame features into smoothed session-level signals.
    """

    def __init__(self, session_id: str = "", candidate_id: str = ""):
        self._buffer: Deque[FrameFeatures] = deque(maxlen=FRAME_BUFFER_SIZE)
        self._session_id = session_id
        self._candidate_id = candidate_id
        self._frame_count = 0
        self._agg = AggregatedFeatures(session_id=session_id, candidate_id=candidate_id)

        self._ema = {
            "identity_score": 0.5,       # Start neutral until verified
            "gaze_score": 0.5,
            "posture_score": 0.5,        # Neutral until calibrated — no free credit
            "fidget_score": 0.0,
            "smile_ratio": 0.0,
            "speech_energy_score": 0.3,  # Must earn via actual speech
            "grooming_score": 0.5,       # Neutral until assessed
            "punctuality_score": 1.0,    # On-time unless proven late
        }

        self._identity_reference_active: bool = False  # set True when first cosine signal received
        self._grooming_has_run: bool = False             # set True when YOLO first produces a result
        self._slouch_start: float = 0.0
        self._slouch_flagged = False
        self._gaze_off_start: float = 0.0
        self._gaze_flagged = False
        self._no_smile_frames: int = 0
        self._low_speech_frames: int = 0
        self._low_identity_frames: int = 0
        self._fidget_frames: int = 0
        self._no_face_frames: int = 0

        # Persistent red flags — accumulate across frames, deduplicated by module+reason
        self._red_flags: List[RedFlag] = []
        self._flag_keys: set = set()  # prevent duplicate flags

    def ingest(self, features: FrameFeatures) -> None:
        self._buffer.append(features)
        self._frame_count += 1
        self._update_ema(features)
        self._check_temporal_rules(features)

    def _ema_update(self, key: str, new_value: float, alpha: float = EMA_ALPHA) -> float:
        self._ema[key] = alpha * new_value + (1 - alpha) * self._ema[key]
        return self._ema[key]

    def _add_flag(self, module: str, reason: str, severity: RedFlagSeverity, confidence: float) -> None:
        key = f"{module}:{reason[:40]}"
        if key not in self._flag_keys:
            self._flag_keys.add(key)
            self._red_flags.append(RedFlag(
                module=module, reason=reason,
                severity=severity, confidence=confidence,
            ))

    @staticmethod
    def _remap_identity(raw_sim: float) -> float:
        """Remap cosine similarity to a fairer identity score.
        Aadhaar-vs-webcam typically gives 0.30-0.55 due to print quality,
        lighting, and angle differences. Actual curve:
          0.25 -> 0.45,  0.35 -> 0.59,  0.45 -> 0.73,  0.55 -> 0.87, 0.60+ -> 0.90+
        """
        if raw_sim >= 0.70:
            return min(1.0, 0.95 + (raw_sim - 0.70) * 0.20)
        if raw_sim >= 0.55:
            return min(0.95, 0.87 + (raw_sim - 0.55) * 0.60)
        if raw_sim >= 0.25:
            return 0.45 + (raw_sim - 0.25) * 1.40
        return max(0.1, raw_sim * 1.5)

    def _update_ema(self, f: FrameFeatures) -> None:
        # ── Identity ──────────────────────────────────────────────────────
        if f.face_cosine_similarity is not None:
            identity_score = self._remap_identity(f.face_cosine_similarity)
            # Identity should converge faster than other signals once a face is matched.
            self._ema_update("identity_score", identity_score, alpha=0.45)
            self._identity_reference_active = True  # Aadhaar reference has been uploaded & compared
        elif not f.face_in_frame:
            # Face completely absent — drive identity toward 0
            self._ema_update("identity_score", 0.0)
        elif not f.face_detected:
            # Identity module skipped this frame — slow decay, no artificial floor
            self._ema_update("identity_score", self._ema["identity_score"] * 0.95)

        # ── Body Language — only meaningful when face is visible ───────────
        if not f.face_in_frame:
            # Nobody present: drive all presence signals toward 0
            self._ema_update("gaze_score", 0.0)
            self._ema_update("posture_score", 0.0)
        else:
            gaze_val = 1.0 if f.gaze_on_camera else max(0.0, 1.0 - f.gaze_off_seconds / GAZE_OFF_CAMERA_THRESHOLD_SEC)
            self._ema_update("gaze_score", gaze_val)

            posture_val = max(0.0, 1.0 - f.posture_angle_deg / (SLOUCH_ANGLE_THRESHOLD_DEG * 2))
            self._ema_update("posture_score", posture_val)

        self._ema_update("fidget_score", f.fidget_score)

        smile_val = 1.0 if f.smile_detected else 0.0
        self._ema_update("smile_ratio", smile_val)

        # Speech: dead mic (RMS ~0) → low neutral; actual audio → compute real energy
        if f.speech_rms < 0.001:
            speech_val = 0.4   # low neutral — mic silent, no credit awarded
        else:
            speech_val = min(1.0, f.speech_rms / max(0.001, SPEECH_RMS_THRESHOLD * 2))
        self._ema_update("speech_energy_score", speech_val)

        if f.grooming_score is not None:
            # Use higher alpha for grooming: it runs every 15s (low frequency),
            # so EMA_ALPHA=0.15 would take many minutes to converge.
            # alpha=0.50 makes the score reflect the 2 most recent YOLO runs.
            self._ema_update("grooming_score", f.grooming_score, alpha=0.50)
            self._grooming_has_run = True

    def _check_temporal_rules(self, f: FrameFeatures) -> None:
        now = time.time()

        # ── Gaze off camera ──────────────────────────────────────────
        if not f.gaze_on_camera:
            if self._gaze_off_start == 0.0:
                self._gaze_off_start = now
            duration = now - self._gaze_off_start
            if duration > GAZE_OFF_CAMERA_THRESHOLD_SEC and not self._gaze_flagged:
                self._add_flag(
                    "body_language",
                    f"Gaze off-camera for {duration:.0f}s — possible distraction",
                    RedFlagSeverity.MEDIUM, 0.90,
                )
                self._gaze_flagged = True
        else:
            self._gaze_off_start = 0.0
            self._gaze_flagged = False

        # ── Sustained slouch ─────────────────────────────────────────
        if f.posture_slouch:
            if self._slouch_start == 0.0:
                self._slouch_start = now
            duration = now - self._slouch_start
            if duration > 5.0 and not self._slouch_flagged:
                self._add_flag(
                    "body_language",
                    f"Slouch for {duration:.0f}s — {f.posture_angle_deg:.1f}° from baseline",
                    RedFlagSeverity.MEDIUM, 0.85,
                )
                self._slouch_flagged = True
        else:
            self._slouch_start = 0.0
            self._slouch_flagged = False

        # ── Excessive fidgeting ──────────────────────────────────────
        if f.fidget_score > FIDGET_MOTION_THRESHOLD:
            self._fidget_frames += 1
            if self._fidget_frames == 8:
                self._add_flag(
                    "body_language",
                    f"Excessive movement (fidget={f.fidget_score:.2f})",
                    RedFlagSeverity.LOW, 0.75,
                )
        else:
            self._fidget_frames = max(0, self._fidget_frames - 1)

        # ── No smile throughout session ───────────────────────────────
        if not f.smile_detected:
            self._no_smile_frames += 1
            if self._no_smile_frames == 20:
                self._add_flag(
                    "first_impression",
                    "No smile detected — candidate appears disengaged",
                    RedFlagSeverity.LOW, 0.70,
                )
        else:
            self._no_smile_frames = max(0, self._no_smile_frames - 2)

        # ── Low speech energy ─────────────────────────────────────────
        if f.speech_rms < SPEECH_RMS_THRESHOLD:
            self._low_speech_frames += 1
            if self._low_speech_frames == 60:
                self._add_flag(
                    "first_impression",
                    f"Very low speech energy (rms={f.speech_rms:.4f}) — speak louder",
                    RedFlagSeverity.LOW, 0.65,
                )
        else:
            self._low_speech_frames = max(0, self._low_speech_frames - 2)

        # ── Identity mismatch ─────────────────────────────────────────
        if f.face_cosine_similarity is not None and f.face_cosine_similarity < IDENTITY_COSINE_THRESHOLD:
            self._low_identity_frames += 1
            if self._low_identity_frames == 5:
                self._add_flag(
                    "identity",
                    f"Face mismatch (similarity={f.face_cosine_similarity:.2f}, threshold={IDENTITY_COSINE_THRESHOLD}) — candidate may not match Aadhaar photo",
                    RedFlagSeverity.HIGH, 0.92,
                )
        else:
            self._low_identity_frames = max(0, self._low_identity_frames - 1)

        # ── No face in frame ──────────────────────────────────────────────
        if not f.face_in_frame:
            self._no_face_frames += 1
            if self._no_face_frames == 12:  # ~4 seconds at 3 fps
                self._add_flag(
                    "identity",
                    "Candidate not visible in frame — possible absence or camera issue",
                    RedFlagSeverity.HIGH, 0.95,
                )
        else:
            self._no_face_frames = max(0, self._no_face_frames - 3)

        # ── Grooming / casual wear ────────────────────────────────────
        if f.attire_class and f.attire_class.lower() in ["t-shirt", "hoodie", "tank-top", "shorts", "casual"]:
            self._add_flag(
                "grooming",
                f"Casual attire detected: '{f.attire_class}'",
                RedFlagSeverity.MEDIUM, 0.80,
            )

    def get_aggregated(self) -> AggregatedFeatures:
        self._agg.frame_count = self._frame_count
        self._agg.is_warmed_up = self._frame_count >= WARMUP_FRAMES

        self._agg.identity_score = self._ema["identity_score"]
        # Verified = reference was compared AND remapped EMA meets the threshold
        self._agg.identity_verified = (
            self._identity_reference_active
            and self._ema["identity_score"] >= self._remap_identity(IDENTITY_COSINE_THRESHOLD)
        )
        self._agg.identity_reference_active = self._identity_reference_active
        self._agg.gaze_score = self._ema["gaze_score"]
        self._agg.posture_score = self._ema["posture_score"]
        self._agg.fidget_score = self._ema["fidget_score"]
        self._agg.smile_ratio = self._ema["smile_ratio"]
        self._agg.speech_energy_score = self._ema["speech_energy_score"]
        self._agg.grooming_score = self._ema["grooming_score"]
        self._agg.grooming_has_run = self._grooming_has_run
        self._agg.punctuality_score = self._ema["punctuality_score"]

        body_language_score = (
            self._ema["gaze_score"] * 0.4 +
            self._ema["posture_score"] * 0.4 +
            max(0.0, 1.0 - self._ema["fidget_score"] / max(FIDGET_MOTION_THRESHOLD, 0.01)) * 0.2
        )
        self._agg.emotion_score = body_language_score

        # Snapshot of current red flags (persistent — not cleared)
        self._agg.red_flags = list(self._red_flags)

        return self._agg
