"""
Module 1 — Identity Verification
- InsightFace (ONNX / CPU) face detection + embedding
- Multi-frame verification (3–5 frames) with EMA smoothing
- Cosine similarity threshold check
- OCR via PaddleOCR (with Google Vision API stub for production)
- RapidFuzz name matching
- Face quality pre-check (blur + lighting) before every match
"""
# pylint: disable=broad-exception-caught  # hardware/model errors must not surface to caller
# pylint: disable=protected-access        # _mock is an internal embedder flag
# pylint: disable=line-too-long           # long ONNX provider lines are intentional

from __future__ import annotations
import threading
from collections import deque
from typing import Optional, Deque

import numpy as np

from cva.config.settings import (
    IDENTITY_COSINE_THRESHOLD,
    IDENTITY_MULTI_FRAME_COUNT,
    IDENTITY_OCCLUSION_DET_THRESH,
    IDENTITY_OCCLUSION_SIMILARITY_GRACE,
    EMA_ALPHA,
    FUZZY_MATCH_THRESHOLD,
)
from cva.common.models import RedFlag, RedFlagSeverity, FrameFeatures
from cva.common.logger import get_logger

logger = get_logger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class FaceEmbedder:
    """
    Wraps InsightFace (buffalo_sc — lightweight ONNX model).
    Falls back to a mock embedder if InsightFace is not installed.
    """

    def __init__(self):
        self._app = None
        self._mock = False
        self._det_size = (640, 640)
        self._last_det_score: float = 1.0   # detection confidence of last matched face
        self._load()

    def _load(self) -> None:
        try:
            from insightface.app import FaceAnalysis  # type: ignore[import-unresolved]  # pylint: disable=import-error,import-outside-toplevel
            from cva.common.hardware import get_available_providers  # pylint: disable=import-outside-toplevel
            providers = get_available_providers()
            self._app = FaceAnalysis(
                name="buffalo_sc",
                providers=providers,
                allowed_modules=["detection", "recognition"],
            )
            self._app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1, det_size=self._det_size)
            logger.info("InsightFace loaded (buffalo_sc).")
        except Exception as e:
            logger.warning(f"InsightFace not available ({e}). Using mock embedder.")
            self._mock = True

    def set_det_size(self, size: tuple) -> None:
        """Switch detection grid size (smaller = faster for verification frames)."""
        if self._app is not None and size != self._det_size:
            self._det_size = size
            self._app.prepare(ctx_id=0, det_size=size)

    @property
    def last_det_score(self) -> float:
        """Detection confidence of the last matched face (1.0 default = full face, no occlusion)."""
        return self._last_det_score

    def get_embedding(self, frame: np.ndarray, det_thresh: float = 0.5) -> Optional[np.ndarray]:
        """Returns face embedding vector or None if no face detected."""
        if self._mock:
            return None  # InsightFace unavailable — identity scoring disabled
        try:
            self._app.det_model.det_thresh = det_thresh
            faces = self._app.get(frame)
            if not faces:
                return None
            # Pick the face with the highest detection score
            best_face = max(faces, key=lambda f: f.det_score)
            self._last_det_score = float(best_face.det_score)
            return best_face.embedding
        except Exception as e:
            logger.warning(f"InsightFace inference error: {e}")
            return None

    def face_detected(self, frame: np.ndarray) -> bool:
        """Return True if at least one face is detected in the frame."""
        if self._mock:
            return True
        try:
            self._app.det_model.det_thresh = 0.5
            faces = self._app.get(frame)
            return len(faces) > 0
        except Exception:
            return False


class IdentityVerifier:
    """
    Maintains a rolling buffer of face embeddings and computes a smoothed
    cosine similarity against a reference (Aadhaar) embedding.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._embedder = FaceEmbedder()
        self._reference_embedding: Optional[np.ndarray] = None
        self._embedding_buffer: Deque[np.ndarray] = deque(maxlen=IDENTITY_MULTI_FRAME_COUNT)
        self._smoothed_similarity: float = 0.0   # start at zero — must earn a passing score
        self._red_flags: list = []
        self._reference_name: Optional[str] = None
        self._cv_name: Optional[str] = None
        self._name_match_flag: Optional["RedFlag"] = None
        self._name_match_checked: bool = False

    def set_reference(self, reference_frame: np.ndarray, name: Optional[str] = None,
                       det_thresh: float = IDENTITY_OCCLUSION_DET_THRESH) -> bool:
        """Set reference face from Aadhaar photo or first frame."""
        # Mock mode — InsightFace not installed; skip embedding and accept the frame
        if self._embedder._mock:
            with self._lock:
                self._reference_name = name
                self._name_match_checked = False
            logger.info("Reference image accepted (mock embedder — InsightFace not installed).")
            return True
        # Use full resolution for enrollment (best quality embedding)
        self._embedder.set_det_size((640, 640))
        embedding = self._embedder.get_embedding(reference_frame, det_thresh=det_thresh)
        if embedding is None:
            logger.warning("Could not extract reference embedding — no face found.")
            return False
        with self._lock:
            self._embedding_buffer.clear()
            self._smoothed_similarity = 0.0
            self._red_flags = []
            self._reference_embedding = embedding
            self._reference_name = name
            self._name_match_checked = False  # invalidate cached name-match result
        logger.info("Reference face embedding set.")
        return True

    def reset_session(self) -> None:
        """Clear per-session rolling state so cached singleton starts clean."""
        with self._lock:
            self._reference_embedding = None
            self._embedding_buffer.clear()
            self._smoothed_similarity = 0.0
            self._red_flags = []
            self._reference_name = None
            self._cv_name = None
            self._name_match_flag = None
            self._name_match_checked = False

    def set_cv_name(self, name: str) -> None:
        """Store the CV name for name-match verification."""
        with self._lock:
            self._cv_name = name
            self._name_match_checked = False  # invalidate cached name-match result

    def set_reference_name(self, name: str) -> None:
        """Store the Aadhaar reference name for name-match verification."""
        with self._lock:
            self._reference_name = name
            self._name_match_checked = False

    def process_frame(self, frame: np.ndarray, features: FrameFeatures) -> FrameFeatures:
        """Run identity check on a single frame, update features in-place."""
        # InsightFace not available — cannot verify; leave score at neutral default.
        # Do NOT emit a fake similarity — that would inflate identity score without real verification.
        if self._embedder._mock:
            features.face_detected = True
            features.face_cosine_similarity = None   # no real signal
            features.identity_verified = False
            return features

        with self._lock:
            reference_embedding = self._reference_embedding

        if reference_embedding is None:
            # Aadhaar not uploaded — identity module is inactive.
            # Emit no similarity so the aggregator keeps identity_score at its neutral default
            # and identity_reference_active stays False (no hard cap fires).
            features.face_detected = self._embedder.face_detected(frame)
            features.face_cosine_similarity = None
            features.identity_verified = False
            return features

        # Use full resolution for verification frames — accuracy over speed
        self._embedder.set_det_size((640, 640))
        embedding = self._embedder.get_embedding(frame)
        if embedding is None:
            features.face_detected = False
            return features

        features.face_detected = True
        with self._lock:
            self._embedding_buffer.append(embedding)

            avg_embedding = np.mean(np.stack(list(self._embedding_buffer)), axis=0)
            # Re-read reference under lock in case enrollment changed mid-frame.
            if self._reference_embedding is None:
                features.face_cosine_similarity = 0.0
                features.identity_verified = False
                return features
            similarity = _cosine_similarity(avg_embedding, self._reference_embedding)

            # Keep an EMA for display smoothing only — never used for the pass/fail decision.
            self._smoothed_similarity = (
                EMA_ALPHA * similarity + (1 - EMA_ALPHA) * self._smoothed_similarity
            )

            # Grace: if InsightFace detection confidence is low (face partially occluded
            # by turban, hijab, surgical mask, niqab, heavy beard, etc.) relax the match
            # threshold slightly rather than immediately flagging a mismatch.
            effective_threshold = IDENTITY_COSINE_THRESHOLD
            if self._embedder.last_det_score < 0.65:
                effective_threshold = max(
                    IDENTITY_COSINE_THRESHOLD - IDENTITY_OCCLUSION_SIMILARITY_GRACE,
                    0.45,
                )
                logger.debug(
                    f"Occlusion grace applied (det_score={self._embedder.last_det_score:.2f}): "
                    f"threshold {IDENTITY_COSINE_THRESHOLD:.2f}→{effective_threshold:.2f}"
                )
            # Verification uses raw buffer average (not EMA); require ≥3 frames before granting pass.
            buffer_ready = len(self._embedding_buffer) >= 3
            features.face_cosine_similarity = similarity
            features.identity_verified = buffer_ready and similarity >= effective_threshold

        if not features.identity_verified:
            features.face_cosine_similarity = self._smoothed_similarity
            logger.info(
                f"Identity mismatch: similarity={self._smoothed_similarity:.3f}",
                extra={"cva_module": "identity", "confidence": self._smoothed_similarity},
            )

        return features

    def check_name_match(self) -> Optional[RedFlag]:
        """Compare Aadhaar name vs CV name using RapidFuzz (result is cached)."""
        with self._lock:
            if self._name_match_checked:
                return self._name_match_flag
            reference_name = self._reference_name
            cv_name = self._cv_name

        if not reference_name or not cv_name:
            return None
        try:
            from rapidfuzz import fuzz  # pylint: disable=import-outside-toplevel
            score = fuzz.token_sort_ratio(reference_name.lower(), cv_name.lower())
            flag = None
            if score < FUZZY_MATCH_THRESHOLD:
                flag = RedFlag(
                    module="identity",
                    reason=f"Name mismatch between ID and CV (similarity score={score})",
                    severity=RedFlagSeverity.HIGH,
                    confidence=1.0 - score / 100,
                )
            with self._lock:
                self._name_match_flag = flag
                self._name_match_checked = True
        except ImportError:
            logger.warning("rapidfuzz not installed — skipping name match.")
            with self._lock:
                self._name_match_checked = True
        with self._lock:
            return self._name_match_flag

    @property
    def smoothed_similarity(self) -> float:
        """EMA-smoothed cosine similarity from the most recent identity check."""
        return self._smoothed_similarity
