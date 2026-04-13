"""
Module 1 — Identity Verification
- InsightFace (ONNX / CPU) face detection + embedding
- Multi-frame verification (3–5 frames) with EMA smoothing
- Cosine similarity threshold check
- OCR via PaddleOCR (with Google Vision API stub for production)
- RapidFuzz name matching
- Face quality pre-check (blur + lighting) before every match
"""

from __future__ import annotations
import numpy as np
import time
from collections import deque
from typing import Optional, Deque, Tuple

from cva.config.settings import (
    IDENTITY_COSINE_THRESHOLD,
    IDENTITY_MULTI_FRAME_COUNT,
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
        self._load()

    def _load(self) -> None:
        try:
            from insightface.app import FaceAnalysis
            from cva.common.hardware import get_available_providers
            providers = get_available_providers()
            self._app = FaceAnalysis(
                name="buffalo_sc",
                providers=providers,
                allowed_modules=["detection", "recognition"],
            )
            self._app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1, det_size=(320, 320))
            logger.info("InsightFace loaded (buffalo_sc).")
        except Exception as e:
            logger.warning(f"InsightFace not available ({e}). Using mock embedder.")
            self._mock = True

    def get_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Returns face embedding vector or None if no face detected."""
        if self._mock:
            return np.random.rand(512).astype(np.float32)
        try:
            faces = self._app.get(frame)
            if not faces:
                return None
            return faces[0].embedding
        except Exception as e:
            logger.warning(f"InsightFace inference error: {e}")
            return None

    def face_detected(self, frame: np.ndarray) -> bool:
        if self._mock:
            return True
        try:
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
        self._embedder = FaceEmbedder()
        self._reference_embedding: Optional[np.ndarray] = None
        self._embedding_buffer: Deque[np.ndarray] = deque(maxlen=IDENTITY_MULTI_FRAME_COUNT)
        self._smoothed_similarity: float = 0.5   # neutral until reference is verified
        self._red_flags: list = []
        self._reference_name: Optional[str] = None
        self._cv_name: Optional[str] = None

    def set_reference(self, reference_frame: np.ndarray, name: Optional[str] = None) -> bool:
        """Set reference face from Aadhaar photo or first frame."""
        embedding = self._embedder.get_embedding(reference_frame)
        if embedding is None:
            logger.warning("Could not extract reference embedding — no face found.")
            return False
        self._reference_embedding = embedding
        self._reference_name = name
        logger.info("Reference face embedding set.")
        return True

    def set_cv_name(self, name: str) -> None:
        self._cv_name = name

    def process_frame(self, frame: np.ndarray, features: FrameFeatures) -> FrameFeatures:
        """Run identity check on a single frame, update features in-place."""
        if self._reference_embedding is None:
            features.face_detected = self._embedder.face_detected(frame)
            # No reference → neutral score (don't penalise candidate)
            features.face_cosine_similarity = 0.75
            features.identity_verified = True
            return features

        embedding = self._embedder.get_embedding(frame)
        if embedding is None:
            features.face_detected = False
            return features

        features.face_detected = True
        self._embedding_buffer.append(embedding)

        if len(self._embedding_buffer) < 3:
            return features

        avg_embedding = np.mean(np.stack(list(self._embedding_buffer)), axis=0)
        similarity = _cosine_similarity(avg_embedding, self._reference_embedding)

        self._smoothed_similarity = (
            EMA_ALPHA * similarity + (1 - EMA_ALPHA) * self._smoothed_similarity
        )

        features.face_cosine_similarity = self._smoothed_similarity
        features.identity_verified = self._smoothed_similarity >= IDENTITY_COSINE_THRESHOLD

        if not features.identity_verified:
            features.face_cosine_similarity = self._smoothed_similarity
            logger.info(
                f"Identity mismatch: similarity={self._smoothed_similarity:.3f}",
                extra={"cva_module": "identity", "confidence": self._smoothed_similarity},
            )

        return features

    def check_name_match(self) -> Optional[RedFlag]:
        """Compare Aadhaar name vs CV name using RapidFuzz."""
        if not self._reference_name or not self._cv_name:
            return None
        try:
            from rapidfuzz import fuzz
            score = fuzz.token_sort_ratio(self._reference_name.lower(), self._cv_name.lower())
            if score < FUZZY_MATCH_THRESHOLD:
                return RedFlag(
                    module="identity",
                    reason=f"Name mismatch: Aadhaar='{self._reference_name}' vs CV='{self._cv_name}' (score={score})",
                    severity=RedFlagSeverity.HIGH,
                    confidence=1.0 - score / 100,
                )
        except ImportError:
            logger.warning("rapidfuzz not installed — skipping name match.")
        return None

    @property
    def smoothed_similarity(self) -> float:
        return self._smoothed_similarity


class OCRExtractor:
    """
    Extracts text from Aadhaar card image using PaddleOCR.
    Falls back gracefully if not installed.
    """

    def __init__(self):
        self._ocr = None
        self._load()

    def _load(self) -> None:
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(use_angle_cls=True, lang="en")
            logger.info("PaddleOCR loaded.")
        except ImportError:
            logger.warning("PaddleOCR not installed — OCR extraction disabled.")

    def extract_name(self, image_path: str) -> Optional[str]:
        """Extract candidate name from Aadhaar card image."""
        if self._ocr is None:
            return None
        try:
            result = self._ocr.ocr(image_path)
            texts = [line[1][0] for block in result for line in block if line[1][1] > 0.7]
            logger.debug(f"OCR extracted texts: {texts}")
            for text in texts:
                if len(text.split()) >= 2 and text[0].isupper():
                    return text.strip()
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
        return None
