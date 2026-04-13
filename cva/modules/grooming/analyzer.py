"""
Module 4 — Grooming & Dressing (Lightweight / Optional)
- YOLOv8n (nano model) for attire + accessory detection
- Context-aware: ethnic wear is NOT flagged as informal
- Runs at low frequency (every GROOMING_INTERVAL seconds)
- Gracefully skipped if system is under load or model unavailable
"""

from __future__ import annotations
import numpy as np
import cv2
import time
from typing import Optional, List

from cva.config.settings import (
    GROOMING_CONFIDENCE_THRESHOLD,
    GROOMING_INTERVAL,
    ETHNIC_WEAR_CLASSES,
    CASUAL_WEAR_CLASSES,
    MODELS_DIR,
)
from cva.common.models import FrameFeatures, RedFlag, RedFlagSeverity
from cva.common.logger import get_logger

logger = get_logger(__name__)


class GroomingAnalyzer:
    """
    Lightweight attire analysis using YOLOv8n.
    Only runs at low frequency. Gracefully degrades if model not available.
    """

    def __init__(self):
        self._model = None
        self._mock = False
        self._last_run_time: float = 0.0
        self._last_result: Optional[dict] = None
        self._load()

    def _load(self) -> None:
        try:
            from ultralytics import YOLO
            from cva.common.hardware import get_torch_device
            model_path = MODELS_DIR / "yolov8n.pt"
            self._model = YOLO(str(model_path))
            self._device = get_torch_device()
            logger.info(f"YOLOv8n grooming model loaded (device={self._device}).")
        except ImportError:
            logger.warning("ultralytics not installed — grooming module in mock mode.")
            self._mock = True
        except Exception as e:
            logger.warning(f"YOLOv8n load failed ({e}) — grooming module in mock mode.")
            self._mock = True

    def should_run(self) -> bool:
        """Returns True if enough time has passed since last grooming check."""
        from cva.config.settings import GROOMING_ENABLED
        if not GROOMING_ENABLED:
            return False
        return (time.time() - self._last_run_time) >= GROOMING_INTERVAL

    def process_frame(self, frame: np.ndarray, features: FrameFeatures) -> FrameFeatures:
        if not self.should_run():
            if self._last_result:
                features.attire_class = self._last_result.get("attire_class")
                features.grooming_score = self._last_result.get("grooming_score", 1.0)
            return features

        self._last_run_time = time.time()

        if self._mock:
            features.attire_class = "formal"
            features.grooming_score = 0.9
            self._last_result = {"attire_class": "formal", "grooming_score": 0.9}
            return features

        try:
            results = self._model(frame, verbose=False, conf=GROOMING_CONFIDENCE_THRESHOLD, device=self._device, imgsz=320)
            attire_class, grooming_score = self._parse_results(results)
            features.attire_class = attire_class
            features.grooming_score = grooming_score
            self._last_result = {"attire_class": attire_class, "grooming_score": grooming_score}
        except Exception as e:
            logger.warning(f"Grooming inference error: {e}")
            features.grooming_score = None

        return features

    def _parse_results(self, results) -> tuple:
        """
        Parse YOLO detections into attire class and grooming score.
        Ethnic wear is mapped to 'formal' — not flagged.
        """
        detected_classes = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names.get(cls_id, "unknown").lower()
                conf = float(box.conf[0])
                if conf >= GROOMING_CONFIDENCE_THRESHOLD:
                    detected_classes.append(cls_name)

        if not detected_classes:
            return ("undetected", 0.5)  # no detections — neutral, no credit awarded

        for cls in detected_classes:
            if any(ethnic in cls for ethnic in ETHNIC_WEAR_CLASSES):
                return ("formal", 1.0)

        for cls in detected_classes:
            if any(casual in cls for casual in CASUAL_WEAR_CLASSES):
                return (cls, 0.3)

        # COCO "tie" is a strong formal indicator
        if "tie" in detected_classes:
            return ("formal", 0.95)

        # Only generic COCO objects detected (person, chair, etc.) — cannot determine attire
        clothing_keywords = set(CASUAL_WEAR_CLASSES + ETHNIC_WEAR_CLASSES + ["tie", "suit", "jacket", "blazer", "shirt"])
        has_clothing = any(
            any(kw in cls for kw in clothing_keywords) for cls in detected_classes
        )
        if not has_clothing:
            return ("undetected", 0.55)  # generic COCO detections — cannot determine attire

        return ("formal", 0.85)

    def get_red_flags(self, features: FrameFeatures) -> List[RedFlag]:
        flags = []
        if features.attire_class and features.attire_class in CASUAL_WEAR_CLASSES:
            flags.append(RedFlag(
                module="grooming",
                reason=f"Casual attire detected: '{features.attire_class}'",
                severity=RedFlagSeverity.MEDIUM,
                confidence=0.85,
            ))
        return flags
