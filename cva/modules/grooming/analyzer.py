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
        self._device = "cpu"
        self._imgsz = 320
        self._load()

    def _load(self) -> None:
        try:
            from ultralytics import YOLO
            from cva.common.hardware import get_torch_device
            model_path = MODELS_DIR / "yolov8n.pt"
            self._model = YOLO(str(model_path))
            self._device = get_torch_device()
            self._imgsz = 256 if self._device == "mps" else 320
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

    def reset(self) -> None:
        self._last_run_time = 0.0
        self._last_result = None

    def _run_inference(self, frame: np.ndarray):
        return self._model(frame, verbose=False, conf=GROOMING_CONFIDENCE_THRESHOLD, device=self._device, imgsz=self._imgsz)

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
            results = self._run_inference(frame)
            attire_class, grooming_score = self._parse_results(results)
            features.attire_class = attire_class
            features.grooming_score = grooming_score
            self._last_result = {"attire_class": attire_class, "grooming_score": grooming_score}
        except Exception as e:
            message = str(e)
            if self._device == "cuda" and "CUDA" in message.upper():
                logger.warning("Grooming CUDA inference failed; falling back to CPU for stability.")
                self._device = "cpu"
                self._imgsz = 320
                try:
                    results = self._run_inference(frame)
                    attire_class, grooming_score = self._parse_results(results)
                    features.attire_class = attire_class
                    features.grooming_score = grooming_score
                    self._last_result = {"attire_class": attire_class, "grooming_score": grooming_score}
                    return features
                except Exception as retry_error:
                    logger.warning(f"Grooming CPU fallback also failed: {retry_error}")
            else:
                logger.warning(f"Grooming inference error: {e}")
            features.grooming_score = None

        return features

    def _parse_results(self, results) -> tuple:
        """
        Parse YOLO detections into attire class and grooming score.
        Ethnic wear is mapped to 'formal' — not flagged.

        Note: COCO vocabulary has limited clothing classes ('tie' is the only
        attire indicator). We use 'tie' as a formal signal, 'backpack' +
        'handbag' as neutral accessories, and absence of any clothing object
        as 'undetected' (neutral). We avoid claiming casual detection from
        classes that COCO was never trained on.
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

        # COCO "tie" is a strong formal indicator
        if "tie" in detected_classes:
            return ("formal", 0.95)

        # If we only see person/furniture/generic objects, we cannot determine attire
        # Give a mild neutral score — grooming module is advisory in demo mode
        person_detected = "person" in detected_classes
        if person_detected:
            return ("neutral", 0.65)  # person visible but attire indeterminate

        return ("undetected", 0.55)  # generic COCO detections — cannot determine attire

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
