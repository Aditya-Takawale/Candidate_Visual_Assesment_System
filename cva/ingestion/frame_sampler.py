"""
Video Ingestion — Adaptive Frame Sampler
- OpenCV webcam / video file input
- Resize to ≤ 640x480
- Adaptive FPS (1–3) based on motion activity
- Frame quality checks: blur + lighting
"""

from __future__ import annotations
import cv2
import time
import numpy as np
import threading
from typing import Optional, Callable, Tuple
from queue import Queue, Full

from cva.config.settings import (
    VIDEO_SOURCE, FRAME_WIDTH, FRAME_HEIGHT,
    FPS_IDLE, FPS_ACTIVE,
    BLUR_THRESHOLD, BRIGHTNESS_MIN, BRIGHTNESS_MAX,
)
from cva.common.logger import get_logger

logger = get_logger(__name__)


def check_frame_quality(frame: np.ndarray) -> Tuple[bool, str]:
    """
    Returns (is_good, reason).
    Checks blur (Laplacian variance) and brightness (mean pixel value).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < BLUR_THRESHOLD:
        return False, f"blurry ({blur_score:.1f})"

    brightness = gray.mean()
    if brightness < BRIGHTNESS_MIN:
        return False, f"too dark ({brightness:.1f})"
    if brightness > BRIGHTNESS_MAX:
        return False, f"overexposed ({brightness:.1f})"

    return True, "ok"


def compute_motion_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """
    Fast motion detection using frame difference.
    Returns normalised score 0–1.
    """
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(diff.mean()) / 255.0


class FrameSampler:
    """
    Continuously captures frames from a camera or video file.
    Applies adaptive FPS, resize, and quality filtering.
    Pushes valid frames into an output queue.
    """

    def __init__(self, output_queue: Queue, on_fps_update: Optional[Callable[[float], None]] = None):
        self.output_queue = output_queue
        self.on_fps_update = on_fps_update

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._current_fps = FPS_IDLE
        self._frame_id = 0

    def start(self) -> None:
        self._cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        # Warm up: discard first 30 frames so camera auto-exposure settles
        for _ in range(30):
            self._cap.read()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("FrameSampler started.")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        logger.info("FrameSampler stopped.")

    def _capture_loop(self) -> None:
        last_capture_time = 0.0
        fps_frame_count = 0
        fps_window_start = time.time()

        while self._running:
            now = time.time()
            interval = 1.0 / self._current_fps

            if now - last_capture_time < interval:
                time.sleep(0.005)
                continue

            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Failed to read frame — source may have ended.")
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            is_good, reason = check_frame_quality(frame)
            if not is_good:
                logger.debug(f"Frame dropped: {reason}")
                last_capture_time = now
                continue

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self._prev_gray is not None:
                motion = compute_motion_score(self._prev_gray, curr_gray)
                self._current_fps = FPS_ACTIVE if motion > 0.02 else FPS_IDLE
            self._prev_gray = curr_gray

            self._frame_id += 1
            payload = {"frame": frame, "frame_id": self._frame_id, "timestamp": now}

            try:
                self.output_queue.put_nowait(payload)
            except Full:
                logger.debug("Frame queue full — dropping frame (backpressure).")

            last_capture_time = now
            fps_frame_count += 1

            elapsed = now - fps_window_start
            if elapsed >= 2.0:
                measured_fps = fps_frame_count / elapsed
                if self.on_fps_update:
                    self.on_fps_update(measured_fps)
                fps_frame_count = 0
                fps_window_start = now

    @property
    def current_fps(self) -> float:
        return self._current_fps
