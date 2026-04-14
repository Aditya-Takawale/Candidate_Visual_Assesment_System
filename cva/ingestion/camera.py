"""
Shared webcam capture — single reader thread model.
One background thread reads from the camera; all consumers (MJPEG feed,
FrameSampler) read from the latest frame buffer.  This eliminates the race
condition where two threads calling cap.read() starve each other on Windows.
"""

from __future__ import annotations
import cv2
import platform
import threading
import time
import numpy as np
from typing import Optional

from cva.config.settings import VIDEO_SOURCE, FRAME_WIDTH, FRAME_HEIGHT
from cva.common.logger import get_logger

logger = get_logger(__name__)

_instance: Optional["SharedCamera"] = None
_init_lock = threading.Lock()
_camera_source = VIDEO_SOURCE


def _create_capture(source: int | str) -> cv2.VideoCapture:
    """Open a camera source with Windows-friendly backend selection."""
    if isinstance(source, int) and platform.system() == "Windows":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(source)


def list_available_cameras(max_devices: int = 6) -> list[dict]:
    """Probe camera indices and return basic metadata for selectable devices."""
    devices: list[dict] = []
    for index in range(max_devices):
        cap = _create_capture(index)
        if not cap.isOpened():
            cap.release()
            continue

        ok, frame = cap.read()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if ok and frame is not None and (width <= 0 or height <= 0):
            height, width = frame.shape[:2]
        cap.release()

        label = f"Camera {index}"
        if width > 0 and height > 0:
            label = f"Camera {index} ({width}x{height})"

        devices.append({
            "index": index,
            "label": label,
            "width": width,
            "height": height,
        })
    return devices


class SharedCamera:
    """Thread-safe camera with a single reader thread."""

    def __init__(self, source: Optional[int | str] = None):
        self._source = VIDEO_SOURCE if source is None else source
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_id: int = 0
        self._open()

    def _open(self) -> None:
        self._cap = _create_capture(self._source)
        if not self._cap.isOpened():
            logger.warning(f"Failed to open camera source: {self._source}")
            self._cap = None
            return
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Flush initial frames so auto-exposure settles
        for _ in range(5):
            self._cap.read()
        logger.info(f"Camera opened: source={self._source}, {FRAME_WIDTH}x{FRAME_HEIGHT}")

    def start(self) -> None:
        if self._running:
            return
        if self._cap is None or not self._cap.isOpened():
            self._open()
        if self._cap is None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="camera-reader")
        self._thread.start()
        logger.info("Camera reader thread started.")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Camera released.")

    def _reader_loop(self) -> None:
        """Single thread that continuously reads frames at ~30 FPS."""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            with self._frame_lock:
                self._latest_frame = frame
                self._frame_id += 1
            time.sleep(0.01)  # ~60+ reads/sec, plenty for all consumers

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Get the latest frame (non-blocking, thread-safe)."""
        with self._frame_lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    @property
    def frame_id(self) -> int:
        with self._frame_lock:
            return self._frame_id

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def source(self) -> int | str:
        return self._source


def get_camera() -> SharedCamera:
    """Return the singleton SharedCamera instance, with auto-fallback."""
    global _instance, _camera_source
    with _init_lock:
        if _instance is None:
            _instance = SharedCamera(_camera_source)
            if _instance.is_opened:
                _instance.start()
                return _instance

            # Default source failed — probe for any working camera
            logger.warning(f"Default camera source {_camera_source} failed, probing alternatives...")
            devices = list_available_cameras(max_devices=6)
            for dev in devices:
                idx = dev["index"]
                if idx == _camera_source:
                    continue
                candidate = SharedCamera(idx)
                if candidate.is_opened:
                    logger.info(f"Auto-selected camera {idx}: {dev['label']}")
                    _instance = candidate
                    _camera_source = idx
                    _instance.start()
                    return _instance
                candidate.stop()

            # Nothing worked — return the failed instance so callers see is_opened=False
            logger.error("No working camera found on any index.")
    return _instance


def get_capture() -> Optional["SharedCamera"]:
    """Compatibility alias — returns the SharedCamera (duck-types enough for MJPEG)."""
    cam = get_camera()
    return cam if cam.is_opened else None


def release():
    """Release the shared camera."""
    global _instance
    with _init_lock:
        if _instance is not None:
            _instance.stop()
            _instance = None


def get_camera_source() -> int | str:
    return _camera_source


def set_camera_source(source: int | str) -> bool:
    """Switch the singleton camera source and restart the shared reader."""
    global _instance, _camera_source
    with _init_lock:
        if _instance is not None and _instance.source == source and _instance.is_opened:
            _camera_source = source
            return True

        old_instance = _instance
        _instance = None
        if old_instance is not None:
            old_instance.stop()

        new_instance = SharedCamera(source)
        if not new_instance.is_opened:
            if old_instance is not None:
                _instance = old_instance
                _instance.start()
            return False

        new_instance.start()
        _instance = new_instance
        _camera_source = source
        return True
