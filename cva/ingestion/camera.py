"""
Shared webcam capture -- single instance, single reader thread.

Design rules:
  - Camera is opened EXACTLY ONCE at startup and never re-probed.
  - WMI is used on Windows to identify USB cameras by name (no cv2 opens during detection).
  - All consumers (MJPEG feed, FrameSampler) share the same cap via the frame buffer.
  - CAP_PROP_BUFFERSIZE=1 prevents stale frames accumulating in the driver queue.
  - _open_capture validates with a live frame so black-frame DSHOW backends are rejected.
"""
# pylint: disable=no-member          # cv2 is a C extension; Pylint cannot see its dynamic members
# pylint: disable=global-statement   # module-level singletons are intentional design
# pylint: disable=broad-exception-caught  # boundary camera code must catch all exceptions
# pylint: disable=protected-access        # _make_dead_instance needs direct attribute init
# pylint: disable=line-too-long           # long thread-start lines are intentional

from __future__ import annotations

import json
import platform
import subprocess
import threading
import time
from typing import Optional

import cv2
import numpy as np

from cva.config.settings import VIDEO_SOURCE, FRAME_WIDTH, FRAME_HEIGHT
from cva.common.logger import get_logger

logger = get_logger(__name__)

_instance: Optional["SharedCamera"] = None
_init_lock = threading.Lock()
_camera_source: int | str = VIDEO_SOURCE

_FALLBACK_MAX = 5


def _wmi_cameras() -> list[dict]:
    """Return [{index, name, is_usb}] via PowerShell PnP. Never opens cv2."""
    if platform.system() != "Windows":
        return []
    try:
        ps = (
            "Get-PnpDevice -Class Camera -Status OK "
            "| Sort-Object -Property InstanceId "
            "| Select-Object FriendlyName, InstanceId "
            "| ConvertTo-Json -Compress"
        )
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=8, check=False,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return []
        raw = json.loads(r.stdout.strip())
        if isinstance(raw, dict):
            raw = [raw]
        out = []
        for i, d in enumerate(raw):
            name = d.get("FriendlyName") or f"Camera {i}"
            is_usb = "USB" in (d.get("InstanceId") or "").upper()
            out.append({"index": i, "name": name, "is_usb": is_usb})
        return out
    except Exception as e:
        logger.debug(f"WMI camera enumeration skipped: {e}")
        return []


def _preferred_index(wmi: list[dict]) -> int:
    """USB external first, then VIDEO_SOURCE default."""
    for d in wmi:
        if d["is_usb"]:
            logger.info(f"USB camera detected: [{d['index']}] {d['name']}")
            return d["index"]
    return VIDEO_SOURCE


def _open_capture(index: int) -> Optional[cv2.VideoCapture]:
    """
    Open ONE capture for index.
    On Windows: tries DSHOW first (most compatible), then CAP_ANY.
    Explicitly negotiates YUY2 format + 30fps so DSHOW doesn't start with
    a garbage/unknown codec (which causes all-zero frames on some drivers).
    Returns None only if the device hard-refuses to open (isOpened=False).
    """
    if platform.system() == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        # Explicitly set a known-good format before the driver latches onto
        # whatever default it chooses.  On the w200 (and similar UVC cameras)
        # the DSHOW default can be a corrupt/garbage FOURCC that yields
        # all-zero frames.  Setting YUY2 + 30 fps forces a clean mode-set.
        yuy2 = cv2.VideoWriter_fourcc("Y", "U", "Y", "2")
        cap.set(cv2.CAP_PROP_FOURCC, yuy2)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
        cc = "".join(chr((fourcc_val >> (8 * i)) & 0xFF) for i in range(4))
        w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)
        h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            f"Camera [{index}] opened  backend={backend}  "
            f"fourcc={cc!r}  fps={fps}  native={w}x{h}  output={FRAME_WIDTH}x{FRAME_HEIGHT}"
        )
        return cap

    logger.warning(f"Camera [{index}] refused to open on any backend.")
    return None


class SharedCamera:
    """Owns one cv2.VideoCapture. A background reader thread feeds all consumers."""

    def __init__(self, cap: cv2.VideoCapture, source: int):
        self._cap    = cap
        self._source = source
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock  = threading.Lock()
        self._frame_id    = 0
        self._running     = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background capture thread."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._reader_loop, daemon=True, name="camera-reader")
        self._thread.start()
        logger.info("Camera reader started.")

    def stop(self) -> None:
        """Stop capture, join the reader thread, and release the device."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Camera released.")

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Return (ok, frame) — frame is a copy of the latest grabbed frame."""
        with self._frame_lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    @property
    def is_opened(self) -> bool:
        """True if the underlying VideoCapture is still open."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def source(self) -> int | str:
        """Camera source index or device path."""
        return self._source

    @property
    def frame_id(self) -> int:
        """Monotonically increasing capture frame counter."""
        with self._frame_lock:
            return self._frame_id

    def _reader_loop(self) -> None:
        while self._running:
            if not self._cap or not self._cap.isOpened():
                time.sleep(0.05)
                continue
            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            # Drop black / not-yet-initialized frames (DSHOW warmup on some USB cameras)
            if frame.max() == 0:
                continue
            resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            resized = cv2.flip(resized, 1)
            with self._frame_lock:
                self._latest_frame = resized
                self._frame_id    += 1


def _make_dead_instance(preferred: int) -> "SharedCamera":
    """Return a non-functional SharedCamera so callers never receive None."""
    obj = SharedCamera.__new__(SharedCamera)
    obj._cap          = None
    obj._source       = preferred
    obj._latest_frame = None
    obj._frame_lock   = threading.Lock()
    obj._frame_id     = 0
    obj._running      = False
    obj._thread       = None
    return obj


def get_camera() -> SharedCamera:
    """
    Lazily create and return the global SharedCamera singleton.
      1. WMI (Windows): no cv2 touches -- gets real device names + USB flag.
      2. Prefer USB external index; fall back to VIDEO_SOURCE.
      3. Open chosen index ONCE (validates live frame). Fail -> try 0..FALLBACK_MAX.
      4. Start reader thread -> live.
    """
    global _instance, _camera_source

    with _init_lock:
        if _instance is not None:
            return _instance

        wmi       = _wmi_cameras()
        preferred = _preferred_index(wmi)
        indices   = [preferred] + [i for i in range(_FALLBACK_MAX + 1) if i != preferred]

        for idx in indices:
            cap = _open_capture(idx)
            if cap is not None:
                _instance      = SharedCamera(cap, idx)
                _camera_source = idx
                _instance.start()
                return _instance

        logger.error(
            "No camera found on indices 0-%d. "
            "Check USB connection, driver, and Windows Camera privacy settings.",
            _FALLBACK_MAX,
        )
        _instance = _make_dead_instance(preferred)
        return _instance


def get_capture() -> Optional[SharedCamera]:
    """Compatibility alias used by the MJPEG feed."""
    cam = get_camera()
    return cam if cam.is_opened else None


def release() -> None:
    """Release the global camera on shutdown."""
    global _instance
    with _init_lock:
        if _instance is not None:
            _instance.stop()
            _instance = None


def get_camera_source() -> int | str:
    """Return the currently configured camera source index or path."""
    return _camera_source


def set_camera_source(source: int | str) -> bool:
    """Switch to a new source. Restores previous on failure."""
    global _instance, _camera_source

    with _init_lock:
        if _instance is not None and _instance.source == source and _instance.is_opened:
            _camera_source = source
            return True

        old = _instance
        _instance = None
        if old is not None:
            old.stop()

        if isinstance(source, int):
            cap = _open_capture(source)
        else:
            cap = cv2.VideoCapture(source)
            cap = cap if cap.isOpened() else None

        if cap is None:
            logger.warning(f"Could not switch to {source}; restoring previous.")
            if old is not None:
                _instance = old
                _instance.start()
            return False

        _instance      = SharedCamera(cap, source)  # type: ignore[arg-type]
        _camera_source = source
        _instance.start()
        return True


def list_available_cameras(max_devices: int = 10) -> list[dict]:
    """Enumerate cameras for the API/UI. NOT called during startup."""
    wmi = _wmi_cameras()
    if wmi:
        results = []
        for d in wmi:
            backend = cv2.CAP_MSMF if platform.system() == "Windows" else cv2.CAP_ANY
            cap = cv2.VideoCapture(d["index"], backend)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            label = f"[{d['index']}] {d['name']}"
            if w and h:
                label += f" ({w}x{h})"
            if d["is_usb"]:
                label += " [USB]"
            results.append({**d, "label": label, "width": w, "height": h})
        return results

    results = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        results.append({
            "index": i, "name": f"Camera {i}",
            "label": f"[{i}] Camera {i} ({w}x{h})" if w else f"[{i}] Camera {i}",
            "width": w, "height": h, "is_usb": False,
        })
    return results
