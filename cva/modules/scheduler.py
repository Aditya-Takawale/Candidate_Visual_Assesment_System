"""
Smart Module Scheduler
- Controls which CV modules run on each frame based on timing rules
- Implements backpressure: drops low-priority modules when queue is deep
- Priority order: Identity > Body Language > First Impression > Grooming
- Tracks active/skipped/degraded module status for system health
"""
# pylint: disable=line-too-long

from __future__ import annotations
import time
from typing import Dict, Optional

from cva.config.settings import (
    IDENTITY_INTERVAL,
    GROOMING_INTERVAL,
    FIRST_IMPRESSION_DURATION,
    GROOMING_ENABLED,
    CPU_IDENTITY_INTERVAL,
    CPU_BODY_LANGUAGE_FRAME_STRIDE,
    CPU_GROOMING_INTERVAL,
    CPU_GROOMING_ENABLED,
    APPLE_SILICON_IDENTITY_INTERVAL,
    APPLE_SILICON_BODY_LANGUAGE_FRAME_STRIDE,
    APPLE_SILICON_GROOMING_INTERVAL,
    APPLE_SILICON_GROOMING_ENABLED,
)
from cva.common.models import ModuleStatus
from cva.common.logger import get_logger
from cva.common.hardware import get_runtime_profile

logger = get_logger(__name__)

BACKPRESSURE_QUEUE_THRESHOLD = 10


class ModuleScheduler:
    """
    Decides per-frame which modules should execute.
    Applies timing rules and backpressure degradation.
    """

    def __init__(self, session_start_time: Optional[float] = None):
        self._session_start = session_start_time or time.time()
        self._last_identity_run: float = 0.0
        self._last_grooming_run: float = 0.0
        self._runtime_profile = get_runtime_profile()

        if self._runtime_profile == "cpu":
            self._identity_interval = CPU_IDENTITY_INTERVAL
            self._grooming_interval = CPU_GROOMING_INTERVAL
            self._body_language_stride = max(1, CPU_BODY_LANGUAGE_FRAME_STRIDE)
            self._grooming_enabled = CPU_GROOMING_ENABLED
        elif self._runtime_profile == "apple_silicon":
            self._identity_interval = APPLE_SILICON_IDENTITY_INTERVAL
            self._grooming_interval = APPLE_SILICON_GROOMING_INTERVAL
            self._body_language_stride = max(1, APPLE_SILICON_BODY_LANGUAGE_FRAME_STRIDE)
            self._grooming_enabled = APPLE_SILICON_GROOMING_ENABLED
        else:
            self._identity_interval = IDENTITY_INTERVAL
            self._grooming_interval = GROOMING_INTERVAL
            self._body_language_stride = 1
            self._grooming_enabled = GROOMING_ENABLED
        self._status: Dict[str, ModuleStatus] = {
            "identity": ModuleStatus.WARMING_UP,
            "body_language": ModuleStatus.WARMING_UP,
            "first_impression": ModuleStatus.WARMING_UP,
            "grooming": ModuleStatus.WARMING_UP,
        }
        logger.info(
            "Scheduler runtime profile: %s | identity=%.1fs | body_stride=%s | grooming=%.1fs | grooming_enabled=%s",
            self._runtime_profile,
            self._identity_interval,
            self._body_language_stride,
            self._grooming_interval,
            self._grooming_enabled,
        )

    def should_run(self, module: str, frame_id: int, queue_depth: int = 0) -> bool:
        """Return True if the given module should execute on this frame."""
        now = time.time()
        elapsed_session = now - self._session_start
        backpressure = queue_depth >= BACKPRESSURE_QUEUE_THRESHOLD

        if module == "identity":
            # Identity is security-critical; under backpressure we slow it down
            # but do not disable it entirely.
            interval = self._identity_interval * 2.0 if backpressure else self._identity_interval
            should = (now - self._last_identity_run) >= interval
            if should:
                self._last_identity_run = now
            self._status["identity"] = ModuleStatus.ACTIVE if should else ModuleStatus.SKIPPED
            return should

        if module == "body_language":
            if backpressure:
                self._status["body_language"] = ModuleStatus.SKIPPED
                return False
            should = frame_id % self._body_language_stride == 0
            self._status["body_language"] = ModuleStatus.ACTIVE if should else ModuleStatus.SKIPPED
            return should

        if module == "first_impression":
            if elapsed_session > FIRST_IMPRESSION_DURATION:
                self._status["first_impression"] = ModuleStatus.SKIPPED
                return False
            if backpressure:
                self._status["first_impression"] = ModuleStatus.SKIPPED
                return False
            self._status["first_impression"] = ModuleStatus.ACTIVE
            return True

        if module == "grooming":
            if not self._grooming_enabled:
                self._status["grooming"] = ModuleStatus.SKIPPED
                return False
            if backpressure:
                self._status["grooming"] = ModuleStatus.SKIPPED
                return False
            should = (now - self._last_grooming_run) >= self._grooming_interval
            if should:
                self._last_grooming_run = now
            self._status["grooming"] = ModuleStatus.ACTIVE if should else ModuleStatus.SKIPPED
            return should

        return False

    def mark_degraded(self, module: str) -> None:
        """Mark a module as degraded (e.g., after repeated inference errors)."""
        self._status[module] = ModuleStatus.DEGRADED
        logger.warning(f"Module '{module}' marked as DEGRADED.")

    def get_active_modules(self):
        """Return list of currently active module names."""
        return [m for m, s in self._status.items() if s == ModuleStatus.ACTIVE]

    def get_skipped_modules(self):
        """Return list of currently skipped module names."""
        return [m for m, s in self._status.items() if s == ModuleStatus.SKIPPED]

    def get_degraded_modules(self):
        """Return list of currently degraded module names."""
        return [m for m, s in self._status.items() if s == ModuleStatus.DEGRADED]

    def get_status(self) -> Dict[str, str]:
        """Return a dict mapping module name to its current status string."""
        return {m: s.value for m, s in self._status.items()}
