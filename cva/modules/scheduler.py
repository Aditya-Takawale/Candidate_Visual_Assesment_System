"""
Smart Module Scheduler
- Controls which CV modules run on each frame based on timing rules
- Implements backpressure: drops low-priority modules when queue is deep
- Priority order: Identity > Body Language > First Impression > Grooming
- Tracks active/skipped/degraded module status for system health
"""

from __future__ import annotations
import time
from typing import Dict, Optional

from cva.config.settings import (
    IDENTITY_INTERVAL,
    GROOMING_INTERVAL,
    FIRST_IMPRESSION_DURATION,
    GROOMING_ENABLED,
)
from cva.common.models import ModuleStatus
from cva.common.logger import get_logger

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
        self._status: Dict[str, ModuleStatus] = {
            "identity": ModuleStatus.WARMING_UP,
            "body_language": ModuleStatus.WARMING_UP,
            "first_impression": ModuleStatus.WARMING_UP,
            "grooming": ModuleStatus.WARMING_UP,
        }

    def should_run(self, module: str, frame_id: int, queue_depth: int = 0) -> bool:
        now = time.time()
        elapsed_session = now - self._session_start
        backpressure = queue_depth >= BACKPRESSURE_QUEUE_THRESHOLD

        if module == "identity":
            if backpressure:
                self._status["identity"] = ModuleStatus.SKIPPED
                return False
            should = (now - self._last_identity_run) >= IDENTITY_INTERVAL
            if should:
                self._last_identity_run = now
            self._status["identity"] = ModuleStatus.ACTIVE if should else ModuleStatus.SKIPPED
            return should

        if module == "body_language":
            if backpressure:
                self._status["body_language"] = ModuleStatus.SKIPPED
                return False
            self._status["body_language"] = ModuleStatus.ACTIVE
            return True

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
            if not GROOMING_ENABLED:
                self._status["grooming"] = ModuleStatus.SKIPPED
                return False
            if backpressure:
                self._status["grooming"] = ModuleStatus.SKIPPED
                return False
            should = (now - self._last_grooming_run) >= GROOMING_INTERVAL
            if should:
                self._last_grooming_run = now
            self._status["grooming"] = ModuleStatus.ACTIVE if should else ModuleStatus.SKIPPED
            return should

        return False

    def mark_degraded(self, module: str) -> None:
        self._status[module] = ModuleStatus.DEGRADED
        logger.warning(f"Module '{module}' marked as DEGRADED.")

    def get_active_modules(self):
        return [m for m, s in self._status.items() if s == ModuleStatus.ACTIVE]

    def get_skipped_modules(self):
        return [m for m, s in self._status.items() if s == ModuleStatus.SKIPPED]

    def get_degraded_modules(self):
        return [m for m, s in self._status.items() if s == ModuleStatus.DEGRADED]

    def get_status(self) -> Dict[str, str]:
        return {m: s.value for m, s in self._status.items()}
