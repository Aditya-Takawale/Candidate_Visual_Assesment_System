"""
Session Orchestrator
- Owns the main processing loop
- Ties together: FrameSampler → Scheduler → CV Modules → Aggregator → Scorer → Store
- Publishes results via asyncio queue for FastAPI WebSocket consumption
- Tracks SystemHealth in real time
"""

from __future__ import annotations
import asyncio
import threading
import time
import uuid
from queue import Queue, Empty
from typing import Optional, Callable

from cva.config.settings import SCORING_INTERVAL_SEC, WARMUP_FRAMES
from cva.common.models import FrameFeatures, SystemHealth, ScoringResult
from cva.common.hardware import get_primary_backend
from cva.common.logger import get_logger
from cva.ingestion.frame_sampler import FrameSampler
from cva.modules.scheduler import ModuleScheduler
from cva.modules.aggregator import MultiFrameAggregator
from cva.modules.identity.verifier import IdentityVerifier
from cva.modules.body_language.analyzer import BodyLanguageAnalyzer
from cva.modules.first_impression.analyzer import FirstImpressionAnalyzer
from cva.modules.grooming.analyzer import GroomingAnalyzer
from cva.scoring.engine import ScoringEngine
from cva.storage.feature_store import get_feature_store

logger = get_logger(__name__)


class SessionOrchestrator:
    """
    Manages a single candidate assessment session end-to-end.
    Runs in a background thread; exposes results via asyncio queue.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        role: str = "developer",
        scheduled_start: Optional[float] = None,
        on_score: Optional[Callable[[ScoringResult], None]] = None,
        on_health: Optional[Callable[[SystemHealth], None]] = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.candidate_id = candidate_id or "candidate_001"
        self._role = role
        self._on_score = on_score
        self._on_health = on_health

        self._frame_queue: Queue = Queue(maxsize=30)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._sampler = FrameSampler(
            output_queue=self._frame_queue,
            on_fps_update=self._update_fps,
        )
        self._scheduler = ModuleScheduler(session_start_time=scheduled_start)
        self._aggregator = MultiFrameAggregator(
            session_id=self.session_id,
            candidate_id=self.candidate_id,
        )
        self._identity = IdentityVerifier()
        self._body_language = BodyLanguageAnalyzer()
        self._first_impression = FirstImpressionAnalyzer(scheduled_start_time=scheduled_start)
        self._grooming = GroomingAnalyzer()
        self._scorer = ScoringEngine(role=role)
        self._store = get_feature_store()

        self._health = SystemHealth(hardware_backend=get_primary_backend())
        self._last_score_time: float = 0.0
        self._last_snapshot_time: float = 0.0
        self._measured_fps: float = 0.0

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._sampler.start()
        self._thread = threading.Thread(target=self._process_loop, daemon=True, name="cva-session")
        self._thread.start()
        logger.info(f"Session '{self.session_id}' started. Backend: {self._health.hardware_backend}")

    def stop(self) -> None:
        self._running = False
        self._sampler.stop()
        self._first_impression.stop()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(f"Session '{self.session_id}' stopped.")

    def set_reference_image(self, frame) -> bool:
        return self._identity.set_reference(frame)

    def set_candidate_names(self, aadhaar_name: str, cv_name: str) -> None:
        self._identity._reference_name = aadhaar_name
        self._identity.set_cv_name(cv_name)

    # ──────────────────────────────────────────
    # Internal processing loop
    # ──────────────────────────────────────────

    def _process_loop(self) -> None:
        while self._running:
            try:
                payload = self._frame_queue.get(timeout=0.1)
            except Empty:
                continue

            frame = payload["frame"]
            frame_id = payload["frame_id"]
            timestamp = payload["timestamp"]
            queue_depth = self._frame_queue.qsize()

            features = FrameFeatures(timestamp=timestamp, frame_id=frame_id)
            active = []
            skipped = []

            # ── Identity ──────────────────────────────
            if self._scheduler.should_run("identity", frame_id, queue_depth):
                try:
                    features = self._identity.process_frame(frame, features)
                    name_flag = self._identity.check_name_match()
                    active.append("identity")
                except Exception as e:
                    logger.warning(f"Identity module error: {e}")
                    self._scheduler.mark_degraded("identity")
            else:
                skipped.append("identity")

            # ── Body Language ─────────────────────────
            if self._scheduler.should_run("body_language", frame_id, queue_depth):
                try:
                    features = self._body_language.process_frame(frame, features)
                    active.append("body_language")
                except Exception as e:
                    logger.warning(f"Body language module error: {e}")
                    self._scheduler.mark_degraded("body_language")
            else:
                skipped.append("body_language")

            # ── First Impression ──────────────────────
            if self._scheduler.should_run("first_impression", frame_id, queue_depth):
                try:
                    features = self._first_impression.process_frame(frame, features)
                    active.append("first_impression")
                except Exception as e:
                    logger.warning(f"First impression module error: {e}")
                    self._scheduler.mark_degraded("first_impression")
            else:
                skipped.append("first_impression")

            # ── Grooming ──────────────────────────────
            if self._scheduler.should_run("grooming", frame_id, queue_depth):
                try:
                    features = self._grooming.process_frame(frame, features)
                    active.append("grooming")
                except Exception as e:
                    logger.warning(f"Grooming module error: {e}")
                    self._scheduler.mark_degraded("grooming")
            else:
                skipped.append("grooming")

            # ── Aggregate ─────────────────────────────
            self._aggregator.ingest(features)
            agg = self._aggregator.get_aggregated()

            # ── Periodic scoring ──────────────────────
            now = time.time()
            if (now - self._last_score_time) >= SCORING_INTERVAL_SEC:
                result = self._scorer.score(agg)
                if result:
                    self._last_score_time = now
                    if self._on_score:
                        self._on_score(result)
                    if hasattr(self._store, "save_final_score"):
                        self._store.save_final_score(result)

            # ── Periodic feature snapshot ─────────────
            if (now - self._last_snapshot_time) >= SCORING_INTERVAL_SEC:
                self._store.save_snapshot(agg)
                self._last_snapshot_time = now

            # ── Health update ─────────────────────────
            self._health.fps = self._measured_fps
            self._health.active_modules = active
            self._health.skipped_modules = skipped
            self._health.degraded_modules = self._scheduler.get_degraded_modules()
            self._health.frame_queue_depth = queue_depth
            self._health.warmup_remaining = max(0, WARMUP_FRAMES - agg.frame_count)
            if self._on_health:
                self._on_health(self._health)

    def _update_fps(self, fps: float) -> None:
        self._measured_fps = fps
        self._health.fps = fps

    @property
    def health(self) -> SystemHealth:
        return self._health
