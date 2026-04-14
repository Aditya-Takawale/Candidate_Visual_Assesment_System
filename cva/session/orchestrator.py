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


# ── Module singleton cache (loaded once, reused across sessions) ──
_cached_identity: Optional[IdentityVerifier] = None
_cached_body_language: Optional[BodyLanguageAnalyzer] = None
_cached_grooming: Optional[GroomingAnalyzer] = None
_cached_scorer: dict[str, ScoringEngine] = {}


def _get_identity() -> IdentityVerifier:
    global _cached_identity
    if _cached_identity is None:
        _cached_identity = IdentityVerifier()
    return _cached_identity


def _get_body_language() -> BodyLanguageAnalyzer:
    global _cached_body_language
    if _cached_body_language is None:
        _cached_body_language = BodyLanguageAnalyzer()
    return _cached_body_language


def _get_grooming() -> GroomingAnalyzer:
    global _cached_grooming
    if _cached_grooming is None:
        _cached_grooming = GroomingAnalyzer()
    return _cached_grooming


def _get_scorer(role: str) -> ScoringEngine:
    if role not in _cached_scorer:
        _cached_scorer[role] = ScoringEngine(role=role)
    return _cached_scorer[role]


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
        self.session_id = session_id or str(uuid.uuid4())
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
        # Reuse cached module instances (models loaded only once)
        self._identity = _get_identity()
        self._body_language = _get_body_language()
        self._identity.reset_session()
        self._body_language.reset()
        self._first_impression = FirstImpressionAnalyzer(scheduled_start_time=scheduled_start)
        self._grooming = _get_grooming()
        self._grooming.reset()
        self._scorer = _get_scorer(role)
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

    def set_reference_image(self, frame, det_thresh: float = 0.5) -> bool:
        return self._identity.set_reference(frame, det_thresh=det_thresh)

    def set_candidate_names(self, aadhaar_name: str, cv_name: str) -> None:
        self._identity._reference_name = aadhaar_name
        self._identity.set_cv_name(cv_name)

    # ──────────────────────────────────────────
    # Internal processing loop
    # ──────────────────────────────────────────

    def _process_loop(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cva-mod")

        # Define module runners once — not inside the while loop
        def _run_identity(f, feat):
            feat = self._identity.process_frame(f, feat)
            name_flag = self._identity.check_name_match()
            return feat, name_flag

        def _run_body_language(f, feat):
            return self._body_language.process_frame(f, feat)

        def _run_first_impression(f, feat):
            return self._first_impression.process_frame(f, feat)

        def _run_grooming(f, feat):
            return self._grooming.process_frame(f, feat)

        while self._running:
            try:
                payload = self._frame_queue.get(timeout=0.1)
            except Empty:
                continue

            # ── Smart frame skipping: drop stale frames ──
            queue_depth = self._frame_queue.qsize()
            if queue_depth > 3:
                # Skip to latest frame — discard stale ones
                while not self._frame_queue.empty():
                    try:
                        payload = self._frame_queue.get_nowait()
                    except Empty:
                        break
                queue_depth = 0

            frame = payload["frame"]
            frame_id = payload["frame_id"]
            timestamp = payload["timestamp"]

            features = FrameFeatures(timestamp=timestamp, frame_id=frame_id)
            active = []
            skipped = []
            futures = {}

            # ── Submit modules in parallel ────────────
            if self._scheduler.should_run("identity", frame_id, queue_depth):
                f_id = FrameFeatures(timestamp=timestamp, frame_id=frame_id)
                futures["identity"] = pool.submit(_run_identity, frame, f_id)
            else:
                skipped.append("identity")

            if self._scheduler.should_run("body_language", frame_id, queue_depth):
                f_bl = FrameFeatures(timestamp=timestamp, frame_id=frame_id)
                futures["body_language"] = pool.submit(_run_body_language, frame, f_bl)
            else:
                skipped.append("body_language")

            if self._scheduler.should_run("first_impression", frame_id, queue_depth):
                f_fi = FrameFeatures(timestamp=timestamp, frame_id=frame_id)
                futures["first_impression"] = pool.submit(_run_first_impression, frame, f_fi)
            else:
                skipped.append("first_impression")

            if self._scheduler.should_run("grooming", frame_id, queue_depth):
                f_gr = FrameFeatures(timestamp=timestamp, frame_id=frame_id)
                futures["grooming"] = pool.submit(_run_grooming, frame, f_gr)
            else:
                skipped.append("grooming")

            # ── Collect results and merge into features ──
            for module, future in futures.items():
                try:
                    result = future.result(timeout=2.0)
                    if module == "identity":
                        mod_feat, name_flag = result
                        features.face_detected = mod_feat.face_detected
                        features.face_cosine_similarity = mod_feat.face_cosine_similarity
                        features.identity_verified = mod_feat.identity_verified
                        if name_flag:
                            self._aggregator._add_flag(
                                name_flag.module, name_flag.reason,
                                name_flag.severity, name_flag.confidence,
                            )
                    elif module == "body_language":
                        mod_feat = result
                        features.face_in_frame = mod_feat.face_in_frame
                        features.gaze_on_camera = mod_feat.gaze_on_camera
                        features.gaze_off_seconds = mod_feat.gaze_off_seconds
                        features.posture_angle_deg = mod_feat.posture_angle_deg
                        features.posture_slouch = mod_feat.posture_slouch
                        features.fidget_score = mod_feat.fidget_score
                        features.emotion = mod_feat.emotion
                    elif module == "first_impression":
                        mod_feat = result
                        features.smile_detected = mod_feat.smile_detected
                        features.speech_rms = mod_feat.speech_rms
                        features.speech_active = mod_feat.speech_active
                    elif module == "grooming":
                        mod_feat = result
                        features.attire_class = mod_feat.attire_class
                        features.grooming_score = mod_feat.grooming_score
                    active.append(module)
                except Exception as e:
                    logger.warning(f"{module} module error: {e}")
                    self._scheduler.mark_degraded(module)

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

        pool.shutdown(wait=False)

    def _update_fps(self, fps: float) -> None:
        self._measured_fps = fps
        self._health.fps = fps

    @property
    def health(self) -> SystemHealth:
        return self._health
