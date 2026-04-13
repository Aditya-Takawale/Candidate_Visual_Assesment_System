"""
Feature Store — Config-driven backend: "memory" | "sqlite"
Stores per-session feature snapshots every 30 seconds.
In production this would be replaced by Feast.
"""

from __future__ import annotations
import json
import time
import sqlite3
import threading
from typing import Dict, List, Optional
from dataclasses import asdict

from cva.config.settings import STORAGE_BACKEND, SQLITE_DB_PATH, SCORING_INTERVAL_SEC
from cva.common.models import AggregatedFeatures
from cva.common.logger import get_logger

logger = get_logger(__name__)


class InMemoryFeatureStore:
    def __init__(self):
        self._store: Dict[str, List[dict]] = {}

    def save_snapshot(self, features: AggregatedFeatures) -> None:
        key = features.session_id or "default"
        if key not in self._store:
            self._store[key] = []
        snapshot = {
            **asdict(features),
            "snapshot_ts": time.time(),
        }
        snapshot.pop("red_flags", None)
        self._store[key].append(snapshot)

    def get_latest(self, session_id: str) -> Optional[dict]:
        snaps = self._store.get(session_id, [])
        return snaps[-1] if snaps else None

    def get_history(self, session_id: str) -> List[dict]:
        return self._store.get(session_id, [])


class SQLiteFeatureStore:
    def __init__(self):
        self._conn = sqlite3.connect(str(SQLITE_DB_PATH), check_same_thread=False)
        self._lock = threading.Lock()  # serialize all DB access across threads
        self._create_tables()
        logger.info(f"SQLite feature store at {SQLITE_DB_PATH}")

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                candidate_id TEXT,
                snapshot_ts REAL NOT NULL,
                frame_count INTEGER,
                identity_score REAL,
                gaze_score REAL,
                posture_score REAL,
                fidget_score REAL,
                smile_ratio REAL,
                speech_energy_score REAL,
                grooming_score REAL,
                payload TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS red_flags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                module TEXT,
                reason TEXT,
                severity TEXT,
                confidence REAL,
                timestamp REAL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS final_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                candidate_id TEXT,
                role TEXT,
                final_score REAL,
                module_scores TEXT,
                shap_breakdown TEXT,
                model_version TEXT,
                scored_at REAL
            )
        """)
        self._conn.commit()

    def save_snapshot(self, features: AggregatedFeatures) -> None:
        d = asdict(features)
        d.pop("red_flags", None)
        with self._lock:
            self._conn.execute("""
                INSERT INTO feature_snapshots
                (session_id, candidate_id, snapshot_ts, frame_count,
                 identity_score, gaze_score, posture_score, fidget_score,
                 smile_ratio, speech_energy_score, grooming_score, payload)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                features.session_id, features.candidate_id, time.time(),
                features.frame_count, features.identity_score, features.gaze_score,
                features.posture_score, features.fidget_score, features.smile_ratio,
                features.speech_energy_score, features.grooming_score,
                json.dumps(d),
            ))
            self._conn.commit()

    def save_red_flag(self, session_id: str, flag) -> None:
        with self._lock:
            self._conn.execute("""
                INSERT INTO red_flags (session_id, module, reason, severity, confidence, timestamp)
                VALUES (?,?,?,?,?,?)
            """, (session_id, flag.module, flag.reason, flag.severity, flag.confidence, flag.timestamp))
            self._conn.commit()

    def save_final_score(self, result) -> None:
        with self._lock:
            self._conn.execute("""
                INSERT INTO final_scores
                (session_id, candidate_id, role, final_score, module_scores, shap_breakdown, model_version, scored_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                result.session_id, result.candidate_id, result.role,
                result.final_score,
                json.dumps(result.module_scores),
                json.dumps(result.shap_breakdown),
                result.model_version, result.timestamp,
            ))
            self._conn.commit()

    def get_latest(self, session_id: str) -> Optional[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT payload FROM feature_snapshots WHERE session_id=? ORDER BY snapshot_ts DESC LIMIT 1",
                (session_id,)
            )
            row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def get_history(self, session_id: str) -> List[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT payload FROM feature_snapshots WHERE session_id=? ORDER BY snapshot_ts ASC",
                (session_id,)
            )
            rows = cur.fetchall()
        return [json.loads(r[0]) for r in rows]


def get_feature_store():
    """Factory: returns the configured feature store backend."""
    if STORAGE_BACKEND == "sqlite":
        return SQLiteFeatureStore()
    return InMemoryFeatureStore()
