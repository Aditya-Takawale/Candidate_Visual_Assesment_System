"""
Shared Pydantic data models used across all CVA modules.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import time


class RedFlagSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ModuleStatus(str, Enum):
    ACTIVE = "active"
    SKIPPED = "skipped"
    DEGRADED = "degraded"
    WARMING_UP = "warming_up"


@dataclass
class RedFlag:
    module: str
    reason: str
    severity: RedFlagSeverity
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    detail: Optional[str] = None


@dataclass
class FrameFeatures:
    """Raw per-frame output from each CV module."""
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0

    # Identity
    face_detected: bool = False
    face_cosine_similarity: Optional[float] = None
    identity_verified: bool = False

    # Body Language
    gaze_on_camera: bool = True
    gaze_off_seconds: float = 0.0
    posture_angle_deg: float = 0.0
    posture_slouch: bool = False
    fidget_score: float = 0.0
    emotion: Optional[str] = None
    emotion_negative_seconds: float = 0.0

    # First Impression
    smile_detected: bool = False
    speech_rms: float = 0.0
    speech_active: bool = False

    # Grooming
    attire_class: Optional[str] = None
    grooming_score: Optional[float] = None


@dataclass
class AggregatedFeatures:
    """Smoothed features after multi-frame EMA aggregation."""
    session_id: str = ""
    candidate_id: str = ""
    frame_count: int = 0
    is_warmed_up: bool = False

    # Identity
    identity_score: float = 1.0
    identity_verified: bool = False
    identity_red_flags: int = 0

    # Body Language
    gaze_score: float = 1.0        # 1.0 = perfect eye contact
    posture_score: float = 1.0     # 1.0 = upright
    fidget_score: float = 0.0      # 0.0 = no fidgeting
    emotion_score: float = 1.0     # 1.0 = positive/neutral

    # First Impression
    smile_ratio: float = 0.0
    speech_energy_score: float = 1.0
    punctuality_score: float = 1.0

    # Grooming
    grooming_score: float = 1.0

    # Red flags accumulated
    red_flags: List[RedFlag] = field(default_factory=list)


@dataclass
class ScoringResult:
    """Final output of the scoring engine."""
    session_id: str
    candidate_id: str
    role: str
    timestamp: float = field(default_factory=time.time)

    final_score: float = 0.0           # 0–100
    module_scores: Dict[str, float] = field(default_factory=dict)
    shap_breakdown: Dict[str, float] = field(default_factory=dict)
    red_flags: List[RedFlag] = field(default_factory=list)
    score_reason: str = ""
    model_version: str = "demo-v1"


@dataclass
class SystemHealth:
    """Live system performance metrics for dashboard."""
    fps: float = 0.0
    target_fps: int = 3
    hardware_backend: str = "CPU"
    active_modules: List[str] = field(default_factory=list)
    skipped_modules: List[str] = field(default_factory=list)
    degraded_modules: List[str] = field(default_factory=list)
    frame_queue_depth: int = 0
    warmup_remaining: int = 0
