"""
CVA System — Central Configuration
All thresholds, paths, and flags are config-driven. Never hardcoded in modules.
"""
# pylint: disable=line-too-long

from __future__ import annotations
import os
from pathlib import Path
from typing import Literal, Optional

# ─────────────────────────────────────────────
# MODE
# ─────────────────────────────────────────────
MODE: Literal["demo", "production"] = os.getenv("CVA_MODE", "demo")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

for _d in (MODELS_DIR, LOGS_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# VIDEO INGESTION
# ─────────────────────────────────────────────
VIDEO_SOURCE = int(os.getenv("CVA_VIDEO_SOURCE", "0"))   # 0 = webcam; path for file
FRAME_WIDTH = 480
FRAME_HEIGHT = 360
FPS_IDLE = 1
FPS_ACTIVE = 4
BLUR_THRESHOLD = 5.0            # Laplacian variance — below this = blurry frame
BRIGHTNESS_MIN = 20             # Mean pixel brightness below this = too dark
BRIGHTNESS_MAX = 220            # Above this = overexposed

# ─────────────────────────────────────────────
# SMART SCHEDULER INTERVALS (seconds)
# ─────────────────────────────────────────────
IDENTITY_INTERVAL = 2           # Run identity check every N seconds (more responsive in real-time mode)
BODY_LANGUAGE_INTERVAL = 0      # Continuous (every frame)
FIRST_IMPRESSION_DURATION = 300 # Only during first N seconds of session
GROOMING_INTERVAL = 15          # Run grooming every N seconds (relaxed for CPU)
GROOMING_ENABLED = True
CPU_IDENTITY_INTERVAL = float(os.getenv("CVA_CPU_IDENTITY_INTERVAL", "3.0"))
CPU_BODY_LANGUAGE_FRAME_STRIDE = int(os.getenv("CVA_CPU_BODY_LANGUAGE_FRAME_STRIDE", "2"))
CPU_GROOMING_INTERVAL = float(os.getenv("CVA_CPU_GROOMING_INTERVAL", "30.0"))
CPU_GROOMING_ENABLED = os.getenv("CVA_CPU_GROOMING_ENABLED", "false").lower() == "true"
APPLE_SILICON_IDENTITY_INTERVAL = float(os.getenv("CVA_APPLE_IDENTITY_INTERVAL", "2.5"))
APPLE_SILICON_BODY_LANGUAGE_FRAME_STRIDE = int(os.getenv("CVA_APPLE_BODY_LANGUAGE_FRAME_STRIDE", "2"))
APPLE_SILICON_GROOMING_INTERVAL = float(os.getenv("CVA_APPLE_GROOMING_INTERVAL", "20.0"))
APPLE_SILICON_GROOMING_ENABLED = os.getenv("CVA_APPLE_GROOMING_ENABLED", "true").lower() == "true"

# ─────────────────────────────────────────────
# MULTI-FRAME AGGREGATION
# ─────────────────────────────────────────────
EMA_ALPHA = 0.15                # Smoothing factor (lower = smoother at low FPS)
FRAME_BUFFER_SIZE = 10          # Number of frames to aggregate over
WARMUP_FRAMES = 5               # Hold scoring until this many frames collected
BASELINE_CALIBRATION_FRAMES = 5 * FPS_ACTIVE  # First N frames = baseline calibration

# ─────────────────────────────────────────────
# IDENTITY MODULE
# ─────────────────────────────────────────────
IDENTITY_COSINE_THRESHOLD = 0.60        # Below this = face mismatch red flag (strict: InsightFace same-person ~0.55–0.80)
IDENTITY_MULTI_FRAME_COUNT = 10         # Average embeddings over N frames (more frames = more robust)
IDENTITY_LIVENESS_ENABLED = True
IDENTITY_MISMATCH_SCORE_CAP = 35        # Final score hard-cap when confirmed identity mismatch
IDENTITY_OCCLUSION_DET_THRESH = 0.40   # Relaxed detection threshold for head coverings (turban, hijab, mask)
IDENTITY_OCCLUSION_SIMILARITY_GRACE = 0.08  # Similarity grace when InsightFace det_score < 0.65 (partial occlusion)
FUZZY_MATCH_THRESHOLD = 80              # RapidFuzz score below this = name mismatch

# ─────────────────────────────────────────────
# BODY LANGUAGE MODULE
# ─────────────────────────────────────────────
GAZE_OFF_CAMERA_THRESHOLD_SEC = 5.0     # Flag if gaze off-camera > N seconds
SLOUCH_ANGLE_THRESHOLD_DEG = 20.0       # Flag if slouch > N degrees from baseline
FIDGET_MOTION_THRESHOLD = 0.15          # Normalised optical flow magnitude threshold
FIDGET_WINDOW_SEC = 5.0                 # Rolling window for fidget detection
NEGATIVE_EMOTION_THRESHOLD_SEC = 10.0  # Flag if negative emotion continuous > N sec

# ─────────────────────────────────────────────
# FIRST IMPRESSION MODULE
# ─────────────────────────────────────────────
PUNCTUALITY_GRACE_PERIOD_SEC = 120      # Flag if joins > N seconds late
SMILE_LANDMARK_THRESHOLD = 0.3         # Lip corner ratio threshold for smile
SPEECH_RMS_THRESHOLD = 0.02            # librosa RMS below this = low energy
SPEECH_PAUSE_MAX_SEC = 3.0             # Flag pause longer than N seconds
FILLER_WORDS = ["uh", "um", "er", "hmm", "like"]

# ─────────────────────────────────────────────
# GROOMING MODULE
# ─────────────────────────────────────────────
GROOMING_CONFIDENCE_THRESHOLD = 0.30    # Lower threshold catches more clothing signals (was 0.5)
GROOMING_UNCONFIRMED_SCORE = 0.35       # Score when person is visible but no formal indicator detected
GROOMING_ABSOLUTE_SCORE_CAP = 50        # Final score cap when confirmed casual / unconfirmed attire
SLOUCH_ABSOLUTE_MAX_DEG = 12.0          # Head tilt beyond this = slouch regardless of personal baseline
ETHNIC_WEAR_CLASSES = [                 # Allowed formal ethnic wear (not flagged)
    "kurta", "saree", "sherwani", "salwar", "dhoti", "churidar"
]
CASUAL_WEAR_CLASSES = ["t-shirt", "hoodie", "tank-top", "shorts"]

# ─────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────
SCORING_INTERVAL_SEC = 8                # Recompute score every N seconds (more frames at low FPS)
SHAP_REFRESH_INTERVAL_SEC = float(os.getenv("CVA_SHAP_REFRESH_INTERVAL_SEC", "15"))
MODULE_WEIGHTS = {
    "identity": 0.20,
    "body_language": 0.40,
    "first_impression": 0.20,
    "grooming": 0.20,
}
# Role-based weight overrides
ROLE_WEIGHTS = {
    "developer": {"identity": 0.20, "body_language": 0.40, "first_impression": 0.15, "grooming": 0.25},
    "sales":     {"identity": 0.15, "body_language": 0.30, "first_impression": 0.35, "grooming": 0.20},
    "hr":        {"identity": 0.15, "body_language": 0.30, "first_impression": 0.30, "grooming": 0.25},
}
DEFAULT_ROLE = os.getenv("CVA_ROLE", "developer")

# ─────────────────────────────────────────────
# FEATURE STORAGE
# ─────────────────────────────────────────────
STORAGE_BACKEND: Literal["memory", "sqlite"] = os.getenv("CVA_STORAGE", "memory")
SQLITE_DB_PATH = DATA_DIR / "cva_features.db"

# ─────────────────────────────────────────────
# FASTAPI / DASHBOARD
# ─────────────────────────────────────────────
API_HOST = os.getenv("CVA_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("CVA_API_PORT", "8000"))
_cors_raw = os.getenv("CVA_CORS_ORIGINS", "*")
CORS_ORIGINS: list = [o.strip() for o in _cors_raw.split(",")]
# Set CVA_API_KEY env var to require auth on all session endpoints (None = open/demo)
API_KEY: Optional[str] = os.getenv("CVA_API_KEY") or None
# Maximum base64 payload accepted on upload endpoints (~10 MB image → ~13.4 MB base64)
MAX_UPLOAD_BYTES: int = int(os.getenv("CVA_MAX_UPLOAD_BYTES", str(14 * 1024 * 1024)))

# ─────────────────────────────────────────────
# PRODUCTION STUBS (design only — not active in demo)
# ─────────────────────────────────────────────
# KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
# REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
# FEAST_REPO_PATH = BASE_DIR / "infra" / "feast"
# TRITON_URL = os.getenv("TRITON_URL", "localhost:8001")
