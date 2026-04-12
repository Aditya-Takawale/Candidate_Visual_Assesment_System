"""
CVA System — Central Configuration
All thresholds, paths, and flags are config-driven. Never hardcoded in modules.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Literal

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
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_IDLE = 1
FPS_ACTIVE = 3
BLUR_THRESHOLD = 5.0            # Laplacian variance — below this = blurry frame
BRIGHTNESS_MIN = 20             # Mean pixel brightness below this = too dark
BRIGHTNESS_MAX = 220            # Above this = overexposed

# ─────────────────────────────────────────────
# SMART SCHEDULER INTERVALS (seconds)
# ─────────────────────────────────────────────
IDENTITY_INTERVAL = 3           # Run identity check every N seconds
BODY_LANGUAGE_INTERVAL = 0      # Continuous (every frame)
FIRST_IMPRESSION_DURATION = 300 # Only during first N seconds of session
GROOMING_INTERVAL = 10          # Run grooming every N seconds (0 = disabled)
GROOMING_ENABLED = True

# ─────────────────────────────────────────────
# MULTI-FRAME AGGREGATION
# ─────────────────────────────────────────────
EMA_ALPHA = 0.3                 # Exponential moving average smoothing factor
FRAME_BUFFER_SIZE = 10          # Number of frames to aggregate over
WARMUP_FRAMES = 5               # Hold scoring until this many frames collected
BASELINE_CALIBRATION_FRAMES = 5 * FPS_ACTIVE  # First N frames = baseline calibration

# ─────────────────────────────────────────────
# IDENTITY MODULE
# ─────────────────────────────────────────────
IDENTITY_COSINE_THRESHOLD = 0.6         # Below this = face mismatch red flag
IDENTITY_MULTI_FRAME_COUNT = 5          # Average embeddings over N frames
IDENTITY_LIVENESS_ENABLED = True
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
GROOMING_CONFIDENCE_THRESHOLD = 0.5
ETHNIC_WEAR_CLASSES = [                 # Allowed formal ethnic wear (not flagged)
    "kurta", "saree", "sherwani", "salwar", "dhoti", "churidar"
]
CASUAL_WEAR_CLASSES = ["t-shirt", "hoodie", "tank-top", "shorts"]

# ─────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────
SCORING_INTERVAL_SEC = 5                # Recompute full score every N seconds
MODULE_WEIGHTS = {
    "identity": 0.30,
    "body_language": 0.35,
    "first_impression": 0.20,
    "grooming": 0.15,
}
# Role-based weight overrides
ROLE_WEIGHTS = {
    "developer": {"identity": 0.30, "body_language": 0.35, "first_impression": 0.15, "grooming": 0.20},
    "sales":     {"identity": 0.25, "body_language": 0.30, "first_impression": 0.30, "grooming": 0.15},
    "hr":        {"identity": 0.25, "body_language": 0.25, "first_impression": 0.30, "grooming": 0.20},
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
API_HOST = os.getenv("CVA_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("CVA_API_PORT", "8000"))
CORS_ORIGINS = ["*"]

# ─────────────────────────────────────────────
# PRODUCTION STUBS (design only — not active in demo)
# ─────────────────────────────────────────────
# KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
# REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
# FEAST_REPO_PATH = BASE_DIR / "infra" / "feast"
# TRITON_URL = os.getenv("TRITON_URL", "localhost:8001")
