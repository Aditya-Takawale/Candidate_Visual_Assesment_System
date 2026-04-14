# Candidate Visual Assessment System (CVA)
**Demo Mode — Local Laptop Execution**

---

## Architecture

```
Webcam/Video
    ↓
FrameSampler (OpenCV, adaptive 1–3 FPS, quality filter)
    ↓
ModuleScheduler (smart timing + backpressure)
    ├── Identity     (InsightFace ONNX, every 5s)
    ├── Body Language (MediaPipe, continuous)
    ├── First Impression (FaceMesh + audio, first 30s)
    └── Grooming     (YOLOv8n, every 15s, optional)
    ↓
MultiFrameAggregator (EMA smoothing, temporal rules)
    ↓
ScoringEngine (XGBoost + SHAP, role-based weights)
    ↓
FastAPI + WebSocket → Dashboard (http://localhost:8000)
```

---

## Quick Start

```bash
# 1. Clone / open project
cd Candidate_Visual_Assesment

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure (optional)
cp .env.example .env
# Edit .env as needed

# 5. Run
python run.py
```

Open **http://localhost:8000** in your browser.

---

## Hardware Backends (Auto-Detected)

| Hardware | Backend Used |
|---|---|
| NVIDIA GPU | CUDA (onnxruntime-gpu) |
| Apple Silicon | CoreML / MPS |
| Windows GPU | DirectML |
| CPU only | CPUExecutionProvider |

No manual selection required — auto-detected at startup.

For Apple Silicon testers, the app now uses:
- ONNX Runtime via `onnxruntime-silicon` for InsightFace where available
- PyTorch `mps` for YOLOv8 grooming inference
- Automatic fallback to CPU if Metal acceleration is unavailable

Expected behaviour on macOS:
- Apple Silicon: functionality should match closely, but speed will usually be below an RTX laptop
- Intel Mac: works in CPU mode, but noticeably slower

Apple Silicon runtime profile:
- identity runs slightly less often than on CUDA
- body-language analysis samples every 2 frames instead of every frame
- grooming stays enabled, but runs less often and uses a smaller YOLO input size on MPS

Install notes:
- Windows/Linux with NVIDIA: install the CUDA PyTorch wheel
- Apple Silicon Mac: `pip install torch torchvision` is sufficient for MPS

---

## Project Structure

```
cva/
├── config/
│   └── settings.py          # All thresholds + config (never hardcoded)
├── common/
│   ├── hardware.py          # ONNX backend auto-detection
│   ├── logger.py            # Structured logging
│   └── models.py            # Shared dataclasses
├── ingestion/
│   └── frame_sampler.py     # OpenCV adaptive frame sampler
├── modules/
│   ├── identity/            # InsightFace + OCR + name match
│   ├── body_language/       # MediaPipe pose + gaze + fidget
│   ├── first_impression/    # Smile + VAD + speech energy
│   ├── grooming/            # YOLOv8n attire detection
│   ├── scheduler.py         # Smart module scheduler
│   └── aggregator.py        # Multi-frame EMA aggregator
├── scoring/
│   └── engine.py            # XGBoost + SHAP scoring
├── storage/
│   └── feature_store.py     # Memory / SQLite feature store
├── session/
│   └── orchestrator.py      # Main session processing loop
├── api/
│   └── main.py              # FastAPI backend + WebSocket
└── dashboard/
    ├── index.html           # Dashboard UI
    └── app.js               # Dashboard JS
```

---

## Configuration

All thresholds are in `cva/config/settings.py` — **never hardcoded** in modules:

| Setting | Default | Description |
|---|---|---|
| `IDENTITY_INTERVAL` | 5s | How often identity runs |
| `GAZE_OFF_CAMERA_THRESHOLD_SEC` | 5s | Gaze off-camera red flag |
| `SLOUCH_ANGLE_THRESHOLD_DEG` | 20° | Slouch deviation from baseline |
| `FIRST_IMPRESSION_DURATION` | 30s | Window for first impression module |
| `GROOMING_INTERVAL` | 15s | How often grooming runs |
| `SCORING_INTERVAL_SEC` | 30s | Score recompute frequency |
| `STORAGE_BACKEND` | memory | `memory` or `sqlite` |

---

## Red Flag Categories

| Module | Trigger |
|---|---|
| Identity | Face cosine < 0.6, name mismatch < 80 |
| Body Language | Gaze off > 5s, slouch > 20°, fidget high |
| First Impression | Late > 2min, no smile, low speech RMS, long pause |
| Grooming | Casual attire detected |

---

## Role-Based Scoring Weights

| Module | Developer | Sales | HR |
|---|---|---|---|
| Identity | 30% | 25% | 25% |
| Body Language | 35% | 30% | 25% |
| First Impression | 15% | 30% | 30% |
| Grooming | 20% | 15% | 20% |

---

## Production Upgrade Path

See `CVA_Production_Architecture.docx` for the full production design:
- OpenCV → **NVIDIA DeepStream**
- Local inference → **NVIDIA Triton Inference Server**
- Task queue → **Ray Serve / Apache Flink**
- Storage → **Redis + Feast feature store**
- Deployment → **Kubernetes (AWS EKS)**
- Monitoring → **Prometheus + Grafana + OpenTelemetry**
- Drift detection → **Evidently AI**

---

## Compliance Notes (Demo)
- No Aadhaar data is stored in demo mode (`STORAGE_BACKEND=memory`)
- Face embeddings are held in-process memory only; not persisted
- For production: AES-256 at rest, TLS 1.3 in transit, Aadhaar last-4 only



c:/Developer/Candidate_Visual_Assesment/.venv/Scripts/python.exe run.py