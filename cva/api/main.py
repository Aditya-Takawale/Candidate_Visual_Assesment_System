"""
FastAPI Backend — CVA Demo Mode
- REST endpoints: session start/stop, health, latest score
- WebSocket: streams live score + red flags + system health to dashboard
- CORS enabled for dashboard on any origin (demo only)
"""

from __future__ import annotations
import asyncio
import base64
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cva.config.settings import API_HOST, API_PORT, CORS_ORIGINS, DEFAULT_ROLE
from cva.common.models import ScoringResult, SystemHealth
from cva.common.hardware import get_primary_backend
from cva.common.logger import get_logger
from cva.session.orchestrator import SessionOrchestrator

_event_loop: Optional[asyncio.AbstractEventLoop] = None


@asynccontextmanager
async def lifespan(app_: FastAPI):
    global _event_loop
    _event_loop = asyncio.get_running_loop()
    yield

logger = get_logger(__name__)

app = FastAPI(title="CVA Demo", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global session state (single session for demo) ────────────────────────────
_session: Optional[SessionOrchestrator] = None
_latest_score: Optional[dict] = None
_latest_health: Optional[dict] = None
_ws_clients: list[WebSocket] = []


# ── Pydantic request/response schemas ─────────────────────────────────────────

class StartSessionRequest(BaseModel):
    candidate_id: str = "candidate_001"
    role: str = DEFAULT_ROLE
    aadhaar_name: Optional[str] = None
    cv_name: Optional[str] = None
    scheduled_start: Optional[float] = None


class SessionResponse(BaseModel):
    session_id: str
    status: str
    hardware_backend: str


# ── Callbacks (called from orchestrator background thread) ────────────────────

def _on_score(result: ScoringResult) -> None:
    global _latest_score
    _latest_score = _serialise(asdict(result))
    _dispatch({"type": "score", "data": _latest_score})


def _on_health(health: SystemHealth) -> None:
    global _latest_health
    _latest_health = _serialise(asdict(health))
    _dispatch({"type": "health", "data": _latest_health})


def _dispatch(message: dict) -> None:
    """Thread-safe: schedule broadcast onto the FastAPI event loop."""
    if _event_loop is not None and not _event_loop.is_closed():
        asyncio.run_coroutine_threadsafe(_broadcast(message), _event_loop)


async def _broadcast(message: dict) -> None:
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


def _serialise(obj):
    """Recursively make dicts JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(i) for i in obj]
    if isinstance(obj, float) and (obj != obj):  # NaN guard
        return None
    return obj


# ── REST Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def api_health():
    return {"status": "ok", "hardware_backend": get_primary_backend(), "ts": time.time()}


@app.post("/session/start", response_model=SessionResponse)
def start_session(req: StartSessionRequest):
    global _session, _latest_score, _latest_health
    if _session is not None:
        _session.stop()

    _latest_score = None
    _latest_health = None

    _session = SessionOrchestrator(
        candidate_id=req.candidate_id,
        role=req.role,
        scheduled_start=req.scheduled_start,
        on_score=_on_score,
        on_health=_on_health,
    )

    if req.aadhaar_name and req.cv_name:
        _session.set_candidate_names(req.aadhaar_name, req.cv_name)

    _session.start()
    return SessionResponse(
        session_id=_session.session_id,
        status="started",
        hardware_backend=get_primary_backend(),
    )


@app.post("/session/stop")
def stop_session():
    global _session
    if _session is None:
        raise HTTPException(status_code=404, detail="No active session.")
    _session.stop()
    sid = _session.session_id
    _session = None
    return {"session_id": sid, "status": "stopped"}


@app.get("/session/score")
def get_latest_score():
    if _latest_score is None:
        return {"status": "warming_up", "score": None}
    return {"status": "ok", "score": _latest_score}


@app.get("/session/health")
def get_session_health():
    if _session is None:
        return {"status": "no_session"}
    return asdict(_session.health)


@app.post("/session/reference")
async def upload_reference(data: dict):
    """
    Accept a base64-encoded reference image (Aadhaar photo) for identity.
    Body: { "image_b64": "<base64 string>" }
    """
    global _session
    if _session is None:
        raise HTTPException(status_code=404, detail="No active session.")
    try:
        img_bytes = base64.b64decode(data["image_b64"])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        ok = _session.set_reference_image(frame)
        return {"status": "reference_set" if ok else "no_face_detected"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── MJPEG Video Feed ──────────────────────────────────────────────────────────

_video_cap: Optional[cv2.VideoCapture] = None
_video_lock = asyncio.Lock()


def _get_capture() -> Optional[cv2.VideoCapture]:
    global _video_cap
    if _video_cap is None or not _video_cap.isOpened():
        from cva.config.settings import VIDEO_SOURCE, FRAME_WIDTH, FRAME_HEIGHT
        _video_cap = cv2.VideoCapture(VIDEO_SOURCE)
        if _video_cap.isOpened():
            _video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            _video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            for _ in range(10):
                _video_cap.read()
    return _video_cap if (_video_cap and _video_cap.isOpened()) else None


def _annotate_frame(frame: np.ndarray) -> np.ndarray:
    """Draw a lightweight HUD on the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent top bar
    cv2.rectangle(overlay, (0, 0), (w, 36), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Session ID
    sid = _session.session_id if _session else "no session"
    cv2.putText(frame, f"Session: {sid}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 210, 255), 1, cv2.LINE_AA)

    # Score badge (top-right)
    if _latest_score:
        score_val = _latest_score.get("final_score", 0)
        colour = (74, 222, 128) if score_val >= 75 else (251, 191, 36) if score_val >= 50 else (248, 113, 113)
        label = f"Score: {score_val:.0f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(frame, label, (w - tw - 10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)

    # Active module badges (bottom bar)
    if _latest_health:
        active = _latest_health.get("active_modules", [])
        x = 10
        for mod in active:
            label = mod.replace("_", " ").title()
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (x - 4, h - 28), (x + tw + 4, h - 8), (22, 163, 74), -1)
            cv2.putText(frame, label, (x, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            x += tw + 14

    # FPS
    if _latest_health:
        fps = _latest_health.get("fps", 0)
        cv2.putText(frame, f"{fps:.1f} fps", (10, h - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 116, 139), 1, cv2.LINE_AA)

    return frame


async def _mjpeg_generator():
    while True:
        cap = _get_capture()
        if cap is None:
            # Send a black placeholder frame
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera not available", (160, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 116, 139), 2)
            _, buf = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
            await asyncio.sleep(0.5)
            continue

        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.05)
            continue

        frame = _annotate_frame(frame)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        await asyncio.sleep(1 / 15)   # cap at 15 fps for dashboard preview


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info(f"WebSocket client connected. Total: {len(_ws_clients)}")
    try:
        if _latest_score:
            await websocket.send_json({"type": "score", "data": _latest_score})
        if _latest_health:
            await websocket.send_json({"type": "health", "data": _latest_health})
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)
        logger.info(f"WebSocket client disconnected. Remaining: {len(_ws_clients)}")


# ── Serve Dashboard Static Files ──────────────────────────────────────────────

_dashboard_dir = Path(__file__).parent.parent / "dashboard"
if _dashboard_dir.exists():
    app.mount("/", StaticFiles(directory=str(_dashboard_dir), html=True), name="dashboard")
