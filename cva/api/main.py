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
import re
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, Header, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cva.config.settings import API_HOST, API_PORT, API_KEY, CORS_ORIGINS, DEFAULT_ROLE, MAX_UPLOAD_BYTES
from cva.common.models import ScoringResult, SystemHealth
from cva.common.hardware import get_primary_backend
from cva.common.logger import get_logger
from cva.session.orchestrator import SessionOrchestrator

_event_loop: Optional[asyncio.AbstractEventLoop] = None
_ocr_reader = None          # EasyOCR singleton — initialized on first upload
_ocr_lock   = threading.Lock()  # guards singleton initialization


@asynccontextmanager
async def lifespan(app_: FastAPI):
    global _event_loop
    _event_loop = asyncio.get_running_loop()
    # Pre-load all heavy models at startup so first session is instant
    import time as _t
    _t0 = _t.time()
    logger.info("Pre-loading CV models...")
    from cva.session.orchestrator import (
        _get_identity, _get_body_language, _get_grooming, _get_scorer,
    )
    _get_identity()
    _get_body_language()
    _get_grooming()
    _get_scorer("developer")
    logger.info(f"All models ready in {_t.time()-_t0:.1f}s")
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


class ReferenceUploadRequest(BaseModel):
    """Typed model for reference image upload — enforces size cap."""
    image_b64: str = Field(..., max_length=MAX_UPLOAD_BYTES)


class AadhaarUploadRequest(BaseModel):
    """Typed model for Aadhaar upload — enforces size cap."""
    image_b64: str = Field(..., max_length=MAX_UPLOAD_BYTES)
    manual_name: Optional[str] = Field(None, max_length=200)


# ── Optional API-key authentication ──────────────────────────────────────────────────────────────────────────────

def _require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """FastAPI dependency: enforces X-Api-Key header when CVA_API_KEY env var is set."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


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
def start_session(req: StartSessionRequest, _: None = Depends(_require_api_key)):
    global _session, _latest_score, _latest_health
    try:
        if _session is not None:
            _session.stop()
            _session = None

        _latest_score = None
        _latest_health = None

        _session = SessionOrchestrator(
            candidate_id=req.candidate_id,
            role=req.role,
            scheduled_start=req.scheduled_start,
            on_score=_on_score,
            on_health=_on_health,
        )

        if req.aadhaar_name or req.cv_name:
            _session.set_candidate_names(
                aadhaar_name=req.aadhaar_name or "",
                cv_name=req.cv_name or req.aadhaar_name or "",
            )

        _session.start()
        return SessionResponse(
            session_id=_session.session_id,
            status="started",
            hardware_backend=get_primary_backend(),
        )
    except Exception as e:
        logger.error(f"Session start failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/stop")
def stop_session(_: None = Depends(_require_api_key)):
    global _session
    if _session is None:
        raise HTTPException(status_code=404, detail="No active session.")
    _session.stop()
    sid = _session.session_id
    _session = None
    return {"session_id": sid, "status": "stopped"}


@app.get("/session/score")
def get_latest_score(_: None = Depends(_require_api_key)):
    if _latest_score is None:
        return {"status": "warming_up", "score": None}
    return {"status": "ok", "score": _latest_score}


@app.get("/session/health")
def get_session_health(_: None = Depends(_require_api_key)):
    if _session is None:
        return {"status": "no_session"}
    return asdict(_session.health)


@app.post("/session/reference")
async def upload_reference(data: ReferenceUploadRequest, _: None = Depends(_require_api_key)):
    """Legacy endpoint — kept for compatibility."""
    global _session
    if _session is None:
        raise HTTPException(status_code=404, detail="No active session.")
    try:
        img_bytes = base64.b64decode(data.image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
        session_ref = _session
        ok = await asyncio.to_thread(session_ref.set_reference_image, frame)
        return {"status": "reference_set" if ok else "no_face_detected"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _get_ocr_reader():
    """Thread-safe EasyOCR singleton initialization."""
    global _ocr_reader
    with _ocr_lock:
        if _ocr_reader is None:
            import easyocr
            _ocr_reader = easyocr.Reader(["en", "hi"], gpu=True, verbose=False)
    return _ocr_reader


def _aadhaar_process_sync(card_img: np.ndarray, manual_name: str, session) -> dict:
    """CPU-bound OCR + face detection — called via asyncio.to_thread to avoid blocking the event loop."""
    result: dict = {
        "status": "ok",
        "face_detected": False,
        "name_detected": None,
        "name_used": None,
        "ocr_texts": [],
    }

    # ── 1. OCR — extract name from ID card ──────────────────────────────
    ocr_name: Optional[str] = None
    try:
        reader = _get_ocr_reader()

        # ── Image preprocessing for robust OCR ──
        proc = card_img.copy()
        h, w = proc.shape[:2]
        # Upscale small images (phone photos of cards can be tiny)
        if max(h, w) < 1000:
            scale = 1500 / max(h, w)
            proc = cv2.resize(proc, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # Convert to grayscale, enhance contrast, denoise
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        gray = cv2.fastNlMeansDenoising(gray, h=12)
        # Sharpen
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        gray = cv2.filter2D(gray, -1, kernel)

        # Run EasyOCR with English + Hindi for Indian ID cards
        ocr_results = reader.readtext(gray, paragraph=False)
        # Keep text + confidence, sorted by vertical position (top to bottom)
        raw_lines = [(bbox, text.strip(), conf) for (bbox, text, conf) in ocr_results if text.strip()]
        raw_lines.sort(key=lambda x: x[0][0][1])  # sort by y-coordinate
        texts = [text for (_, text, conf) in raw_lines if conf > 0.25]
        result["ocr_texts"] = texts
        logger.info(f"OCR completed: {len(texts)} text lines found")

        # ── Strategy 1: Label-based detection ──
        # Indian IDs have "Name" / "नाम" / "नाम / Name" labels; the actual
        # name is usually the NEXT line after such a label.
        name_labels = ["name", "नाम", "naam"]
        skip_labels = ["father", "पिता", "mother", "माता", "husband", "पति",
                        "signature", "हस्ताक्षर"]
        for i, t in enumerate(texts):
            t_lower = t.lower().strip()
            # Check if this line IS a name label (or contains one)
            is_name_label = any(lbl in t_lower for lbl in name_labels)
            is_skip_label = any(lbl in t_lower for lbl in skip_labels)
            if is_name_label and not is_skip_label:
                # The name might be on the SAME line after "Name" or on the NEXT line
                # Case 1: "Name SHREYAS SHRIPAD WAKHARE" or "नाम / Name"
                # Extract text after the label
                after = re.split(r'(?i)name\s*[:/]?\s*', t)
                if len(after) > 1 and len(after[-1].strip()) > 3:
                    candidate = after[-1].strip()
                    # Verify it looks like a name (mostly letters)
                    alpha_ratio = sum(1 for c in candidate if c.isalpha() or c == ' ') / max(len(candidate), 1)
                    if alpha_ratio > 0.8:
                        ocr_name = candidate
                        logger.info("OCR name extracted (same-line label method)")
                        break
                # Case 2: Name is on the next line
                if i + 1 < len(texts):
                    candidate = texts[i + 1].strip()
                    alpha_ratio = sum(1 for c in candidate if c.isalpha() or c == ' ') / max(len(candidate), 1)
                    if alpha_ratio > 0.7 and len(candidate) > 3:
                        ocr_name = candidate
                        logger.info("OCR name extracted (next-line label method)")
                        break

        # ── Strategy 2: Fallback heuristic if label-based failed ──
        if not ocr_name:
            skip_words = {
                "government", "india", "aadhaar", "unique", "identification",
                "authority", "male", "female", "transgender", "of", "the",
                "income", "tax", "department", "permanent", "account", "number", "card",
                "address", "digilocker", "powered", "tap", "zoom", "download",
                "date", "birth", "dob", "year", "vid", "help", "to", "govt",
                "signature", "father", "mother", "husband", "name",
                "maharashtra", "karnataka", "delhi", "tamil", "nadu",
                "andhra", "pradesh", "telangana", "gujarat", "rajasthan",
                "campus", "phase", "road", "city", "next", "pune",
                "nagar", "wadi", "dighi", "camp", "tirupati", "pin",
                "issue", "enrolment", "enrollment", "no", "qr", "code",
            }
            for t in texts:
                clean = t.strip()
                if len(clean) < 4:
                    continue
                # Skip lines with >30% digits
                digit_ratio = sum(1 for c in clean if c.isdigit()) / max(len(clean), 1)
                if digit_ratio > 0.3:
                    continue
                # Skip date patterns
                if "/" in clean or (any(c == '-' for c in clean) and any(c.isdigit() for c in clean)):
                    continue
                # Skip Hindi-only lines (Devanagari characters)
                devanagari_ratio = sum(1 for c in clean if '\u0900' <= c <= '\u097F') / max(len(clean), 1)
                if devanagari_ratio > 0.5:
                    continue
                # Skip lines that are mostly keywords
                words = clean.split()
                lower_words = [w.lower().rstrip(".:,/") for w in words]
                keyword_count = sum(1 for w in lower_words if w in skip_words)
                if keyword_count >= len(words) * 0.5:
                    continue
                # Must be mostly alphabetic
                alpha_ratio = sum(1 for c in clean if c.isalpha() or c == ' ') / max(len(clean), 1)
                if alpha_ratio < 0.8:
                    continue
                # Must have 2+ words that look like names (capitalized or all-caps)
                if len(words) >= 2:
                    ocr_name = clean
                    logger.info("OCR name extracted (fallback heuristic)")
                    break

        if not ocr_name:
            logger.warning(f"OCR could not extract name from {len(texts)} text lines.")
        result["name_detected"] = ocr_name
    except ImportError:
        logger.warning("easyocr not installed — skipping OCR name extraction.")
    except Exception as e:
        logger.warning(f"OCR failed: {e}", exc_info=True)

    # Use manual name if provided, else OCR name
    name_to_use = manual_name or ocr_name
    result["name_used"] = name_to_use

    # ── 2. Face extraction — set as identity reference ────────────────────
    # Try original size first, then progressively larger upscales with lower threshold
    h, w = card_img.shape[:2]
    face_ok = False

    for scale_label, scale, thresh in [
        ("original", 1, 0.5),
        ("2x",       2, 0.4),
        ("3x",       3, 0.3),
        ("4x",       4, 0.25),
    ]:
        img = card_img if scale == 1 else cv2.resize(
            card_img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC
        )
        try:
            face_ok = session.set_reference_image(img, det_thresh=thresh)
            if face_ok:
                logger.info(f"Face found in Aadhaar card ({scale_label}, {img.shape[1]}x{img.shape[0]}, thresh={thresh})")
                break
        except Exception as e:
            logger.debug(f"Face detection failed at {scale_label}: {e}")

    result["face_detected"] = face_ok
    if not face_ok:
        result["status"] = "face_not_found"
        logger.warning("No face detected in Aadhaar card image.")

    return result


@app.post("/session/aadhaar")
async def upload_aadhaar(data: AadhaarUploadRequest, _: None = Depends(_require_api_key)):
    """
    Full Aadhaar card processing:
      1. Decode the uploaded card image (base64)
      2. Run EasyOCR to extract the candidate name (non-blocking)
      3. Run InsightFace to extract face embedding as identity reference (non-blocking)
      4. Optionally accept manually entered name override
    Body: { "image_b64": "<base64>", "manual_name": "<optional override>" }
    Returns: { "status", "name_detected", "face_detected", "name_used" }
    """
    global _session
    if _session is None:
        raise HTTPException(status_code=404, detail="No active session.")

    try:
        img_bytes = base64.b64decode(data.image_b64)
        nparr    = np.frombuffer(img_bytes, np.uint8)
        card_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if card_img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode error: {e}")

    manual_name = (data.manual_name or "").strip()
    session_ref = _session  # capture reference before entering thread

    # Heavy: OCR + face detection → run in thread pool (non-blocking)
    result = await asyncio.to_thread(_aadhaar_process_sync, card_img, manual_name, session_ref)

    # Apply name to session (fast, back in async context)
    name_to_use = result.get("name_used")
    if name_to_use and _session is not None:
        _session.set_candidate_names(aadhaar_name=name_to_use, cv_name=name_to_use)
        logger.info("Aadhaar document processed — names set for session.")

    return result


# ── MJPEG Video Feed ──────────────────────────────────────────────────────────

from cva.ingestion.camera import get_capture as _get_capture


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
    _cam = _get_capture()
    while True:
        if _cam is None or not _cam.is_opened:
            _cam = _get_capture()
        if _cam is None:
            # Send a black placeholder frame
            from cva.config.settings import FRAME_WIDTH, FRAME_HEIGHT
            blank = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera not available", (int(FRAME_WIDTH * 0.2), FRAME_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 116, 139), 2)
            _, buf = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
            await asyncio.sleep(0.5)
            continue

        ret, frame = _cam.read()
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
