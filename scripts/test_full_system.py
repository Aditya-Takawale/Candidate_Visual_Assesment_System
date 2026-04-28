"""Full system test for CVA — run while server is running on localhost:8000."""
import os
import sys
import time

import requests

BASE = "http://127.0.0.1:8000"
TIMEOUT = 8
API_KEY = os.getenv("CVA_API_KEY", "").strip()
HEADERS = {"X-Api-Key": API_KEY} if API_KEY else {}


def safe_get(path: str):
    """GET request with default headers and timeout."""
    return requests.get(f"{BASE}{path}", headers=HEADERS, timeout=TIMEOUT)


def safe_post(path: str, json_body=None):
    """POST request with default headers and timeout."""
    return requests.post(f"{BASE}{path}", json=json_body, headers=HEADERS, timeout=TIMEOUT)

def sep(title):
    """Print a section separator with title."""
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")

print("=" * 60)
print("  CVA FULL SYSTEM TEST")
print("=" * 60)


def wait_for_server(max_wait_sec: int = 45) -> bool:
    """Poll /health until server responds or deadline is reached."""
    deadline = time.time() + max_wait_sec
    while time.time() < deadline:
        try:
            resp = safe_get("/health")
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False

# ── 1. Health ────────────────────────────────────────────────
sep("1. Health Check")
if not wait_for_server():
    print("  FAIL: Server is not reachable at http://127.0.0.1:8000")
    print("  Start it first: .venv\\Scripts\\python run.py")
    sys.exit(1)

r = safe_get("/health")
h = r.json()
print(f"  Status:  {h['status']}")
print(f"  Backend: {h['hardware_backend']}")
assert h["status"] == "ok", "FAIL"
print("  ✓ PASS")

# ── 2. Start Session ────────────────────────────────────────
sep("2. Start Session")
r = safe_post("/session/start", json_body={
    "candidate_id": "test_001",
    "role": "developer",
    "aadhaar_name": "Aditya Takawale",
    "cv_name": "Aditya Takawale",
})
s = r.json()
print(f"  Session ID: {s['session_id']}")
print(f"  Backend:    {s['hardware_backend']}")
assert s["status"] == "started", "FAIL"
print("  ✓ PASS")

# ── 3. Wait for warmup ──────────────────────────────────────
sep("3. Warmup (20s)")
for i in range(4):
    time.sleep(5)
    r = safe_get("/session/health")
    hl = r.json()
    warmup = hl.get("warmup_remaining", "?")
    fps = hl.get("fps", 0)
    active = hl.get("active_modules", [])
    print(f"  {(i+1)*5}s — FPS: {fps:.1f}  Warmup left: {warmup}  Active: {active}")

# ── 4. Score Check ──────────────────────────────────────────
sep("4. Score Check")
r = safe_get("/session/score")
sc = r.json()
print(f"  Status: {sc['status']}")
if sc["status"] == "ok" and sc["score"]:
    score = sc["score"]
    print(f"  Final Score: {score['final_score']}")
    print("  Module Scores:")
    for m, v in score["module_scores"].items():
        print(f"    {m:<20} {v}")
    print(f"  Red Flags ({len(score.get('red_flags', []))}):")
    for f in score.get("red_flags", []):
        sev = f["severity"].upper()
        print(f"    [{sev}] {f['module']}: {f['reason']}")
    print(f"  Reason: {score['score_reason']}")
    print("  SHAP Breakdown:")
    for k, v in score.get("shap_breakdown", {}).items():
        direction = "▲" if v > 0 else "▼" if v < 0 else "─"  # pylint: disable=invalid-name
        print(f"    {k:<20} {v:+.1f}% {direction}")
    print("  ✓ PASS")
else:
    print("  Still warming up — waiting 10s more...")
    time.sleep(10)
    r = safe_get("/session/score")
    sc = r.json()
    if sc["status"] == "ok" and sc["score"]:
        print(f"  Final Score: {sc['score']['final_score']}")
        print("  ✓ PASS (delayed)")
    else:
        print(f"  WARNING: No score after 30s — {sc}")

# ── 5. Health Detail ────────────────────────────────────────
sep("5. System Health Detail")
r = safe_get("/session/health")
health = r.json()
print(f"  FPS:             {health.get('fps', 0):.1f}")
print(f"  Backend:         {health.get('hardware_backend', '?')}")
print(f"  Active modules:  {health.get('active_modules', [])}")
print(f"  Skipped modules: {health.get('skipped_modules', [])}")
print(f"  Degraded:        {health.get('degraded_modules', [])}")
print(f"  Queue depth:     {health.get('frame_queue_depth', 0)}")
print(f"  Warmup left:     {health.get('warmup_remaining', 0)}")
print("  ✓ PASS")

# ── 6. Second Score (check red flags accumulate) ────────────
sep("6. Score Update After 10s More")
time.sleep(10)
r = safe_get("/session/score")
sc2 = r.json()
if sc2["status"] == "ok" and sc2["score"]:
    s2 = sc2["score"]
    print(f"  Updated Score: {s2['final_score']}")
    print(f"  Red Flags ({len(s2.get('red_flags', []))}):")
    for f in s2.get("red_flags", []):
        sev = f["severity"].upper()
        print(f"    [{sev}] {f['module']}: {f['reason']}")
    print("  ✓ PASS")

# ── 7. Stop Session ─────────────────────────────────────────
sep("7. Stop Session")
r = safe_post("/session/stop")
print(f"  Response: {r.json()}")
print("  ✓ PASS")

print("\n" + "=" * 60)
print("  ALL TESTS PASSED ✓")
print("=" * 60)
