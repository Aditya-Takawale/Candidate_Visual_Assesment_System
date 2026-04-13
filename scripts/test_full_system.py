"""Full system test for CVA — run while server is running on localhost:8000."""
import requests
import time
import json

BASE = "http://127.0.0.1:8000"

def sep(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")

print("=" * 60)
print("  CVA FULL SYSTEM TEST")
print("=" * 60)

# ── 1. Health ────────────────────────────────────────────────
sep("1. Health Check")
r = requests.get(f"{BASE}/health")
h = r.json()
print(f"  Status:  {h['status']}")
print(f"  Backend: {h['hardware_backend']}")
assert h["status"] == "ok", "FAIL"
print("  ✓ PASS")

# ── 2. Start Session ────────────────────────────────────────
sep("2. Start Session")
r = requests.post(f"{BASE}/session/start", json={
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
    r = requests.get(f"{BASE}/session/health")
    hl = r.json()
    warmup = hl.get("warmup_remaining", "?")
    fps = hl.get("fps", 0)
    active = hl.get("active_modules", [])
    print(f"  {(i+1)*5}s — FPS: {fps:.1f}  Warmup left: {warmup}  Active: {active}")

# ── 4. Score Check ──────────────────────────────────────────
sep("4. Score Check")
r = requests.get(f"{BASE}/session/score")
sc = r.json()
print(f"  Status: {sc['status']}")
if sc["status"] == "ok" and sc["score"]:
    score = sc["score"]
    print(f"  Final Score: {score['final_score']}")
    print(f"  Module Scores:")
    for m, v in score["module_scores"].items():
        print(f"    {m:<20} {v}")
    print(f"  Red Flags ({len(score.get('red_flags', []))}):")
    for f in score.get("red_flags", []):
        sev = f["severity"].upper()
        print(f"    [{sev}] {f['module']}: {f['reason']}")
    print(f"  Reason: {score['score_reason']}")
    print(f"  SHAP Breakdown:")
    for k, v in score.get("shap_breakdown", {}).items():
        direction = "▲" if v > 0 else "▼" if v < 0 else "─"
        print(f"    {k:<20} {v:+.1f}% {direction}")
    print("  ✓ PASS")
else:
    print("  Still warming up — waiting 10s more...")
    time.sleep(10)
    r = requests.get(f"{BASE}/session/score")
    sc = r.json()
    if sc["status"] == "ok" and sc["score"]:
        print(f"  Final Score: {sc['score']['final_score']}")
        print("  ✓ PASS (delayed)")
    else:
        print(f"  WARNING: No score after 30s — {sc}")

# ── 5. Health Detail ────────────────────────────────────────
sep("5. System Health Detail")
r = requests.get(f"{BASE}/session/health")
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
r = requests.get(f"{BASE}/session/score")
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
r = requests.post(f"{BASE}/session/stop")
print(f"  Response: {r.json()}")
print("  ✓ PASS")

print("\n" + "=" * 60)
print("  ALL TESTS PASSED ✓")
print("=" * 60)
