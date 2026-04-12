"""
Train XGBoost scoring model with synthetic data.
Covers all red flag scenarios with realistic distributions.
Run: python scripts/train_scoring_model.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "cva" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
N = 10000

def _clip(x): return np.clip(x, 0.0, 1.0)

# ── Feature columns ──────────────────────────────────────────────────
# identity, gaze, posture, fidget_inv, emotion, smile, speech_energy, grooming, punctuality

samples = []
labels  = []

def make_sample(
    identity=0.8, gaze=0.85, posture=0.85, fidget=0.05,
    emotion=0.8, smile=0.5, speech=0.6, grooming=0.8, punctuality=1.0,
    noise=0.05
):
    n = np.random.normal
    return [
        _clip(identity   + n(0, noise)),
        _clip(gaze       + n(0, noise)),
        _clip(posture    + n(0, noise)),
        _clip(1.0 - fidget + n(0, noise)),   # fidget_inv
        _clip(emotion    + n(0, noise)),
        _clip(smile      + n(0, noise)),
        _clip(speech     + n(0, noise)),
        _clip(grooming   + n(0, noise)),
        _clip(punctuality+ n(0, noise)),
    ]

# ─── GOOD candidates (label=1) ────────────────────────────────────────────────

# Strong overall performer
for _ in range(1000):
    samples.append(make_sample(
        identity=0.92, gaze=0.90, posture=0.88,
        fidget=0.03, emotion=0.85, smile=0.75,
        speech=0.80, grooming=0.90, punctuality=1.0,
    ))
    labels.append(1)

# Good but slightly nervous (some fidget, moderate speech)
for _ in range(600):
    samples.append(make_sample(
        identity=0.85, gaze=0.82, posture=0.80,
        fidget=0.10, emotion=0.75, smile=0.60,
        speech=0.65, grooming=0.85, punctuality=1.0,
    ))
    labels.append(1)

# Formal attire but not very smiley
for _ in range(400):
    samples.append(make_sample(
        identity=0.88, gaze=0.85, posture=0.85,
        fidget=0.05, emotion=0.80, smile=0.30,
        speech=0.70, grooming=0.90, punctuality=1.0,
    ))
    labels.append(1)

# Slightly late but otherwise strong
for _ in range(300):
    samples.append(make_sample(
        identity=0.90, gaze=0.88, posture=0.86,
        fidget=0.04, emotion=0.82, smile=0.65,
        speech=0.75, grooming=0.88, punctuality=0.70,
    ))
    labels.append(1)

# ─── BAD candidates (label=0) ────────────────────────────────────────────────

# Gaze off-camera (RED FLAG)
for _ in range(700):
    samples.append(make_sample(
        identity=0.75, gaze=0.20, posture=0.70,
        fidget=0.08, emotion=0.55, smile=0.25,
        speech=0.50, grooming=0.70, punctuality=0.90,
    ))
    labels.append(0)

# Identity mismatch (RED FLAG)
for _ in range(700):
    samples.append(make_sample(
        identity=0.30, gaze=0.70, posture=0.72,
        fidget=0.07, emotion=0.60, smile=0.40,
        speech=0.55, grooming=0.75, punctuality=0.90,
    ))
    labels.append(0)

# Slouching badly (RED FLAG)
for _ in range(600):
    samples.append(make_sample(
        identity=0.80, gaze=0.78, posture=0.20,
        fidget=0.06, emotion=0.40, smile=0.35,
        speech=0.50, grooming=0.70, punctuality=1.0,
    ))
    labels.append(0)

# Excessive fidgeting (RED FLAG)
for _ in range(500):
    samples.append(make_sample(
        identity=0.80, gaze=0.75, posture=0.70,
        fidget=0.35, emotion=0.50, smile=0.30,
        speech=0.55, grooming=0.72, punctuality=0.95,
    ))
    labels.append(0)

# No smile + low speech (disengaged)
for _ in range(500):
    samples.append(make_sample(
        identity=0.82, gaze=0.80, posture=0.78,
        fidget=0.06, emotion=0.55, smile=0.05,
        speech=0.10, grooming=0.75, punctuality=0.90,
    ))
    labels.append(0)

# Casual attire (grooming RED FLAG)
for _ in range(400):
    samples.append(make_sample(
        identity=0.83, gaze=0.82, posture=0.80,
        fidget=0.06, emotion=0.65, smile=0.40,
        speech=0.60, grooming=0.20, punctuality=0.95,
    ))
    labels.append(0)

# Very late (punctuality RED FLAG)
for _ in range(300):
    samples.append(make_sample(
        identity=0.80, gaze=0.78, posture=0.78,
        fidget=0.07, emotion=0.65, smile=0.40,
        speech=0.60, grooming=0.78, punctuality=0.10,
    ))
    labels.append(0)

# Multiple red flags together (most severe)
for _ in range(800):
    samples.append(make_sample(
        identity=0.35, gaze=0.25, posture=0.25,
        fidget=0.30, emotion=0.30, smile=0.10,
        speech=0.15, grooming=0.25, punctuality=0.20,
        noise=0.08,
    ))
    labels.append(0)

# Borderline cases (adds nuance)
for _ in range(700):
    label = int(np.random.random() > 0.5)
    base = 0.65 if label else 0.45
    samples.append(make_sample(
        identity=base, gaze=base, posture=base,
        fidget=0.12, emotion=base, smile=base - 0.1,
        speech=base - 0.05, grooming=base,
        punctuality=base + 0.1, noise=0.10,
    ))
    labels.append(label)

# ── Build arrays ──────────────────────────────────────────────────────────────
X = np.array(samples, dtype=np.float32)
y = np.array(labels, dtype=np.int32)

print(f"Dataset: {len(X)} samples — {y.sum()} good / {(1-y).sum()} poor")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def _check_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

# ── Train XGBoost ─────────────────────────────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Poor", "Good"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# ── Feature importance ────────────────────────────────────────────────────────
feature_names = ["identity", "gaze", "posture", "fidget_inv",
                 "emotion", "smile", "speech_energy", "grooming", "punctuality"]
importance = dict(zip(feature_names, model.feature_importances_))
print("\nFeature Importance:")
for k, v in sorted(importance.items(), key=lambda x: -x[1]):
    bar = "█" * int(v * 40)
    print(f"  {k:<18} {v:.4f}  {bar}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = MODELS_DIR / "scoring_model.json"
model.save_model(str(out_path))
print(f"\nModel saved to: {out_path}")
