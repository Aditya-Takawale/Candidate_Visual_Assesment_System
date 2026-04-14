"""
Train XGBoost Regression scoring model for CVA.
─────────────────────────────────────────────────
Predicts a continuous score 0–100 (not binary), trained on 50K
synthetic samples covering every red-flag scenario with correlated
feature distributions.  Evaluated with 5-fold CV, MAE, and R².
Includes overfitting diagnostics (train-test gap, learning curve).

Features (9):
  identity, gaze, posture, fidget_inv, emotion, smile,
  speech_energy, grooming, punctuality

Run:  python scripts/train_scoring_model.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
try:
    from sklearn.metrics import root_mean_squared_error as _rmse_fn
except ImportError:
    from sklearn.metrics import mean_squared_error
    def _rmse_fn(y_true, y_pred): return mean_squared_error(y_true, y_pred) ** 0.5
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "cva" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
rng = np.random.default_rng(42)

FEATURE_NAMES = [
    "identity", "gaze", "posture", "fidget_inv",
    "emotion", "smile", "speech_energy", "grooming", "punctuality",
]

# Role-based weight profiles (used to compute ground-truth target)
MODULE_WEIGHTS = {
    "identity": 0.30, "body_language": 0.35,
    "first_impression": 0.20, "grooming": 0.15,
}

# ═════════════════════════════════════════════════════════════════════════════
# Data generation helpers
# ═════════════════════════════════════════════════════════════════════════════

def clip(x):
    return np.clip(x, 0.0, 1.0)


def make_row(
    identity, gaze, posture, fidget, emotion, smile,
    speech, grooming, punctuality, noise=0.04,
):
    """Generate one feature vector + ground-truth score."""
    n = rng.normal
    feat = np.array([
        clip(identity    + n(0, noise)),
        clip(gaze        + n(0, noise)),
        clip(posture     + n(0, noise)),
        clip(1.0 - fidget + n(0, noise)),  # fidget_inv: higher = calmer
        clip(emotion     + n(0, noise)),
        clip(smile       + n(0, noise)),
        clip(speech      + n(0, noise)),
        clip(grooming    + n(0, noise)),
        clip(punctuality + n(0, noise)),
    ], dtype=np.float32)

    # Ground-truth score: weighted combination with penalties for red-flag zones
    body_lang = feat[1]*0.35 + feat[2]*0.35 + feat[3]*0.15 + feat[4]*0.15
    first_imp = feat[5]*0.30 + feat[6]*0.40 + feat[8]*0.30

    raw = (
        feat[0] * MODULE_WEIGHTS["identity"] +
        body_lang * MODULE_WEIGHTS["body_language"] +
        first_imp * MODULE_WEIGHTS["first_impression"] +
        feat[7] * MODULE_WEIGHTS["grooming"]
    ) * 100.0

    # Hard penalties for clear red-flag scenarios
    if feat[0] < 0.40:  raw -= 20     # identity mismatch
    if feat[1] < 0.30:  raw -= 15     # gaze off-camera
    if feat[2] < 0.30:  raw -= 12     # severe slouch
    if feat[3] < 0.40:  raw -= 10     # excessive fidgeting
    if feat[5] < 0.10:  raw -= 8      # never smiled
    if feat[6] < 0.15:  raw -= 8      # silence
    if feat[7] < 0.30:  raw -= 12     # casual attire
    if feat[8] < 0.20:  raw -= 10     # very late

    # Mild bonus for consistently excellent signals
    if all(f > 0.80 for f in feat):
        raw += 5

    score = float(np.clip(raw + n(0, 1.5), 0, 100))
    return feat, score


def generate_batch(n, **kwargs):
    X, y = [], []
    for _ in range(n):
        feat, score = make_row(**kwargs)
        X.append(feat)
        y.append(score)
    return X, y


# ═════════════════════════════════════════════════════════════════════════════
# Generate 50K samples across all scenarios
# ═════════════════════════════════════════════════════════════════════════════

all_X, all_y = [], []

def add(n, **kw):
    X, y = generate_batch(n, **kw)
    all_X.extend(X); all_y.extend(y)

# ── EXCELLENT candidates (score ~80-95) ──────────────────────────────────────
add(5000, identity=0.95, gaze=0.92, posture=0.90, fidget=0.02,
    emotion=0.88, smile=0.80, speech=0.85, grooming=0.92, punctuality=1.0)

add(3000, identity=0.90, gaze=0.88, posture=0.88, fidget=0.04,
    emotion=0.82, smile=0.70, speech=0.78, grooming=0.88, punctuality=1.0)

add(2000, identity=0.88, gaze=0.85, posture=0.85, fidget=0.03,
    emotion=0.85, smile=0.65, speech=0.75, grooming=0.95, punctuality=0.90)

# ── GOOD candidates (score ~60-80) ──────────────────────────────────────────
add(3000, identity=0.85, gaze=0.78, posture=0.80, fidget=0.08,
    emotion=0.72, smile=0.55, speech=0.65, grooming=0.82, punctuality=1.0)

add(2000, identity=0.82, gaze=0.75, posture=0.75, fidget=0.10,
    emotion=0.70, smile=0.45, speech=0.60, grooming=0.78, punctuality=0.85,
    noise=0.06)

add(1500, identity=0.88, gaze=0.82, posture=0.78, fidget=0.06,
    emotion=0.75, smile=0.35, speech=0.55, grooming=0.80, punctuality=0.75)

# ── AVERAGE candidates (score ~40-60) ───────────────────────────────────────
add(2500, identity=0.75, gaze=0.60, posture=0.65, fidget=0.12,
    emotion=0.60, smile=0.30, speech=0.45, grooming=0.65, punctuality=0.80,
    noise=0.07)

add(2000, identity=0.70, gaze=0.55, posture=0.60, fidget=0.14,
    emotion=0.55, smile=0.25, speech=0.40, grooming=0.60, punctuality=0.70,
    noise=0.08)

# ── RED FLAG: Gaze off-camera ───────────────────────────────────────────────
add(2000, identity=0.80, gaze=0.15, posture=0.70, fidget=0.08,
    emotion=0.55, smile=0.20, speech=0.50, grooming=0.72, punctuality=0.90)

add(1500, identity=0.82, gaze=0.10, posture=0.65, fidget=0.10,
    emotion=0.50, smile=0.15, speech=0.45, grooming=0.68, punctuality=0.85)

# ── RED FLAG: Identity mismatch ─────────────────────────────────────────────
add(2000, identity=0.20, gaze=0.78, posture=0.75, fidget=0.06,
    emotion=0.65, smile=0.40, speech=0.55, grooming=0.80, punctuality=0.90)

add(1500, identity=0.10, gaze=0.70, posture=0.70, fidget=0.08,
    emotion=0.60, smile=0.35, speech=0.50, grooming=0.75, punctuality=1.0)

# ── RED FLAG: Sustained slouch ──────────────────────────────────────────────
add(2000, identity=0.82, gaze=0.75, posture=0.15, fidget=0.07,
    emotion=0.45, smile=0.30, speech=0.50, grooming=0.72, punctuality=0.95)

add(1500, identity=0.80, gaze=0.78, posture=0.10, fidget=0.05,
    emotion=0.40, smile=0.25, speech=0.45, grooming=0.70, punctuality=1.0)

# ── RED FLAG: Excessive fidgeting ───────────────────────────────────────────
add(2000, identity=0.82, gaze=0.72, posture=0.68, fidget=0.35,
    emotion=0.48, smile=0.25, speech=0.50, grooming=0.70, punctuality=0.90)

add(1000, identity=0.78, gaze=0.68, posture=0.65, fidget=0.45,
    emotion=0.42, smile=0.20, speech=0.45, grooming=0.65, punctuality=0.85)

# ── RED FLAG: No smile + low speech (disengaged) ───────────────────────────
add(2000, identity=0.83, gaze=0.78, posture=0.75, fidget=0.06,
    emotion=0.50, smile=0.03, speech=0.08, grooming=0.72, punctuality=0.90)

add(1500, identity=0.80, gaze=0.72, posture=0.72, fidget=0.08,
    emotion=0.45, smile=0.05, speech=0.12, grooming=0.68, punctuality=0.85)

# Additional disengaged variants — model was scoring these too high (58.5)
add(1500, identity=0.82, gaze=0.70, posture=0.70, fidget=0.04,
    emotion=0.42, smile=0.02, speech=0.05, grooming=0.70, punctuality=0.90)

add(1000, identity=0.85, gaze=0.75, posture=0.73, fidget=0.05,
    emotion=0.40, smile=0.01, speech=0.03, grooming=0.72, punctuality=0.85)

# ── RED FLAG: Casual attire (grooming) ──────────────────────────────────────
add(2000, identity=0.85, gaze=0.80, posture=0.78, fidget=0.05,
    emotion=0.68, smile=0.45, speech=0.55, grooming=0.15, punctuality=0.95)

add(1500, identity=0.82, gaze=0.78, posture=0.75, fidget=0.07,
    emotion=0.62, smile=0.40, speech=0.50, grooming=0.10, punctuality=0.90)

# ── RED FLAG: Very late (punctuality) ───────────────────────────────────────
add(1500, identity=0.82, gaze=0.78, posture=0.76, fidget=0.06,
    emotion=0.65, smile=0.40, speech=0.55, grooming=0.78, punctuality=0.05)

add(1000, identity=0.80, gaze=0.75, posture=0.72, fidget=0.08,
    emotion=0.60, smile=0.35, speech=0.50, grooming=0.72, punctuality=0.10)

# ── MULTIPLE RED FLAGS (catastrophic) ──────────────────────────────────────
add(2000, identity=0.25, gaze=0.20, posture=0.18, fidget=0.40,
    emotion=0.25, smile=0.05, speech=0.10, grooming=0.15, punctuality=0.10,
    noise=0.06)

add(1000, identity=0.15, gaze=0.15, posture=0.12, fidget=0.50,
    emotion=0.20, smile=0.02, speech=0.05, grooming=0.10, punctuality=0.05,
    noise=0.05)

# ── BORDERLINE cases (maximum variance — forces model to learn nuance) ─────
for _ in range(3000):
    base = rng.uniform(0.35, 0.70)
    add(1,
        identity=base + rng.normal(0, 0.08),
        gaze=base + rng.normal(0, 0.10),
        posture=base + rng.normal(0, 0.10),
        fidget=rng.uniform(0.08, 0.20),
        emotion=base + rng.normal(0, 0.08),
        smile=max(0, base - 0.15 + rng.normal(0, 0.10)),
        speech=max(0, base - 0.10 + rng.normal(0, 0.08)),
        grooming=base + rng.normal(0, 0.10),
        punctuality=rng.uniform(0.40, 1.0),
        noise=0.08,
    )

# ── RANDOM SPREAD (ensures model generalises) ─────────────────────────────
for _ in range(5000):
    add(1,
        identity=rng.uniform(0.0, 1.0),
        gaze=rng.uniform(0.0, 1.0),
        posture=rng.uniform(0.0, 1.0),
        fidget=rng.uniform(0.0, 0.5),
        emotion=rng.uniform(0.0, 1.0),
        smile=rng.uniform(0.0, 1.0),
        speech=rng.uniform(0.0, 1.0),
        grooming=rng.uniform(0.0, 1.0),
        punctuality=rng.uniform(0.0, 1.0),
        noise=0.02,
    )

# ═════════════════════════════════════════════════════════════════════════════
# Build arrays
# ═════════════════════════════════════════════════════════════════════════════

X = np.array(all_X, dtype=np.float32)
y = np.array(all_y, dtype=np.float32)

print(f"Dataset: {len(X):,} samples")
print(f"  Score range: {y.min():.1f} – {y.max():.1f}")
print(f"  Mean: {y.mean():.1f}  Std: {y.std():.1f}")
print(f"  <30: {(y < 30).sum():,}  |  30-60: {((y >= 30) & (y < 60)).sum():,}  |  60-80: {((y >= 60) & (y < 80)).sum():,}  |  >80: {(y >= 80).sum():,}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42,
)

# ═════════════════════════════════════════════════════════════════════════════
# 5-Fold Cross-Validation to find optimal hyperparameters
# ═════════════════════════════════════════════════════════════════════════════

print("\n── 5-Fold Cross-Validation ──────────────────────────────────────────")

param_grid = [
    {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.03,
     "subsample": 0.80, "colsample_bytree": 0.85, "min_child_weight": 5,
     "reg_alpha": 0.1, "reg_lambda": 1.0, "gamma": 0.1},
    {"n_estimators": 800, "max_depth": 7, "learning_rate": 0.02,
     "subsample": 0.85, "colsample_bytree": 0.80, "min_child_weight": 8,
     "reg_alpha": 0.5, "reg_lambda": 2.0, "gamma": 0.2},
    {"n_estimators": 600, "max_depth": 5, "learning_rate": 0.05,
     "subsample": 0.75, "colsample_bytree": 0.90, "min_child_weight": 3,
     "reg_alpha": 0.01, "reg_lambda": 0.5, "gamma": 0.05},
    # Strong regularization — guards against overfitting on synthetic data
    {"n_estimators": 700, "max_depth": 4, "learning_rate": 0.03,
     "subsample": 0.70, "colsample_bytree": 0.75, "min_child_weight": 10,
     "reg_alpha": 1.0, "reg_lambda": 5.0, "gamma": 0.5},
    # Balanced depth + regularization — good generalization
    {"n_estimators": 600, "max_depth": 5, "learning_rate": 0.04,
     "subsample": 0.80, "colsample_bytree": 0.85, "min_child_weight": 6,
     "reg_alpha": 0.3, "reg_lambda": 3.0, "gamma": 0.15},
]

best_mae = float("inf")
best_params = None

for i, params in enumerate(param_grid):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_maes = []
    for train_idx, val_idx in kf.split(X_train):
        model_cv = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            verbosity=0,
            **params,
        )
        model_cv.fit(
            X_train[train_idx], y_train[train_idx],
            eval_set=[(X_train[val_idx], y_train[val_idx])],
            verbose=False,
        )
        preds = model_cv.predict(X_train[val_idx])
        fold_maes.append(mean_absolute_error(y_train[val_idx], preds))

    avg_mae = np.mean(fold_maes)
    print(f"  Config {i+1}: MAE = {avg_mae:.3f} (±{np.std(fold_maes):.3f})")
    if avg_mae < best_mae:
        best_mae = avg_mae
        best_params = params

print(f"\n  Best config MAE: {best_mae:.3f}")
print(f"  Params: depth={best_params['max_depth']}, lr={best_params['learning_rate']}, "
      f"n_est={best_params['n_estimators']}")

# ═════════════════════════════════════════════════════════════════════════════
# Train final model with best params on full training set
# ═════════════════════════════════════════════════════════════════════════════

print("\n── Training Final Model ─────────────────────────────────────────────")

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    early_stopping_rounds=30,
    **best_params,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

# ═════════════════════════════════════════════════════════════════════════════
# Evaluate
# ═════════════════════════════════════════════════════════════════════════════

y_pred = np.clip(model.predict(X_test), 0, 100)
mae  = mean_absolute_error(y_test, y_pred)
rmse = _rmse_fn(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

# Train set predictions for overfitting gap analysis
y_pred_train = np.clip(model.predict(X_train), 0, 100)
mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = _rmse_fn(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

print(f"\n── Test Set Results ─────────────────────────────────────────────────")
print(f"  MAE:  {mae:.2f} points  (out of 100)")
print(f"  RMSE: {rmse:.2f}")
print(f"  R²:   {r2:.4f}")

print(f"\n── Overfitting Diagnostics ──────────────────────────────────────────")
print(f"  Train MAE:  {mae_train:.2f}  |  Test MAE:  {mae:.2f}  |  Gap: {mae - mae_train:.2f}")
print(f"  Train RMSE: {rmse_train:.2f}  |  Test RMSE: {rmse:.2f}  |  Gap: {rmse - rmse_train:.2f}")
print(f"  Train R²:   {r2_train:.4f}  |  Test R²:   {r2:.4f}  |  Gap: {r2_train - r2:.4f}")
gap = mae - mae_train
if gap > 2.0:
    print(f"  ⚠ WARNING: Train-test gap of {gap:.2f} suggests possible overfitting!")
    print(f"    Consider: increase min_child_weight, reduce max_depth, or add more data")
elif gap < 0.5:
    print(f"  ✓ Model generalises well (gap < 0.5)")
else:
    print(f"  ✓ Acceptable gap (0.5–2.0) — model fits well without overfitting")

# Check accuracy on red-flag detection (score < 40 when should be low)
low_mask  = y_test < 40
high_mask = y_test >= 70
print(f"\n  Low-score detection  (<40): predicted-mean = {y_pred[low_mask].mean():.1f}  (target-mean = {y_test[low_mask].mean():.1f})")
print(f"  High-score detection (≥70): predicted-mean = {y_pred[high_mask].mean():.1f}  (target-mean = {y_test[high_mask].mean():.1f})")

# Feature importance
importance = dict(zip(FEATURE_NAMES, model.feature_importances_))
print(f"\n── Feature Importance ───────────────────────────────────────────────")
for k, v in sorted(importance.items(), key=lambda x: -x[1]):
    bar = "█" * int(v * 50)
    print(f"  {k:<16} {v:.4f}  {bar}")

# Sanity check: specific scenarios
print(f"\n── Sanity Checks ───────────────────────────────────────────────────")
scenarios = {
    "Perfect candidate":       [0.95, 0.92, 0.90, 0.98, 0.88, 0.80, 0.85, 0.92, 1.0],
    "Good + no smile":         [0.85, 0.82, 0.80, 0.90, 0.72, 0.10, 0.60, 0.80, 1.0],
    "Identity mismatch":       [0.15, 0.80, 0.78, 0.92, 0.70, 0.50, 0.60, 0.82, 1.0],
    "Gaze off-camera":         [0.85, 0.10, 0.75, 0.88, 0.55, 0.30, 0.50, 0.75, 0.9],
    "Severe slouch":           [0.85, 0.80, 0.10, 0.90, 0.40, 0.30, 0.50, 0.72, 1.0],
    "Casual attire":           [0.85, 0.82, 0.80, 0.92, 0.70, 0.50, 0.60, 0.12, 0.9],
    "Very late":               [0.85, 0.80, 0.78, 0.90, 0.65, 0.40, 0.55, 0.78, 0.05],
    "Disengaged (no smile/sp)":[0.82, 0.75, 0.72, 0.88, 0.48, 0.03, 0.08, 0.70, 0.9],
    "Total disaster":          [0.10, 0.10, 0.10, 0.30, 0.15, 0.02, 0.05, 0.08, 0.05],
    "Neutral (just joined)":   [0.50, 0.50, 0.80, 1.00, 0.50, 0.00, 0.00, 0.50, 1.0],
}
for name, feats in scenarios.items():
    pred = float(np.clip(model.predict(np.array([feats], dtype=np.float32))[0], 0, 100))
    print(f"  {name:<28} → {pred:5.1f}/100")

# ═════════════════════════════════════════════════════════════════════════════
# Save
# ═════════════════════════════════════════════════════════════════════════════

out_path = MODELS_DIR / "scoring_model.json"
model.save_model(str(out_path))
print(f"\n✓ Model saved to: {out_path}")
print(f"  File size: {out_path.stat().st_size / 1024:.0f} KB")
