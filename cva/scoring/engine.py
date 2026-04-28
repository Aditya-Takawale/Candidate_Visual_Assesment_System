"""
Scoring Engine
- XGBoost model aggregates module scores into a final weighted candidate score
- SHAP explainability: per-module score contribution shown to recruiter
- Role-based weight profiles: developer / sales / hr
- Cold start: returns None until warmup is complete
- Falls back to rule-based weighted average if XGBoost model not available
"""
# pylint: disable=broad-exception-caught  # ML inference errors must not surface to caller
# pylint: disable=line-too-long           # long log/SHAP lines are intentional

from __future__ import annotations
import json
import time
import threading
from dataclasses import replace
from typing import Optional, Dict

import numpy as np

from cva.config.settings import (
    MODULE_WEIGHTS,
    ROLE_WEIGHTS,
    DEFAULT_ROLE,
    MODELS_DIR,
    SHAP_REFRESH_INTERVAL_SEC,
    IDENTITY_MISMATCH_SCORE_CAP,
    GROOMING_UNCONFIRMED_SCORE,
    GROOMING_ABSOLUTE_SCORE_CAP,
)
from cva.common.models import AggregatedFeatures, ScoringResult
from cva.common.logger import get_logger

logger = get_logger(__name__)


def _build_feature_vector(agg: AggregatedFeatures) -> np.ndarray:
    """Convert aggregated features to a flat numpy vector for XGBoost."""
    return np.array([
        agg.identity_score,
        agg.gaze_score,
        agg.posture_score,
        1.0 - min(1.0, agg.fidget_score / 0.15),
        agg.emotion_score,
        agg.smile_ratio,
        agg.speech_energy_score,
        agg.grooming_score,
        agg.punctuality_score,
    ], dtype=np.float32).reshape(1, -1)


def _rule_based_score(agg: AggregatedFeatures, _weights: Dict[str, float]) -> Dict[str, float]:  # pylint: disable=unused-argument
    """Fallback weighted rule-based scoring when XGBoost is unavailable."""
    body_language = (
        agg.gaze_score * 0.35 +
        agg.posture_score * 0.35 +
        (1.0 - min(1.0, agg.fidget_score / 0.15)) * 0.15 +
        agg.emotion_score * 0.15
    )
    first_impression = (
        agg.smile_ratio * 0.35 +
        agg.speech_energy_score * 0.20 +
        agg.punctuality_score * 0.45
    )
    return {
        "identity": float(np.clip(agg.identity_score, 0.0, 1.0)),
        "body_language": float(np.clip(body_language, 0.0, 1.0)),
        "first_impression": float(np.clip(first_impression, 0.0, 1.0)),
        "grooming": float(np.clip(agg.grooming_score, 0.0, 1.0)),
    }


class ScoringEngine:
    """
    Computes final candidate score with SHAP explanations.
    """

    def __init__(self, role: str = DEFAULT_ROLE):
        self._role = role
        self._weights = ROLE_WEIGHTS.get(role, MODULE_WEIGHTS)
        self._xgb_model = None
        self._shap_explainer = None
        self._shap_init_started = False
        self._last_shap_breakdown: Optional[Dict[str, float]] = None
        self._last_shap_at: float = 0.0
        self._shap_refresh_interval_sec = SHAP_REFRESH_INTERVAL_SEC
        self._feature_names = [
            "identity", "gaze", "posture", "fidget_inv",
            "emotion", "smile", "speech_energy", "grooming", "punctuality"
        ]
        self._load_model()

    def _load_model(self) -> None:
        model_path = MODELS_DIR / "scoring_model.json"
        if not model_path.exists():
            logger.warning("XGBoost scoring model not found — using rule-based fallback.")
            return
        try:
            import xgboost as xgb  # pylint: disable=import-outside-toplevel  # type: ignore[import-unresolved]
            self._xgb_model = xgb.XGBRegressor()
            self._xgb_model.load_model(str(model_path))
            logger.info("XGBoost scoring model loaded (regressor).")
            # Validate model quality against the sidecar written by train_scoring_model.py.
            # Auto-reject overfit / underfit models so the system falls back to rule-based.
            _meta_path = MODELS_DIR / "scoring_model_meta.json"
            if _meta_path.exists():
                try:
                    with open(_meta_path, encoding="utf-8") as _mf:
                        _meta = json.load(_mf)
                    _mae = _meta.get("mae_test", 0.0)
                    _gap = _meta.get("mae_gap",  0.0)
                    _r2  = _meta.get("r2_test",  1.0)
                    _r2g = _meta.get("r2_gap",   0.0)
                    if _gap > 2.0 or _mae > 5.0 or _r2 < 0.85 or _r2g > 0.05:
                        logger.warning(
                            f"XGBoost model failed quality gate "
                            f"(MAE={_mae:.2f}, gap={_gap:.2f}, "
                            f"R²={_r2:.4f}, R²_gap={_r2g:.4f}) "
                            f"— falling back to rule-based scoring."
                        )
                        self._xgb_model = None
                        return
                    logger.info(
                        f"Model quality validated: MAE={_mae:.2f}, "
                        f"gap={_gap:.2f}, R²={_r2:.4f}"
                    )
                except Exception as _meta_e:
                    logger.warning(
                        f"Could not read scoring_model_meta.json ({_meta_e}) "
                        f"— proceeding without quality validation."
                    )
            else:
                logger.warning(
                    "scoring_model_meta.json not found — "
                    "re-run train_scoring_model.py to generate validated metrics."
                )
            # SHAP explainer is deferred to first score() call (~6s init)
        except ImportError:
            logger.warning("xgboost not installed — using rule-based fallback scoring.")
        except Exception as e:
            logger.warning(f"XGBoost model load error: {e} — using rule-based fallback.")

    def _ensure_shap(self) -> None:
        """Lazy-init SHAP explainer on first scoring call (avoids slow startup)."""
        if self._shap_explainer is not None or self._xgb_model is None:
            return
        try:
            import shap  # pylint: disable=import-outside-toplevel  # type: ignore[import-unresolved]
            self._shap_explainer = shap.TreeExplainer(self._xgb_model)
            logger.info("SHAP explainer ready (deferred init).")
        except ImportError:
            logger.warning("shap not installed — SHAP explanations disabled.")
        except Exception as e:
            logger.debug(f"SHAP init error: {e}")

    def warmup_explainability(self) -> None:
        """Pre-initialize SHAP in a background thread to avoid first-score latency."""
        if self._shap_init_started or self._xgb_model is None:
            return
        self._shap_init_started = True
        try:
            t = threading.Thread(target=self._ensure_shap, daemon=True, name="cva-shap-warmup")
            t.start()
        except Exception as e:
            logger.debug(f"SHAP warmup thread error: {e}")

    def score(self, agg: AggregatedFeatures) -> Optional[ScoringResult]:
        """Returns ScoringResult or None if still in warmup phase."""
        if not agg.is_warmed_up:
            return None

        # Lazy-init SHAP on first real scoring call (saves ~6s from startup)
        self._ensure_shap()

        # Sanitise features: replace absent/hardware-failure values with neutral defaults
        safe = self._apply_data_guards(agg)

        module_scores = _rule_based_score(safe, self._weights)

        xgb_score = None
        if self._xgb_model is not None:
            try:
                fv = _build_feature_vector(safe)
                xgb_score = float(np.clip(self._xgb_model.predict(fv)[0], 0, 100))
            except Exception as e:
                logger.warning(f"XGBoost inference error: {e} — using rule-based scores.")

        # Compute final score: XGBoost if available, else rule-based
        rule_score = sum(
            module_scores.get(m, 0.5) * w for m, w in self._weights.items()
        ) * 100.0

        if xgb_score is not None:
            # Blend: 60% XGBoost (trained on penalties), 40% rule-based (smooth)
            final_score = 0.60 * xgb_score + 0.40 * rule_score
        else:
            final_score = rule_score

        final_score = float(np.clip(final_score, 0.0, 100.0))

        # Presence gate: if gaze is near-zero the candidate is absent.
        # Identity is only factored in when it has actually been verified —
        # an un-uploaded Aadhaar must not drag down the presence score.
        identity_for_presence = safe.identity_score if safe.identity_verified else 0.5
        presence = safe.gaze_score + identity_for_presence * 0.5
        if presence < 0.2:
            presence_multiplier = max(0.15, presence / 0.2)
            final_score = final_score * presence_multiplier

        # Hard cap: confirmed identity mismatch tanks the total score regardless of other signals.
        # Only fires when Aadhaar was actually uploaded & compared (reference_active=True)
        # and the face still doesn’t match (identity_verified=False).
        if safe.identity_reference_active and not safe.identity_verified:
            final_score = min(final_score, float(IDENTITY_MISMATCH_SCORE_CAP))
        # Grooming hard cap: once YOLO has run and cannot confirm formal attire
        # (score ≤ GROOMING_UNCONFIRMED_SCORE = 0.35), cap the total at 50.
        # Prevents a perfectly-behaved candidate in a t-shirt from exceeding 50/100.
        if agg.grooming_has_run and agg.grooming_score <= GROOMING_UNCONFIRMED_SCORE:
            final_score = min(final_score, float(GROOMING_ABSOLUTE_SCORE_CAP))
        shap_breakdown = self._compute_shap(safe, module_scores)

        reason = self._build_reason(shap_breakdown)

        result = ScoringResult(
            session_id=agg.session_id,
            candidate_id=agg.candidate_id,
            role=self._role,
            final_score=round(final_score, 1),
            module_scores={k: round(v * 100, 1) for k, v in module_scores.items()},
            shap_breakdown=shap_breakdown,
            red_flags=agg.red_flags,
            score_reason=reason,
            identity_reference_active=agg.identity_reference_active,
            grooming_has_run=agg.grooming_has_run,
        )
        return result

    def _compute_shap(self, agg: AggregatedFeatures, module_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Compute SHAP-based contribution per module as % impact on final score.
        Falls back to weight-proportional decomposition if SHAP unavailable.
        """
        now = time.time()
        if self._last_shap_breakdown is not None and (now - self._last_shap_at) < self._shap_refresh_interval_sec:
            return dict(self._last_shap_breakdown)

        if self._shap_explainer is not None:
            try:
                fv = _build_feature_vector(agg)
                shap_values = self._shap_explainer.shap_values(fv)
                vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
                total = sum(abs(v) for v in vals) + 1e-9
                breakdown = {
                    name: round(float(v / total) * 100, 1)
                    for name, v in zip(self._feature_names, vals)
                }
                self._last_shap_breakdown = breakdown
                self._last_shap_at = now
                return breakdown
            except Exception as e:
                logger.debug(f"SHAP computation error: {e}")

        baseline = sum(module_scores.get(m, 0.5) * w for m, w in self._weights.items())
        breakdown = {}
        for module, weight in self._weights.items():
            score = module_scores.get(module, 0.5)
            contribution = (score * weight / (baseline + 1e-9)) * 100.0
            breakdown[module] = round(contribution - 25.0, 1)
        self._last_shap_breakdown = breakdown
        self._last_shap_at = now
        return breakdown

    def _build_reason(self, shap_breakdown: Dict[str, float]) -> str:
        negatives = sorted(
            [(k, v) for k, v in shap_breakdown.items() if v < 0],
            key=lambda x: x[1]
        )
        if not negatives:
            return "Strong performance across all modules."
        parts = [f"{k.replace('_', ' ').title()} ({v:+.1f}%)" for k, v in negatives[:3]]
        return "Score impacted by: " + ", ".join(parts)

    def _apply_data_guards(self, agg: AggregatedFeatures) -> AggregatedFeatures:
        """
        Replace missing/hardware-failure values with neutral defaults.
        Rule: penalise behaviour, not absent data or hardware problems.
        """
        kwargs: Dict[str, object] = {}

        # Identity guard: Aadhaar NOT uploaded (no cosine signal ever received) → neutral (0.75), not guilty.
        # If reference IS active and score is low it is a real mismatch — do NOT neutralize it.
        if not agg.identity_reference_active and agg.identity_score < 0.1:
            kwargs["identity_score"] = 0.75

        # Microphone guard: near-zero energy almost always means a dead mic, not silence
        if agg.speech_energy_score < 0.05:
            kwargs["speech_energy_score"] = 0.5

        # Grooming guard: only neutralise when grooming module has produced no signal at all
        # (score still at near-zero initialisation). Once YOLO has run — even returning 0.35
        # for unconfirmed attire — do NOT override it with a generous neutral.
        if agg.grooming_score < 0.10:
            kwargs["grooming_score"] = 0.50

        return replace(agg, **kwargs) if kwargs else agg

    def set_role(self, role: str) -> None:
        """Update the scoring role and its associated module weights."""
        self._role = role
        self._weights = ROLE_WEIGHTS.get(role, MODULE_WEIGHTS)
        logger.info(f"Scoring role updated to: {role}")
