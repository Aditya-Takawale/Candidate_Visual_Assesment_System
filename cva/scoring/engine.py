"""
Scoring Engine
- XGBoost model aggregates module scores into a final weighted candidate score
- SHAP explainability: per-module score contribution shown to recruiter
- Role-based weight profiles: developer / sales / hr
- Cold start: returns None until warmup is complete
- Falls back to rule-based weighted average if XGBoost model not available
"""

from __future__ import annotations
import numpy as np
import time
from typing import Optional, Dict

from cva.config.settings import (
    MODULE_WEIGHTS,
    ROLE_WEIGHTS,
    DEFAULT_ROLE,
    MODELS_DIR,
    WARMUP_FRAMES,
)
from cva.common.models import AggregatedFeatures, ScoringResult, RedFlag
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


def _rule_based_score(agg: AggregatedFeatures, weights: Dict[str, float]) -> Dict[str, float]:
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
            import xgboost as xgb
            self._xgb_model = xgb.XGBRegressor()
            self._xgb_model.load_model(str(model_path))
            logger.info("XGBoost scoring model loaded (regressor).")
            try:
                import shap
                self._shap_explainer = shap.TreeExplainer(self._xgb_model)
                logger.info("SHAP explainer ready.")
            except ImportError:
                logger.warning("shap not installed — SHAP explanations disabled.")
        except ImportError:
            logger.warning("xgboost not installed — using rule-based fallback scoring.")
        except Exception as e:
            logger.warning(f"XGBoost model load error: {e} — using rule-based fallback.")

    def score(self, agg: AggregatedFeatures) -> Optional[ScoringResult]:
        """Returns ScoringResult or None if still in warmup phase."""
        if not agg.is_warmed_up:
            return None

        module_scores = _rule_based_score(agg, self._weights)

        xgb_score = None
        if self._xgb_model is not None:
            try:
                fv = _build_feature_vector(agg)
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

        shap_breakdown = self._compute_shap(agg, module_scores)

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
        )
        return result

    def _compute_shap(self, agg: AggregatedFeatures, module_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Compute SHAP-based contribution per module as % impact on final score.
        Falls back to weight-proportional decomposition if SHAP unavailable.
        """
        if self._shap_explainer is not None:
            try:
                fv = _build_feature_vector(agg)
                shap_values = self._shap_explainer.shap_values(fv)
                vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
                total = sum(abs(v) for v in vals) + 1e-9
                return {
                    name: round(float(v / total) * 100, 1)
                    for name, v in zip(self._feature_names, vals)
                }
            except Exception as e:
                logger.debug(f"SHAP computation error: {e}")

        total_weight = sum(self._weights.values()) + 1e-9
        baseline = sum(module_scores.get(m, 0.5) * w for m, w in self._weights.items())
        breakdown = {}
        for module, weight in self._weights.items():
            score = module_scores.get(module, 0.5)
            contribution = (score * weight / (baseline + 1e-9)) * 100.0
            breakdown[module] = round(contribution - 25.0, 1)
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

    def set_role(self, role: str) -> None:
        self._role = role
        self._weights = ROLE_WEIGHTS.get(role, MODULE_WEIGHTS)
        logger.info(f"Scoring role updated to: {role}")
