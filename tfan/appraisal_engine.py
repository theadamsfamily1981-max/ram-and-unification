# tfan/appraisal_engine.py
# Layer 2: Cognitive Appraisal Engine (EMA/CoRE-style interface)
#
# Implements the cognitive appraisal layer from the Generative Affective
# Cognition framework. Takes world state + homeostatic summary and outputs
# appraisal dimensions + discrete emotion label + explanation.
#
# Per Marsella & Gratch EMA model: separates appraisal (fast feature detection)
# from inference (perception, planning, reasoning).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math


# ============================================================================
#  Appraisal Dimensions (CoRE benchmark dimensions)
# ============================================================================

APPRAISAL_DIMENSIONS = [
    "pleasantness",       # How pleasant/unpleasant is this situation?
    "goal_congruence",    # Does this align with my goals?
    "control",            # How much control do I have?
    "certainty",          # How certain am I about the outcome?
    "responsibility_self",  # Am I responsible for this?
    "responsibility_other", # Is someone else responsible?
    "goal_obstacle",      # Is there an obstacle to my goals?
    "coping_potential",   # Can I cope with this situation?
]

# Emotion labels mapped from appraisal patterns (simplified OCC-style)
EMOTION_PATTERNS = {
    "joy": {"pleasantness": 0.8, "goal_congruence": 0.7, "certainty": 0.6},
    "relief": {"pleasantness": 0.6, "goal_congruence": 0.5, "certainty": 0.8},
    "hope": {"pleasantness": 0.5, "goal_congruence": 0.6, "certainty": -0.3},
    "fear": {"pleasantness": -0.7, "goal_congruence": -0.6, "certainty": -0.5, "control": -0.5},
    "anxiety": {"pleasantness": -0.5, "goal_congruence": -0.4, "certainty": -0.7},
    "anger": {"pleasantness": -0.6, "goal_congruence": -0.7, "responsibility_other": 0.7},
    "frustration": {"pleasantness": -0.5, "goal_obstacle": 0.7, "coping_potential": -0.4},
    "sadness": {"pleasantness": -0.6, "goal_congruence": -0.5, "control": -0.6},
    "confusion": {"certainty": -0.8, "coping_potential": -0.3},
    "curiosity": {"pleasantness": 0.3, "certainty": -0.4, "goal_congruence": 0.2},
    "satisfaction": {"pleasantness": 0.7, "goal_congruence": 0.8, "coping_potential": 0.6},
    "neutral": {},
}


# ============================================================================
#  Appraisal Result
# ============================================================================

@dataclass
class AppraisalResult:
    """Result of cognitive appraisal."""
    dims: Dict[str, float]  # Appraisal dimension scores (-1 to 1)
    emotion_label: str      # Discrete emotion label
    confidence: float       # Confidence in the emotion label
    explanation: str        # Human-readable explanation
    raw_context: Optional[Dict[str, Any]] = None


# ============================================================================
#  Appraisal Engine (Rule-based stub)
# ============================================================================

class AppraisalEngineStub:
    """
    Stub appraisal engine using rule-based heuristics.

    This is a placeholder for a proper CoRE-finetuned LLM.
    Maps homeostatic state + context signals to appraisal dimensions.
    """

    def __init__(self):
        self.history: List[AppraisalResult] = []

    def appraise(
        self,
        homeostatic_state: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> AppraisalResult:
        """
        Perform cognitive appraisal based on homeostatic state and context.

        Args:
            homeostatic_state: From HomeostaticCore.get_state()
                - drive_total, valence, n_energy, n_integrity, etc.
            context: Optional additional context
                - "event_type": str (e.g., "training_step", "user_message")
                - "loss_trend": float (-1 to 1, negative = improving)
                - "uncertainty": float (0 to 1)
                - "user_sentiment": float (-1 to 1)

        Returns:
            AppraisalResult with dimension scores and emotion label
        """
        context = context or {}
        dims = {d: 0.0 for d in APPRAISAL_DIMENSIONS}

        # Extract homeostatic signals
        drive = homeostatic_state.get("drive_total", 0.5)
        valence = homeostatic_state.get("valence", 0.0)
        n_energy = homeostatic_state.get("n_energy", 0.5)
        n_integrity = homeostatic_state.get("n_integrity", 0.5)
        n_cogload = homeostatic_state.get("n_cogload", 0.5)
        n_safety = homeostatic_state.get("n_safety", 0.5)
        n_novelty = homeostatic_state.get("n_novelty", 0.5)

        # Extract context signals
        loss_trend = context.get("loss_trend", 0.0)
        uncertainty = context.get("uncertainty", 0.5)
        user_sentiment = context.get("user_sentiment", 0.0)

        # === Map homeostatic state to appraisal dimensions ===

        # Pleasantness: inverse of drive, modified by valence
        dims["pleasantness"] = -drive * 0.5 + valence * 0.5

        # Goal congruence: high when drive is low and valence positive
        dims["goal_congruence"] = (1.0 - drive) * 0.6 + (-loss_trend) * 0.4

        # Control: high when cognitive load is low and safety is high
        dims["control"] = (1.0 - n_cogload) * 0.5 + n_safety * 0.3 + n_integrity * 0.2

        # Certainty: inverse of uncertainty, modified by integrity
        dims["certainty"] = (1.0 - uncertainty) * 0.7 + n_integrity * 0.3 - 0.5

        # Coping potential: high when energy and safety are good
        dims["coping_potential"] = n_energy * 0.4 + n_safety * 0.3 + (1.0 - n_cogload) * 0.3

        # Goal obstacle: high when drive is increasing (valence negative)
        dims["goal_obstacle"] = max(0, -valence) * 0.7 + drive * 0.3

        # Responsibility: context-dependent
        dims["responsibility_self"] = 0.3 if loss_trend > 0 else 0.1
        dims["responsibility_other"] = max(0, -user_sentiment) * 0.5

        # Clamp all dimensions to [-1, 1]
        for k in dims:
            dims[k] = max(-1.0, min(1.0, dims[k]))

        # === Determine emotion label ===
        emotion_label, confidence = self._match_emotion(dims)

        # === Generate explanation ===
        explanation = self._generate_explanation(dims, emotion_label, homeostatic_state)

        result = AppraisalResult(
            dims=dims,
            emotion_label=emotion_label,
            confidence=confidence,
            explanation=explanation,
            raw_context=context,
        )

        self.history.append(result)
        if len(self.history) > 100:
            self.history.pop(0)

        return result

    def _match_emotion(self, dims: Dict[str, float]) -> Tuple[str, float]:
        """Match appraisal dimensions to closest emotion pattern."""
        best_match = "neutral"
        best_score = -float("inf")

        for emotion, pattern in EMOTION_PATTERNS.items():
            if not pattern:
                continue

            score = 0.0
            count = 0
            for dim, target in pattern.items():
                if dim in dims:
                    # Higher score when dimension matches target
                    diff = abs(dims[dim] - target)
                    score += 1.0 - diff
                    count += 1

            if count > 0:
                score /= count
                if score > best_score:
                    best_score = score
                    best_match = emotion

        confidence = max(0.0, min(1.0, best_score))
        return best_match, confidence

    def _generate_explanation(
        self,
        dims: Dict[str, float],
        emotion: str,
        homeo: Dict[str, float],
    ) -> str:
        """Generate a brief explanation for the appraisal."""
        parts = []

        if emotion == "neutral":
            return "System operating normally."

        # Top contributing dimensions
        sorted_dims = sorted(dims.items(), key=lambda x: abs(x[1]), reverse=True)
        top_dims = sorted_dims[:2]

        for dim, val in top_dims:
            if abs(val) > 0.3:
                direction = "high" if val > 0 else "low"
                parts.append(f"{dim.replace('_', ' ')} is {direction}")

        if parts:
            return f"Feeling {emotion}: {'; '.join(parts)}."
        return f"Feeling {emotion}."

    def get_dominant_dims(self, n: int = 3) -> List[Tuple[str, float]]:
        """Get the N most extreme appraisal dimensions from last appraisal."""
        if not self.history:
            return []
        dims = self.history[-1].dims
        sorted_dims = sorted(dims.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_dims[:n]


# ============================================================================
#  Appraisal Engine Client (LLM backend)
# ============================================================================

class AppraisalEngineClient:
    """
    Layer 2: Cognitive appraisal via LLM backend.

    This is the interface for a CoRE-finetuned LLM that performs
    cognitive appraisal. Uses the stub as fallback.
    """

    def __init__(self, backend: Optional[Any] = None):
        """
        Args:
            backend: Optional LLM backend with run_appraisal() method.
                     Falls back to rule-based stub if None.
        """
        self.backend = backend
        self._stub = AppraisalEngineStub()

    def appraise(
        self,
        homeostatic_state: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> AppraisalResult:
        """
        Perform cognitive appraisal.

        If an LLM backend is available, uses it. Otherwise falls back to stub.
        """
        if self.backend is not None:
            try:
                return self._appraise_llm(homeostatic_state, context)
            except Exception:
                pass

        return self._stub.appraise(homeostatic_state, context)

    def _appraise_llm(
        self,
        homeostatic_state: Dict[str, float],
        context: Optional[Dict[str, Any]],
    ) -> AppraisalResult:
        """Run appraisal through LLM backend."""
        # Prepare prompt context
        llm_context = {
            "homeostasis": homeostatic_state,
            "observations": context or {},
        }

        raw = self.backend.run_appraisal(llm_context)

        dims = {k: float(raw.get("dims", {}).get(k, 0.0)) for k in APPRAISAL_DIMENSIONS}
        emotion_label = raw.get("emotion_label", "neutral")
        confidence = raw.get("confidence", 0.5)
        explanation = raw.get("explanation", "")

        return AppraisalResult(
            dims=dims,
            emotion_label=emotion_label,
            confidence=confidence,
            explanation=explanation,
            raw_context=context,
        )

    def get_state(self) -> Dict[str, Any]:
        """Get current appraisal state for telemetry."""
        if self._stub.history:
            last = self._stub.history[-1]
            return {
                "emotion_label": last.emotion_label,
                "emotion_confidence": last.confidence,
                "appraisal_dims": last.dims,
                "explanation": last.explanation,
            }
        return {
            "emotion_label": "neutral",
            "emotion_confidence": 0.0,
            "appraisal_dims": {d: 0.0 for d in APPRAISAL_DIMENSIONS},
            "explanation": "",
        }


# ============================================================================
#  Factory Function
# ============================================================================

def create_appraisal_engine(backend: Optional[Any] = None) -> AppraisalEngineClient:
    """Create an appraisal engine with optional LLM backend."""
    return AppraisalEngineClient(backend=backend)
