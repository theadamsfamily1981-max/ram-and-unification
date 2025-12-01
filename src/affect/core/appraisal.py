"""Layer 3: Appraisal Engine - Cognitive interpretation of affect.

This module implements the Amygdala-analogue appraisal system that:
- Integrates Layer 1 (homeostatic state) and Layer 2 (sensory features)
- Performs EMA-style cognitive appraisal
- Generates the agent's TRUE internal affective state
- Produces discrete emotion labels for interpretability

The appraisal engine can be backed by:
- An LLM (Claude) for rich semantic appraisal
- A learned neural network for fast inference
- Rule-based heuristics for testing

Reference: EMA (Elliott, Ortony, & Arbib), OCC (Ortony, Clore, Collins)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import json
import time

from .homeostatic import HomeostaticState, HomeostaticCore
from .interoceptive import InteroceptiveEvent, PADEstimate


# =============================================================================
# Emotion Taxonomy
# =============================================================================

class DiscreteEmotion(Enum):
    """Discrete emotion labels (OCC + extensions)."""
    # Positive
    JOY = "joy"
    HOPE = "hope"
    RELIEF = "relief"
    PRIDE = "pride"
    GRATITUDE = "gratitude"
    LOVE = "love"
    CURIOSITY = "curiosity"
    INTEREST = "interest"

    # Negative
    DISTRESS = "distress"
    FEAR = "fear"
    ANGER = "anger"
    DISGUST = "disgust"
    SHAME = "shame"
    GUILT = "guilt"
    FRUSTRATION = "frustration"
    SADNESS = "sadness"

    # Complex/Mixed
    CONFUSION = "confusion"
    SURPRISE = "surprise"
    ANTICIPATION = "anticipation"
    BOREDOM = "boredom"

    # Neutral
    NEUTRAL = "neutral"


# =============================================================================
# Appraisal Dimensions
# =============================================================================

@dataclass
class AppraisalDimensions:
    """EMA-style appraisal dimensions.

    These dimensions are computed from context and determine emotion.
    """
    # Goal-relevance
    goal_congruence: float = 0.0    # [-1, 1]: helps (+) vs hinders (-) goals
    goal_importance: float = 0.5    # [0, 1]: how much this matters

    # Agency/Responsibility
    causal_agent: str = "unknown"   # "self", "other", "situation", "unknown"
    responsibility: str = "unknown" # "internal", "external", "mixed"
    intentionality: float = 0.5     # [0, 1]: was action intentional?

    # Control/Coping
    coping_potential: float = 0.5   # [0, 1]: can agent handle this?
    control: float = 0.5            # [0, 1]: does agent have control?
    predictability: float = 0.5     # [0, 1]: was this expected?

    # Certainty
    certainty: float = 0.5          # [0, 1]: how sure is agent about situation?

    # Moral/Normative
    moral_valence: float = 0.0      # [-1, 1]: good (+) vs bad (-) morally
    norm_violation: float = 0.0     # [0, 1]: degree of norm violation

    # Future-orientation
    future_expectancy: float = 0.5  # [0, 1]: expectation of positive outcome

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_congruence": self.goal_congruence,
            "goal_importance": self.goal_importance,
            "causal_agent": self.causal_agent,
            "responsibility": self.responsibility,
            "intentionality": self.intentionality,
            "coping_potential": self.coping_potential,
            "control": self.control,
            "predictability": self.predictability,
            "certainty": self.certainty,
            "moral_valence": self.moral_valence,
            "norm_violation": self.norm_violation,
            "future_expectancy": self.future_expectancy,
        }

    def to_tensor(self) -> torch.Tensor:
        """Convert numeric dimensions to tensor."""
        return torch.tensor([
            self.goal_congruence,
            self.goal_importance,
            self.intentionality,
            self.coping_potential,
            self.control,
            self.predictability,
            self.certainty,
            self.moral_valence,
            self.norm_violation,
            self.future_expectancy,
        ], dtype=torch.float32)


# =============================================================================
# Appraisal Result
# =============================================================================

@dataclass
class AppraisalResult:
    """Output from the appraisal engine."""
    # Discrete emotion
    discrete_emotion: DiscreteEmotion = DiscreteEmotion.NEUTRAL

    # Generated PAD (the agent's TRUE internal state)
    generated_pad: PADEstimate = field(default_factory=PADEstimate)

    # Appraisal dimensions that led to this emotion
    appraisal_dims: AppraisalDimensions = field(default_factory=AppraisalDimensions)

    # Epistemic state
    epistemic_uncertainty: float = 0.5  # [0, 1]

    # Explanation
    explanation: str = ""

    # Metadata
    timestamp: float = field(default_factory=time.time)
    appraisal_method: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "discrete_emotion": self.discrete_emotion.value,
            "generated_pad": self.generated_pad.to_dict(),
            "appraisal_dims": self.appraisal_dims.to_dict(),
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "explanation": self.explanation,
            "timestamp": self.timestamp,
            "appraisal_method": self.appraisal_method,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# Rule-Based Appraisal (baseline)
# =============================================================================

def rule_based_appraisal(
    homeostatic_state: HomeostaticState,
    homeostatic_drive: float,
    interoceptive_event: InteroceptiveEvent,
    context: Optional[str] = None,
) -> AppraisalResult:
    """Simple rule-based appraisal for baseline/testing.

    Maps homeostatic state + sensor features to emotions using heuristics.
    """
    # Extract key signals
    pad = interoceptive_event.pad_estimate
    uncertainty = interoceptive_event.uncertainty

    # Compute appraisal dimensions from state
    dims = AppraisalDimensions()

    # Goal congruence from homeostatic drive
    # Low drive = congruent (things are good)
    dims.goal_congruence = 1.0 - 2.0 * min(homeostatic_drive, 1.0)

    # Importance from safety and integrity
    dims.goal_importance = 0.5 * homeostatic_state.safety + 0.5 * homeostatic_state.integrity

    # Coping potential from coherence and energy
    dims.coping_potential = 0.5 * homeostatic_state.coherence + 0.5 * homeostatic_state.energy

    # Control from dominance estimate
    dims.control = (pad.dominance + 1.0) / 2.0

    # Certainty from inverse of epistemic uncertainty
    dims.certainty = 1.0 - uncertainty.epistemic

    # Predictability (placeholder)
    dims.predictability = 0.5

    # Moral valence from toxicity and safety
    if interoceptive_event.text_features:
        dims.moral_valence = -interoceptive_event.text_features.toxicity
    dims.norm_violation = 1.0 - homeostatic_state.safety

    # Future expectancy from current trajectory
    dims.future_expectancy = dims.coping_potential * (1.0 + dims.goal_congruence) / 2.0

    # Map appraisal to discrete emotion
    emotion = _map_appraisal_to_emotion(dims, homeostatic_state, pad)

    # Generate internal PAD from appraisal
    generated_pad = _generate_pad_from_appraisal(dims, homeostatic_drive)

    # Build explanation
    explanation = _generate_explanation(emotion, dims, homeostatic_state)

    return AppraisalResult(
        discrete_emotion=emotion,
        generated_pad=generated_pad,
        appraisal_dims=dims,
        epistemic_uncertainty=uncertainty.epistemic,
        explanation=explanation,
        appraisal_method="rule_based",
    )


def _map_appraisal_to_emotion(
    dims: AppraisalDimensions,
    state: HomeostaticState,
    expressed_pad: PADEstimate,
) -> DiscreteEmotion:
    """Map appraisal dimensions to discrete emotion."""

    # High goal congruence + high certainty
    if dims.goal_congruence > 0.5 and dims.certainty > 0.5:
        if dims.future_expectancy > 0.7:
            return DiscreteEmotion.JOY
        elif dims.coping_potential > 0.7:
            return DiscreteEmotion.PRIDE
        else:
            return DiscreteEmotion.RELIEF

    # High goal congruence + low certainty
    if dims.goal_congruence > 0.3 and dims.certainty < 0.4:
        return DiscreteEmotion.HOPE

    # Negative goal congruence + external responsibility
    if dims.goal_congruence < -0.3:
        if dims.responsibility == "external" or dims.causal_agent == "other":
            if dims.control < 0.3:
                return DiscreteEmotion.ANGER
            else:
                return DiscreteEmotion.FRUSTRATION
        elif dims.responsibility == "internal" or dims.causal_agent == "self":
            if dims.moral_valence < -0.3:
                return DiscreteEmotion.GUILT
            else:
                return DiscreteEmotion.SHAME

    # Low coping potential + negative congruence
    if dims.coping_potential < 0.3 and dims.goal_congruence < 0:
        if dims.certainty > 0.6:
            return DiscreteEmotion.FEAR
        else:
            return DiscreteEmotion.DISTRESS

    # High novelty need (boredom)
    if state.novelty < 0.2:
        return DiscreteEmotion.BOREDOM

    # High novelty + uncertainty
    if state.novelty > 0.7 and dims.certainty < 0.4:
        if dims.goal_congruence > 0:
            return DiscreteEmotion.CURIOSITY
        else:
            return DiscreteEmotion.CONFUSION

    # Surprise
    if dims.predictability < 0.3:
        return DiscreteEmotion.SURPRISE

    # Moral violation
    if dims.norm_violation > 0.5:
        return DiscreteEmotion.DISGUST

    return DiscreteEmotion.NEUTRAL


def _generate_pad_from_appraisal(dims: AppraisalDimensions, drive: float) -> PADEstimate:
    """Generate internal PAD from appraisal dimensions."""
    # Pleasure from goal congruence and drive
    pleasure = 0.6 * dims.goal_congruence + 0.4 * (1.0 - min(drive, 1.0) * 2)

    # Arousal from importance, uncertainty, and drive magnitude
    arousal = 0.4 * dims.goal_importance + 0.3 * (1 - dims.certainty) + 0.3 * min(drive, 1.0)

    # Dominance from control and coping
    dominance = 0.5 * dims.control + 0.5 * dims.coping_potential

    # Shift to [-1, 1] range
    pleasure = float(np.clip(pleasure, -1, 1))
    arousal = float(np.clip(arousal * 2 - 0.5, -1, 1))
    dominance = float(np.clip(dominance * 2 - 1, -1, 1))

    return PADEstimate(
        pleasure=pleasure,
        arousal=arousal,
        dominance=dominance,
        confidence=dims.certainty,
    )


def _generate_explanation(
    emotion: DiscreteEmotion,
    dims: AppraisalDimensions,
    state: HomeostaticState,
) -> str:
    """Generate human-readable explanation of appraisal."""
    parts = []

    if emotion == DiscreteEmotion.JOY:
        parts.append("Goals are being met and situation is favorable.")
    elif emotion == DiscreteEmotion.ANGER:
        parts.append("Goals are being blocked by external factors.")
    elif emotion == DiscreteEmotion.FEAR:
        parts.append("Threat detected with limited coping resources.")
    elif emotion == DiscreteEmotion.CURIOSITY:
        parts.append("Novel situation with potential for positive outcomes.")
    elif emotion == DiscreteEmotion.CONFUSION:
        parts.append("High uncertainty about situation and appropriate response.")
    elif emotion == DiscreteEmotion.FRUSTRATION:
        parts.append("Goals blocked but some control remains.")
    elif emotion == DiscreteEmotion.BOREDOM:
        parts.append("Need for novelty not being met.")

    if state.safety < 0.7:
        parts.append(f"Safety concern detected (level: {state.safety:.2f}).")
    if state.cogload > 0.7:
        parts.append(f"High cognitive load (level: {state.cogload:.2f}).")

    return " ".join(parts) if parts else "Neutral affective state."


# =============================================================================
# Neural Appraisal Network (learned)
# =============================================================================

class NeuralAppraisalNetwork(nn.Module):
    """Learned appraisal network for fast inference.

    Takes homeostatic state + interoceptive features and outputs:
    - Appraisal dimensions
    - PAD values
    - Emotion logits
    """

    def __init__(
        self,
        homeostatic_dim: int = 8,
        interoceptive_dim: int = 25,
        hidden_dim: int = 256,
        num_emotions: int = len(DiscreteEmotion),
    ):
        super().__init__()
        input_dim = homeostatic_dim + interoceptive_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Appraisal head
        self.appraisal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 10),  # 10 appraisal dimensions
            nn.Tanh(),
        )

        # PAD head
        self.pad_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # PAD
            nn.Tanh(),
        )

        # Emotion head
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_emotions),
        )

        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        homeostatic: torch.Tensor,  # [batch, 8]
        interoceptive: torch.Tensor,  # [batch, 25]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            appraisal: [batch, 10] appraisal dimensions
            pad: [batch, 3] PAD values
            emotion_logits: [batch, num_emotions]
            uncertainty: [batch, 1]
        """
        x = torch.cat([homeostatic, interoceptive], dim=-1)
        h = self.encoder(x)

        appraisal = self.appraisal_head(h)
        pad = self.pad_head(h)
        emotion_logits = self.emotion_head(h)
        uncertainty = self.uncertainty_head(h)

        return appraisal, pad, emotion_logits, uncertainty


# =============================================================================
# Appraisal Engine (main interface)
# =============================================================================

class AppraisalEngine:
    """Layer 3: Cognitive appraisal engine.

    Integrates homeostatic state and interoceptive events to produce
    the agent's true internal affective state.

    Can use:
    - Rule-based appraisal (default)
    - Neural network (fast learned)
    - LLM backend (rich semantic)
    """

    def __init__(
        self,
        method: str = "rule_based",
        neural_model: Optional[NeuralAppraisalNetwork] = None,
        llm_backend: Optional[Any] = None,  # LLM client
    ):
        self.method = method
        self.neural_model = neural_model
        self.llm_backend = llm_backend

        self._history: List[AppraisalResult] = []
        self._history_len = 50

    def appraise(
        self,
        homeostatic_core: HomeostaticCore,
        interoceptive_event: InteroceptiveEvent,
        context: Optional[str] = None,
    ) -> AppraisalResult:
        """Perform cognitive appraisal.

        Args:
            homeostatic_core: Layer 1 homeostatic state
            interoceptive_event: Layer 2 sensory event
            context: Optional natural language context

        Returns:
            AppraisalResult with emotion, PAD, and explanation
        """
        if self.method == "neural" and self.neural_model is not None:
            result = self._neural_appraise(homeostatic_core, interoceptive_event)
        elif self.method == "llm" and self.llm_backend is not None:
            result = self._llm_appraise(homeostatic_core, interoceptive_event, context)
        else:
            result = rule_based_appraisal(
                homeostatic_core.state,
                homeostatic_core.current_drive,
                interoceptive_event,
                context,
            )

        # Track history
        self._history.append(result)
        if len(self._history) > self._history_len:
            self._history = self._history[-self._history_len:]

        return result

    def _neural_appraise(
        self,
        homeostatic_core: HomeostaticCore,
        interoceptive_event: InteroceptiveEvent,
    ) -> AppraisalResult:
        """Appraisal using neural network."""
        # Prepare inputs
        homeo = homeostatic_core.get_observation().unsqueeze(0)  # [1, 11]
        intero = interoceptive_event.to_observation_vector().unsqueeze(0)  # [1, 25]

        with torch.no_grad():
            appraisal_t, pad_t, emotion_logits, uncertainty_t = self.neural_model(
                homeo[:, :8],  # State only
                intero,
            )

        # Convert outputs
        emotion_idx = emotion_logits.argmax(dim=-1).item()
        emotions = list(DiscreteEmotion)
        emotion = emotions[emotion_idx] if emotion_idx < len(emotions) else DiscreteEmotion.NEUTRAL

        pad = PADEstimate(
            pleasure=pad_t[0, 0].item(),
            arousal=pad_t[0, 1].item(),
            dominance=pad_t[0, 2].item(),
            confidence=1.0 - uncertainty_t[0, 0].item(),
        )

        # Placeholder appraisal dims
        dims = AppraisalDimensions()

        return AppraisalResult(
            discrete_emotion=emotion,
            generated_pad=pad,
            appraisal_dims=dims,
            epistemic_uncertainty=uncertainty_t[0, 0].item(),
            explanation="Neural appraisal result.",
            appraisal_method="neural",
        )

    def _llm_appraise(
        self,
        homeostatic_core: HomeostaticCore,
        interoceptive_event: InteroceptiveEvent,
        context: Optional[str],
    ) -> AppraisalResult:
        """Appraisal using LLM backend."""
        # TODO: Implement LLM-based appraisal
        # Would call Claude with the Layer 3 system prompt
        return rule_based_appraisal(
            homeostatic_core.state,
            homeostatic_core.current_drive,
            interoceptive_event,
            context,
        )

    def get_emotion_trajectory(self, window: int = 10) -> List[str]:
        """Get recent emotion trajectory."""
        recent = self._history[-window:]
        return [r.discrete_emotion.value for r in recent]

    def get_mean_pad(self, window: int = 10) -> Dict[str, float]:
        """Get mean PAD over recent window."""
        recent = self._history[-window:]
        if not recent:
            return {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0}

        p = np.mean([r.generated_pad.pleasure for r in recent])
        a = np.mean([r.generated_pad.arousal for r in recent])
        d = np.mean([r.generated_pad.dominance for r in recent])

        return {"pleasure": float(p), "arousal": float(a), "dominance": float(d)}


__all__ = [
    # Enums
    "DiscreteEmotion",
    # Data structures
    "AppraisalDimensions",
    "AppraisalResult",
    # Functions
    "rule_based_appraisal",
    # Networks
    "NeuralAppraisalNetwork",
    # Main interface
    "AppraisalEngine",
]
