"""Layer 2: Interoceptive Monitor (Pulse v2) - Multimodal sensing and affect estimation.

This module formalizes Pulse as a typed event emitter that provides:
- Environmental features from text/audio/vision modalities
- Expressed PAD estimate (what emotion is being displayed)
- Uncertainty estimates (aleatoric and epistemic)

The interoceptive monitor is the "sensory" layer that feeds into:
- Layer 3 (Appraisal Engine) for cognitive interpretation
- Layer 4 (Policy) for direct reactive control
"""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import time
import json


# =============================================================================
# Feature Schemas
# =============================================================================

@dataclass
class TextFeatures:
    """Features extracted from text modality."""
    toxicity: float = 0.0           # Toxicity score [0, 1]
    valence_pred: float = 0.0       # Predicted valence [-1, 1]
    arousal_pred: float = 0.0       # Predicted arousal [-1, 1]
    sarcasm_prob: float = 0.0       # Sarcasm probability [0, 1]
    formality: float = 0.5          # Formality level [0, 1]
    complexity: float = 0.5         # Linguistic complexity [0, 1]
    topic_vector: Optional[List[float]] = None  # Topic embedding

    def to_dict(self) -> Dict[str, Any]:
        return {
            "toxicity": self.toxicity,
            "valence_pred": self.valence_pred,
            "arousal_pred": self.arousal_pred,
            "sarcasm_prob": self.sarcasm_prob,
            "formality": self.formality,
            "complexity": self.complexity,
            "topic_vector": self.topic_vector,
        }


@dataclass
class AudioFeatures:
    """Features extracted from audio modality."""
    vad_active: bool = False        # Voice activity detected
    pitch_mean: float = 0.0         # Mean pitch (Hz)
    pitch_std: float = 0.0          # Pitch variability
    intensity: float = 0.0          # Volume/intensity [0, 1]
    jitter: float = 0.0             # Voice jitter (perturbation)
    shimmer: float = 0.0            # Voice shimmer (amplitude variation)
    speech_rate: float = 0.0        # Words per second
    prosody_valence: float = 0.0    # Prosodic valence [-1, 1]
    prosody_arousal: float = 0.0    # Prosodic arousal [-1, 1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vad_active": self.vad_active,
            "pitch_mean": self.pitch_mean,
            "pitch_std": self.pitch_std,
            "intensity": self.intensity,
            "jitter": self.jitter,
            "shimmer": self.shimmer,
            "speech_rate": self.speech_rate,
            "prosody_valence": self.prosody_valence,
            "prosody_arousal": self.prosody_arousal,
        }


@dataclass
class VisionFeatures:
    """Features extracted from vision modality."""
    face_detected: bool = False
    # Action Units (FACS)
    action_units: Dict[str, float] = field(default_factory=dict)
    # Gaze
    gaze_contact: float = 0.0       # Eye contact probability [0, 1]
    gaze_direction: Optional[List[float]] = None  # [x, y, z] direction
    # Expression
    smile_prob: float = 0.0
    frown_prob: float = 0.0
    surprise_prob: float = 0.0
    # Head pose
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "face_detected": self.face_detected,
            "action_units": self.action_units,
            "gaze_contact": self.gaze_contact,
            "gaze_direction": self.gaze_direction,
            "smile_prob": self.smile_prob,
            "frown_prob": self.frown_prob,
            "surprise_prob": self.surprise_prob,
            "head_pitch": self.head_pitch,
            "head_yaw": self.head_yaw,
            "head_roll": self.head_roll,
        }


@dataclass
class UncertaintyEstimates:
    """Uncertainty quantification for sensor readings."""
    aleatoric: float = 0.5          # Data/measurement noise [0, 1]
    epistemic: float = 0.5          # Model uncertainty [0, 1]
    modality_agreement: float = 0.5 # Cross-modal consistency [0, 1]

    @property
    def total(self) -> float:
        """Combined uncertainty."""
        return 0.5 * self.aleatoric + 0.5 * self.epistemic

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aleatoric": self.aleatoric,
            "epistemic": self.epistemic,
            "modality_agreement": self.modality_agreement,
            "total": self.total,
        }


@dataclass
class PADEstimate:
    """Estimated Pleasure-Arousal-Dominance from expressed behavior."""
    pleasure: float = 0.0   # [-1, 1]
    arousal: float = 0.0    # [-1, 1]
    dominance: float = 0.0  # [-1, 1]
    confidence: float = 0.5 # [0, 1]

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.pleasure, self.arousal, self.dominance], dtype=torch.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pleasure": self.pleasure,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "confidence": self.confidence,
        }


# =============================================================================
# Interoceptive Event (main output)
# =============================================================================

@dataclass
class InteroceptiveEvent:
    """Complete output from Layer 2 interoceptive monitor.

    This is the "Pulse event v2" schema that feeds into:
    - Layer 3 (Appraisal Engine)
    - Layer 4 (Policy/Regulator)
    """
    # Metadata
    timestamp: float = field(default_factory=time.time)
    event_id: str = ""
    source_modalities: List[str] = field(default_factory=list)

    # Per-modality features
    text_features: Optional[TextFeatures] = None
    audio_features: Optional[AudioFeatures] = None
    vision_features: Optional[VisionFeatures] = None

    # Aggregated estimates
    pad_estimate: PADEstimate = field(default_factory=PADEstimate)
    uncertainty: UncertaintyEstimates = field(default_factory=UncertaintyEstimates)

    # Context
    interaction_turn: int = 0
    session_id: str = ""
    user_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/transmission."""
        return {
            "timestamp": self.timestamp,
            "event_id": self.event_id,
            "source_modalities": self.source_modalities,
            "text_features": self.text_features.to_dict() if self.text_features else None,
            "audio_features": self.audio_features.to_dict() if self.audio_features else None,
            "vision_features": self.vision_features.to_dict() if self.vision_features else None,
            "pad_estimate": self.pad_estimate.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "interaction_turn": self.interaction_turn,
            "session_id": self.session_id,
            "user_id": self.user_id,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_observation_vector(self) -> torch.Tensor:
        """Convert to flat tensor for neural network input.

        Returns:
            Tensor of shape [~30] with key features
        """
        features = []

        # Text features (6)
        if self.text_features:
            features.extend([
                self.text_features.toxicity,
                self.text_features.valence_pred,
                self.text_features.arousal_pred,
                self.text_features.sarcasm_prob,
                self.text_features.formality,
                self.text_features.complexity,
            ])
        else:
            features.extend([0.0] * 6)

        # Audio features (6)
        if self.audio_features:
            features.extend([
                float(self.audio_features.vad_active),
                self.audio_features.intensity,
                self.audio_features.jitter,
                self.audio_features.speech_rate / 5.0,  # Normalize
                self.audio_features.prosody_valence,
                self.audio_features.prosody_arousal,
            ])
        else:
            features.extend([0.0] * 6)

        # Vision features (6)
        if self.vision_features:
            features.extend([
                float(self.vision_features.face_detected),
                self.vision_features.gaze_contact,
                self.vision_features.smile_prob,
                self.vision_features.frown_prob,
                self.vision_features.surprise_prob,
                (self.vision_features.head_pitch + 90) / 180,  # Normalize to [0, 1]
            ])
        else:
            features.extend([0.0] * 6)

        # PAD estimate (4)
        features.extend([
            self.pad_estimate.pleasure,
            self.pad_estimate.arousal,
            self.pad_estimate.dominance,
            self.pad_estimate.confidence,
        ])

        # Uncertainty (3)
        features.extend([
            self.uncertainty.aleatoric,
            self.uncertainty.epistemic,
            self.uncertainty.modality_agreement,
        ])

        return torch.tensor(features, dtype=torch.float32)


# =============================================================================
# Interoceptive Monitor (main interface)
# =============================================================================

class InteroceptiveMonitor:
    """Layer 2: Multimodal sensing and affect estimation.

    This is the upgraded "Pulse" that:
    - Processes text, audio, and vision inputs
    - Estimates expressed PAD (what emotion is being displayed)
    - Quantifies uncertainty
    - Emits typed InteroceptiveEvents

    Usage:
        monitor = InteroceptiveMonitor()
        event = monitor.process(
            text="I'm frustrated with this!",
            audio_features=AudioFeatures(prosody_arousal=0.8),
        )
    """

    def __init__(self):
        self._event_counter = 0
        self._session_id = ""
        self._user_id = ""
        self._turn_counter = 0

        # Calibration state (for dissonance detection)
        self._pad_history: List[PADEstimate] = []
        self._history_len = 50

    def set_session(self, session_id: str, user_id: str = ""):
        """Set session context."""
        self._session_id = session_id
        self._user_id = user_id
        self._turn_counter = 0

    def process(
        self,
        text: Optional[str] = None,
        text_features: Optional[TextFeatures] = None,
        audio_features: Optional[AudioFeatures] = None,
        vision_features: Optional[VisionFeatures] = None,
    ) -> InteroceptiveEvent:
        """Process multimodal inputs and emit an InteroceptiveEvent.

        Args:
            text: Raw text input (will extract features if text_features not provided)
            text_features: Pre-extracted text features
            audio_features: Audio features
            vision_features: Vision features

        Returns:
            InteroceptiveEvent with all features and estimates
        """
        self._event_counter += 1
        self._turn_counter += 1

        modalities = []

        # Text processing
        if text is not None and text_features is None:
            text_features = self._extract_text_features(text)
        if text_features is not None:
            modalities.append("text")

        # Audio (assume pre-extracted)
        if audio_features is not None:
            modalities.append("audio")

        # Vision (assume pre-extracted)
        if vision_features is not None:
            modalities.append("vision")

        # Fuse PAD estimates
        pad_estimate = self._fuse_pad_estimates(text_features, audio_features, vision_features)

        # Estimate uncertainty
        uncertainty = self._estimate_uncertainty(text_features, audio_features, vision_features)

        # Create event
        event = InteroceptiveEvent(
            timestamp=time.time(),
            event_id=f"pulse_{self._event_counter}",
            source_modalities=modalities,
            text_features=text_features,
            audio_features=audio_features,
            vision_features=vision_features,
            pad_estimate=pad_estimate,
            uncertainty=uncertainty,
            interaction_turn=self._turn_counter,
            session_id=self._session_id,
            user_id=self._user_id,
        )

        # Track history
        self._pad_history.append(pad_estimate)
        if len(self._pad_history) > self._history_len:
            self._pad_history = self._pad_history[-self._history_len:]

        return event

    def _extract_text_features(self, text: str) -> TextFeatures:
        """Extract features from text (placeholder - integrate actual models)."""
        # TODO: Integrate actual toxicity, sentiment, sarcasm models
        # For now, return placeholder features
        length = len(text)
        complexity = min(length / 500, 1.0)

        return TextFeatures(
            toxicity=0.0,
            valence_pred=0.0,
            arousal_pred=0.0,
            sarcasm_prob=0.0,
            formality=0.5,
            complexity=complexity,
        )

    def _fuse_pad_estimates(
        self,
        text_features: Optional[TextFeatures],
        audio_features: Optional[AudioFeatures],
        vision_features: Optional[VisionFeatures],
    ) -> PADEstimate:
        """Fuse PAD estimates from multiple modalities."""
        pad_sources = []
        weights = []

        # Text PAD
        if text_features is not None:
            pad_sources.append((text_features.valence_pred, text_features.arousal_pred, 0.0))
            weights.append(0.4)

        # Audio PAD
        if audio_features is not None:
            pad_sources.append((audio_features.prosody_valence, audio_features.prosody_arousal, 0.0))
            weights.append(0.3)

        # Vision PAD (derive from expressions)
        if vision_features is not None:
            v_valence = vision_features.smile_prob - vision_features.frown_prob
            v_arousal = vision_features.surprise_prob
            v_dominance = vision_features.gaze_contact - 0.5
            pad_sources.append((v_valence, v_arousal, v_dominance))
            weights.append(0.3)

        if not pad_sources:
            return PADEstimate()

        # Weighted average
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        pleasure = sum(p[0] * w for p, w in zip(pad_sources, weights))
        arousal = sum(p[1] * w for p, w in zip(pad_sources, weights))
        dominance = sum(p[2] * w for p, w in zip(pad_sources, weights))

        # Confidence based on number of modalities
        confidence = min(len(pad_sources) / 3.0, 1.0)

        return PADEstimate(
            pleasure=float(np.clip(pleasure, -1, 1)),
            arousal=float(np.clip(arousal, -1, 1)),
            dominance=float(np.clip(dominance, -1, 1)),
            confidence=confidence,
        )

    def _estimate_uncertainty(
        self,
        text_features: Optional[TextFeatures],
        audio_features: Optional[AudioFeatures],
        vision_features: Optional[VisionFeatures],
    ) -> UncertaintyEstimates:
        """Estimate uncertainty from available modalities."""
        n_modalities = sum([
            text_features is not None,
            audio_features is not None,
            vision_features is not None,
        ])

        # More modalities = lower epistemic uncertainty
        epistemic = 1.0 - (n_modalities / 3.0) * 0.6

        # Aleatoric based on signal quality (placeholder)
        aleatoric = 0.3

        # Modality agreement (placeholder)
        modality_agreement = 0.7 if n_modalities > 1 else 0.3

        return UncertaintyEstimates(
            aleatoric=aleatoric,
            epistemic=epistemic,
            modality_agreement=modality_agreement,
        )

    def compute_dissonance(self, appraisal_pad: Dict[str, float]) -> float:
        """Compute dissonance between expressed PAD and appraised PAD.

        Dissonance indicates mismatch between what is sensed (Layer 2)
        and what is cognitively interpreted (Layer 3).

        Args:
            appraisal_pad: PAD from Layer 3 appraisal engine

        Returns:
            Dissonance score [0, 1] where 0 = agreement, 1 = full disagreement
        """
        if not self._pad_history:
            return 0.5

        expressed = self._pad_history[-1]
        appraised = appraisal_pad

        # Compare signs (most important)
        sign_disagreement = 0.0
        if np.sign(expressed.pleasure) != np.sign(appraised.get("pleasure", 0)):
            sign_disagreement += 0.33
        if np.sign(expressed.arousal) != np.sign(appraised.get("arousal", 0)):
            sign_disagreement += 0.33
        if np.sign(expressed.dominance) != np.sign(appraised.get("dominance", 0)):
            sign_disagreement += 0.34

        # Compare magnitudes
        magnitude_diff = (
            abs(expressed.pleasure - appraised.get("pleasure", 0)) +
            abs(expressed.arousal - appraised.get("arousal", 0)) +
            abs(expressed.dominance - appraised.get("dominance", 0))
        ) / 6.0  # Max diff is 2 per dimension

        # Combined dissonance
        dissonance = 0.6 * sign_disagreement + 0.4 * magnitude_diff
        return float(np.clip(dissonance, 0, 1))


__all__ = [
    # Feature schemas
    "TextFeatures",
    "AudioFeatures",
    "VisionFeatures",
    "UncertaintyEstimates",
    "PADEstimate",
    # Main event
    "InteroceptiveEvent",
    # Monitor
    "InteroceptiveMonitor",
]
