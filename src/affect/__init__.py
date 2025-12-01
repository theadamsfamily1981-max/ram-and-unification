"""Affect Module - Grounded emotion as gradient signal.

This module implements the 4-layer affect architecture that treats
emotion as the engine of cognition rather than a side-channel.

Layers:
    1. Homeostatic Core - Internal needs and drives
    2. Interoceptive Monitor - Multimodal sensing (Pulse)
    3. Appraisal Engine - Cognitive interpretation
    4. Policy/Regulator - Action selection (gates)

The key insight: emotion grounds valence/arousal/dominance in
homeostatic dynamics, making affect a first-class gradient signal
for long-horizon reinforcement learning.
"""

from .core import (
    # Layer 1
    HomeostaticCore,
    HomeostaticState,
    HomeostaticSetpoints,
    NeedType,
    compute_homeostatic_drive,
    compute_homeostatic_reward,
    # Layer 2
    InteroceptiveMonitor,
    InteroceptiveEvent,
    PADEstimate,
    TextFeatures,
    AudioFeatures,
    VisionFeatures,
    # Layer 3
    AppraisalEngine,
    AppraisalResult,
    DiscreteEmotion,
    AppraisalDimensions,
    # Layer 4
    PolicyRegulator,
    AffectiveAction,
    SNNPolicyInterface,
)

__version__ = "0.1.0"

__all__ = [
    # Layer 1
    "HomeostaticCore",
    "HomeostaticState",
    "HomeostaticSetpoints",
    "NeedType",
    "compute_homeostatic_drive",
    "compute_homeostatic_reward",
    # Layer 2
    "InteroceptiveMonitor",
    "InteroceptiveEvent",
    "PADEstimate",
    "TextFeatures",
    "AudioFeatures",
    "VisionFeatures",
    # Layer 3
    "AppraisalEngine",
    "AppraisalResult",
    "DiscreteEmotion",
    "AppraisalDimensions",
    # Layer 4
    "PolicyRegulator",
    "AffectiveAction",
    "SNNPolicyInterface",
]
