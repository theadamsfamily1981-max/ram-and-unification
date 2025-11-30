"""4-Layer Grounded Affect Architecture.

This module implements a biologically-inspired affective system where:
- Emotion is the gradient signal for long-horizon control
- Valence, arousal, and dominance are grounded in homeostatic dynamics
- Policy actions are "coping behaviors" that minimize homeostatic drive

Architecture:
    Layer 1 (Insula) - Homeostatic Core
        Internal needs and set-points, drive computation, reward signal

    Layer 2 (Pulse) - Interoceptive Monitor
        Multimodal sensing, PAD estimation, uncertainty quantification

    Layer 3 (Amygdala) - Appraisal Engine
        Cognitive interpretation, discrete emotions, generated internal PAD

    Layer 4 (PFC) - Policy/Regulator
        Action selection (gates), RL optimization, SNN hardware interface

Usage:
    from affect.core import (
        HomeostaticCore,
        InteroceptiveMonitor,
        AppraisalEngine,
        PolicyRegulator,
    )

    # Initialize layers
    core = HomeostaticCore()
    monitor = InteroceptiveMonitor()
    appraisal = AppraisalEngine()
    policy = PolicyRegulator()

    # Process interaction
    core.update_state(energy=0.7, social=0.6)
    event = monitor.process(text="I'm confused by this")
    result = appraisal.appraise(core, event)
    action = policy.select_action(core, event, result)

    # Use action gates
    if action.request_clarification:
        ask_user_for_help()
"""

# Layer 1: Homeostatic Core
from .homeostatic import (
    NeedType,
    HomeostaticState,
    HomeostaticSetpoints,
    compute_homeostatic_drive,
    compute_homeostatic_reward,
    compute_valence_arousal_dominance,
    HomeostaticCore,
    estimate_energy_from_metrics,
    estimate_integrity_from_metrics,
    estimate_social_from_interaction,
    estimate_novelty_from_signals,
)

# Layer 2: Interoceptive Monitor
from .interoceptive import (
    TextFeatures,
    AudioFeatures,
    VisionFeatures,
    UncertaintyEstimates,
    PADEstimate,
    InteroceptiveEvent,
    InteroceptiveMonitor,
)

# Layer 3: Appraisal Engine
from .appraisal import (
    DiscreteEmotion,
    AppraisalDimensions,
    AppraisalResult,
    rule_based_appraisal,
    NeuralAppraisalNetwork,
    AppraisalEngine,
)

# Layer 4: Policy/Regulator
from .policy import (
    AffectiveAction,
    build_policy_observation,
    heuristic_policy,
    AffectivePolicyNetwork,
    PolicyRegulator,
    SNNPolicyInterface,
)


__all__ = [
    # Layer 1
    "NeedType",
    "HomeostaticState",
    "HomeostaticSetpoints",
    "compute_homeostatic_drive",
    "compute_homeostatic_reward",
    "compute_valence_arousal_dominance",
    "HomeostaticCore",
    "estimate_energy_from_metrics",
    "estimate_integrity_from_metrics",
    "estimate_social_from_interaction",
    "estimate_novelty_from_signals",
    # Layer 2
    "TextFeatures",
    "AudioFeatures",
    "VisionFeatures",
    "UncertaintyEstimates",
    "PADEstimate",
    "InteroceptiveEvent",
    "InteroceptiveMonitor",
    # Layer 3
    "DiscreteEmotion",
    "AppraisalDimensions",
    "AppraisalResult",
    "rule_based_appraisal",
    "NeuralAppraisalNetwork",
    "AppraisalEngine",
    # Layer 4
    "AffectiveAction",
    "build_policy_observation",
    "heuristic_policy",
    "AffectivePolicyNetwork",
    "PolicyRegulator",
    "SNNPolicyInterface",
]
