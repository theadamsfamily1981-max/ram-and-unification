"""TFF - Topological Feature Fusion Framework.

This module implements a unified framework for:
1. Multi-modal feature fusion via MCCA alignment
2. Topological feature extraction via persistent homology
3. Topological regularization for representation stability
4. UDK (Unified Dynamics and Knowledge) adaptive control
5. COS (Cognitive Offloading Subsystem) for NCE decision making

Architecture Overview:
    ┌─────────────────────────────────────────────────────────────┐
    │                    TFF Framework                            │
    ├─────────────────────────────────────────────────────────────┤
    │  ┌───────────────┐    ┌───────────────┐    ┌─────────────┐  │
    │  │ TopFusion     │───▶│ TopologyHead  │───▶│ Regularizer │  │
    │  │ Encoder       │    │ (PH Engine)   │    │ (Betti EMA) │  │
    │  └───────────────┘    └───────────────┘    └─────────────┘  │
    │         │                    │                    │         │
    │         ▼                    ▼                    ▼         │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │              UDK Controller                            │  │
    │  │  σ (Belief Cost) + ε (Turbulence) + κ (Curvature)     │  │
    │  │  ─────────────────▶ UTCF ─────────────────▶            │  │
    │  │  Learning Rate × Temperature × Memory Write            │  │
    │  └───────────────────────────────────────────────────────┘  │
    │         │                                                   │
    │         ▼                                                   │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │           Cognitive Offloading Subsystem               │  │
    │  │  NCE Policy: Internal vs Tool vs Memory vs Human       │  │
    │  └───────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘

Usage:
    from tff import (
        TopFusionEncoder,
        TopologyHead,
        TopologicalRegularizer,
        UDKController,
        CognitiveOffloadingSubsystem,
    )

    # Multi-modal fusion
    encoder = TopFusionEncoder(config)
    fused = encoder([text_emb, audio_emb, vision_emb])

    # Topological features
    topo_head = TopologyHead(config)
    topo_features = topo_head(fused["fused"])

    # Adaptive control
    udk = UDKController(config)
    control = udk(cost_R=kl_div, precision_gain=info_gain)

    # Offloading decisions
    cos = CognitiveOffloadingSubsystem(config)
    decision = cos(complexity=0.8, uncertainty=0.3)
"""

# TopFusionEncoder and MCCA components
from .encoder import (
    MCCAConfig,
    TopFusionConfig,
    MCCAProjection,
    MCCAWrapper,
    TopFusionEncoder,
    ModalityEncoder,
    create_top_fusion_encoder,
)

# Topology components
from .topology import (
    HomologyDimension,
    PHConfig,
    PIConfig,
    TopologyHeadConfig,
    PersistenceDiagram,
    PersistentHomologyEngine,
    PersistenceImageEncoder,
    PersistenceLandscapeEncoder,
    TopologyHead,
    create_topology_head,
)

# Regularization components
from .regularizer import (
    RegularizerConfig,
    BettiTracker,
    TopologicalRegularizer,
    TopologicalRegularizerWrapper,
    create_topological_regularizer,
)

# UDK Controller components
from .udk import (
    ControlSignal,
    SigmaConfig,
    EpsilonConfig,
    KappaConfig,
    UDKConfig,
    SigmaProxy,
    EpsilonProxy,
    KappaProxy,
    UDKController,
    UDKOptimizer,
    create_udk_controller,
)

# Cognitive Offloading components
from .cos import (
    OffloadTarget,
    OffloadReason,
    ResourceCosts,
    NCEConfig,
    OffloadDecision,
    WorkingMemoryState,
    ToolRegistry,
    NCEPolicy,
    CognitiveOffloadingSubsystem,
    create_cos,
)


__version__ = "0.1.0"

__all__ = [
    # Encoder
    "MCCAConfig",
    "TopFusionConfig",
    "MCCAProjection",
    "MCCAWrapper",
    "TopFusionEncoder",
    "ModalityEncoder",
    "create_top_fusion_encoder",
    # Topology
    "HomologyDimension",
    "PHConfig",
    "PIConfig",
    "TopologyHeadConfig",
    "PersistenceDiagram",
    "PersistentHomologyEngine",
    "PersistenceImageEncoder",
    "PersistenceLandscapeEncoder",
    "TopologyHead",
    "create_topology_head",
    # Regularizer
    "RegularizerConfig",
    "BettiTracker",
    "TopologicalRegularizer",
    "TopologicalRegularizerWrapper",
    "create_topological_regularizer",
    # UDK
    "ControlSignal",
    "SigmaConfig",
    "EpsilonConfig",
    "KappaConfig",
    "UDKConfig",
    "SigmaProxy",
    "EpsilonProxy",
    "KappaProxy",
    "UDKController",
    "UDKOptimizer",
    "create_udk_controller",
    # COS
    "OffloadTarget",
    "OffloadReason",
    "ResourceCosts",
    "NCEConfig",
    "OffloadDecision",
    "WorkingMemoryState",
    "ToolRegistry",
    "NCEPolicy",
    "CognitiveOffloadingSubsystem",
    "create_cos",
]
