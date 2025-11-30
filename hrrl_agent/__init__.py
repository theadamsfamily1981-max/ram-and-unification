"""
HRRL Agent: Homeostatic Reinforcement Regulated Learning

A complete agent framework implementing:

Hard Spec (fully implemented):
- L1: Need vector n(t), drive d(t), F_int, HRRL reward r_t = -ΔF_int
- L2: Hyperbolic appraisal on M = ℍ^128, a(t) = W_app log_0(z_s ⊕ z_b)
- L3: MLP gating over [V, A, D, appraisal, epistemic, aleatoric]
      outputs: τ, lr_scale, mem_write_p, empathy_w
- L4: Memory with salience = ||d||·||a|| + βU_epi
      Replay distribution with (salience, Π_q, identity distance)
      LoRA adapters with homeostatic regularization
      "Reject update if F_int rises too much"
- Online & Sleep training loops

TGSFN Substrate (V. Edge of Chaos):
- Criticality control via Π_q minimization (g → 1)
- Antifragile loop with Jacobian spectral norm monitoring
- Hardware mandates: 16-bit fixed point, manifold recentering, K-FAC fidelity
- Fast Learnable Time Warping (FLTW) with O(N·T) complexity
- Avalanche exponent validation: α = 1.63 ± 0.04

Experimental/Guarded:
- Hyperbolic identity manifold (thresholds tunable)
- DAU: Very conservative (tiny step, identity/value bans, logging)
- Π_q auto-tuning: Disabled by default (measurement first)

Example usage:

    from hrrl_agent import create_agent, HRRLAgent, HRRLConfig
    from hrrl_agent import create_tgsfn_substrate, TGSFNConfig

    # Quick agent creation
    agent = create_agent(obs_dim=64, action_dim=8)

    # TGSFN substrate
    substrate = create_tgsfn_substrate(
        input_dim=64, hidden_dims=[256, 128], output_dim=32
    )

    # Use agent
    action, info = agent(observation)
    result = agent.step(observation, action, control_input)

    # Force consolidation
    agent.consolidate()
"""

# Core components
from .config import (
    HRRLConfig,
    L1Config,
    L2Config,
    L3Config,
    L4Config,
    IdentityConfig,
    DAUConfig,
    ThermodynamicsConfig,
    TrainingConfig
)

# L1: Homeostatic Core
from .l1_homeostat import (
    HomeostatL1,
    BatchedHomeostatL1,
    HomeostatState
)

# L2: Hyperbolic Appraisal
from .l2_hyperbolic import (
    HyperbolicAppraisalL2,
    HyperbolicAppraisalWithDrive,
    AppraisalOutput,
    PoincareOperations,
    HyperbolicLinear
)

# L3: Gating Controller
from .l3_gating import (
    GatingControllerL3,
    ResidualGatingController,
    GatingControllerWithHistory,
    GatingOutputs
)

# L4: Memory & Personalization
from .l4_memory import (
    ReplayBuffer,
    MemoryEntry,
    LoRALayer,
    LoRALinear,
    PersonalizationModule,
    HomeostaticRejectionGate,
    MemoryConsolidation
)

# Experimental: Identity
from .identity import (
    HyperbolicIdentity,
    IdentityState,
    IdentityAlertLevel,
    IdentityAwareEncoder,
    IdentityPreservingOptimizer
)

# Guarded: DAU
from .dau import (
    DynamicArchitectureUpdate,
    DAUGuard,
    DAUCheckpoint,
    DAUAction,
    DAUProposal
)

# Thermodynamics
from .thermodynamics import (
    EntropyProductionMonitor,
    ThermodynamicLoss,
    CriticalityAnalyzer,
    ThermodynamicsSnapshot
)

# Training Loops
from .loops import (
    OnlineLoop,
    SleepLoop,
    DualLoopTrainer,
    StepResult
)

# Main Agent
from .agent import (
    HRRLAgent,
    PolicyNetwork,
    create_agent
)

# Criticality Control (Edge of Chaos)
from .criticality import (
    CriticalityController,
    CriticalityConfig,
    CriticalityState,
    CriticalityRegime,
    CriticalInitializer,
    EffectiveGainEstimator,
    AvalancheAnalyzer,
    EIBalanceMonitor
)

# Antifragile Loop
from .antifragile import (
    AntifragileLoop,
    AntifragileConfig,
    AntifragileState,
    StabilityStatus,
    JacobianMonitor,
    AxiomCorrector
)

# Hardware Mandates
from .hardware import (
    FixedPoint16,
    FixedPointHyperbolic,
    ManifoldRecenterer,
    Orthonormalizer,
    KFACTracker,
    FastLearnableTimeWarping
)

# TGSFN Substrate
from .tgsfn import (
    TGSFNConfig,
    TGSFNState,
    TGSFNLoss,
    TGSFNLayer,
    TGSFNSubstrate,
    create_tgsfn_substrate
)

__version__ = "0.2.0"
__author__ = "HRRL Framework"

__all__ = [
    # Config
    "HRRLConfig",
    "L1Config",
    "L2Config",
    "L3Config",
    "L4Config",
    "IdentityConfig",
    "DAUConfig",
    "ThermodynamicsConfig",
    "TrainingConfig",

    # L1
    "HomeostatL1",
    "BatchedHomeostatL1",
    "HomeostatState",

    # L2
    "HyperbolicAppraisalL2",
    "HyperbolicAppraisalWithDrive",
    "AppraisalOutput",
    "PoincareOperations",
    "HyperbolicLinear",

    # L3
    "GatingControllerL3",
    "ResidualGatingController",
    "GatingControllerWithHistory",
    "GatingOutputs",

    # L4
    "ReplayBuffer",
    "MemoryEntry",
    "LoRALayer",
    "LoRALinear",
    "PersonalizationModule",
    "HomeostaticRejectionGate",
    "MemoryConsolidation",

    # Identity
    "HyperbolicIdentity",
    "IdentityState",
    "IdentityAlertLevel",
    "IdentityAwareEncoder",
    "IdentityPreservingOptimizer",

    # DAU
    "DynamicArchitectureUpdate",
    "DAUGuard",
    "DAUCheckpoint",
    "DAUAction",
    "DAUProposal",

    # Thermodynamics
    "EntropyProductionMonitor",
    "ThermodynamicLoss",
    "CriticalityAnalyzer",
    "ThermodynamicsSnapshot",

    # Loops
    "OnlineLoop",
    "SleepLoop",
    "DualLoopTrainer",
    "StepResult",

    # Agent
    "HRRLAgent",
    "PolicyNetwork",
    "create_agent",

    # Criticality (Edge of Chaos)
    "CriticalityController",
    "CriticalityConfig",
    "CriticalityState",
    "CriticalityRegime",
    "CriticalInitializer",
    "EffectiveGainEstimator",
    "AvalancheAnalyzer",
    "EIBalanceMonitor",

    # Antifragile
    "AntifragileLoop",
    "AntifragileConfig",
    "AntifragileState",
    "StabilityStatus",
    "JacobianMonitor",
    "AxiomCorrector",

    # Hardware
    "FixedPoint16",
    "FixedPointHyperbolic",
    "ManifoldRecenterer",
    "Orthonormalizer",
    "KFACTracker",
    "FastLearnableTimeWarping",

    # TGSFN
    "TGSFNConfig",
    "TGSFNState",
    "TGSFNLoss",
    "TGSFNLayer",
    "TGSFNSubstrate",
    "create_tgsfn_substrate",
]
