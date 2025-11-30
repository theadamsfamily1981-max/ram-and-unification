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

Experimental/Guarded:
- Hyperbolic identity manifold (thresholds tunable)
- DAU: Very conservative (tiny step, identity/value bans, logging)
- Π_q-based criticality: Measurement + logging (auto-tuning disabled)

Example usage:

    from hrrl_agent import create_agent, HRRLAgent, HRRLConfig

    # Quick creation with defaults
    agent = create_agent(obs_dim=64, action_dim=8)

    # Full control
    config = HRRLConfig()
    config.l1.num_needs = 8
    config.identity.enabled = True
    agent = HRRLAgent(config, obs_dim=64, action_dim=8)

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

__version__ = "0.1.0"
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
]
