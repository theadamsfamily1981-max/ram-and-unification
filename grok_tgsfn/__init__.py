# grok_tgsfn/__init__.py
# Unified Homeostatic-Affective-Geometric Framework for Antifragile Agents
#
# This package implements the rigorous mathematical framework from the Grok memo:
#   - L1: HomeostatL1 - Continuous-time homeostatic dynamics
#   - L2: HyperbolicAppraisalL2 - Poincaré ball cognitive appraisal
#   - L3: GatingControllerL3 - Explicit gating equations
#   - L4: MemoryL4 - Salience, replay, personalization
#   - TGSFN: Thermodynamic-Geometric Spiking Field Network substrate
#   - Thermodynamics: Π_q monitoring and loss
#   - Coupling: Affective neuromodulation and prosocial coupling
#
# Key equations implemented:
#   F_int(n) = ½ n^T Σ^{-1} n          (free energy)
#   dn/dt = -Γ d(t) + ξ(t) + u(t)       (homeostatic dynamics)
#   r_t = -ΔF_int                       (HRRL intrinsic reward)
#   a(t) = W_app · log_μ(z_s ⊕ z_b)    (hyperbolic appraisal)
#   τ(t) = σ(3 - 2A + U_epi)           (temperature gating)
#   sal(t) = ||d|| · ||a|| + β U_epi   (salience)

from .config import (
    L1Config,
    L2Config,
    L3Config,
    L4Config,
    TGSFNConfig,
    ThermoConfig,
    CouplingConfig,
)

from .l1_homeostat import HomeostatL1, HomeostatState
from .l2_appraisal import HyperbolicAppraisalL2
from .l3_gating import GatingControllerL3, GatingOutputs
from .l4_memory import MemoryL4, MemoryItem

from .tgsfn_substrate import TGSFNLayer, TGSFNState
from .thermodynamics import ThermodynamicMonitor
from .coupling import AffectiveCoupler

__version__ = "0.1.0"

__all__ = [
    # Config
    "L1Config",
    "L2Config",
    "L3Config",
    "L4Config",
    "TGSFNConfig",
    "ThermoConfig",
    "CouplingConfig",
    # L1 - Homeostatic Core
    "HomeostatL1",
    "HomeostatState",
    # L2 - Hyperbolic Appraisal
    "HyperbolicAppraisalL2",
    # L3 - Gating Controller
    "GatingControllerL3",
    "GatingOutputs",
    # L4 - Memory
    "MemoryL4",
    "MemoryItem",
    # TGSFN Substrate
    "TGSFNLayer",
    "TGSFNState",
    # Thermodynamics
    "ThermodynamicMonitor",
    # Coupling
    "AffectiveCoupler",
]
