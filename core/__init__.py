# core/__init__.py
# TGSFN Core Modules
#
# This package contains the core mathematical objects for the TGSFN architecture:
#   - pi_q: Thermodynamic regularizer Π_q
#   - control_law: L5 Controller with Riemannian projection
#   - dau: Dynamic Axiom Updater for antifragility
#   - neurons: LIF neuron dynamics with GCSD

from .pi_q import compute_pi_q, EntropyProductionMonitor
from .control_law import L5Controller
from .dau import DynamicAxiomUpdater

__all__ = [
    "compute_pi_q",
    "EntropyProductionMonitor",
    "L5Controller",
    "DynamicAxiomUpdater",
]
