# tgsfn_core/__init__.py
# TGSFN Core - Thermodynamic-Geometric Spiking Field Network
#
# This package implements the core SNN components for the TGSFN architecture:
#   - LIF neuron dynamics with E/I balance
#   - Thermodynamic regularizer (Π_q)
#   - Avalanche detection and criticality metrics

from .snn_model import LIFLayer, TGSFNNetwork
from .piq_loss import compute_piq, compute_internal_free_energy, tgsfn_loss
from .avalanches import detect_avalanches, compute_avalanche_stats
from .metrics import CriticalityMetrics, compute_branching_ratio

__all__ = [
    "LIFLayer",
    "TGSFNNetwork",
    "compute_piq",
    "compute_internal_free_energy",
    "tgsfn_loss",
    "detect_avalanches",
    "compute_avalanche_stats",
    "CriticalityMetrics",
    "compute_branching_ratio",
]

__version__ = "0.1.0"
