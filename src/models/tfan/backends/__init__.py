"""Compute backends for TF-A-N.

Provides different execution backends:
- snn_emu: SNN emulation backend (CPU/GPU)
- fpga: FPGA hardware backend (Ara-SYNERGY)
"""

from .snn_emu import SNNBackend

__all__ = ["SNNBackend"]
