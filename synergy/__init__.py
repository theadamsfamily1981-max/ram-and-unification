"""Ara-SYNERGY FPGA Integration.

Provides hardware acceleration for SNN fabric via FPGA.
The FpgaFabric class implements the same step() API as SNNFabric
but executes on FPGA hardware.
"""

from .fpga_device import FpgaFabric, FpgaHandle, FpgaConfig

__all__ = [
    "FpgaFabric",
    "FpgaHandle",
    "FpgaConfig",
]
