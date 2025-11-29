"""
Cathedral Avatar System - Hybrid Accelerated Architecture (HAA)

Python orchestration package for FPGA + GPU heterogeneous avatar generation.
"""

from .orchestrator import (
    CathedralAvatarOrchestrator,
    PersonalityMode,
    HAAConfig,
    FPGATTSManager,
    GPUAnimationManager,
)

__version__ = "0.1.0"
__author__ = "Cathedral Project"

__all__ = [
    "CathedralAvatarOrchestrator",
    "PersonalityMode",
    "HAAConfig",
    "FPGATTSManager",
    "GPUAnimationManager",
]
