# analysis/__init__.py
# TGSFN Analysis Modules
#
# This package contains tools for validating criticality and measuring
# avalanche dynamics in spiking neural networks.
#
# Modules:
#   - avalanches: Avalanche extraction and statistics
#   - criticality: C-DP universality class validation

from .avalanches import (
    extract_avalanches,
    AvalancheStats,
    AvalancheAnalyzer,
)
from .criticality import (
    fit_power_law,
    validate_cdp_scaling,
    finite_size_analysis,
    CriticalityValidator,
)

__all__ = [
    "extract_avalanches",
    "AvalancheStats",
    "AvalancheAnalyzer",
    "fit_power_law",
    "validate_cdp_scaling",
    "finite_size_analysis",
    "CriticalityValidator",
]
