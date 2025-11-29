"""SNN Fabric - Multi-population spiking neural network simulation.

The fabric provides:
- Multi-population architecture with configurable neuron types
- CI-compliant projections with TLS sparsity and low-rank weights
- Time-stepped simulation with external input support
- FPGA-compatible export format
"""

from .fabric import SNNFabric, load_fabric_config, build_fabric_from_config
from .export import (
    export_fabric_to_dict,
    export_fabric_to_binary,
    load_fabric_from_export,
    FPGAExporter,
    quantize_weights,
)

__all__ = [
    # Fabric
    "SNNFabric",
    "load_fabric_config",
    "build_fabric_from_config",
    # Export
    "export_fabric_to_dict",
    "export_fabric_to_binary",
    "load_fabric_from_export",
    "FPGAExporter",
    "quantize_weights",
]
