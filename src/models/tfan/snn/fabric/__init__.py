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
    export_kitten_fabric,
    export_kitten_from_snns_fabric,
)
from .from_model import (
    KittenFabricData,
    build_kitten_fabric_from_model,
    build_kitten_fabric_from_state_dict,
    quantize_q5_10,
    quantize_q1_14,
    quantize_q1_6,
    dense_to_csr,
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
    # Kitten FPGA export
    "export_kitten_fabric",
    "export_kitten_from_snns_fabric",
    "KittenFabricData",
    "build_kitten_fabric_from_model",
    "build_kitten_fabric_from_state_dict",
    "quantize_q5_10",
    "quantize_q1_14",
    "quantize_q1_6",
    "dense_to_csr",
]
