# tfan/snn/fabric/__init__.py
"""
SNN Fabric: Hardware-shaped multi-population spiking neural network framework.

This module provides abstractions for building SNN systems that map cleanly
to FPGA/neuromorphic hardware while remaining trainable and testable in software.

Core concepts:
    - Population: Group of neurons (LIF, input, readout, etc.)
    - Projection: Synaptic connectivity (low-rank masked, dense, etc.)
    - Fabric: Graph of populations + projections with discrete-time stepping

Hardware mapping:
    - Populations → memory blocks (state) + compute kernels (dynamics)
    - Projections → sparse MatVec units (CSR format)
    - Fabric → control FSM orchestrating updates

Example:
    >>> from tfan.snn.fabric import build_fabric_from_config
    >>> fabric = build_fabric_from_config("configs/snn/fabric_toy.yaml")
    >>> results = fabric.run(timesteps=256, batch=8, device="cuda")
    >>> print(f"Spike rate: {results['overall_spike_rate']:.3f}")
"""

# Core types
from .types import (
    SpikeBatch,
    PopulationState,
    ProjectionParams,
    NEURON_MODELS,
    LIF_STATE_KEYS,
    ALIF_STATE_KEYS,
    INPUT_STATE_KEYS,
    READOUT_STATE_KEYS,
)

# Populations
from .populations import (
    Population,
    LIFPopulation,
    InputPopulation,
    ReadoutPopulation,
    SurrogateHeaviside,
)

# Projections
from .projections import (
    Projection,
    LowRankProjection,
    LowRankMaskedSynapse,
    DenseProjection,
)

# Fabric
from .fabric import SNNFabric

# Config
from .config import (
    PopulationConfig,
    ProjectionConfig,
    FabricConfig,
    build_fabric_from_config,
    build_feedforward_config,
)

__all__ = [
    # Types
    "SpikeBatch",
    "PopulationState",
    "ProjectionParams",
    "NEURON_MODELS",
    "LIF_STATE_KEYS",
    "ALIF_STATE_KEYS",
    "INPUT_STATE_KEYS",
    "READOUT_STATE_KEYS",
    # Populations
    "Population",
    "LIFPopulation",
    "InputPopulation",
    "ReadoutPopulation",
    "SurrogateHeaviside",
    # Projections
    "Projection",
    "LowRankProjection",
    "LowRankMaskedSynapse",
    "DenseProjection",
    # Fabric
    "SNNFabric",
    # Config
    "PopulationConfig",
    "ProjectionConfig",
    "FabricConfig",
    "build_fabric_from_config",
    "build_feedforward_config",
]
