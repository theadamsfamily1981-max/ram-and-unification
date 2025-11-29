"""SNN (Spiking Neural Network) module for TF-A-N.

Provides spiking neural network simulation with:
- Multi-population fabric architecture
- TLS (Top-k Landmark Selection) sparse connectivity
- Low-rank weight decomposition for CI compliance
- FPGA-compatible export format
"""

from .fabric import (
    SNNFabric,
    load_fabric_config,
    build_fabric_from_config,
)
from .types import (
    PopulationState,
    SpikeBatch,
    NeuronParams,
    SynapseParams,
    ProjectionParams,
)
from .populations import Population, LIFPopulation
from .projections import Projection, LowRankProjection
from .synapses import LowRankMaskedSynapse
from .encoders import RateCodeEncoder, PoissonEncoder, LatencyEncoder
from .model import SNNFabricModel

__all__ = [
    # Fabric
    "SNNFabric",
    "load_fabric_config",
    "build_fabric_from_config",
    # Types
    "PopulationState",
    "SpikeBatch",
    "NeuronParams",
    "SynapseParams",
    "ProjectionParams",
    # Populations
    "Population",
    "LIFPopulation",
    # Projections
    "Projection",
    "LowRankProjection",
    # Synapses
    "LowRankMaskedSynapse",
    # Encoders
    "RateCodeEncoder",
    "PoissonEncoder",
    "LatencyEncoder",
    # Model
    "SNNFabricModel",
]
