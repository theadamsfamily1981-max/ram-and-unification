# tfan/snn/fabric/types.py
"""
Core types for SNN fabric: events, states, and projection parameters.

These are thin wrappers that can later be swapped for FPGA buffers or
hardware-accelerated representations without changing the high-level API.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import torch


@dataclass
class SpikeBatch:
    """
    Logical spike/event container for a population.

    Attributes:
        spikes: [batch, N] binary or count spikes for a population
        meta: Optional metadata (time index, encoder info, etc.)

    Design rationale:
        - Simple tensor-based interface that's hardware-friendly
        - Can be serialized to CSR format for FPGA event queues
        - Meta dict allows extensibility without breaking interface
    """
    spikes: torch.Tensor
    meta: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate spike tensor shape."""
        assert self.spikes.ndim == 2, f"Expected [batch, N], got {self.spikes.shape}"

    @property
    def batch_size(self) -> int:
        return self.spikes.shape[0]

    @property
    def num_neurons(self) -> int:
        return self.spikes.shape[1]

    @property
    def event_count(self) -> int:
        """Total number of spike events across batch."""
        return self.spikes.sum().item()

    @property
    def sparsity(self) -> float:
        """Fraction of neurons that did NOT spike."""
        total = self.spikes.numel()
        return 1.0 - (self.event_count / total) if total > 0 else 1.0


@dataclass
class PopulationState:
    """
    Generic container for a population's internal state.

    For LIF neurons: data = {"v": membrane_voltage, "s": spikes}
    For other neuron types: data can hold arbitrary tensors

    Design rationale:
        - Generic dict allows different neuron models without changing interface
        - Keys are standardized per neuron type (e.g., "v", "s" for LIF)
        - Can be extended with plasticity traces, homeostatic variables, etc.
    """
    data: Dict[str, torch.Tensor]

    def __post_init__(self):
        """Validate that at least voltage/spikes are present for standard neurons."""
        if "v" in self.data and "s" in self.data:
            assert self.data["v"].shape == self.data["s"].shape, \
                f"Voltage and spike shapes must match: {self.data['v'].shape} vs {self.data['s'].shape}"

    @property
    def device(self) -> torch.device:
        """Device of the first tensor in state."""
        return next(iter(self.data.values())).device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the first tensor in state."""
        return next(iter(self.data.values())).dtype


@dataclass
class ProjectionParams:
    """
    Parameters of a synaptic projection (pre -> post connectivity).

    Attributes:
        N_pre: Number of presynaptic neurons
        N_post: Number of postsynaptic neurons
        k: Sparsity (average connections per neuron)
        r: Low-rank dimension (for low-rank factorization)
        delay: Synaptic delay in timesteps (default: 0)
        plasticity: Whether this projection supports learning (default: True)

    Design rationale:
        - Matches Ara-SYNERGY FPGA constraints (k=64, r=32 typical)
        - k and r are used by mask builders and low-rank synapse layers
        - Delay will be used for event-driven FPGA scheduling
        - Plasticity flag allows freezing certain projections

    CI Gates (matching scripts/bench_snn.py):
        - k ≤ 0.02 * N_pre (avg_degree gate)
        - r ≤ 0.02 * N_pre (rank gate)
        - Sparsity ≥ 0.98 (implied by k constraint)
    """
    N_pre: int
    N_post: int
    k: int
    r: int
    delay: int = 0
    plasticity: bool = True

    def __post_init__(self):
        """Validate projection parameters against CI gates."""
        assert self.N_pre > 0 and self.N_post > 0, "Population sizes must be positive"
        assert self.k > 0 and self.k <= self.N_pre, f"k={self.k} must be in (0, N_pre={self.N_pre}]"
        assert self.r > 0 and self.r <= min(self.N_pre, self.N_post), \
            f"r={self.r} must be in (0, min(N_pre={self.N_pre}, N_post={self.N_post})]"
        assert self.delay >= 0, f"Delay must be non-negative, got {self.delay}"

        # Warning if violating typical gates (not hard error, just warning)
        if self.k > 0.02 * self.N_pre:
            import warnings
            warnings.warn(
                f"k={self.k} exceeds recommended gate of 0.02*N_pre={0.02*self.N_pre:.1f}. "
                f"This may fail CI gates in bench_snn.py"
            )
        if self.r > 0.02 * self.N_pre:
            import warnings
            warnings.warn(
                f"r={self.r} exceeds recommended gate of 0.02*N_pre={0.02*self.N_pre:.1f}. "
                f"This may fail CI gates in bench_snn.py"
            )

    @property
    def sparsity(self) -> float:
        """Expected sparsity of the projection (fraction of zero weights)."""
        total_possible = self.N_pre * self.N_post
        actual_connections = self.N_pre * self.k  # Assuming k connections per pre neuron
        return 1.0 - (actual_connections / total_possible) if total_possible > 0 else 1.0

    @property
    def param_count_dense(self) -> int:
        """Parameter count for dense connectivity."""
        return self.N_pre * self.N_post

    @property
    def param_count_lowrank_sparse(self) -> int:
        """
        Parameter count for low-rank masked synapses: W ≈ M ⊙ (U V^T).

        Returns:
            U: N_post × r
            V: N_pre × r
            M: N_pre × k (sparse mask indices/values)
        """
        U_params = self.N_post * self.r
        V_params = self.N_pre * self.r
        M_params = self.N_pre * self.k  # k connections per pre neuron
        return U_params + V_params + M_params

    @property
    def param_reduction_pct(self) -> float:
        """Percentage parameter reduction vs dense."""
        dense = self.param_count_dense
        sparse = self.param_count_lowrank_sparse
        return 100.0 * (1.0 - sparse / dense) if dense > 0 else 0.0


# Neuron model type hints (for type checking and documentation)
NEURON_MODELS = ["lif", "alif", "input", "readout"]

# Standard state keys for different neuron types
LIF_STATE_KEYS = ["v", "s"]  # voltage, spikes
ALIF_STATE_KEYS = ["v", "s", "a"]  # voltage, spikes, adaptation
INPUT_STATE_KEYS = ["x"]  # raw input
READOUT_STATE_KEYS = ["y"]  # output activations
