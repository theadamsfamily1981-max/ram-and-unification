"""SNNFabric - Core multi-population spiking neural network.

Orchestrates populations and projections for time-stepped simulation.
"""

from __future__ import annotations

import yaml
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ..types import PopulationState, SpikeBatch, FabricConfig
from ..populations import Population, create_population
from ..projections import Projection, create_projection


def load_fabric_config(config_path: str | Path) -> FabricConfig:
    """Load fabric configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        FabricConfig instance
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    return FabricConfig.from_dict(raw)


def build_fabric_from_config(
    config: FabricConfig,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> "SNNFabric":
    """Build SNNFabric from configuration.

    Args:
        config: FabricConfig instance
        device: Target device
        dtype: Data type for weights

    Returns:
        Configured SNNFabric instance
    """
    fabric = SNNFabric(
        name=config.name,
        time_steps=config.time_steps,
        dt=config.dt,
        device=device,
        dtype=dtype,
    )

    # Create populations
    for pop_name, pop_cfg in config.populations.items():
        N = pop_cfg["N"]
        neuron_type = pop_cfg.get("neuron_type", "lif")
        params = pop_cfg.get("params", {})

        pop = create_population(N, neuron_type, params, name=pop_name)
        fabric.add_population(pop_name, pop)

    # Create projections
    pop_sizes = {name: pop.N for name, pop in fabric.populations.items()}
    for proj_cfg in config.projections:
        proj = create_projection(proj_cfg, pop_sizes, device=device, dtype=dtype)
        fabric.add_projection(proj)

    # Apply CI constraints if configured
    if config.ci_constraints.get("enabled", False):
        fabric.enforce_ci_constraints(
            max_rank=config.ci_constraints.get("max_rank", 32),
            max_nnz_per_row=config.ci_constraints.get("max_nnz_per_row", 64),
        )

    return fabric


class SNNFabric(nn.Module):
    """Multi-population spiking neural network fabric.

    Manages populations of neurons connected by projections,
    providing time-stepped simulation with external input support.

    Args:
        name: Fabric identifier
        time_steps: Default number of simulation timesteps
        dt: Timestep duration (ms)
        device: Target device
        dtype: Data type for computations
    """

    def __init__(
        self,
        name: str = "snn_fabric",
        time_steps: int = 256,
        dt: float = 1.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.name = name
        self.time_steps = time_steps
        self.dt = dt
        self.device = device
        self.dtype = dtype

        # Populations and projections
        self.populations: Dict[str, Population] = nn.ModuleDict()
        self.projections: List[Projection] = nn.ModuleList()

        # Index projections by target population for efficient lookup
        self._proj_by_post: Dict[str, List[Projection]] = {}

    def add_population(self, name: str, population: Population):
        """Add a population to the fabric.

        Args:
            name: Population identifier
            population: Population instance
        """
        population.to(self.device)
        self.populations[name] = population
        self._proj_by_post[name] = []

    def add_projection(self, projection: Projection):
        """Add a projection to the fabric.

        Args:
            projection: Projection instance connecting two populations
        """
        if projection.pre not in self.populations:
            raise ValueError(f"Presynaptic population '{projection.pre}' not found")
        if projection.post not in self.populations:
            raise ValueError(f"Postsynaptic population '{projection.post}' not found")

        projection.to(self.device)
        self.projections.append(projection)
        self._proj_by_post[projection.post].append(projection)

    def init_state(
        self,
        batch: int = 1,
        device: Optional[str] = None,
    ) -> Dict[str, PopulationState]:
        """Initialize state for all populations.

        Args:
            batch: Batch size
            device: Target device (uses fabric device if None)

        Returns:
            Dictionary mapping population names to initial states
        """
        device = device or self.device
        return {
            name: pop.init_state(batch, device)
            for name, pop in self.populations.items()
        }

    def step(
        self,
        states: Dict[str, PopulationState],
        external_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, PopulationState], Dict[str, SpikeBatch]]:
        """Advance fabric by one timestep.

        Args:
            states: Current population states
            external_inputs: Optional external currents by population name

        Returns:
            new_states: Updated population states
            spikes: Spike batches for each population
        """
        external_inputs = external_inputs or {}

        # Collect spikes from previous timestep for projection inputs
        # (In first step, states may not have "last_spikes" so we handle that)
        prev_spikes: Dict[str, SpikeBatch] = {}
        for name, state in states.items():
            if "last_spikes" in state.data:
                prev_spikes[name] = SpikeBatch(spikes=state.data["last_spikes"])
            else:
                # No previous spikes - create zeros
                batch = state.v.shape[0] if state.v is not None else 1
                N = self.populations[name].N
                prev_spikes[name] = SpikeBatch(
                    spikes=torch.zeros(batch, N, device=self.device)
                )

        # Compute synaptic currents for each population
        synaptic_currents: Dict[str, torch.Tensor] = {}
        for pop_name, pop in self.populations.items():
            batch = states[pop_name].v.shape[0]
            total_current = torch.zeros(batch, pop.N, device=self.device, dtype=self.dtype)

            # Add contributions from all projections targeting this population
            for proj in self._proj_by_post.get(pop_name, []):
                if proj.pre in prev_spikes:
                    current = proj(prev_spikes[proj.pre])
                    total_current = total_current + current

            # Add external input if provided
            if pop_name in external_inputs:
                total_current = total_current + external_inputs[pop_name]

            synaptic_currents[pop_name] = total_current

        # Update each population
        new_states: Dict[str, PopulationState] = {}
        new_spikes: Dict[str, SpikeBatch] = {}

        for pop_name, pop in self.populations.items():
            state = states[pop_name]
            current = synaptic_currents[pop_name]

            # Advance population
            new_state, spikes = pop(state, current)

            # Store spikes for next timestep's projection computation
            new_state.data["last_spikes"] = spikes.spikes

            new_states[pop_name] = new_state
            new_spikes[pop_name] = spikes

        return new_states, new_spikes

    def run(
        self,
        external_inputs: Optional[Dict[str, torch.Tensor]] = None,
        time_steps: Optional[int] = None,
        batch: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Run full simulation.

        Args:
            external_inputs: External input currents [batch, time, N] or [batch, N]
            time_steps: Number of timesteps (uses fabric default if None)
            batch: Batch size

        Returns:
            spike_accumulators: Accumulated spikes per population [batch, N]
            aux: Auxiliary metrics (spike rates, event counts, etc.)
        """
        time_steps = time_steps or self.time_steps
        device = self.device

        # Initialize state
        states = self.init_state(batch, device)

        # Prepare external inputs
        if external_inputs is not None:
            # Ensure time dimension
            for key, val in external_inputs.items():
                if val.dim() == 2:
                    external_inputs[key] = val.unsqueeze(1).repeat(1, time_steps, 1)

        # Spike accumulators
        spike_accumulators = {
            name: torch.zeros(batch, pop.N, device=device)
            for name, pop in self.populations.items()
        }

        total_spikes = 0.0
        total_events = 0

        # Run simulation
        for t in range(time_steps):
            # Get external inputs for this timestep
            ext_t = None
            if external_inputs is not None:
                ext_t = {
                    name: inp[:, t, :] if inp.dim() == 3 else inp
                    for name, inp in external_inputs.items()
                }

            # Step simulation
            states, spikes = self.step(states, ext_t)

            # Accumulate spikes
            for name, spike_batch in spikes.items():
                spike_accumulators[name] += spike_batch.spikes
                total_spikes += spike_batch.spikes.sum().item()
                total_events += spike_batch.count

        # Compute metrics
        total_neurons = sum(pop.N for pop in self.populations.values())
        spike_rate = total_spikes / (batch * total_neurons * time_steps)
        spike_sparsity = 1.0 - spike_rate

        aux = {
            "spike_rate": spike_rate,
            "spike_sparsity": spike_sparsity,
            "active_events": total_events,
            "time_steps": time_steps,
            "total_neurons": total_neurons,
        }

        return spike_accumulators, aux

    def enforce_ci_constraints(self, max_rank: int = 32, max_nnz_per_row: int = 64):
        """Enforce CI constraints on all projections.

        Args:
            max_rank: Maximum low-rank factor
            max_nnz_per_row: Maximum nonzeros per row
        """
        for proj in self.projections:
            if hasattr(proj, "synapse") and hasattr(proj.synapse, "r"):
                if proj.synapse.r > max_rank:
                    raise ValueError(
                        f"Projection {proj.name} has rank {proj.synapse.r} > {max_rank}"
                    )
                if proj.synapse.k > max_nnz_per_row:
                    raise ValueError(
                        f"Projection {proj.name} has k={proj.synapse.k} > {max_nnz_per_row}"
                    )

    def ci_audit(self) -> Dict[str, Any]:
        """Audit CI compliance of entire fabric.

        Returns:
            Dictionary with compliance status and details
        """
        results = {"compliant": True, "projections": {}}

        for proj in self.projections:
            if hasattr(proj, "ci_audit"):
                audit = proj.ci_audit()
                results["projections"][proj.name] = audit
                if not audit.get("ci_compliant", True):
                    results["compliant"] = False

        return results

    def summary(self) -> str:
        """Generate human-readable fabric summary."""
        lines = [
            f"=== SNN Fabric: {self.name} ===",
            f"Device: {self.device}, dtype: {self.dtype}",
            f"Time steps: {self.time_steps}, dt: {self.dt}ms",
            "",
            "Populations:",
        ]

        total_neurons = 0
        for name, pop in self.populations.items():
            lines.append(f"  {name}: N={pop.N}")
            total_neurons += pop.N

        lines.append(f"  Total neurons: {total_neurons}")
        lines.append("")
        lines.append("Projections:")

        total_params = 0
        for proj in self.projections:
            info = f"  {proj.name}: {proj.pre} -> {proj.post}"
            if hasattr(proj, "synapse"):
                syn = proj.synapse
                params = syn.U.numel() + syn.V.numel()
                total_params += params
                info += f" (r={syn.r}, k={syn.k}, params={params:,})"
            lines.append(info)

        lines.append(f"  Total projection params: {total_params:,}")

        return "\n".join(lines)


__all__ = [
    "SNNFabric",
    "load_fabric_config",
    "build_fabric_from_config",
]
