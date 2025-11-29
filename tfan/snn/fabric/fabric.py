# tfan/snn/fabric/fabric.py
"""
SNN Fabric: Multi-population, multi-projection graph container.

The fabric is the top-level orchestrator that:
    - Holds populations (neuron groups)
    - Holds projections (synaptic connections)
    - Maintains per-population state
    - Steps the entire network forward in time

Design philosophy:
    - Simple synchronous discrete-time stepping (for now)
    - Clear boundaries suitable for hardware mapping
    - Event-driven optimization where beneficial
    - Extensible to asynchronous, event-queue-based scheduling

Hardware mapping:
    - Fabric becomes the top-level control FSM
    - Populations → memory blocks + compute kernels
    - Projections → sparse MatVec units
    - Step() → microcoded sequence of population updates
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from .types import PopulationState, SpikeBatch
from .populations import Population
from .projections import Projection


class SNNFabric(nn.Module):
    """
    Logical SNN fabric: multi-population, multi-projection graph.

    The fabric manages:
        - Multiple neuron populations (input, hidden, output)
        - Synaptic projections connecting them
        - Temporal evolution via step()

    Current implementation:
        - Synchronous discrete-time stepping (all populations update together)
        - Fixed topological order of populations
        - No delays or event queues (future extension)

    Future extensions:
        - Event-driven scheduling (only process active populations)
        - Per-population clocks (heterogeneous time scales)
        - Synaptic delays via event queues
        - FPGA/neuromorphic hardware backends
    """

    def __init__(
        self,
        populations: Dict[str, Population],
        projections: List[Projection],
        validate: bool = True,
    ):
        """
        Args:
            populations: Dict mapping name -> Population
            projections: List of Projection objects connecting populations
            validate: If True, validate graph structure (no orphan projections, etc.)
        """
        super().__init__()

        # Store populations and projections as nn.ModuleDict/List
        self.populations = nn.ModuleDict(populations)
        self.projections = nn.ModuleList(projections)

        # Build adjacency: post_name -> list of projections feeding it
        self._incoming: Dict[str, List[Projection]] = {name: [] for name in populations}
        for proj in self.projections:
            if proj.post not in self._incoming:
                raise ValueError(
                    f"Projection '{proj.name}' targets unknown population '{proj.post}'. "
                    f"Known populations: {list(populations.keys())}"
                )
            self._incoming[proj.post].append(proj)

        # Validate graph structure if requested
        if validate:
            self._validate_graph()

    def _validate_graph(self):
        """
        Validate fabric graph structure.

        Checks:
            - All projections reference existing populations
            - No orphan populations (optional warning)
            - No self-loops with zero delay (can cause issues in sync stepping)
        """
        pop_names = set(self.populations.keys())

        for proj in self.projections:
            # Check presynaptic population exists
            if proj.pre not in pop_names:
                raise ValueError(
                    f"Projection '{proj.name}' has unknown presynaptic population '{proj.pre}'. "
                    f"Known populations: {list(pop_names)}"
                )

            # Check postsynaptic population exists
            if proj.post not in pop_names:
                raise ValueError(
                    f"Projection '{proj.name}' has unknown postsynaptic population '{proj.post}'. "
                    f"Known populations: {list(pop_names)}"
                )

            # Warn about self-loops with zero delay
            if proj.pre == proj.post and proj.params.delay == 0:
                import warnings
                warnings.warn(
                    f"Projection '{proj.name}' is a self-loop with zero delay. "
                    f"This may cause instability in synchronous stepping."
                )

    def init_state(self, batch: int, device: str) -> Dict[str, PopulationState]:
        """
        Initialize state for all populations.

        Args:
            batch: Batch size
            device: Device string (e.g., "cuda:0", "cpu")

        Returns:
            Dict mapping population name -> PopulationState
        """
        return {
            name: pop.init_state(batch=batch, device=device)
            for name, pop in self.populations.items()
        }

    def step(
        self,
        states: Dict[str, PopulationState],
        external_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, PopulationState], Dict[str, SpikeBatch]]:
        """
        Step the entire fabric forward one logical timestep.

        This is a synchronous update: all populations read their current state,
        compute inputs from projections, update, and write new state.

        Args:
            states: Current state for each population
            external_inputs: Optional external currents per population [batch, N_pop]

        Returns:
            new_states: Updated state for each population
            spikes_out: Emitted spikes for each population

        Design notes:
            - For now, simple sequential update in population order
            - Later: topological sort, parallel updates, event-driven scheduling
            - Hardware: this becomes a microcoded FSM on FPGA
        """
        external_inputs = external_inputs or {}
        new_states: Dict[str, PopulationState] = {}
        spikes_out: Dict[str, SpikeBatch] = {}

        # Step each population in order
        # TODO: Topological sort to respect dependencies
        for name, pop in self.populations.items():
            # 1. Get external input (if any)
            ext_input = external_inputs.get(name, None)
            if ext_input is None:
                # No external input, initialize to zero
                batch = list(states.values())[0].data[list(states.values())[0].data.keys().__iter__().__next__()].shape[0]
                device = states[name].device
                total_current = torch.zeros(batch, pop.N, device=device)
            else:
                total_current = ext_input.clone()

            # 2. Aggregate synaptic input from all incoming projections
            for proj in self._incoming[name]:
                # Get presynaptic spikes from this timestep
                # NOTE: In synchronous stepping, we use spikes from CURRENT timestep
                # This creates a feedforward data flow
                pre_spikes = spikes_out.get(proj.pre, None)
                if pre_spikes is not None:
                    # Project spikes to postsynaptic input
                    syn_current = proj(pre_spikes)
                    total_current = total_current + syn_current

            # 3. Step population with aggregated input
            state_new, spikes = pop.step(states[name], total_current)
            new_states[name] = state_new
            spikes_out[name] = spikes

        return new_states, spikes_out

    def run(
        self,
        timesteps: int,
        batch: int,
        device: str = "cpu",
        external_inputs: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        return_spikes: bool = True,
        return_states: bool = False,
    ) -> Dict[str, any]:
        """
        Run the fabric for multiple timesteps.

        Args:
            timesteps: Number of timesteps to simulate
            batch: Batch size
            device: Device to run on
            external_inputs: Optional dict mapping timestep -> {pop_name: input}
            return_spikes: If True, accumulate and return all spikes
            return_states: If True, return final states

        Returns:
            Dict containing:
                - "spike_counts": {pop_name: total_spike_count} if return_spikes
                - "spike_rate": {pop_name: avg_spike_rate} if return_spikes
                - "final_states": {pop_name: PopulationState} if return_states
                - "total_events": total number of spike events across all pops

        Design notes:
            - This is the main entry point for simulation
            - For training: call this, compute loss on outputs, backprop
            - For FPGA: this loop becomes the main control sequence
        """
        external_inputs = external_inputs or {}

        # Initialize state
        states = self.init_state(batch=batch, device=device)

        # Accumulate spike counts
        spike_accumulators = {
            name: torch.zeros(batch, pop.N, device=device)
            for name, pop in self.populations.items()
        }
        total_events = 0

        # Time loop
        for t in range(timesteps):
            # Get external inputs for this timestep (if any)
            ext_t = external_inputs.get(t, {})

            # Step fabric
            states, spikes = self.step(states, external_inputs=ext_t)

            # Accumulate spikes
            if return_spikes:
                for name, spike_batch in spikes.items():
                    spike_accumulators[name] += spike_batch.spikes
                    total_events += spike_batch.event_count

        # Compute statistics
        results = {}

        if return_spikes:
            # Total spike counts per population
            results["spike_counts"] = {
                name: acc.sum().item()
                for name, acc in spike_accumulators.items()
            }

            # Average spike rate per population
            total_neurons = sum(pop.N for pop in self.populations.values())
            total_possible = batch * total_neurons * timesteps
            results["spike_rate"] = {
                name: spike_accumulators[name].sum().item() / (batch * self.populations[name].N * timesteps)
                for name in spike_accumulators.keys()
            }

            results["total_events"] = total_events
            results["overall_spike_rate"] = total_events / total_possible if total_possible > 0 else 0.0

        if return_states:
            results["final_states"] = states

        return results

    @property
    def total_neurons(self) -> int:
        """Total number of neurons across all populations."""
        return sum(pop.N for pop in self.populations.values())

    @property
    def total_synapses(self) -> int:
        """Total number of synaptic parameters across all projections."""
        return sum(
            proj.params.param_count_lowrank_sparse
            for proj in self.projections
        )

    @property
    def total_synapses_dense_equivalent(self) -> int:
        """Total synapses if all projections were dense."""
        return sum(
            proj.params.param_count_dense
            for proj in self.projections
        )

    @property
    def param_reduction_pct(self) -> float:
        """Overall parameter reduction percentage vs dense."""
        dense = self.total_synapses_dense_equivalent
        sparse = self.total_synapses
        return 100.0 * (1.0 - sparse / dense) if dense > 0 else 0.0

    def summary(self) -> str:
        """Generate a human-readable summary of the fabric."""
        lines = [
            "=" * 70,
            "SNN Fabric Summary",
            "=" * 70,
            f"Populations: {len(self.populations)}",
            f"Projections: {len(self.projections)}",
            f"Total neurons: {self.total_neurons:,}",
            f"Total synapses (low-rank sparse): {self.total_synapses:,}",
            f"Dense equivalent: {self.total_synapses_dense_equivalent:,}",
            f"Parameter reduction: {self.param_reduction_pct:.2f}%",
            "",
            "Populations:",
        ]

        for name, pop in self.populations.items():
            lines.append(f"  {name}: {pop.__class__.__name__}(N={pop.N})")

        lines.append("")
        lines.append("Projections:")
        for proj in self.projections:
            lines.append(
                f"  {proj.name}: {proj.pre} → {proj.post} "
                f"({proj.params.N_pre}×{proj.params.N_post}, k={proj.params.k}, r={proj.params.r})"
            )

        lines.append("=" * 70)
        return "\n".join(lines)
