"""SNNFabricModel - Neural network wrapper for SNN fabric.

Wraps the SNNFabric to expose a standard (output, aux) interface
compatible with the TF-A-N training stack (FDT, topology, hooks, etc.).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from .fabric import SNNFabric, load_fabric_config, build_fabric_from_config
from .types import SpikeBatch


class SNNFabricModel(nn.Module):
    """Time-unrolled model wrapping SNNFabric.

    Provides the same (output, aux) interface as other TF-A-N models,
    enabling integration with:
    - FDTController (learning rate / gradient clipping adaptation)
    - TopologyHead (topological regularization)
    - SNNHooks (spike rate monitoring, sparsity tracking)

    Args:
        fabric: SNNFabric instance
        time_steps: Number of simulation timesteps
        input_pop: Name of population to receive external input (default: "input")
        output_pop: Name of population to use as output (default: "output")
    """

    def __init__(
        self,
        fabric: SNNFabric,
        time_steps: int = 256,
        input_pop: str = "input",
        output_pop: str = "output",
    ):
        super().__init__()
        self.fabric = fabric
        self.time_steps = time_steps
        self.input_pop = input_pop
        self.output_pop = output_pop

    def forward(
        self,
        x: torch.Tensor,
        time_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass through SNN fabric.

        Args:
            x: Input tensor [batch, N] or [batch, time, N]
            time_steps: Override default timesteps

        Returns:
            output: Spike accumulator for output population [batch, N_out]
            aux: Dictionary with spike metrics
        """
        time_steps = time_steps or self.time_steps
        batch = x.shape[0]
        device = x.device

        # Initialize fabric state
        states = self.fabric.init_state(batch=batch, device=str(device))

        # Make time axis explicit if needed
        if x.dim() == 2:
            # Replicate input across all timesteps
            x = x.unsqueeze(1).repeat(1, time_steps, 1)

        # Spike accumulators
        spike_accumulators = {
            name: torch.zeros(batch, pop.N, device=device)
            for name, pop in self.fabric.populations.items()
        }

        total_spikes = 0.0
        total_events = 0

        # Time-stepped simulation
        for t in range(time_steps):
            # External input to designated population
            ext_inputs = {
                self.input_pop: x[:, t, :]
            }

            # Step fabric
            states, spikes = self.fabric.step(states, external_inputs=ext_inputs)

            # Accumulate spikes
            for name, sb in spikes.items():
                spike_accumulators[name] += sb.spikes
                total_spikes += sb.spikes.sum().item()
                total_events += (sb.spikes > 0).sum().item()

        # Get output
        if self.output_pop in spike_accumulators:
            out = spike_accumulators[self.output_pop]
        else:
            # Fallback: concatenate all populations
            out = torch.cat(
                [spike_accumulators[n] for n in sorted(spike_accumulators.keys())],
                dim=-1,
            )

        # Compute metrics
        total_neurons = sum(pop.N for pop in self.fabric.populations.values())
        spike_rate = total_spikes / (batch * total_neurons * time_steps)
        spike_sparsity = 1.0 - spike_rate

        aux = {
            "spike_rate": spike_rate,
            "spike_sparsity": spike_sparsity,
            "active_events": total_events,
            "time_steps": time_steps,
            "spike_accumulators": spike_accumulators,
        }

        return out, aux

    @classmethod
    def from_config(
        cls,
        config_path: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> "SNNFabricModel":
        """Create model from configuration file.

        Args:
            config_path: Path to fabric YAML config
            device: Target device
            dtype: Data type
            **kwargs: Additional arguments for SNNFabricModel

        Returns:
            Configured SNNFabricModel
        """
        fabric_cfg = load_fabric_config(config_path)
        fabric = build_fabric_from_config(fabric_cfg, device=device, dtype=dtype)

        time_steps = kwargs.pop("time_steps", fabric_cfg.time_steps)

        return cls(fabric=fabric, time_steps=time_steps, **kwargs)


class SNNBackendModel(nn.Module):
    """Backend-compatible SNN model with encoder integration.

    Extends SNNFabricModel with:
    - Input encoding (continuous → spike-driving current)
    - Output decoding (spike accumulator → continuous)
    - Optional linear readout layer

    Args:
        fabric: SNNFabric instance
        d_input: Input feature dimension
        d_output: Output feature dimension
        time_steps: Simulation timesteps
        input_pop: Input population name
        output_pop: Output population name
        use_readout: Whether to add linear readout layer
    """

    def __init__(
        self,
        fabric: SNNFabric,
        d_input: int,
        d_output: int,
        time_steps: int = 256,
        input_pop: str = "input",
        output_pop: str = "output",
        use_readout: bool = True,
    ):
        super().__init__()

        self.fabric = fabric
        self.d_input = d_input
        self.d_output = d_output
        self.time_steps = time_steps
        self.input_pop = input_pop
        self.output_pop = output_pop

        # Get population sizes
        input_N = fabric.populations[input_pop].N
        output_N = fabric.populations[output_pop].N

        # Input encoder: continuous features → SNN input current
        self.encoder = nn.Linear(d_input, input_N)

        # Output decoder: spike accumulator → continuous output
        if use_readout:
            self.decoder = nn.Linear(output_N, d_output)
        else:
            self.decoder = None

        # Wrapped fabric model
        self.snn_model = SNNFabricModel(
            fabric=fabric,
            time_steps=time_steps,
            input_pop=input_pop,
            output_pop=output_pop,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with encoding/decoding.

        Args:
            x: Input features [batch, d_input] or [batch, seq, d_input]

        Returns:
            output: Output features [batch, d_output]
            aux: Spike metrics dictionary
        """
        # Handle sequence input (pool over sequence)
        if x.dim() == 3:
            x = x.mean(dim=1)

        # Encode to SNN input
        encoded = torch.relu(self.encoder(x))

        # Run SNN
        spike_out, aux = self.snn_model(encoded)

        # Decode to output
        if self.decoder is not None:
            output = self.decoder(spike_out)
        else:
            output = spike_out

        return output, aux


class SNNEmbeddingModel(nn.Module):
    """SNN model for embedding-based tasks.

    Takes token embeddings and produces spike-based representations.
    Useful for integrating SNN with transformer embeddings.

    Args:
        fabric: SNNFabric instance
        d_model: Embedding dimension
        time_steps: Simulation timesteps
        pooling: Pooling method ("mean", "max", "last")
    """

    def __init__(
        self,
        fabric: SNNFabric,
        d_model: int,
        time_steps: int = 256,
        pooling: str = "mean",
    ):
        super().__init__()

        self.fabric = fabric
        self.d_model = d_model
        self.time_steps = time_steps
        self.pooling = pooling

        # Find input and output population sizes
        pop_names = list(fabric.populations.keys())
        self.input_pop = pop_names[0]  # First population as input
        self.output_pop = pop_names[-1]  # Last population as output

        input_N = fabric.populations[self.input_pop].N
        output_N = fabric.populations[self.output_pop].N

        # Projection layers
        self.embed_to_snn = nn.Linear(d_model, input_N)
        self.snn_to_embed = nn.Linear(output_N, d_model)

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process embeddings through SNN.

        Args:
            embeddings: Token embeddings [batch, seq, d_model]

        Returns:
            output_embeddings: Processed embeddings [batch, seq, d_model]
            aux: Spike metrics
        """
        batch, seq_len, _ = embeddings.shape
        device = embeddings.device

        # Project to SNN input dimension
        snn_input = torch.relu(self.embed_to_snn(embeddings))  # [batch, seq, N_input]

        # Initialize fabric state
        states = self.fabric.init_state(batch=batch, device=str(device))

        # Process each position
        outputs = []
        total_spikes = 0.0
        total_events = 0
        total_neurons = sum(pop.N for pop in self.fabric.populations.values())

        for pos in range(seq_len):
            pos_input = snn_input[:, pos, :]

            # Run for time_steps
            spike_acc = torch.zeros(
                batch, self.fabric.populations[self.output_pop].N, device=device
            )

            for t in range(self.time_steps):
                ext_inputs = {self.input_pop: pos_input}
                states, spikes = self.fabric.step(states, external_inputs=ext_inputs)

                spike_acc += spikes[self.output_pop].spikes
                total_spikes += sum(sb.spikes.sum().item() for sb in spikes.values())
                total_events += sum((sb.spikes > 0).sum().item() for sb in spikes.values())

            # Project back to embedding dimension
            pos_output = self.snn_to_embed(spike_acc)
            outputs.append(pos_output)

        output_embeddings = torch.stack(outputs, dim=1)  # [batch, seq, d_model]

        spike_rate = total_spikes / (batch * seq_len * total_neurons * self.time_steps)

        aux = {
            "spike_rate": spike_rate,
            "spike_sparsity": 1.0 - spike_rate,
            "active_events": total_events,
            "time_steps": self.time_steps,
        }

        return output_embeddings, aux


__all__ = [
    "SNNFabricModel",
    "SNNBackendModel",
    "SNNEmbeddingModel",
]
