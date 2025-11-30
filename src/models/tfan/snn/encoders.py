"""Spike encoders for SNN fabric.

Provides encoding schemes to convert continuous-valued inputs
(e.g., token embeddings, features) into spike-driving currents
or direct spike trains.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RateCodeEncoder(nn.Module):
    """Rate coding encoder.

    Maps continuous features to spike-driving current via linear projection.
    The output current determines the firing rate of input neurons.

    Args:
        d_model: Input feature dimension
        N_pop: Output population size
        bias: Whether to use bias in projection
    """

    def __init__(
        self,
        d_model: int,
        N_pop: int,
        bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.N_pop = N_pop

        self.proj = nn.Linear(d_model, N_pop, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode features to spike-driving current.

        Args:
            x: Input features [batch, seq, d_model] or [batch, d_model]

        Returns:
            Current [batch, N_pop] (pooled if sequence input)
        """
        if x.dim() == 3:
            # Pool over sequence
            x = x.mean(dim=1)

        return torch.relu(self.proj(x))


class TemporalRateEncoder(nn.Module):
    """Temporal rate coding encoder.

    Encodes sequence of features into time-varying currents,
    preserving temporal structure for SNN processing.

    Args:
        d_model: Input feature dimension
        N_pop: Output population size
        time_scale: How many SNN timesteps per input timestep
    """

    def __init__(
        self,
        d_model: int,
        N_pop: int,
        time_scale: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.N_pop = N_pop
        self.time_scale = time_scale

        self.proj = nn.Linear(d_model, N_pop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence to time-varying current.

        Args:
            x: Input sequence [batch, seq, d_model]

        Returns:
            Current [batch, seq * time_scale, N_pop]
        """
        batch, seq, _ = x.shape

        # Project
        currents = torch.relu(self.proj(x))  # [batch, seq, N_pop]

        # Expand in time
        currents = currents.unsqueeze(2).repeat(1, 1, self.time_scale, 1)
        currents = currents.view(batch, seq * self.time_scale, self.N_pop)

        return currents


class PoissonEncoder(nn.Module):
    """Poisson spike encoding.

    Generates Poisson-distributed spike trains where the rate
    is proportional to the input intensity.

    Args:
        d_model: Input feature dimension
        N_pop: Output population size
        max_rate: Maximum firing rate (spikes per timestep)
        time_steps: Number of timesteps to generate
    """

    def __init__(
        self,
        d_model: int,
        N_pop: int,
        max_rate: float = 0.5,
        time_steps: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.N_pop = N_pop
        self.max_rate = max_rate
        self.time_steps = time_steps

        self.proj = nn.Linear(d_model, N_pop)

    def forward(
        self,
        x: torch.Tensor,
        time_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate Poisson spike train.

        Args:
            x: Input features [batch, d_model] or [batch, seq, d_model]
            time_steps: Override default timesteps

        Returns:
            Spike train [batch, time_steps, N_pop]
        """
        time_steps = time_steps or self.time_steps

        if x.dim() == 3:
            x = x.mean(dim=1)

        # Compute rates (0 to max_rate)
        rates = torch.sigmoid(self.proj(x)) * self.max_rate  # [batch, N_pop]

        # Expand to time
        rates = rates.unsqueeze(1).expand(-1, time_steps, -1)  # [batch, time, N_pop]

        # Sample Poisson spikes
        spikes = (torch.rand_like(rates) < rates).float()

        return spikes


class LatencyEncoder(nn.Module):
    """Latency (time-to-first-spike) encoding.

    Encodes input intensity as spike timing: stronger inputs
    cause earlier spikes.

    Args:
        d_model: Input feature dimension
        N_pop: Output population size
        time_steps: Number of timesteps
        tau: Time constant for latency mapping
    """

    def __init__(
        self,
        d_model: int,
        N_pop: int,
        time_steps: int = 256,
        tau: float = 10.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.N_pop = N_pop
        self.time_steps = time_steps
        self.tau = tau

        self.proj = nn.Linear(d_model, N_pop)

    def forward(
        self,
        x: torch.Tensor,
        time_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate latency-coded spike train.

        Args:
            x: Input features [batch, d_model] or [batch, seq, d_model]
            time_steps: Override default timesteps

        Returns:
            Spike train [batch, time_steps, N_pop]
        """
        time_steps = time_steps or self.time_steps

        if x.dim() == 3:
            x = x.mean(dim=1)

        # Compute intensity (higher = earlier spike)
        intensity = torch.sigmoid(self.proj(x))  # [batch, N_pop]

        # Map intensity to spike time
        # intensity = 1 → spike at t=0
        # intensity = 0 → spike at t=time_steps (or never)
        spike_times = ((1 - intensity) * time_steps * self.tau / (self.tau + 1)).long()
        spike_times = torch.clamp(spike_times, 0, time_steps - 1)

        # Generate spike train
        batch, N_pop = intensity.shape
        spikes = torch.zeros(batch, time_steps, N_pop, device=x.device)

        # Place single spike at computed time
        batch_idx = torch.arange(batch, device=x.device).unsqueeze(1).expand(-1, N_pop)
        neuron_idx = torch.arange(N_pop, device=x.device).unsqueeze(0).expand(batch, -1)
        spikes[batch_idx, spike_times, neuron_idx] = 1.0

        return spikes


class DeltaEncoder(nn.Module):
    """Delta (temporal difference) encoding.

    Encodes changes in input over time, generating spikes
    for positive and negative changes.

    Args:
        d_model: Input feature dimension
        N_pop: Output population size (should be 2x for pos/neg)
        threshold: Minimum change to trigger spike
    """

    def __init__(
        self,
        d_model: int,
        N_pop: int,
        threshold: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.N_pop = N_pop
        self.threshold = threshold

        # Two projections: one for increases, one for decreases
        self.proj_pos = nn.Linear(d_model, N_pop // 2)
        self.proj_neg = nn.Linear(d_model, N_pop // 2)

        self.prev_value = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate delta-coded spikes.

        Args:
            x: Input features [batch, d_model]

        Returns:
            Spike current [batch, N_pop]
        """
        if self.prev_value is None:
            self.prev_value = x.detach().clone()
            return torch.zeros(x.shape[0], self.N_pop, device=x.device)

        # Compute delta
        delta = x - self.prev_value
        self.prev_value = x.detach().clone()

        # Positive changes
        pos_delta = torch.relu(delta)
        pos_spikes = (pos_delta > self.threshold).float()
        pos_current = self.proj_pos(pos_spikes * pos_delta)

        # Negative changes
        neg_delta = torch.relu(-delta)
        neg_spikes = (neg_delta > self.threshold).float()
        neg_current = self.proj_neg(neg_spikes * neg_delta)

        return torch.cat([pos_current, neg_current], dim=-1)

    def reset(self):
        """Reset encoder state."""
        self.prev_value = None


class LearnedEncoder(nn.Module):
    """Learned spike encoding with multiple layers.

    Uses a small MLP to learn optimal encoding from data.

    Args:
        d_model: Input feature dimension
        N_pop: Output population size
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
    """

    def __init__(
        self,
        d_model: int,
        N_pop: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.N_pop = N_pop

        layers = []
        in_dim = d_model
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, N_pop))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode features through learned MLP.

        Args:
            x: Input features [batch, d_model] or [batch, seq, d_model]

        Returns:
            Spike-driving current [batch, N_pop]
        """
        if x.dim() == 3:
            x = x.mean(dim=1)

        return torch.relu(self.encoder(x))


def create_encoder(
    encoder_type: str,
    d_model: int,
    N_pop: int,
    **kwargs,
) -> nn.Module:
    """Factory function to create encoder by type.

    Args:
        encoder_type: Type identifier
        d_model: Input dimension
        N_pop: Output population size
        **kwargs: Additional encoder arguments

    Returns:
        Encoder module
    """
    encoders = {
        "rate": RateCodeEncoder,
        "temporal_rate": TemporalRateEncoder,
        "poisson": PoissonEncoder,
        "latency": LatencyEncoder,
        "delta": DeltaEncoder,
        "learned": LearnedEncoder,
    }

    if encoder_type not in encoders:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Available: {list(encoders.keys())}")

    return encoders[encoder_type](d_model, N_pop, **kwargs)


__all__ = [
    "RateCodeEncoder",
    "TemporalRateEncoder",
    "PoissonEncoder",
    "LatencyEncoder",
    "DeltaEncoder",
    "LearnedEncoder",
    "create_encoder",
]
