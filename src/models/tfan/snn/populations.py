"""Neuron population implementations for SNN fabric.

Provides different neuron models including LIF (Leaky Integrate-and-Fire).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

from .types import NeuronParams, PopulationState, SpikeBatch


class SurrogateSpikeFunction(torch.autograd.Function):
    """Surrogate gradient for non-differentiable spike function.

    Uses fast sigmoid for backward pass while maintaining binary spikes forward.
    """

    scale: float = 25.0

    @staticmethod
    def forward(ctx, v: torch.Tensor, v_th: float) -> torch.Tensor:
        """Forward: binary spike when v >= v_th."""
        ctx.save_for_backward(v)
        ctx.v_th = v_th
        return (v >= v_th).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward: fast sigmoid surrogate gradient."""
        v, = ctx.saved_tensors
        v_th = ctx.v_th
        scale = SurrogateSpikeFunction.scale

        # Fast sigmoid surrogate: scale / (1 + scale * |v - v_th|)^2
        grad_v = grad_output * scale / (1.0 + scale * torch.abs(v - v_th)) ** 2
        return grad_v, None


def surrogate_spike(v: torch.Tensor, v_th: float) -> torch.Tensor:
    """Apply spike function with surrogate gradient."""
    return SurrogateSpikeFunction.apply(v, v_th)


class Population(nn.Module, ABC):
    """Base class for neuron populations.

    Args:
        N: Number of neurons
        params: Neuron parameters
        name: Population identifier
    """

    def __init__(
        self,
        N: int,
        params: Optional[NeuronParams] = None,
        name: str = "population",
    ):
        super().__init__()
        self.N = N
        self.params = params or NeuronParams()
        self.name = name

    @abstractmethod
    def init_state(self, batch: int, device: str = "cpu") -> PopulationState:
        """Initialize population state.

        Args:
            batch: Batch size
            device: Target device

        Returns:
            Initial population state
        """
        pass

    @abstractmethod
    def forward(
        self,
        state: PopulationState,
        input_current: torch.Tensor,
    ) -> Tuple[PopulationState, SpikeBatch]:
        """Advance population by one timestep.

        Args:
            state: Current population state
            input_current: Input current [batch, N]

        Returns:
            new_state: Updated population state
            spikes: Output spikes for this timestep
        """
        pass


class LIFPopulation(Population):
    """Leaky Integrate-and-Fire neuron population.

    Implements the LIF dynamics:
        v[t+1] = alpha * v[t] + (1 - alpha) * (v_rest + i_syn[t])
        i_syn[t+1] = beta * i_syn[t] + input[t]
        spike[t] = (v[t] >= v_th)
        v[t] = v_reset if spike[t] else v[t]

    Args:
        N: Number of neurons
        params: LIF neuron parameters
        name: Population identifier
    """

    def __init__(
        self,
        N: int,
        params: Optional[NeuronParams] = None,
        name: str = "lif_population",
    ):
        super().__init__(N, params, name)

        # Register parameters as buffers (not trainable, but saved with model)
        self.register_buffer("v_th", torch.tensor(self.params.v_th))
        self.register_buffer("v_reset", torch.tensor(self.params.v_reset))
        self.register_buffer("v_rest", torch.tensor(self.params.v_rest))
        self.register_buffer("alpha", torch.tensor(self.params.alpha))
        self.register_buffer("beta", torch.tensor(self.params.beta))

    def init_state(self, batch: int, device: str = "cpu") -> PopulationState:
        """Initialize LIF state with resting potential."""
        return PopulationState(data={
            "v": torch.full((batch, self.N), self.params.v_rest, device=device),
            "i_syn": torch.zeros(batch, self.N, device=device),
            "refractory": torch.zeros(batch, self.N, dtype=torch.int32, device=device),
        })

    def forward(
        self,
        state: PopulationState,
        input_current: torch.Tensor,
    ) -> Tuple[PopulationState, SpikeBatch]:
        """Advance LIF population by one timestep.

        Args:
            state: Current LIF state
            input_current: Input current [batch, N]

        Returns:
            new_state: Updated state
            spikes: Binary spike tensor [batch, N]
        """
        v = state.v
        i_syn = state.i_syn
        refractory = state.refractory

        # Update synaptic current: decay + new input
        i_syn_new = self.beta * i_syn + input_current

        # Update membrane potential (only for non-refractory neurons)
        is_refractory = refractory > 0
        v_new = torch.where(
            is_refractory,
            v,  # Keep voltage during refractory
            self.alpha * v + (1 - self.alpha) * (self.v_rest + i_syn_new)
        )

        # Generate spikes with surrogate gradient
        spikes = surrogate_spike(v_new, self.v_th.item())

        # Reset spiking neurons
        v_new = torch.where(spikes > 0, self.v_reset.expand_as(v_new), v_new)

        # Update refractory counters
        refractory_new = torch.where(
            spikes > 0,
            torch.full_like(refractory, self.params.refractory_steps),
            torch.clamp(refractory - 1, min=0)
        )

        new_state = PopulationState(data={
            "v": v_new,
            "i_syn": i_syn_new,
            "refractory": refractory_new,
        })

        return new_state, SpikeBatch(spikes=spikes)


class AdaptiveLIFPopulation(Population):
    """Adaptive LIF with spike-frequency adaptation.

    Adds an adaptation current that increases after each spike,
    making subsequent spikes harder to generate.

    Args:
        N: Number of neurons
        params: Neuron parameters
        name: Population identifier
        tau_adapt: Adaptation time constant
        adapt_strength: Adaptation increment per spike
    """

    def __init__(
        self,
        N: int,
        params: Optional[NeuronParams] = None,
        name: str = "alif_population",
        tau_adapt: float = 100.0,
        adapt_strength: float = 0.1,
    ):
        super().__init__(N, params, name)
        self.tau_adapt = tau_adapt
        self.adapt_strength = adapt_strength

        # Adaptation decay factor
        adapt_decay = 1.0 - (1.0 / tau_adapt) if tau_adapt > 0 else 0.0
        self.register_buffer("adapt_decay", torch.tensor(adapt_decay))
        self.register_buffer("v_th", torch.tensor(self.params.v_th))
        self.register_buffer("v_reset", torch.tensor(self.params.v_reset))
        self.register_buffer("v_rest", torch.tensor(self.params.v_rest))
        self.register_buffer("alpha", torch.tensor(self.params.alpha))
        self.register_buffer("beta", torch.tensor(self.params.beta))

    def init_state(self, batch: int, device: str = "cpu") -> PopulationState:
        """Initialize ALIF state."""
        return PopulationState(data={
            "v": torch.full((batch, self.N), self.params.v_rest, device=device),
            "i_syn": torch.zeros(batch, self.N, device=device),
            "adapt": torch.zeros(batch, self.N, device=device),
            "refractory": torch.zeros(batch, self.N, dtype=torch.int32, device=device),
        })

    def forward(
        self,
        state: PopulationState,
        input_current: torch.Tensor,
    ) -> Tuple[PopulationState, SpikeBatch]:
        """Advance ALIF population by one timestep."""
        v = state.v
        i_syn = state.i_syn
        adapt = state.data.get("adapt", torch.zeros_like(v))
        refractory = state.refractory

        # Update synaptic current
        i_syn_new = self.beta * i_syn + input_current

        # Effective threshold increases with adaptation
        v_th_eff = self.v_th + adapt

        # Update membrane potential
        is_refractory = refractory > 0
        v_new = torch.where(
            is_refractory,
            v,
            self.alpha * v + (1 - self.alpha) * (self.v_rest + i_syn_new)
        )

        # Generate spikes against adaptive threshold
        spikes = surrogate_spike(v_new, v_th_eff)

        # Reset and update adaptation
        v_new = torch.where(spikes > 0, self.v_reset.expand_as(v_new), v_new)
        adapt_new = self.adapt_decay * adapt + self.adapt_strength * spikes

        # Update refractory
        refractory_new = torch.where(
            spikes > 0,
            torch.full_like(refractory, self.params.refractory_steps),
            torch.clamp(refractory - 1, min=0)
        )

        new_state = PopulationState(data={
            "v": v_new,
            "i_syn": i_syn_new,
            "adapt": adapt_new,
            "refractory": refractory_new,
        })

        return new_state, SpikeBatch(spikes=spikes)


def create_population(
    N: int,
    neuron_type: str,
    params: Dict[str, Any],
    name: str = "population",
) -> Population:
    """Factory function to create population by type.

    Args:
        N: Number of neurons
        neuron_type: Type identifier ("lif", "alif", etc.)
        params: Neuron parameters dict
        name: Population name

    Returns:
        Population instance
    """
    neuron_params = NeuronParams.from_dict(params)

    if neuron_type == "lif":
        return LIFPopulation(N, neuron_params, name)
    elif neuron_type == "alif":
        return AdaptiveLIFPopulation(
            N, neuron_params, name,
            tau_adapt=params.get("tau_adapt", 100.0),
            adapt_strength=params.get("adapt_strength", 0.1),
        )
    else:
        raise ValueError(f"Unknown neuron type: {neuron_type}")


__all__ = [
    "Population",
    "LIFPopulation",
    "AdaptiveLIFPopulation",
    "create_population",
    "surrogate_spike",
    "SurrogateSpikeFunction",
]
