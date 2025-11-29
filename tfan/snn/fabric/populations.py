# tfan/snn/fabric/populations.py
"""
Neuron population abstractions for SNN fabric.

Populations wrap neuron models (LIF, ALIF, etc.) and provide a common interface
for state initialization and stepping. This allows the fabric to treat all
populations uniformly while maintaining hardware-friendly boundaries.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .types import SpikeBatch, PopulationState, LIF_STATE_KEYS


class Population(nn.Module):
    """
    Base class for a neuron population in the fabric.

    A population represents a group of neurons of the same type (e.g., all LIF).
    It provides:
        - init_state(): Creates initial state tensors
        - step(): Updates state given input current, emits spikes

    Hardware mapping:
        - Each population becomes a memory block (state) + update kernel (step)
        - State layout should be amenable to DMA/burst access on FPGA
    """

    def __init__(self, name: str, N: int):
        """
        Args:
            name: Unique identifier for this population
            N: Number of neurons in this population
        """
        super().__init__()
        self.name = name
        self.N = N

    def init_state(self, batch: int, device: str) -> PopulationState:
        """
        Initialize population state for a given batch size.

        Args:
            batch: Batch size
            device: Device string (e.g., "cuda:0", "cpu")

        Returns:
            PopulationState with initialized tensors
        """
        raise NotImplementedError(f"{self.__class__.__name__}.init_state not implemented")

    def step(
        self,
        state: PopulationState,
        input_current: torch.Tensor,
    ) -> Tuple[PopulationState, SpikeBatch]:
        """
        Step the population forward one timestep.

        Args:
            state: Current state (from previous timestep or init_state)
            input_current: External + synaptic input [batch, N]

        Returns:
            new_state: Updated state
            spikes: Emitted spikes for this timestep

        Design notes:
            - input_current is already the sum of all inputs (external + synaptic)
            - This function should be O(N) and amenable to parallelization
            - For FPGA: this becomes a single-cycle update kernel
        """
        raise NotImplementedError(f"{self.__class__.__name__}.step not implemented")


class LIFPopulation(Population):
    """
    Leaky Integrate-and-Fire (LIF) neuron population.

    Dynamics:
        v[t+1] = alpha * v[t] + input_current[t]
        s[t+1] = 1 if v[t+1] >= v_th else 0
        v[t+1] = 0 if s[t+1] == 1 (reset after spike)

    Where:
        v: membrane voltage
        s: binary spike output
        alpha: leak factor (0 < alpha < 1, typically ~0.95)
        v_th: spike threshold

    Hardware mapping:
        - State (v, s) fits in BRAM/URAM on FPGA
        - Update is simple fixed-point arithmetic, no multiplies for basic LIF
        - Surrogate gradient for backprop (not needed on FPGA)
    """

    def __init__(
        self,
        name: str,
        N: int,
        v_th: float = 1.0,
        alpha: float = 0.95,
        surrogate_scale: float = 10.0,
        reset_mode: str = "zero",
    ):
        """
        Args:
            name: Population identifier
            N: Number of neurons
            v_th: Spike threshold
            alpha: Leak factor (membrane time constant)
            surrogate_scale: Scale for surrogate gradient (training only)
            reset_mode: "zero" (hard reset) or "subtract" (soft reset by v_th)
        """
        super().__init__(name=name, N=N)
        self.v_th = v_th
        self.alpha = alpha
        self.surrogate_scale = surrogate_scale
        self.reset_mode = reset_mode

        # Register as buffers (non-trainable constants)
        self.register_buffer("_v_th", torch.tensor(v_th))
        self.register_buffer("_alpha", torch.tensor(alpha))

    def init_state(self, batch: int, device: str) -> PopulationState:
        """Initialize voltage and spikes to zero."""
        v = torch.zeros(batch, self.N, device=device)
        s = torch.zeros(batch, self.N, device=device)
        return PopulationState(data={"v": v, "s": s})

    def step(
        self,
        state: PopulationState,
        input_current: torch.Tensor,
    ) -> Tuple[PopulationState, SpikeBatch]:
        """
        Single LIF timestep with surrogate gradient for training.

        Args:
            state: Current voltage and spikes
            input_current: [batch, N] external + synaptic input

        Returns:
            new_state: Updated voltage and spikes
            spikes: Binary spike tensor [batch, N]
        """
        v = state.data["v"]
        batch, N = v.shape
        assert input_current.shape == (batch, N), \
            f"Input current shape {input_current.shape} != state shape {v.shape}"

        # 1. Leaky integration
        v_new = self.alpha * v + input_current

        # 2. Spike generation with surrogate gradient
        # Forward: hard threshold
        # Backward: soft sigmoid gradient (allows learning)
        s_new = self._spike_function(v_new)

        # 3. Reset
        if self.reset_mode == "zero":
            # Hard reset: v = 0 after spike
            v_new = v_new * (1.0 - s_new)
        elif self.reset_mode == "subtract":
            # Soft reset: v -= v_th after spike
            v_new = v_new - s_new * self.v_th
        else:
            raise ValueError(f"Unknown reset_mode: {self.reset_mode}")

        new_state = PopulationState(data={"v": v_new, "s": s_new})
        spikes = SpikeBatch(spikes=s_new)

        return new_state, spikes

    def _spike_function(self, v: torch.Tensor) -> torch.Tensor:
        """
        Spike function with surrogate gradient.

        Forward: s = 1 if v >= v_th else 0
        Backward: ds/dv = surrogate_scale * sigmoid'(surrogate_scale * (v - v_th))

        This allows gradients to flow through the hard threshold during training.
        """
        return SurrogateHeaviside.apply(v, self.v_th, self.surrogate_scale)


class SurrogateHeaviside(torch.autograd.Function):
    """
    Heaviside step function with sigmoid surrogate gradient.

    This is the standard trick for training SNNs: hard threshold forward,
    soft gradient backward. See Zenke & Ganguli (2018), Neftci et al. (2019).
    """

    @staticmethod
    def forward(ctx, v, v_th, scale):
        """Forward: hard threshold."""
        ctx.save_for_backward(v, v_th, torch.tensor(scale))
        return (v >= v_th).float()

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: scaled sigmoid derivative."""
        v, v_th, scale = ctx.saved_tensors
        # Sigmoid derivative: σ'(x) = σ(x)(1 - σ(x))
        # We use: ds/dv = scale * σ'(scale * (v - v_th))
        x = scale * (v - v_th)
        sigmoid = torch.sigmoid(x)
        grad_surrogate = scale * sigmoid * (1.0 - sigmoid)
        return grad_output * grad_surrogate, None, None


class InputPopulation(Population):
    """
    Input population that just passes through external inputs (no dynamics).

    Used for:
        - Sensor inputs (e.g., vision, audio)
        - Encoded spike trains from RateEncoder/LatencyEncoder
        - Direct injection of spike patterns

    No internal state, just emits whatever is fed as input_current.
    """

    def __init__(self, name: str, N: int):
        super().__init__(name=name, N=N)

    def init_state(self, batch: int, device: str) -> PopulationState:
        """Input population has no internal state."""
        x = torch.zeros(batch, self.N, device=device)
        return PopulationState(data={"x": x})

    def step(
        self,
        state: PopulationState,
        input_current: torch.Tensor,
    ) -> Tuple[PopulationState, SpikeBatch]:
        """
        Just emit the input as spikes (no dynamics).

        Typically input_current is already binary (from encoders).
        """
        batch, N = input_current.shape
        assert N == self.N, f"Input dimension {N} != population size {self.N}"

        # Clamp to [0, 1] to ensure binary-like
        spikes = torch.clamp(input_current, 0.0, 1.0)

        new_state = PopulationState(data={"x": spikes})
        return new_state, SpikeBatch(spikes=spikes)


class ReadoutPopulation(Population):
    """
    Readout population for converting spikes to continuous outputs.

    Uses a simple linear integration (no spiking):
        y[t+1] = beta * y[t] + input_current[t]

    Where:
        y: continuous output
        beta: temporal smoothing factor (< 1.0)

    This is commonly used for:
        - Classification logits (sum spikes over time)
        - Regression outputs
        - Acoustic features (in Ara-SYNERGY)
    """

    def __init__(
        self,
        name: str,
        N: int,
        beta: float = 0.9,
    ):
        """
        Args:
            name: Population identifier
            N: Number of output units
            beta: Temporal smoothing (0 = no memory, 1 = full integration)
        """
        super().__init__(name=name, N=N)
        self.beta = beta
        self.register_buffer("_beta", torch.tensor(beta))

    def init_state(self, batch: int, device: str) -> PopulationState:
        """Initialize output to zero."""
        y = torch.zeros(batch, self.N, device=device)
        return PopulationState(data={"y": y})

    def step(
        self,
        state: PopulationState,
        input_current: torch.Tensor,
    ) -> Tuple[PopulationState, SpikeBatch]:
        """
        Accumulate input with temporal smoothing.

        Returns spikes=y (continuous, not binary) for compatibility with fabric.
        """
        y = state.data["y"]
        y_new = self.beta * y + input_current

        new_state = PopulationState(data={"y": y_new})
        # For readout, "spikes" is actually continuous output
        spikes = SpikeBatch(spikes=y_new, meta={"readout": True})

        return new_state, spikes
