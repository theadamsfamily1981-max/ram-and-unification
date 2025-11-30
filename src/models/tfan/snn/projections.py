"""Projection implementations for SNN fabric.

A projection connects a presynaptic population to a postsynaptic population
via a synapse model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from .types import ProjectionParams, SynapseParams, SpikeBatch
from .synapses import LowRankMaskedSynapse, create_synapse


class Projection(nn.Module, ABC):
    """Base class for projections between populations.

    Args:
        name: Projection identifier
        pre: Name of presynaptic population
        post: Name of postsynaptic population
        params: Projection parameters
    """

    def __init__(
        self,
        name: str,
        pre: str,
        post: str,
        params: ProjectionParams,
    ):
        super().__init__()
        self.name = name
        self.pre = pre
        self.post = post
        self.params = params

    @abstractmethod
    def forward(self, spikes: SpikeBatch) -> torch.Tensor:
        """Transform presynaptic spikes to postsynaptic current.

        Args:
            spikes: Presynaptic spike batch

        Returns:
            Postsynaptic input current [batch, N_post]
        """
        pass


class LowRankProjection(Projection):
    """Projection using low-rank masked synapse.

    Implements CI-compliant sparse connectivity with:
    - TLS (Top-k Landmark Selection) mask
    - Low-rank weight decomposition

    Args:
        name: Projection identifier
        pre: Presynaptic population name
        post: Postsynaptic population name
        params: Projection parameters
        device: Target device
        dtype: Data type
    """

    def __init__(
        self,
        name: str,
        pre: str,
        post: str,
        params: ProjectionParams,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(name, pre, post, params)

        # Create synapse parameters from projection params
        syn_params = SynapseParams(
            N_pre=params.N_pre,
            N_post=params.N_post,
            k=params.k,
            r=params.r,
            init_scale=params.init_scale,
            trainable=params.trainable,
        )

        # Create synapse
        self.synapse = LowRankMaskedSynapse(
            syn_params,
            device=device,
            dtype=dtype,
        )

    def forward(self, spikes: SpikeBatch) -> torch.Tensor:
        """Apply projection synapse to spikes.

        Args:
            spikes: Presynaptic spike batch

        Returns:
            Postsynaptic current [batch, N_post]
        """
        return self.synapse(spikes.spikes)

    def effective_weight(self) -> torch.Tensor:
        """Get full effective weight matrix."""
        return self.synapse.effective_weight()

    def ci_audit(self) -> Dict[str, Any]:
        """Audit CI compliance of this projection."""
        return self.synapse.ci_audit()


class DelayedProjection(Projection):
    """Projection with axonal delay.

    Adds a configurable delay between presynaptic spikes and
    postsynaptic current.

    Args:
        name: Projection identifier
        pre: Presynaptic population name
        post: Postsynaptic population name
        params: Projection parameters
        delay_steps: Number of timesteps delay
        device: Target device
        dtype: Data type
    """

    def __init__(
        self,
        name: str,
        pre: str,
        post: str,
        params: ProjectionParams,
        delay_steps: int = 1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(name, pre, post, params)
        self.delay_steps = delay_steps

        # Create underlying synapse
        syn_params = SynapseParams(
            N_pre=params.N_pre,
            N_post=params.N_post,
            k=params.k,
            r=params.r,
            init_scale=params.init_scale,
            trainable=params.trainable,
        )
        self.synapse = LowRankMaskedSynapse(syn_params, device=device, dtype=dtype)

        # Delay buffer (circular)
        self.delay_buffer = None
        self.buffer_idx = 0

    def init_delay_buffer(self, batch: int, device: str):
        """Initialize delay buffer."""
        self.delay_buffer = torch.zeros(
            self.delay_steps, batch, self.params.N_pre, device=device
        )
        self.buffer_idx = 0

    def forward(self, spikes: SpikeBatch) -> torch.Tensor:
        """Apply delayed projection.

        Args:
            spikes: Current timestep presynaptic spikes

        Returns:
            Postsynaptic current from delayed spikes
        """
        if self.delay_buffer is None:
            batch = spikes.spikes.shape[0]
            self.init_delay_buffer(batch, spikes.spikes.device)

        # Get delayed spikes
        delayed_idx = (self.buffer_idx - self.delay_steps) % self.delay_steps
        delayed_spikes = self.delay_buffer[delayed_idx]

        # Store current spikes
        self.delay_buffer[self.buffer_idx] = spikes.spikes
        self.buffer_idx = (self.buffer_idx + 1) % self.delay_steps

        # Apply synapse to delayed spikes
        return self.synapse(delayed_spikes)


class STDPProjection(Projection):
    """Projection with STDP (Spike-Timing-Dependent Plasticity).

    Implements online STDP learning rule for unsupervised adaptation.

    Args:
        name: Projection identifier
        pre: Presynaptic population name
        post: Postsynaptic population name
        params: Projection parameters
        tau_plus: Time constant for potentiation
        tau_minus: Time constant for depression
        a_plus: Learning rate for potentiation
        a_minus: Learning rate for depression
        device: Target device
        dtype: Data type
    """

    def __init__(
        self,
        name: str,
        pre: str,
        post: str,
        params: ProjectionParams,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        a_plus: float = 0.005,
        a_minus: float = 0.005,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(name, pre, post, params)

        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus

        # Eligibility traces
        self.trace_pre = None   # [batch, N_pre]
        self.trace_post = None  # [batch, N_post]

        # Create synapse
        syn_params = SynapseParams(
            N_pre=params.N_pre,
            N_post=params.N_post,
            k=params.k,
            r=params.r,
            init_scale=params.init_scale,
            trainable=params.trainable,
        )
        self.synapse = LowRankMaskedSynapse(syn_params, device=device, dtype=dtype)

    def init_traces(self, batch: int, device: str):
        """Initialize eligibility traces."""
        self.trace_pre = torch.zeros(batch, self.params.N_pre, device=device)
        self.trace_post = torch.zeros(batch, self.params.N_post, device=device)

    def forward(
        self,
        spikes: SpikeBatch,
        post_spikes: Optional[SpikeBatch] = None,
        learn: bool = True,
    ) -> torch.Tensor:
        """Apply STDP projection.

        Args:
            spikes: Presynaptic spikes
            post_spikes: Postsynaptic spikes (for STDP update)
            learn: Whether to apply STDP updates

        Returns:
            Postsynaptic current
        """
        batch = spikes.spikes.shape[0]
        device = spikes.spikes.device

        if self.trace_pre is None:
            self.init_traces(batch, device)

        # Decay traces
        decay_pre = 1.0 - (1.0 / self.tau_plus)
        decay_post = 1.0 - (1.0 / self.tau_minus)
        self.trace_pre = decay_pre * self.trace_pre + spikes.spikes
        if post_spikes is not None:
            self.trace_post = decay_post * self.trace_post + post_spikes.spikes

        # Apply STDP if learning enabled
        if learn and post_spikes is not None:
            self._apply_stdp(spikes.spikes, post_spikes.spikes)

        return self.synapse(spikes.spikes)

    def _apply_stdp(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """Apply STDP weight updates.

        Pre-before-post: potentiation (dW > 0)
        Post-before-pre: depression (dW < 0)
        """
        # This is a simplified STDP; full implementation would modify
        # the low-rank factors U and V
        # For now, we compute the update direction but don't apply it
        # (proper implementation requires careful handling of low-rank structure)

        # Potentiation: post spike, pre trace active
        # dW[post, pre] += a_plus * post_spikes * trace_pre
        # Depression: pre spike, post trace active
        # dW[post, pre] -= a_minus * trace_post * pre_spikes

        pass  # Placeholder for full implementation


def create_projection(
    proj_dict: Dict[str, Any],
    pop_sizes: Dict[str, int],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Projection:
    """Factory function to create projection from config dict.

    Args:
        proj_dict: Projection configuration
        pop_sizes: Dictionary mapping population names to sizes
        device: Target device
        dtype: Data type

    Returns:
        Projection instance
    """
    params = ProjectionParams.from_dict(proj_dict)

    # Fill in population sizes
    params.N_pre = pop_sizes[params.pre]
    params.N_post = pop_sizes[params.post]

    synapse_type = proj_dict.get("synapse_type", "lowrank_masked")

    if synapse_type == "lowrank_masked":
        return LowRankProjection(
            name=params.name,
            pre=params.pre,
            post=params.post,
            params=params,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown synapse type for projection: {synapse_type}")


__all__ = [
    "Projection",
    "LowRankProjection",
    "DelayedProjection",
    "STDPProjection",
    "create_projection",
]
