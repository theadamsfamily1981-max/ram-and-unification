"""SNN Emulation Backend for TF-A-N.

Provides CPU/GPU-based SNN simulation using the SNNFabric infrastructure.
Supports both single-layer models and full fabric configurations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from ..snn import (
    SNNFabric,
    SNNFabricModel,
    load_fabric_config,
    build_fabric_from_config,
)
from ..snn.types import SynapseParams
from ..snn.populations import LIFPopulation, NeuronParams
from ..snn.synapses import LowRankMaskedSynapse


class Backend:
    """Base class for compute backends."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def build_model(self) -> nn.Module:
        raise NotImplementedError


class SingleLayerSNNModel(nn.Module):
    """Simple single-layer SNN model.

    Provides a minimal SNN with one LIF population and one synapse,
    for testing and comparison purposes.

    Args:
        N: Number of neurons
        r: Low-rank factor
        k: Nonzeros per row
        time_steps: Simulation timesteps
        device: Target device
        dtype: Data type
    """

    def __init__(
        self,
        N: int = 4096,
        r: int = 32,
        k: int = 64,
        time_steps: int = 256,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.N = N
        self.time_steps = time_steps

        # Create single population
        self.population = LIFPopulation(N, name="main")

        # Create recurrent synapse
        syn_params = SynapseParams(N_pre=N, N_post=N, k=k, r=r)
        self.synapse = LowRankMaskedSynapse(syn_params, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass.

        Args:
            x: Input [batch, N] or [batch, time, N]

        Returns:
            output: Spike accumulator [batch, N]
            aux: Metrics dictionary
        """
        batch = x.shape[0]
        device = x.device

        # Initialize state
        state = self.population.init_state(batch, str(device))

        # Make time axis explicit
        if x.dim() == 2:
            x = x.unsqueeze(1).repeat(1, self.time_steps, 1)

        spike_acc = torch.zeros(batch, self.N, device=device)
        total_spikes = 0.0

        for t in range(self.time_steps):
            # External input + recurrent
            current = x[:, t, :]
            if t > 0:
                current = current + self.synapse(spike_acc / (t + 1))

            # Update population
            state, spikes = self.population(state, current)
            spike_acc += spikes.spikes
            total_spikes += spikes.spikes.sum().item()

        spike_rate = total_spikes / (batch * self.N * self.time_steps)

        aux = {
            "spike_rate": spike_rate,
            "spike_sparsity": 1.0 - spike_rate,
            "active_events": int(total_spikes),
            "time_steps": self.time_steps,
        }

        return spike_acc, aux


class SNNBackend(Backend):
    """SNN emulation backend.

    Builds either a simple single-layer model or a full fabric-based model
    depending on configuration.

    Configuration options:
        use_fabric: bool - Whether to use full fabric (default: False)
        fabric_config: str - Path to fabric YAML config
        N: int - Number of neurons (for single-layer)
        r: int - Low-rank factor
        k: int - Nonzeros per row
        time_steps: int - Simulation timesteps
        device: str - Target device
        dtype: str - Data type ("float32", "float16", "bfloat16")
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)

    def build_model(self) -> nn.Module:
        """Build SNN model based on configuration.

        Returns:
            SNN model (SingleLayerSNNModel or SNNFabricModel)
        """
        snn_cfg = self.cfg.get("snn", {})
        device = self.cfg.get("device", "cpu")
        dtype_str = self.cfg.get("dtype", "float32")

        # Parse dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(dtype_str, torch.float32)

        use_fabric = snn_cfg.get("use_fabric", False)

        if use_fabric:
            model = self._build_fabric_model(snn_cfg, device, dtype)
        else:
            model = self._build_single_layer_model(snn_cfg, device, dtype)

        # Print summary
        if isinstance(model, SNNFabricModel):
            print("\n=== SNN FABRIC MODEL ===")
            print(model.fabric.summary())
            print("========================\n")
        else:
            print(f"\n=== SINGLE-LAYER SNN ===")
            print(f"  N={model.N}, time_steps={model.time_steps}")
            print(f"  synapse: r={model.synapse.r}, k={model.synapse.k}")
            print("========================\n")

        return model

    def _build_fabric_model(
        self,
        snn_cfg: Dict[str, Any],
        device: str,
        dtype: torch.dtype,
    ) -> SNNFabricModel:
        """Build full fabric-based model.

        Args:
            snn_cfg: SNN configuration
            device: Target device
            dtype: Data type

        Returns:
            SNNFabricModel instance
        """
        fabric_cfg_path = snn_cfg.get("fabric_config")
        if fabric_cfg_path is None:
            raise ValueError("fabric_config path required when use_fabric=True")

        fabric_cfg = load_fabric_config(fabric_cfg_path)
        fabric = build_fabric_from_config(fabric_cfg, device=device, dtype=dtype)

        time_steps = snn_cfg.get("time_steps", fabric_cfg.time_steps)

        model = SNNFabricModel(
            fabric=fabric,
            time_steps=time_steps,
            input_pop=snn_cfg.get("input_pop", "input"),
            output_pop=snn_cfg.get("output_pop", "output"),
        )

        return model

    def _build_single_layer_model(
        self,
        snn_cfg: Dict[str, Any],
        device: str,
        dtype: torch.dtype,
    ) -> SingleLayerSNNModel:
        """Build simple single-layer model.

        Args:
            snn_cfg: SNN configuration
            device: Target device
            dtype: Data type

        Returns:
            SingleLayerSNNModel instance
        """
        return SingleLayerSNNModel(
            N=snn_cfg.get("N", 4096),
            r=snn_cfg.get("lowrank_rank", 32),
            k=snn_cfg.get("k_per_row", 64),
            time_steps=snn_cfg.get("time_steps", 256),
            device=device,
            dtype=dtype,
        )


class SNNHooks:
    """Training hooks for SNN monitoring and regularization.

    Provides callbacks for:
    - Spike rate monitoring
    - Sparsity tracking
    - Rate regularization
    """

    def __init__(
        self,
        target_rate: float = 0.1,
        lambda_rate: float = 0.01,
    ):
        self.target_rate = target_rate
        self.lambda_rate = lambda_rate

        # Metrics accumulator
        self.metrics = {
            "spike_rates": [],
            "sparsities": [],
            "active_events": [],
        }

    def after_step(
        self,
        model: nn.Module,
        output: torch.Tensor,
        aux: Dict[str, Any],
        loss: torch.Tensor,
    ) -> torch.Tensor:
        """Hook called after each training step.

        Args:
            model: SNN model
            output: Model output
            aux: Auxiliary metrics from forward pass
            loss: Current loss value

        Returns:
            Modified loss with rate regularization
        """
        # Record metrics
        self.metrics["spike_rates"].append(aux.get("spike_rate", 0.0))
        self.metrics["sparsities"].append(aux.get("spike_sparsity", 1.0))
        self.metrics["active_events"].append(aux.get("active_events", 0))

        # Add rate regularization
        spike_rate = aux.get("spike_rate", 0.0)
        rate_penalty = self.lambda_rate * (spike_rate - self.target_rate) ** 2

        return loss + rate_penalty

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics.

        Returns:
            Dictionary with mean metrics
        """
        return {
            "mean_spike_rate": sum(self.metrics["spike_rates"]) / max(len(self.metrics["spike_rates"]), 1),
            "mean_sparsity": sum(self.metrics["sparsities"]) / max(len(self.metrics["sparsities"]), 1),
            "total_events": sum(self.metrics["active_events"]),
        }

    def reset(self):
        """Reset accumulated metrics."""
        self.metrics = {
            "spike_rates": [],
            "sparsities": [],
            "active_events": [],
        }


__all__ = [
    "Backend",
    "SNNBackend",
    "SingleLayerSNNModel",
    "SNNHooks",
]
