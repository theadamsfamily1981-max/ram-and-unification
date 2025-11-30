"""Type definitions for SNN fabric.

Defines data structures for populations, synapses, and spike events.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class NeuronParams:
    """Parameters for a neuron model.

    Args:
        tau_mem: Membrane time constant (ms)
        tau_syn: Synaptic time constant (ms)
        v_th: Spike threshold voltage
        v_reset: Reset voltage after spike
        v_rest: Resting membrane potential
        alpha: Membrane decay factor (exp(-dt/tau_mem))
        beta: Synaptic decay factor (exp(-dt/tau_syn))
        refractory_steps: Refractory period in timesteps
    """
    tau_mem: float = 20.0
    tau_syn: float = 5.0
    v_th: float = 1.0
    v_reset: float = 0.0
    v_rest: float = 0.0
    alpha: float = 0.95
    beta: float = 0.85
    refractory_steps: int = 2

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NeuronParams":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tau_mem": self.tau_mem,
            "tau_syn": self.tau_syn,
            "v_th": self.v_th,
            "v_reset": self.v_reset,
            "v_rest": self.v_rest,
            "alpha": self.alpha,
            "beta": self.beta,
            "refractory_steps": self.refractory_steps,
        }


@dataclass
class SynapseParams:
    """Parameters for a synapse model.

    Args:
        N_pre: Number of presynaptic neurons
        N_post: Number of postsynaptic neurons
        k: Number of nonzeros per row (TLS sparsity)
        r: Low-rank factor
        init_scale: Weight initialization scale
        trainable: Whether weights are trainable
    """
    N_pre: int
    N_post: int
    k: int = 64
    r: int = 32
    init_scale: float = 0.02
    trainable: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SynapseParams":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "N_pre": self.N_pre,
            "N_post": self.N_post,
            "k": self.k,
            "r": self.r,
            "init_scale": self.init_scale,
            "trainable": self.trainable,
        }


@dataclass
class ProjectionParams:
    """Parameters for a projection between populations.

    Args:
        name: Projection identifier
        pre: Name of presynaptic population
        post: Name of postsynaptic population
        synapse_type: Type of synapse ("lowrank_masked", "dense", etc.)
        N_pre: Number of presynaptic neurons
        N_post: Number of postsynaptic neurons
        k: Nonzeros per row
        r: Low-rank factor
        init_scale: Initialization scale
        trainable: Whether projection is trainable
    """
    name: str
    pre: str
    post: str
    synapse_type: str = "lowrank_masked"
    N_pre: int = 0
    N_post: int = 0
    k: int = 64
    r: int = 32
    init_scale: float = 0.02
    trainable: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProjectionParams":
        """Create from dictionary, extracting nested params."""
        params = d.get("params", {})
        return cls(
            name=d["name"],
            pre=d["pre"],
            post=d["post"],
            synapse_type=d.get("synapse_type", "lowrank_masked"),
            N_pre=d.get("N_pre", 0),
            N_post=d.get("N_post", 0),
            k=params.get("k", 64),
            r=params.get("r", 32),
            init_scale=params.get("init_scale", 0.02),
            trainable=params.get("trainable", True),
        )


@dataclass
class PopulationState:
    """State of a neuron population at a given timestep.

    Contains membrane potential, synaptic current, refractory counters, etc.

    Args:
        data: Dictionary mapping state variable names to tensors
    """
    data: Dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def v(self) -> Optional[torch.Tensor]:
        """Membrane potential."""
        return self.data.get("v")

    @v.setter
    def v(self, value: torch.Tensor):
        self.data["v"] = value

    @property
    def i_syn(self) -> Optional[torch.Tensor]:
        """Synaptic current."""
        return self.data.get("i_syn")

    @i_syn.setter
    def i_syn(self, value: torch.Tensor):
        self.data["i_syn"] = value

    @property
    def refractory(self) -> Optional[torch.Tensor]:
        """Refractory counter."""
        return self.data.get("refractory")

    @refractory.setter
    def refractory(self, value: torch.Tensor):
        self.data["refractory"] = value

    def clone(self) -> "PopulationState":
        """Create a deep copy of the state."""
        return PopulationState(data={k: v.clone() for k, v in self.data.items()})

    def detach(self) -> "PopulationState":
        """Detach all tensors from computation graph."""
        return PopulationState(data={k: v.detach() for k, v in self.data.items()})


@dataclass
class SpikeBatch:
    """Batch of spike events from a population.

    Args:
        spikes: Binary spike tensor [batch, N] or spike counts
        times: Optional spike times within the timestep
    """
    spikes: torch.Tensor
    times: Optional[torch.Tensor] = None

    @property
    def count(self) -> int:
        """Total number of spikes in batch."""
        return int((self.spikes > 0).sum().item())

    @property
    def rate(self) -> float:
        """Average firing rate (spikes per neuron)."""
        return self.spikes.float().mean().item()

    def to(self, device: torch.device) -> "SpikeBatch":
        """Move to device."""
        return SpikeBatch(
            spikes=self.spikes.to(device),
            times=self.times.to(device) if self.times is not None else None,
        )


@dataclass
class FabricConfig:
    """Configuration for an SNN fabric.

    Args:
        name: Fabric identifier
        description: Human-readable description
        time_steps: Default number of simulation timesteps
        dt: Timestep duration (ms)
        populations: Dict of population configs
        projections: List of projection configs
        ci_constraints: CI constraint settings
        training: Training-related settings
    """
    name: str = "snn_fabric"
    description: str = ""
    time_steps: int = 256
    dt: float = 1.0
    populations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    projections: list = field(default_factory=list)
    ci_constraints: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FabricConfig":
        """Create from dictionary (e.g., loaded from YAML)."""
        fabric = d.get("fabric", {})
        return cls(
            name=fabric.get("name", "snn_fabric"),
            description=fabric.get("description", ""),
            time_steps=fabric.get("time_steps", 256),
            dt=fabric.get("dt", 1.0),
            populations=d.get("populations", {}),
            projections=d.get("projections", []),
            ci_constraints=d.get("ci_constraints", {}),
            training=d.get("training", {}),
        )


__all__ = [
    "NeuronParams",
    "SynapseParams",
    "ProjectionParams",
    "PopulationState",
    "SpikeBatch",
    "FabricConfig",
]
