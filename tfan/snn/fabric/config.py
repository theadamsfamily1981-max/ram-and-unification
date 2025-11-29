# tfan/snn/fabric/config.py
"""
Configuration schema and loaders for SNN fabric.

Allows defining fabric structure via YAML/JSON config files rather than
imperative Python code. This enables:
    - Easy hyperparameter sweeps
    - CI/benchmarking with multiple configs
    - Reproducible experiments
    - Hardware mapping via config transforms
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Literal
import yaml
import json
from pathlib import Path

import torch
from .types import ProjectionParams
from .populations import Population, LIFPopulation, InputPopulation, ReadoutPopulation
from .projections import Projection, LowRankProjection, DenseProjection
from .fabric import SNNFabric


@dataclass
class PopulationConfig:
    """
    Configuration for a single neuron population.

    Attributes:
        name: Unique identifier
        N: Number of neurons
        kind: Type of population ("lif", "input", "readout")
        v_th: Spike threshold (LIF only)
        alpha: Leak factor (LIF only)
        beta: Smoothing factor (readout only)
        surrogate_scale: Gradient scale for surrogate (LIF only)
        reset_mode: "zero" or "subtract" (LIF only)
    """
    name: str
    N: int
    kind: Literal["lif", "input", "readout"] = "lif"

    # LIF parameters
    v_th: float = 1.0
    alpha: float = 0.95
    surrogate_scale: float = 10.0
    reset_mode: str = "zero"

    # Readout parameters
    beta: float = 0.9

    def to_population(self, dtype: torch.dtype = torch.float32) -> Population:
        """Build a Population object from this config."""
        if self.kind == "lif":
            return LIFPopulation(
                name=self.name,
                N=self.N,
                v_th=self.v_th,
                alpha=self.alpha,
                surrogate_scale=self.surrogate_scale,
                reset_mode=self.reset_mode,
            )
        elif self.kind == "input":
            return InputPopulation(name=self.name, N=self.N)
        elif self.kind == "readout":
            return ReadoutPopulation(name=self.name, N=self.N, beta=self.beta)
        else:
            raise ValueError(f"Unknown population kind: {self.kind}")


@dataclass
class ProjectionConfig:
    """
    Configuration for a synaptic projection.

    Attributes:
        name: Unique identifier
        pre: Presynaptic population name
        post: Postsynaptic population name
        k: Sparse connections per neuron
        r: Low-rank dimension
        delay: Synaptic delay in timesteps (future extension)
        plasticity: Whether weights are learnable
        kind: "lowrank" or "dense"
    """
    name: str
    pre: str
    post: str
    k: int
    r: int
    delay: int = 0
    plasticity: bool = True
    kind: Literal["lowrank", "dense"] = "lowrank"

    def to_projection_params(self, populations: Dict[str, PopulationConfig]) -> ProjectionParams:
        """Build ProjectionParams from this config."""
        pre_pop = populations[self.pre]
        post_pop = populations[self.post]

        return ProjectionParams(
            N_pre=pre_pop.N,
            N_post=post_pop.N,
            k=self.k,
            r=self.r,
            delay=self.delay,
            plasticity=self.plasticity,
        )

    def to_projection(
        self,
        populations: Dict[str, PopulationConfig],
        dtype: torch.dtype = torch.float32,
    ) -> Projection:
        """Build a Projection object from this config."""
        params = self.to_projection_params(populations)

        if self.kind == "lowrank":
            return LowRankProjection(
                name=self.name,
                pre=self.pre,
                post=self.post,
                params=params,
                dtype=dtype,
            )
        elif self.kind == "dense":
            return DenseProjection(
                name=self.name,
                pre=self.pre,
                post=self.post,
                params=params,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unknown projection kind: {self.kind}")


@dataclass
class FabricConfig:
    """
    Top-level fabric configuration.

    Attributes:
        populations: List of population configs
        projections: List of projection configs
        dtype: Data type for all tensors ("float32", "float16", "bfloat16")
        validate: Whether to validate graph structure
    """
    populations: List[PopulationConfig]
    projections: List[ProjectionConfig]
    dtype: str = "float32"
    validate: bool = True

    def to_fabric(self) -> SNNFabric:
        """Build SNNFabric from this config."""
        # Parse dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(self.dtype, torch.float32)

        # Build population lookup
        pop_configs = {pc.name: pc for pc in self.populations}

        # Build populations
        pops = {
            name: pc.to_population(dtype=dtype)
            for name, pc in pop_configs.items()
        }

        # Build projections
        projs = [
            pc.to_projection(pop_configs, dtype=dtype)
            for pc in self.projections
        ]

        # Build fabric
        return SNNFabric(
            populations=pops,
            projections=projs,
            validate=self.validate,
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "FabricConfig":
        """Load config from a dictionary."""
        # Parse populations
        pops = [PopulationConfig(**p) for p in config_dict["populations"]]

        # Parse projections
        projs = [ProjectionConfig(**p) for p in config_dict["projections"]]

        # Get optional top-level params
        dtype = config_dict.get("dtype", "float32")
        validate = config_dict.get("validate", True)

        return cls(populations=pops, projections=projs, dtype=dtype, validate=validate)

    @classmethod
    def from_yaml(cls, path: str) -> "FabricConfig":
        """Load config from a YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, path: str) -> "FabricConfig":
        """Load config from a JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    def to_yaml(self, path: str):
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: str):
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def build_fabric_from_config(config_path: str) -> SNNFabric:
    """
    Convenience function: build fabric directly from config file.

    Args:
        config_path: Path to YAML or JSON config file

    Returns:
        Instantiated SNNFabric

    Example:
        >>> fabric = build_fabric_from_config("configs/snn/fabric_toy.yaml")
        >>> fabric.summary()
    """
    path = Path(config_path)

    if path.suffix in [".yaml", ".yml"]:
        config = FabricConfig.from_yaml(str(path))
    elif path.suffix == ".json":
        config = FabricConfig.from_json(str(path))
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    return config.to_fabric()


# Example: Programmatic config builder for common patterns
def build_feedforward_config(
    N_input: int,
    N_hidden: int,
    N_output: int,
    k: int = 64,
    r: int = 32,
) -> FabricConfig:
    """
    Build a simple feedforward fabric config: input → hidden → output.

    Args:
        N_input: Number of input neurons
        N_hidden: Number of hidden neurons
        N_output: Number of output neurons
        k: Sparse connections per neuron
        r: Low-rank dimension

    Returns:
        FabricConfig for a 3-layer feedforward network

    Example:
        >>> config = build_feedforward_config(4096, 4096, 4096)
        >>> fabric = config.to_fabric()
    """
    populations = [
        PopulationConfig(name="input", N=N_input, kind="input"),
        PopulationConfig(name="hidden", N=N_hidden, kind="lif"),
        PopulationConfig(name="output", N=N_output, kind="readout"),
    ]

    projections = [
        ProjectionConfig(
            name="input_to_hidden",
            pre="input",
            post="hidden",
            k=k,
            r=r,
        ),
        ProjectionConfig(
            name="hidden_to_output",
            pre="hidden",
            post="output",
            k=k,
            r=r,
        ),
    ]

    return FabricConfig(populations=populations, projections=projections)
