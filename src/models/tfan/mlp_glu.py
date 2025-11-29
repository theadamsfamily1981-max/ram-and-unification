"""SwiGLU MLP for TF-A-N 7B.

Gated Linear Unit with SiLU (Swish) activation, following LLaMA architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLUMLP(nn.Module):
    """SwiGLU (Swish-Gated Linear Unit) MLP.

    Implements: output = down_proj(silu(gate_proj(x)) * up_proj(x))

    The gate and up projections are computed in parallel, then combined
    with element-wise multiplication after applying SiLU to the gate.

    Args:
        hidden_size: Model hidden dimension (default: 4096)
        intermediate_size: FFN intermediate dimension (default: 13312)
        hidden_act: Activation function (default: "silu")
        mlp_bias: Whether to use bias in linear layers (default: False)
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 13312,
        hidden_act: str = "silu",
        mlp_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Gate projection: produces gating values
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)

        # Up projection: produces values to be gated
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)

        # Down projection: reduces back to hidden size
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=mlp_bias)

        # Activation function
        self.act_fn = self._get_activation(hidden_act)

    @staticmethod
    def _get_activation(name: str):
        """Get activation function by name."""
        activations = {
            "silu": F.silu,
            "swish": F.silu,  # swish is same as silu
            "gelu": F.gelu,
            "relu": F.relu,
            "gelu_new": lambda x: F.gelu(x, approximate="tanh"),
            "gelu_fast": lambda x: F.gelu(x, approximate="tanh"),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
        return activations[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # SwiGLU: silu(gate) * up, then down
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class SwiGLUMLPFused(nn.Module):
    """Fused SwiGLU MLP with combined gate/up projection.

    More memory-efficient version that fuses gate and up projections
    into a single matrix multiplication.

    Args:
        hidden_size: Model hidden dimension
        intermediate_size: FFN intermediate dimension
        mlp_bias: Whether to use bias
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 13312,
        mlp_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Fused gate+up projection (2x intermediate size)
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=mlp_bias)

        # Down projection
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fused gate/up.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # Single matmul for both gate and up
        gate_up = self.gate_up_proj(x)

        # Split and apply SwiGLU
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up

        return self.down_proj(hidden)


class GatedMLP(nn.Module):
    """Generic Gated MLP supporting various gating mechanisms.

    Supports different gating functions including SwiGLU, GeGLU, ReGLU.

    Args:
        hidden_size: Model hidden dimension
        intermediate_size: FFN intermediate dimension
        gate_activation: Activation for gate ("silu", "gelu", "relu")
        mlp_bias: Whether to use bias
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 13312,
        gate_activation: str = "silu",
        mlp_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=mlp_bias)

        # Select gate activation
        if gate_activation == "silu" or gate_activation == "swish":
            self.gate_fn = F.silu
        elif gate_activation == "gelu":
            self.gate_fn = F.gelu
        elif gate_activation == "relu":
            self.gate_fn = F.relu
        else:
            raise ValueError(f"Unknown gate activation: {gate_activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        return self.down_proj(self.gate_fn(self.gate_proj(x)) * self.up_proj(x))


def compute_intermediate_size(hidden_size: int, ffn_mult: float = 3.25) -> int:
    """Compute intermediate size for SwiGLU MLP.

    For SwiGLU, the effective parameter count requires adjustment.
    The formula ensures the total FFN params match a standard 4x MLP.

    Args:
        hidden_size: Model hidden dimension
        ffn_mult: FFN expansion multiplier

    Returns:
        Intermediate dimension size
    """
    # Standard: 8/3 * hidden for SwiGLU to match 4x standard MLP params
    # Custom: use ffn_mult directly
    intermediate = int(hidden_size * ffn_mult)

    # Round to multiple of 256 for efficiency
    intermediate = ((intermediate + 255) // 256) * 256

    return intermediate


__all__ = [
    "SwiGLUMLP",
    "SwiGLUMLPFused",
    "GatedMLP",
    "compute_intermediate_size",
]
