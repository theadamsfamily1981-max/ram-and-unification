"""RMSNorm implementations for TF-A-N 7B.

Provides:
- RMSNorm: Standard implementation
- RMSNormFused: Optionally compiled for better performance
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm normalizes by the root mean square of the input,
    without centering (no mean subtraction). This is more efficient
    than LayerNorm while achieving similar performance.

    Formula: y = x / RMS(x) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)

    Args:
        dim: Hidden dimension
        eps: Epsilon for numerical stability (default: 1e-6)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm.

        Args:
            x: Input tensor [..., dim]

        Returns:
            Normalized tensor [..., dim]
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RMSNormFused(nn.Module):
    """Fused RMSNorm implementation for better performance.

    Uses torch.compile on the inner norm function when available.
    Falls back to the standard implementation otherwise.

    Args:
        dim: Hidden dimension
        eps: Epsilon for numerical stability (default: 1e-6)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

        # Try to compile the norm function for better performance
        # If compile is unavailable / fails, we just use the plain version
        try:
            self._norm = torch.compile(self._norm_impl)
        except Exception:
            self._norm = self._norm_impl

    def _norm_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization (inner implementation)."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm with optional fusion.

        Args:
            x: Input tensor [..., dim]

        Returns:
            Normalized tensor [..., dim]
        """
        # Do the RMS in float32 for stability, then cast back
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


__all__ = ["RMSNorm", "RMSNormFused"]
