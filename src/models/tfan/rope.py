"""Rotary Positional Embeddings (RoPE) for TF-A-N 7B.

LLaMA-style implementation with optional scaling for long contexts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE).

    Applies 2D rotations to Q and K tensors based on position indices.
    Compatible with LLaMA-style transformers.

    Args:
        dim: Head dimension (must be even)
        max_position_embeddings: Maximum sequence length (default: 32768)
        base: Base for frequency calculation (theta) (default: 10000.0)
        scaling_factor: Linear scaling factor for long contexts (default: 1.0)
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor

        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache for cos/sin values
        self._seq_len_cached = 0
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cos/sin cache if sequence length changed."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len

            # Create position indices with scaling
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            t = t / self.scaling_factor

            # Compute frequencies: outer product of positions and inv_freq
            freqs = torch.outer(t, self.inv_freq.to(device))

            # Compute embeddings: [seq_len, dim/2] -> [seq_len, dim]
            emb = torch.cat([freqs, freqs], dim=-1)

            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin embeddings for given sequence.

        Args:
            x: Input tensor [batch, seq_len, ...] (for shape reference)
            position_ids: Optional position indices [batch, seq_len]

        Returns:
            cos, sin: Embeddings [batch, seq_len, dim] or [1, seq_len, dim]
        """
        seq_len = x.shape[1]

        # Update cache if needed
        self._update_cache(seq_len, x.device, x.dtype)

        if position_ids is None:
            # Standard case: sequential positions
            cos = self._cos_cached[:seq_len].unsqueeze(0)  # [1, seq_len, dim]
            sin = self._sin_cached[:seq_len].unsqueeze(0)
        else:
            # Custom positions (e.g., for generation with KV cache)
            cos = self._cos_cached[position_ids]  # [batch, seq_len, dim]
            sin = self._sin_cached[position_ids]

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input.

    Used for applying rotary embeddings:
    [x1, x2, x3, x4, ...] -> [-x2, x1, -x4, x3, ...]

    Args:
        x: Input tensor [..., dim]

    Returns:
        Rotated tensor [..., dim]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to Q and K tensors.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        cos: Cosine embeddings [batch, seq_len, head_dim] or [1, seq_len, head_dim]
        sin: Sine embeddings [batch, seq_len, head_dim] or [1, seq_len, head_dim]

    Returns:
        q_embed, k_embed: Rotated Q and K tensors
    """
    # Reshape cos/sin to match Q/K: [batch, 1, seq_len, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def apply_rotary_pos_emb_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to a single tensor (Q or K).

    Useful for inference/generation when processing Q and K separately.

    Args:
        x: Input tensor [batch, num_heads, seq_len, head_dim]
        cos: Cosine embeddings [batch, seq_len, head_dim] or [1, seq_len, head_dim]
        sin: Sine embeddings [batch, seq_len, head_dim] or [1, seq_len, head_dim]

    Returns:
        x_embed: Rotated tensor
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


__all__ = [
    "RotaryEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",
    "apply_rotary_pos_emb_single",
]
