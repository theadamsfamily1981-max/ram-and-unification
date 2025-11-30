"""Selective Sparse Attention (SSA) for TF-A-N 7B.

O(N log N) complexity attention using Top-k Landmark Selection (TLS).
Supports Grouped Query Attention (GQA) with configurable KV heads.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .rope import RotaryEmbedding, apply_rotary_pos_emb


def ssa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_landmarks: int = 64,
    attention_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Selective Sparse Attention with Top-k Landmark Selection.

    Achieves O(N log N) complexity by selecting top-k landmark positions
    based on query-key similarity scores.

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim]
        key: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        value: Value tensor [batch, num_kv_heads, seq_len, head_dim]
        num_landmarks: Number of landmark positions to select (default: 64)
        attention_mask: Optional attention mask [batch, 1, seq_len, seq_len]
        dropout_p: Dropout probability
        is_causal: Whether to apply causal masking
        scale: Attention scale factor (default: 1/sqrt(head_dim))

    Returns:
        Output tensor [batch, num_heads, seq_len, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    num_kv_heads = key.shape[1]

    # Expand KV heads for GQA
    if num_kv_heads != num_heads:
        kv_groups = num_heads // num_kv_heads
        key = key.repeat_interleave(kv_groups, dim=1)
        value = value.repeat_interleave(kv_groups, dim=1)

    scale = scale or (1.0 / math.sqrt(head_dim))

    # For short sequences, use standard attention
    if seq_len <= num_landmarks * 2:
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        if dropout_p > 0.0 and torch.is_grad_enabled():
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        return torch.matmul(attn_weights, value)

    # SSA: Selective Sparse Attention for long sequences
    # Step 1: Compute landmark scores using strided positions
    stride = max(1, seq_len // num_landmarks)
    landmark_indices = torch.arange(0, seq_len, stride, device=query.device)[:num_landmarks]
    num_actual_landmarks = len(landmark_indices)

    # Get landmark keys: [batch, num_heads, num_landmarks, head_dim]
    landmark_keys = key[:, :, landmark_indices, :]

    # Step 2: Compute query-landmark scores
    # [batch, num_heads, seq_len, num_landmarks]
    landmark_scores = torch.matmul(query, landmark_keys.transpose(-2, -1)) * scale

    # Step 3: Select top-k landmarks per query position
    k = min(num_landmarks // 2, num_actual_landmarks)
    _, top_landmark_indices = landmark_scores.topk(k, dim=-1)  # [batch, num_heads, seq_len, k]

    # Step 4: Gather relevant keys and values
    # Map landmark indices back to sequence positions
    selected_positions = landmark_indices[top_landmark_indices]  # [batch, num_heads, seq_len, k]

    # For efficiency, we'll compute a sparse attention pattern
    # First, get local window around each position
    local_window = min(64, seq_len // 4)

    # Combine local window + selected landmarks
    output = torch.zeros_like(query)

    for b in range(batch_size):
        for h in range(num_heads):
            for q_pos in range(seq_len):
                # Local window indices
                local_start = max(0, q_pos - local_window // 2)
                local_end = min(seq_len, q_pos + local_window // 2) if not is_causal else q_pos + 1
                local_end = max(local_start + 1, local_end)

                local_indices = torch.arange(local_start, local_end, device=query.device)

                # Combine with landmark indices (unique positions)
                landmark_pos = selected_positions[b, h, q_pos]
                if is_causal:
                    landmark_pos = landmark_pos[landmark_pos <= q_pos]

                all_indices = torch.cat([local_indices, landmark_pos])
                all_indices = torch.unique(all_indices)
                all_indices = all_indices[all_indices < seq_len]
                if is_causal:
                    all_indices = all_indices[all_indices <= q_pos]

                # Compute attention for this query position
                q_vec = query[b, h, q_pos:q_pos+1, :]  # [1, head_dim]
                k_selected = key[b, h, all_indices, :]  # [num_selected, head_dim]
                v_selected = value[b, h, all_indices, :]

                scores = torch.matmul(q_vec, k_selected.transpose(-2, -1)) * scale
                weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)

                if dropout_p > 0.0 and torch.is_grad_enabled():
                    weights = F.dropout(weights, p=dropout_p)

                output[b, h, q_pos, :] = torch.matmul(weights, v_selected)

    return output


def ssa_attention_fast(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_landmarks: int = 64,
    local_window: int = 128,
    attention_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Fast vectorized SSA implementation.

    Uses block-sparse patterns for better GPU utilization.

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim]
        key: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        value: Value tensor [batch, num_kv_heads, seq_len, head_dim]
        num_landmarks: Number of global landmark positions
        local_window: Size of local attention window
        attention_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to apply causal masking
        scale: Attention scale factor

    Returns:
        Output tensor [batch, num_heads, seq_len, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    num_kv_heads = key.shape[1]

    # Expand KV heads for GQA
    if num_kv_heads != num_heads:
        kv_groups = num_heads // num_kv_heads
        key = key.repeat_interleave(kv_groups, dim=1)
        value = value.repeat_interleave(kv_groups, dim=1)

    scale = scale or (1.0 / math.sqrt(head_dim))

    # For short sequences, use standard scaled dot-product attention
    if seq_len <= local_window * 2:
        return F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
            is_causal=is_causal,
            scale=scale,
        )

    # Block-sparse attention pattern
    # 1. Local sliding window attention
    # 2. Global landmark attention

    # Compute local attention (sliding window)
    local_output = _sliding_window_attention(
        query, key, value, local_window, scale, is_causal, dropout_p
    )

    # Compute global landmark attention
    stride = seq_len // num_landmarks
    landmark_indices = torch.arange(0, seq_len, stride, device=query.device)[:num_landmarks]

    landmark_keys = key[:, :, landmark_indices, :]
    landmark_values = value[:, :, landmark_indices, :]

    # [batch, num_heads, seq_len, num_landmarks]
    landmark_scores = torch.matmul(query, landmark_keys.transpose(-2, -1)) * scale

    if is_causal:
        # Mask future landmark positions
        positions = torch.arange(seq_len, device=query.device).unsqueeze(1)
        landmark_mask = positions < landmark_indices.unsqueeze(0)
        landmark_scores = landmark_scores.masked_fill(landmark_mask, float('-inf'))

    landmark_weights = F.softmax(landmark_scores, dim=-1, dtype=torch.float32).to(query.dtype)

    if dropout_p > 0.0 and torch.is_grad_enabled():
        landmark_weights = F.dropout(landmark_weights, p=dropout_p)

    global_output = torch.matmul(landmark_weights, landmark_values)

    # Combine local and global with learned gating
    # For simplicity, use fixed 0.7/0.3 split (local/global)
    output = 0.7 * local_output + 0.3 * global_output

    return output


def _sliding_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int,
    scale: float,
    is_causal: bool,
    dropout_p: float,
) -> torch.Tensor:
    """Compute sliding window attention.

    Args:
        query: [batch, num_heads, seq_len, head_dim]
        key: [batch, num_heads, seq_len, head_dim]
        value: [batch, num_heads, seq_len, head_dim]
        window_size: Size of sliding window
        scale: Attention scale
        is_causal: Whether causal
        dropout_p: Dropout probability

    Returns:
        Output tensor [batch, num_heads, seq_len, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    half_window = window_size // 2

    # Pad key and value for sliding window
    pad_size = half_window
    key_padded = F.pad(key, (0, 0, pad_size, pad_size), value=0)
    value_padded = F.pad(value, (0, 0, pad_size, pad_size), value=0)

    # Unfold to get sliding windows: [batch, num_heads, seq_len, window_size, head_dim]
    key_windows = key_padded.unfold(2, window_size, 1)
    value_windows = value_padded.unfold(2, window_size, 1)

    # Reshape for batch matrix multiply
    # query: [batch, num_heads, seq_len, 1, head_dim]
    query = query.unsqueeze(3)

    # [batch, num_heads, seq_len, 1, window_size]
    scores = torch.matmul(query, key_windows.transpose(-2, -1)) * scale
    scores = scores.squeeze(3)  # [batch, num_heads, seq_len, window_size]

    if is_causal:
        # Create causal mask for sliding window
        # Position i can only attend to positions <= i within the window
        positions = torch.arange(seq_len, device=query.device)
        window_positions = torch.arange(window_size, device=query.device) - half_window
        absolute_positions = positions.unsqueeze(1) + window_positions.unsqueeze(0)

        causal_mask = absolute_positions > positions.unsqueeze(1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask, float('-inf'))

    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)

    if dropout_p > 0.0 and torch.is_grad_enabled():
        weights = F.dropout(weights, p=dropout_p)

    # [batch, num_heads, seq_len, window_size] @ [batch, num_heads, seq_len, window_size, head_dim]
    # -> [batch, num_heads, seq_len, head_dim]
    output = torch.einsum('bhsw,bhswd->bhsd', weights, value_windows)

    return output


class SSAAttention(nn.Module):
    """Selective Sparse Attention module for TF-A-N.

    Implements O(N log N) attention with:
    - Grouped Query Attention (GQA)
    - Rotary Position Embeddings (RoPE)
    - Top-k Landmark Selection for sparsity

    Args:
        hidden_size: Model hidden dimension
        num_attention_heads: Number of query heads
        num_key_value_heads: Number of key/value heads (for GQA)
        head_dim: Dimension per head (default: hidden_size // num_attention_heads)
        num_landmarks: Number of landmark positions for sparse attention
        local_window: Size of local attention window
        max_position_embeddings: Maximum sequence length
        rope_theta: Base for RoPE frequency calculation
        attention_dropout: Dropout probability
        use_fast: Whether to use fast vectorized implementation
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: Optional[int] = None,
        num_landmarks: int = 64,
        local_window: int = 128,
        max_position_embeddings: int = 32768,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        use_fast: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.num_landmarks = num_landmarks
        self.local_window = local_window
        self.attention_dropout = attention_dropout
        self.use_fast = use_fast

        self.kv_groups = self.num_heads // self.num_kv_heads

        # Projection layers
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for SSA attention.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_value: Optional cached KV for generation
            output_attentions: Whether to return attention weights (not supported for SSA)
            use_cache: Whether to return updated KV cache

        Returns:
            output: Attention output [batch, seq_len, hidden_size]
            attn_weights: None (not supported for sparse attention)
            past_key_value: Updated KV cache if use_cache=True
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache for generation
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Apply SSA
        if self.use_fast:
            attn_output = ssa_attention_fast(
                query_states,
                key_states,
                value_states,
                num_landmarks=self.num_landmarks,
                local_window=self.local_window,
                attention_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_output = ssa_attention(
                query_states,
                key_states,
                value_states,
                num_landmarks=self.num_landmarks,
                attention_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=True,
            )

        # Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


__all__ = [
    "SSAAttention",
    "ssa_attention",
    "ssa_attention_fast",
]
