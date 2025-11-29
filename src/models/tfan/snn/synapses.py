"""Synapse implementations for SNN fabric.

Provides sparse synapse models with CI-compliant constraints:
- Low-rank decomposition (W = M ⊙ UV^T)
- TLS (Top-k Landmark Selection) sparsity masks
- CSR format for efficient sparse operations
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np

from .types import SynapseParams


def generate_tls_mask_csr(
    N_pre: int,
    N_post: int,
    k: int,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Generate TLS sparse mask in CSR format.

    Creates a sparse connectivity pattern where each postsynaptic neuron
    receives connections from exactly k presynaptic neurons, selected
    as evenly-spaced landmarks.

    Args:
        N_pre: Number of presynaptic neurons
        N_post: Number of postsynaptic neurons
        k: Number of connections per postsynaptic neuron
        device: Target device

    Returns:
        Dictionary with 'indptr' and 'indices' tensors in CSR format
    """
    k = min(k, N_pre)

    # Generate landmark indices for each row (postsynaptic neuron)
    # Evenly spaced with small random jitter
    indices_list = []
    indptr = [0]

    for row in range(N_post):
        # Evenly spaced base positions
        base = np.linspace(0, N_pre - 1, k, dtype=np.float32)
        # Add small jitter (within ±stride/4)
        stride = N_pre / k if k > 1 else N_pre
        jitter = np.random.uniform(-stride / 4, stride / 4, k)
        positions = np.clip(base + jitter, 0, N_pre - 1).astype(np.int32)
        # Ensure unique
        positions = np.unique(positions)
        # Pad if needed
        while len(positions) < k:
            extra = np.random.randint(0, N_pre)
            if extra not in positions:
                positions = np.append(positions, extra)
        positions = np.sort(positions[:k])

        indices_list.append(positions)
        indptr.append(indptr[-1] + len(positions))

    indices = np.concatenate(indices_list)
    indptr = np.array(indptr, dtype=np.int32)

    return {
        "indptr": torch.tensor(indptr, dtype=torch.int32, device=device),
        "indices": torch.tensor(indices, dtype=torch.int32, device=device),
    }


class LowRankMaskedSynapse(nn.Module):
    """Low-rank synapse with sparse TLS mask.

    Implements W_eff = M ⊙ (U @ V^T) where:
    - M is a fixed sparse mask (CSR format)
    - U is [N_post, r] low-rank factor
    - V is [N_pre, r] low-rank factor

    The effective weight is computed only at mask positions for efficiency.

    Args:
        params: Synapse parameters (N_pre, N_post, k, r, etc.)
        mask_csr: Optional pre-computed mask; generated if None
        device: Target device
        dtype: Data type for weights
    """

    def __init__(
        self,
        params: SynapseParams,
        mask_csr: Optional[Dict[str, torch.Tensor]] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.params = params
        self.N_pre = params.N_pre
        self.N_post = params.N_post
        self.k = params.k
        self.r = params.r

        # Generate or use provided mask
        if mask_csr is None:
            mask_csr = generate_tls_mask_csr(
                self.N_pre, self.N_post, self.k, device=device
            )

        # Register mask as buffers (not trainable)
        self.register_buffer("indptr", mask_csr["indptr"])
        self.register_buffer("indices", mask_csr["indices"])

        # Compute number of nonzeros
        self.nnz = int(self.indices.shape[0])

        # Initialize low-rank factors
        # Using scaled initialization for stability
        scale = params.init_scale / math.sqrt(self.r)
        self.U = nn.Parameter(
            torch.randn(self.N_post, self.r, device=device, dtype=dtype) * scale
        )
        self.V = nn.Parameter(
            torch.randn(self.N_pre, self.r, device=device, dtype=dtype) * scale
        )

        if not params.trainable:
            self.U.requires_grad = False
            self.V.requires_grad = False

    def effective_weight(self) -> torch.Tensor:
        """Compute full effective weight matrix W = M ⊙ (U @ V^T).

        Note: This materializes the full dense matrix. Use forward()
        for efficient sparse computation.

        Returns:
            Weight matrix [N_post, N_pre]
        """
        # Full low-rank product
        W_lr = self.U @ self.V.T  # [N_post, N_pre]

        # Apply mask by zeroing non-mask positions
        mask_dense = torch.zeros_like(W_lr)
        for row in range(self.N_post):
            start = self.indptr[row].item()
            end = self.indptr[row + 1].item()
            cols = self.indices[start:end]
            mask_dense[row, cols] = 1.0

        return W_lr * mask_dense

    def effective_weight_sparse(self) -> torch.Tensor:
        """Compute effective weights only at mask positions.

        Returns:
            Sparse weight values [nnz]
        """
        values = torch.zeros(self.nnz, device=self.U.device, dtype=self.U.dtype)

        for row in range(self.N_post):
            start = self.indptr[row].item()
            end = self.indptr[row + 1].item()
            cols = self.indices[start:end].long()

            # W[row, cols] = U[row, :] @ V[cols, :].T
            u_row = self.U[row]  # [r]
            v_cols = self.V[cols]  # [k, r]
            values[start:end] = v_cols @ u_row  # [k]

        return values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply synapse: y = W_eff @ x using sparse computation.

        Args:
            x: Presynaptic activity [batch, N_pre]

        Returns:
            Postsynaptic current [batch, N_post]
        """
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        # Get sparse weights
        values = self.effective_weight_sparse()

        # CSR sparse matrix-vector product
        output = torch.zeros(batch_size, self.N_post, device=device, dtype=dtype)

        for row in range(self.N_post):
            start = self.indptr[row].item()
            end = self.indptr[row + 1].item()
            if start == end:
                continue

            cols = self.indices[start:end].long()
            weights = values[start:end]

            # Gather and weighted sum: output[:, row] = sum(x[:, cols] * weights)
            output[:, row] = (x[:, cols] * weights.unsqueeze(0)).sum(dim=1)

        return output

    def forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        """Dense forward pass (for debugging/comparison).

        Args:
            x: Presynaptic activity [batch, N_pre]

        Returns:
            Postsynaptic current [batch, N_post]
        """
        W = self.effective_weight()
        return x @ W.T

    @property
    def mask_csr(self) -> Dict[str, torch.Tensor]:
        """Get mask in CSR format."""
        return {"indptr": self.indptr, "indices": self.indices}

    def sparsity(self) -> float:
        """Compute actual sparsity ratio."""
        total = self.N_pre * self.N_post
        return 1.0 - (self.nnz / total)

    def ci_audit(self) -> Dict[str, Any]:
        """Audit CI compliance.

        Returns:
            Dictionary with CI metrics and compliance status
        """
        # Check rank constraint
        rank_ok = self.r <= 32  # CI max rank

        # Check sparsity constraint (k per row)
        max_k = 0
        for row in range(self.N_post):
            start = self.indptr[row].item()
            end = self.indptr[row + 1].item()
            row_k = end - start
            max_k = max(max_k, row_k)
        sparsity_ok = max_k <= 64  # CI max nnz per row

        return {
            "rank": self.r,
            "max_nnz_per_row": max_k,
            "rank_compliant": rank_ok,
            "sparsity_compliant": sparsity_ok,
            "ci_compliant": rank_ok and sparsity_ok,
            "actual_sparsity": self.sparsity(),
        }


class DenseSynapse(nn.Module):
    """Dense synapse for comparison/debugging.

    Simple fully-connected synapse without sparsity constraints.

    Args:
        N_pre: Number of presynaptic neurons
        N_post: Number of postsynaptic neurons
        init_scale: Initialization scale
        trainable: Whether weights are trainable
    """

    def __init__(
        self,
        N_pre: int,
        N_post: int,
        init_scale: float = 0.02,
        trainable: bool = True,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.N_pre = N_pre
        self.N_post = N_post

        self.weight = nn.Parameter(
            torch.randn(N_post, N_pre, device=device, dtype=dtype) * init_scale
        )
        if not trainable:
            self.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dense synapse."""
        return F.linear(x, self.weight)


def create_synapse(
    synapse_type: str,
    params: SynapseParams,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Factory function to create synapse by type.

    Args:
        synapse_type: Type identifier
        params: Synapse parameters
        device: Target device
        dtype: Data type

    Returns:
        Synapse module
    """
    if synapse_type == "lowrank_masked":
        return LowRankMaskedSynapse(params, device=device, dtype=dtype)
    elif synapse_type == "dense":
        return DenseSynapse(
            params.N_pre, params.N_post,
            init_scale=params.init_scale,
            trainable=params.trainable,
            device=device, dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown synapse type: {synapse_type}")


__all__ = [
    "LowRankMaskedSynapse",
    "DenseSynapse",
    "generate_tls_mask_csr",
    "create_synapse",
]
