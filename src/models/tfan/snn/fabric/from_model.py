"""Build SNN fabric from trained TF-A-N model checkpoint.

Provides utilities to:
- Quantize model weights to fixed-point formats
- Convert dense weight matrices to CSR sparse format
- Build SNNFabric instances from model.snn_head
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .fabric import SNNFabric


# =============================================================================
# Fixed-Point Quantization
# =============================================================================

def quantize_q5_10(x: torch.Tensor) -> torch.Tensor:
    """Quantize to signed Q5.10 fixed point (int16).

    Range: [-32.0, 31.999] with resolution ~0.001
    Used for: membrane potential (v), threshold (v_th)

    Args:
        x: Float tensor to quantize

    Returns:
        Quantized int16 tensor
    """
    scale = 1 << 10  # 1024
    q = torch.round(x * scale)
    q = torch.clamp(q, -32768, 32767)
    return q.to(torch.int16)


def quantize_q1_14(x: torch.Tensor) -> torch.Tensor:
    """Quantize to unsigned Q1.14 fixed point (uint16).

    Range: [0.0, 1.999] with resolution ~0.00006
    Used for: alpha (leak factor), parameters

    Args:
        x: Float tensor to quantize (assumed in [0, 2))

    Returns:
        Quantized uint16 tensor (stored as int32 for compatibility)
    """
    scale = 1 << 14  # 16384
    q = torch.round(x * scale)
    q = torch.clamp(q, 0, 65535)
    return q.to(torch.int32)


def quantize_q1_6(x: torch.Tensor) -> torch.Tensor:
    """Quantize to signed Q1.6 fixed point (int8).

    Range: [-2.0, 1.984] with resolution ~0.016
    Used for: synaptic weights (8-bit mode)

    Args:
        x: Float tensor to quantize

    Returns:
        Quantized int8 tensor
    """
    scale = 1 << 6  # 64
    q = torch.round(x * scale)
    q = torch.clamp(q, -128, 127)
    return q.to(torch.int8)


def quantize_q1_14_weights(x: torch.Tensor) -> torch.Tensor:
    """Quantize to signed Q1.14 fixed point (int16).

    Range: [-2.0, 1.99994] with resolution ~0.00006
    Used for: synaptic weights (16-bit mode)

    Args:
        x: Float tensor to quantize

    Returns:
        Quantized int16 tensor
    """
    scale = 1 << 14  # 16384
    q = torch.round(x * scale)
    q = torch.clamp(q, -32768, 32767)
    return q.to(torch.int16)


# =============================================================================
# CSR Conversion
# =============================================================================

def dense_to_csr(
    weight_matrix: torch.Tensor,
    threshold: float = 1e-6,
    quantize_8bit: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert dense [N_pre, N_post] weight matrix to CSR representation.

    CSR format stores the matrix by rows:
    - row_ptr[i] gives the starting index in col_idx/values for row i
    - col_idx[j] gives the column index of the j-th nonzero
    - values[j] gives the quantized weight of the j-th nonzero

    Args:
        weight_matrix: Dense weight tensor [N_pre, N_post]
        threshold: Minimum absolute value to consider nonzero
        quantize_8bit: Use 8-bit (Q1.6) or 16-bit (Q1.14) quantization

    Returns:
        row_ptr: [N_pre + 1] array of row pointers (int32)
        col_idx: [nnz] array of column indices (int32)
        w_vals: [nnz] array of quantized weights (int8 or int16)
    """
    w = weight_matrix.detach().cpu()
    N_pre, N_post = w.shape

    row_ptr = [0]
    col_idx = []
    w_vals_float = []

    for i in range(N_pre):
        row = w[i]
        for j in range(N_post):
            val = row[j].item()
            if abs(val) > threshold:
                col_idx.append(j)
                w_vals_float.append(val)
        row_ptr.append(len(col_idx))

    row_ptr = np.asarray(row_ptr, dtype=np.int32)
    col_idx = np.asarray(col_idx, dtype=np.int32)

    # Quantize weights
    w_tensor = torch.tensor(w_vals_float, dtype=torch.float32)
    if quantize_8bit:
        w_q = quantize_q1_6(w_tensor).numpy()
    else:
        w_q = quantize_q1_14_weights(w_tensor).numpy()

    return row_ptr, col_idx, w_q


def sparse_to_csr(
    indices: torch.Tensor,
    values: torch.Tensor,
    N_pre: int,
    N_post: int,
    quantize_8bit: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert sparse COO representation to CSR.

    Args:
        indices: [2, nnz] tensor of (row, col) indices
        values: [nnz] tensor of weight values
        N_pre: Number of presynaptic neurons (rows)
        N_post: Number of postsynaptic neurons (cols)
        quantize_8bit: Use 8-bit or 16-bit quantization

    Returns:
        row_ptr, col_idx, w_vals in CSR format
    """
    indices_np = indices.detach().cpu().numpy()
    values_np = values.detach().cpu().numpy()

    # Sort by row index
    sort_idx = np.lexsort((indices_np[1], indices_np[0]))
    row_indices = indices_np[0, sort_idx]
    col_indices = indices_np[1, sort_idx]
    sorted_values = values_np[sort_idx]

    # Build CSR
    row_ptr = np.zeros(N_pre + 1, dtype=np.int32)
    for r in row_indices:
        row_ptr[r + 1] += 1
    row_ptr = np.cumsum(row_ptr)

    col_idx = col_indices.astype(np.int32)

    # Quantize
    w_tensor = torch.tensor(sorted_values, dtype=torch.float32)
    if quantize_8bit:
        w_q = quantize_q1_6(w_tensor).numpy()
    else:
        w_q = quantize_q1_14_weights(w_tensor).numpy()

    return row_ptr, col_idx, w_q


# =============================================================================
# Kitten Fabric Data Structure
# =============================================================================

class KittenFabricData:
    """Container for Kitten FPGA fabric data.

    Holds quantized neuron states and CSR projections ready for
    export to binary format.

    Attributes:
        name: Fabric identifier
        num_neurons: Total neuron count
        v_init_fp: Initial membrane potentials (Q5.10)
        v_th_fp: Thresholds (Q5.10)
        alpha_fp: Leak factors (Q1.14)
        projections: List of projection dicts with CSR data
    """

    def __init__(
        self,
        name: str,
        num_neurons: int,
        v_init_fp: np.ndarray,
        v_th_fp: np.ndarray,
        alpha_fp: np.ndarray,
    ):
        self.name = name
        self.num_neurons = num_neurons
        self.v_init_fp = v_init_fp
        self.v_th_fp = v_th_fp
        self.alpha_fp = alpha_fp
        self.projections = []

    def add_projection(
        self,
        name: str,
        pre_start: int,
        pre_end: int,
        post_start: int,
        post_end: int,
        row_ptr: np.ndarray,
        col_idx: np.ndarray,
        weights_fp: np.ndarray,
    ):
        """Add a CSR projection.

        Args:
            name: Projection identifier
            pre_start: Start index of presynaptic population
            pre_end: End index of presynaptic population
            post_start: Start index of postsynaptic population
            post_end: End index of postsynaptic population
            row_ptr: CSR row pointers [N_pre + 1]
            col_idx: CSR column indices [nnz]
            weights_fp: Quantized weights [nnz]
        """
        self.projections.append({
            "name": name,
            "pre_start": pre_start,
            "pre_end": pre_end,
            "post_start": post_start,
            "post_end": post_end,
            "row_ptr": row_ptr,
            "col_idx": col_idx,
            "weights_fp": weights_fp,
            "nnz": len(col_idx),
        })

    @property
    def total_synapses(self) -> int:
        """Total number of synapses across all projections."""
        return sum(p["nnz"] for p in self.projections)


# =============================================================================
# Build Fabric from TF-A-N Model
# =============================================================================

def build_kitten_fabric_from_model(
    model,
    num_neurons: int,
    alpha: float = 0.9,
    v_th: float = 0.5,
    quantize_8bit: bool = True,
    snn_head_attr: str = "snn_head",
) -> KittenFabricData:
    """Build Kitten fabric data from a TF-A-N model with SNN head.

    Assumes the model has an SNN head module with:
    - neuron_thresholds: [N] float tensor
    - neuron_v_init: [N] float tensor (optional)
    - weight_matrix: [N, N] float tensor (dense) or sparse

    Args:
        model: TF-A-N model with SNN head
        num_neurons: Number of neurons to export (can truncate)
        alpha: Default leak factor if not in model
        v_th: Default threshold if not in model
        quantize_8bit: Use 8-bit weight quantization
        snn_head_attr: Attribute name for SNN head module

    Returns:
        KittenFabricData ready for export
    """
    # Get SNN head
    if not hasattr(model, snn_head_attr):
        raise AttributeError(f"Model has no '{snn_head_attr}' attribute")

    snn_head = getattr(model, snn_head_attr)

    # Extract thresholds
    if hasattr(snn_head, "neuron_thresholds"):
        thr = snn_head.neuron_thresholds.detach().cpu()[:num_neurons]
    else:
        thr = torch.full((num_neurons,), v_th)

    # Extract initial membrane potentials
    if hasattr(snn_head, "neuron_v_init"):
        v_init = snn_head.neuron_v_init.detach().cpu()[:num_neurons]
    else:
        v_init = torch.zeros(num_neurons)

    # Extract alpha (leak factor)
    if hasattr(snn_head, "alpha"):
        alpha_val = snn_head.alpha.detach().cpu().item()
    else:
        alpha_val = alpha

    # Quantize neuron parameters
    v_init_fp = quantize_q5_10(v_init).numpy()
    v_th_fp = quantize_q5_10(thr).numpy()
    alpha_fp = quantize_q1_14(torch.tensor([alpha_val])).numpy()
    alpha_fp = np.full(num_neurons, alpha_fp[0], dtype=np.int32)

    # Create fabric data container
    fabric = KittenFabricData(
        name="tfan_snn_fabric",
        num_neurons=num_neurons,
        v_init_fp=v_init_fp,
        v_th_fp=v_th_fp,
        alpha_fp=alpha_fp,
    )

    # Extract weight matrix
    if hasattr(snn_head, "weight_matrix"):
        W = snn_head.weight_matrix.detach().cpu()

        # Handle shape
        if W.dim() == 2:
            W = W[:num_neurons, :num_neurons]

            # Convert to CSR
            row_ptr, col_idx, w_fp = dense_to_csr(W, quantize_8bit=quantize_8bit)

            fabric.add_projection(
                name="main",
                pre_start=0,
                pre_end=num_neurons,
                post_start=0,
                post_end=num_neurons,
                row_ptr=row_ptr,
                col_idx=col_idx,
                weights_fp=w_fp,
            )

    elif hasattr(snn_head, "projections"):
        # Model already has sparse projections
        for proj in snn_head.projections:
            if hasattr(proj, "indptr") and hasattr(proj, "indices"):
                # Already CSR
                row_ptr = proj.indptr.detach().cpu().numpy().astype(np.int32)
                col_idx = proj.indices.detach().cpu().numpy().astype(np.int32)
                values = proj.values.detach().cpu()

                if quantize_8bit:
                    w_fp = quantize_q1_6(values).numpy()
                else:
                    w_fp = quantize_q1_14_weights(values).numpy()

                fabric.add_projection(
                    name=getattr(proj, "name", "proj"),
                    pre_start=0,
                    pre_end=num_neurons,
                    post_start=0,
                    post_end=num_neurons,
                    row_ptr=row_ptr,
                    col_idx=col_idx,
                    weights_fp=w_fp,
                )

    return fabric


def build_kitten_fabric_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    num_neurons: int,
    alpha: float = 0.9,
    v_th: float = 0.5,
    quantize_8bit: bool = True,
    prefix: str = "snn_head.",
) -> KittenFabricData:
    """Build Kitten fabric from a state dict (no model instance needed).

    This is useful when you have a checkpoint but don't want to instantiate
    the full model.

    Args:
        state_dict: Model state dictionary
        num_neurons: Number of neurons to export
        alpha: Default leak factor
        v_th: Default threshold
        quantize_8bit: Use 8-bit weight quantization
        prefix: Key prefix for SNN head tensors

    Returns:
        KittenFabricData ready for export
    """
    # Find SNN head tensors
    def get_tensor(name: str, default: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        key = f"{prefix}{name}"
        if key in state_dict:
            return state_dict[key]
        return default

    # Extract thresholds
    thr = get_tensor("neuron_thresholds")
    if thr is not None:
        thr = thr[:num_neurons]
    else:
        thr = torch.full((num_neurons,), v_th)

    # Extract initial membrane
    v_init = get_tensor("neuron_v_init")
    if v_init is not None:
        v_init = v_init[:num_neurons]
    else:
        v_init = torch.zeros(num_neurons)

    # Quantize
    v_init_fp = quantize_q5_10(v_init).numpy()
    v_th_fp = quantize_q5_10(thr).numpy()
    alpha_fp = quantize_q1_14(torch.tensor([alpha])).numpy()
    alpha_fp = np.full(num_neurons, alpha_fp[0], dtype=np.int32)

    fabric = KittenFabricData(
        name="tfan_snn_fabric",
        num_neurons=num_neurons,
        v_init_fp=v_init_fp,
        v_th_fp=v_th_fp,
        alpha_fp=alpha_fp,
    )

    # Find weight matrix
    W = get_tensor("weight_matrix")
    if W is not None:
        W = W[:num_neurons, :num_neurons]
        row_ptr, col_idx, w_fp = dense_to_csr(W, quantize_8bit=quantize_8bit)

        fabric.add_projection(
            name="main",
            pre_start=0,
            pre_end=num_neurons,
            post_start=0,
            post_end=num_neurons,
            row_ptr=row_ptr,
            col_idx=col_idx,
            weights_fp=w_fp,
        )

    return fabric


__all__ = [
    # Quantization
    "quantize_q5_10",
    "quantize_q1_14",
    "quantize_q1_6",
    "quantize_q1_14_weights",
    # CSR conversion
    "dense_to_csr",
    "sparse_to_csr",
    # Data structures
    "KittenFabricData",
    # Builders
    "build_kitten_fabric_from_model",
    "build_kitten_fabric_from_state_dict",
]
