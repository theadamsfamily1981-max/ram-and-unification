"""FPGA export utilities for SNN fabric.

Provides functions to export fabric structure and weights
into hardware-friendly formats for Ara-SYNERGY FPGA deployment.
"""

from __future__ import annotations

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .fabric import SNNFabric
from ..projections import LowRankProjection
from ..synapses import LowRankMaskedSynapse


def quantize_weights(
    t: torch.Tensor,
    num_bits: int = 16,
    symmetric: bool = True,
) -> Tuple[np.ndarray, float, int]:
    """Quantize tensor to fixed-point format.

    Args:
        t: Input tensor
        num_bits: Number of bits for quantization
        symmetric: Whether to use symmetric quantization

    Returns:
        q_values: Quantized values as int array
        scale: Scale factor for dequantization
        zero_point: Zero point (0 for symmetric)
    """
    t_np = t.detach().cpu().numpy()

    if symmetric:
        max_abs = np.abs(t_np).max() + 1e-8
        qmax = 2 ** (num_bits - 1) - 1
        scale = max_abs / qmax
        q = np.clip(np.round(t_np / scale), -qmax, qmax).astype(np.int16)
        zero_point = 0
    else:
        t_min = t_np.min()
        t_max = t_np.max()
        qmax = 2 ** num_bits - 1
        scale = (t_max - t_min) / qmax + 1e-8
        zero_point = int(np.round(-t_min / scale))
        q = np.clip(np.round(t_np / scale) + zero_point, 0, qmax).astype(np.uint16)

    return q, float(scale), zero_point


def export_fabric_to_dict(
    fabric: SNNFabric,
    quantize: bool = True,
    num_bits: int = 16,
) -> Dict[str, Any]:
    """Export fabric structure and weights to dictionary.

    This is the canonical export format that Ara-SYNERGY's
    FPGA loader will consume.

    Args:
        fabric: SNNFabric instance
        quantize: Whether to quantize weights
        num_bits: Quantization bits

    Returns:
        Dictionary with populations and projections data
    """
    # Export populations
    pops = []
    for name, pop in fabric.populations.items():
        pop_data = {
            "name": name,
            "N": pop.N,
            "neuron_type": pop.__class__.__name__,
            "params": pop.params.to_dict() if hasattr(pop.params, "to_dict") else {},
        }
        pops.append(pop_data)

    # Export projections
    projs = []
    for proj in fabric.projections:
        if not isinstance(proj, LowRankProjection):
            raise TypeError(
                f"Only LowRankProjection is supported for export, got {type(proj)}"
            )

        syn = proj.synapse

        # Get CSR structure
        indptr = syn.indptr.cpu().numpy().astype(np.int32)
        indices = syn.indices.cpu().numpy().astype(np.int32)

        # Compute effective weights at mask positions
        values = syn.effective_weight_sparse()

        if quantize:
            q_values, scale, zero_point = quantize_weights(values, num_bits)
            values_data = {
                "quantized": True,
                "num_bits": num_bits,
                "values": q_values.tolist(),
                "scale": scale,
                "zero_point": zero_point,
            }
        else:
            values_data = {
                "quantized": False,
                "values": values.cpu().numpy().tolist(),
            }

        # Export low-rank factors for potential hardware acceleration
        U = syn.U.detach().cpu().numpy()
        V = syn.V.detach().cpu().numpy()

        if quantize:
            q_U, scale_U, zp_U = quantize_weights(syn.U, num_bits)
            q_V, scale_V, zp_V = quantize_weights(syn.V, num_bits)
            lowrank_data = {
                "U": {"values": q_U.tolist(), "scale": scale_U, "zero_point": zp_U},
                "V": {"values": q_V.tolist(), "scale": scale_V, "zero_point": zp_V},
            }
        else:
            lowrank_data = {
                "U": {"values": U.tolist()},
                "V": {"values": V.tolist()},
            }

        proj_data = {
            "name": proj.name,
            "pre": proj.pre,
            "post": proj.post,
            "N_pre": proj.params.N_pre,
            "N_post": proj.params.N_post,
            "k": proj.params.k,
            "r": proj.params.r,
            # CSR structure
            "indptr": indptr.tolist(),
            "indices": indices.tolist(),
            "nnz": int(syn.nnz),
            # Weight data
            "weights": values_data,
            # Low-rank factors
            "lowrank": lowrank_data,
        }
        projs.append(proj_data)

    return {
        "fabric_name": fabric.name,
        "time_steps": fabric.time_steps,
        "dt": fabric.dt,
        "populations": pops,
        "projections": projs,
        "total_neurons": sum(pop.N for pop in fabric.populations.values()),
        "total_synapses": sum(proj.synapse.nnz for proj in fabric.projections),
    }


def export_fabric_to_binary(
    fabric: SNNFabric,
    output_dir: str | Path,
    quantize: bool = True,
    num_bits: int = 16,
) -> Dict[str, Path]:
    """Export fabric to binary files for FPGA loading.

    Creates:
    - config.json: Fabric metadata
    - populations.bin: Population parameters
    - proj_*.bin: Per-projection CSR data

    Args:
        fabric: SNNFabric instance
        output_dir: Output directory
        quantize: Whether to quantize
        num_bits: Quantization bits

    Returns:
        Dictionary mapping file types to paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    # Export to dict first
    data = export_fabric_to_dict(fabric, quantize, num_bits)

    # Save config JSON
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "fabric_name": data["fabric_name"],
            "time_steps": data["time_steps"],
            "dt": data["dt"],
            "populations": data["populations"],
            "total_neurons": data["total_neurons"],
            "total_synapses": data["total_synapses"],
            "projection_count": len(data["projections"]),
        }, f, indent=2)
    files["config"] = config_path

    # Save projections as binary
    for proj in data["projections"]:
        proj_path = output_dir / f"proj_{proj['name']}.bin"

        with open(proj_path, "wb") as f:
            # Header: N_pre, N_post, k, r, nnz
            header = np.array([
                proj["N_pre"], proj["N_post"],
                proj["k"], proj["r"], proj["nnz"]
            ], dtype=np.int32)
            f.write(header.tobytes())

            # CSR indptr
            indptr = np.array(proj["indptr"], dtype=np.int32)
            f.write(indptr.tobytes())

            # CSR indices
            indices = np.array(proj["indices"], dtype=np.int32)
            f.write(indices.tobytes())

            # Weight values
            weights = proj["weights"]
            if weights["quantized"]:
                w = np.array(weights["values"], dtype=np.int16)
                # Write scale as float32
                scale = np.array([weights["scale"]], dtype=np.float32)
                f.write(scale.tobytes())
                f.write(w.tobytes())
            else:
                w = np.array(weights["values"], dtype=np.float32)
                f.write(w.tobytes())

        files[f"proj_{proj['name']}"] = proj_path

    return files


def load_fabric_from_export(
    export_dir: str | Path,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load exported fabric data.

    Args:
        export_dir: Directory containing exported files
        device: Target device

    Returns:
        Dictionary with loaded data
    """
    export_dir = Path(export_dir)

    # Load config
    with open(export_dir / "config.json", "r") as f:
        config = json.load(f)

    # Load projections
    projections = {}
    for i in range(config["projection_count"]):
        # Find projection files
        proj_files = list(export_dir.glob("proj_*.bin"))
        for proj_file in proj_files:
            proj_name = proj_file.stem.replace("proj_", "")

            with open(proj_file, "rb") as f:
                # Read header
                header = np.frombuffer(f.read(5 * 4), dtype=np.int32)
                N_pre, N_post, k, r, nnz = header

                # Read CSR indptr
                indptr = np.frombuffer(f.read((N_post + 1) * 4), dtype=np.int32)

                # Read CSR indices
                indices = np.frombuffer(f.read(nnz * 4), dtype=np.int32)

                # Read weights (assume quantized for now)
                scale = np.frombuffer(f.read(4), dtype=np.float32)[0]
                values = np.frombuffer(f.read(nnz * 2), dtype=np.int16)

                projections[proj_name] = {
                    "N_pre": N_pre,
                    "N_post": N_post,
                    "k": k,
                    "r": r,
                    "nnz": nnz,
                    "indptr": torch.tensor(indptr, device=device),
                    "indices": torch.tensor(indices, device=device),
                    "values": torch.tensor(values.astype(np.float32) * scale, device=device),
                }

    return {
        "config": config,
        "projections": projections,
    }


class FPGAExporter:
    """High-level exporter for FPGA deployment.

    Manages the export process and provides utilities for
    validating exported data.

    Args:
        fabric: SNNFabric to export
        output_dir: Output directory
        quantize: Whether to quantize weights
        num_bits: Quantization bits
    """

    def __init__(
        self,
        fabric: SNNFabric,
        output_dir: str | Path,
        quantize: bool = True,
        num_bits: int = 16,
    ):
        self.fabric = fabric
        self.output_dir = Path(output_dir)
        self.quantize = quantize
        self.num_bits = num_bits

    def export(self) -> Dict[str, Path]:
        """Run full export.

        Returns:
            Dictionary mapping file types to paths
        """
        return export_fabric_to_binary(
            self.fabric,
            self.output_dir,
            self.quantize,
            self.num_bits,
        )

    def validate(self) -> Dict[str, Any]:
        """Validate exported data matches original fabric.

        Returns:
            Validation results
        """
        # Export to dict
        exported = export_fabric_to_dict(self.fabric, self.quantize, self.num_bits)

        # Check structure
        results = {
            "valid": True,
            "checks": {},
        }

        # Check populations
        orig_pops = set(self.fabric.populations.keys())
        exp_pops = set(p["name"] for p in exported["populations"])
        results["checks"]["populations_match"] = orig_pops == exp_pops
        if not results["checks"]["populations_match"]:
            results["valid"] = False

        # Check projections
        orig_proj_count = len(self.fabric.projections)
        exp_proj_count = len(exported["projections"])
        results["checks"]["projection_count"] = orig_proj_count == exp_proj_count
        if not results["checks"]["projection_count"]:
            results["valid"] = False

        # Check weight reconstruction error
        max_error = 0.0
        for i, proj in enumerate(self.fabric.projections):
            if hasattr(proj, "synapse"):
                orig_weights = proj.synapse.effective_weight_sparse()
                exp_weights = exported["projections"][i]["weights"]

                if exp_weights["quantized"]:
                    recon = torch.tensor(exp_weights["values"]) * exp_weights["scale"]
                else:
                    recon = torch.tensor(exp_weights["values"])

                error = (orig_weights.cpu() - recon).abs().max().item()
                max_error = max(max_error, error)

        results["checks"]["max_weight_error"] = max_error
        results["quantization_error"] = max_error

        return results

    def summary(self) -> str:
        """Generate export summary."""
        data = export_fabric_to_dict(self.fabric, self.quantize, self.num_bits)

        lines = [
            f"=== FPGA Export Summary ===",
            f"Fabric: {data['fabric_name']}",
            f"Output: {self.output_dir}",
            f"Quantization: {self.num_bits}-bit" if self.quantize else "None",
            "",
            f"Populations: {len(data['populations'])}",
            f"Total neurons: {data['total_neurons']:,}",
            "",
            f"Projections: {len(data['projections'])}",
            f"Total synapses: {data['total_synapses']:,}",
            "",
            "Memory estimates:",
        ]

        # Estimate memory
        total_bytes = 0
        for proj in data["projections"]:
            # indptr: (N_post + 1) * 4 bytes
            # indices: nnz * 4 bytes
            # values: nnz * 2 bytes (if quantized)
            indptr_bytes = (proj["N_post"] + 1) * 4
            indices_bytes = proj["nnz"] * 4
            values_bytes = proj["nnz"] * (2 if self.quantize else 4)
            proj_bytes = indptr_bytes + indices_bytes + values_bytes
            total_bytes += proj_bytes
            lines.append(f"  {proj['name']}: {proj_bytes / 1024:.1f} KB")

        lines.append(f"  Total: {total_bytes / 1024:.1f} KB ({total_bytes / 1024 / 1024:.2f} MB)")

        return "\n".join(lines)


def generate_hls_structs(
    fabric: SNNFabric,
    output_path: str | Path,
    num_bits: int = 16,
) -> Path:
    """Generate C/C++ header with struct definitions for HLS/SYCL.

    Creates type definitions that match the binary export format,
    allowing direct memory mapping in HLS/SYCL kernels.

    Args:
        fabric: SNNFabric instance
        output_path: Path to output header file
        num_bits: Weight quantization bits

    Returns:
        Path to generated header file
    """
    output_path = Path(output_path)

    # Determine weight type based on bits
    weight_type = "int8_t" if num_bits <= 8 else "int16_t"

    total_neurons = sum(pop.N for pop in fabric.populations.values())
    total_synapses = sum(proj.synapse.nnz for proj in fabric.projections)

    header = f'''// =============================================================================
// hls_structs.h
//
// Auto-generated HLS/SYCL struct definitions for SNN fabric deployment
// Fabric: {fabric.name}
// Generated for {num_bits}-bit quantization
//
// WARNING: This file is auto-generated. Do not edit manually.
// =============================================================================

#ifndef HLS_STRUCTS_H
#define HLS_STRUCTS_H

#include <stdint.h>

// =============================================================================
// Configuration Constants
// =============================================================================

#define FABRIC_NAME "{fabric.name}"
#define TIME_STEPS {fabric.time_steps}
#define DT_US {int(fabric.dt * 1e6)}

#define NUM_POPULATIONS {len(fabric.populations)}
#define NUM_PROJECTIONS {len(fabric.projections)}
#define TOTAL_NEURONS {total_neurons}
#define TOTAL_SYNAPSES {total_synapses}
#define WEIGHT_BITS {num_bits}

// Population sizes
'''

    # Add population size constants
    for name, pop in fabric.populations.items():
        const_name = f"N_{name.upper()}"
        header += f"#define {const_name} {pop.N}\n"

    header += f'''
// =============================================================================
// Type Definitions
// =============================================================================

typedef {weight_type} weight_q_t;
typedef int16_t  fp_v_t;     // Membrane potential (Q6.10)
typedef int32_t  fp_i_t;     // Input current accumulator
typedef uint16_t fp_param_t; // Neuron parameters (Q1.14)

typedef struct {{
    uint16_t neuron_idx;
    uint8_t  spike;
    uint8_t  _pad;
}} spike_event_t;

// =============================================================================
// Projection Header (at start of proj_*.bin)
// =============================================================================

typedef struct __attribute__((packed)) {{
    int32_t N_pre;
    int32_t N_post;
    int32_t k;
    int32_t r;
    int32_t nnz;
}} proj_header_t;

// =============================================================================
// Neuron State
// =============================================================================

typedef struct __attribute__((packed)) {{
    fp_v_t v;
    fp_v_t v_th;
    fp_param_t alpha;
    uint8_t refractory;
    uint8_t flags;
}} neuron_state_t;

// =============================================================================
// Fixed-Point Conversion
// =============================================================================

#define FP_V_FRAC_BITS 10
#define FP_PARAM_FRAC_BITS 14
#define FLOAT_TO_FP_V(x)     ((fp_v_t)((x) * (1 << FP_V_FRAC_BITS)))
#define FLOAT_TO_FP_PARAM(x) ((fp_param_t)((x) * (1 << FP_PARAM_FRAC_BITS)))

// =============================================================================
// Memory Size Macros
// =============================================================================

#define INDPTR_SIZE(n_post)   (((n_post) + 1) * sizeof(int32_t))
#define INDICES_SIZE(nnz)     ((nnz) * sizeof(int32_t))
#define VALUES_SIZE(nnz)      ((nnz) * sizeof(weight_q_t))

'''

    # Add projection-specific constants
    for proj in fabric.projections:
        name_upper = proj.name.upper().replace("→", "_TO_").replace(" ", "_")
        header += f"// Projection: {proj.name}\n"
        header += f"#define PROJ_{name_upper}_N_PRE {proj.params.N_pre}\n"
        header += f"#define PROJ_{name_upper}_N_POST {proj.params.N_post}\n"
        header += f"#define PROJ_{name_upper}_NNZ {proj.synapse.nnz}\n\n"

    header += "#endif // HLS_STRUCTS_H\n"

    with open(output_path, "w") as f:
        f.write(header)

    return output_path


# =============================================================================
# Kitten FPGA Export (matches HLS kernel interface)
# =============================================================================

def export_kitten_fabric(
    fabric_data: "KittenFabricData",
    out_dir: str | Path,
) -> Dict[str, Path]:
    """Export Kitten fabric data to binary files for FPGA deployment.

    Creates files matching the FABRIC_MAPPING.md specification:
    - neurons.bin: Packed neuron records (v, v_th, flags) - 6 bytes each
    - weights.bin: CSR data (row_ptr, col_idx, weights) for all projections
    - fabric_topology.json: Metadata with offsets and sizes

    Args:
        fabric_data: KittenFabricData from from_model.py
        out_dir: Output directory

    Returns:
        Dictionary mapping file names to paths
    """
    import struct
    from .from_model import KittenFabricData

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_neurons = fabric_data.num_neurons
    files = {}

    # =========================================================================
    # neurons.bin - Packed neuron state records
    # Layout: [v: int16, v_th: int16, flags: uint16] = 6 bytes per neuron
    # =========================================================================
    neuron_record_size = 6
    v_offset = 0
    v_th_offset = 2
    flags_offset = 4

    neurons_bytes = bytearray(neuron_record_size * num_neurons)

    for n in range(num_neurons):
        base = n * neuron_record_size
        # v (Q5.10)
        struct.pack_into("<h", neurons_bytes, base + v_offset, int(fabric_data.v_init_fp[n]))
        # v_th (Q5.10)
        struct.pack_into("<h", neurons_bytes, base + v_th_offset, int(fabric_data.v_th_fp[n]))
        # flags (initially 0)
        struct.pack_into("<H", neurons_bytes, base + flags_offset, 0)

    neurons_path = out_dir / "neurons.bin"
    neurons_path.write_bytes(neurons_bytes)
    files["neurons"] = neurons_path

    # =========================================================================
    # weights.bin - CSR data for all projections
    # Layout: [row_ptr: int32[N+1], col_idx: int32[nnz], weights: int8[nnz]]
    # =========================================================================
    weights_bytes = bytearray()
    projection_metadata = []

    for proj in fabric_data.projections:
        proj_meta = {
            "name": proj["name"],
            "pre_start": proj["pre_start"],
            "pre_end": proj["pre_end"],
            "post_start": proj["post_start"],
            "post_end": proj["post_end"],
            "nnz": proj["nnz"],
        }

        # Row pointer offset
        proj_meta["row_ptr_offset_bytes"] = len(weights_bytes)
        proj_meta["row_ptr_length"] = len(proj["row_ptr"])
        for v in proj["row_ptr"]:
            weights_bytes += struct.pack("<I", int(v))

        # Column indices offset
        proj_meta["col_idx_offset_bytes"] = len(weights_bytes)
        for c in proj["col_idx"]:
            weights_bytes += struct.pack("<I", int(c))

        # Weights offset
        proj_meta["weights_offset_bytes"] = len(weights_bytes)
        for w in proj["weights_fp"]:
            # 8-bit weights
            weights_bytes += struct.pack("<b", int(w))

        projection_metadata.append(proj_meta)

    weights_path = out_dir / "weights.bin"
    weights_path.write_bytes(weights_bytes)
    files["weights"] = weights_path

    # =========================================================================
    # fabric_topology.json - Metadata
    # =========================================================================
    topology = {
        "version": 1,
        "fabric_name": fabric_data.name,
        "endianness": "little",

        "total_neurons": num_neurons,
        "total_synapses": fabric_data.total_synapses,

        "fixed_point": {
            "v_bits": 16,
            "v_frac_bits": 10,
            "w_bits": 8,
            "w_frac_bits": 6,
            "param_bits": 16,
            "param_frac_bits": 14,
        },

        "neuron_state_layout": {
            "record_size_bytes": neuron_record_size,
            "record_count": num_neurons,
            "v_offset_bytes": v_offset,
            "v_stride_bytes": neuron_record_size,
            "threshold_offset_bytes": v_th_offset,
            "threshold_stride_bytes": neuron_record_size,
            "flags_offset_bytes": flags_offset,
            "flags_stride_bytes": neuron_record_size,
        },

        "projections": projection_metadata,
    }

    topology_path = out_dir / "fabric_topology.json"
    with open(topology_path, "w") as f:
        json.dump(topology, f, indent=2)
    files["topology"] = topology_path

    print(f"[Kitten export] Exported to {out_dir}/")
    print(f"  neurons.bin:          {len(neurons_bytes):,} bytes ({num_neurons} neurons)")
    print(f"  weights.bin:          {len(weights_bytes):,} bytes ({fabric_data.total_synapses} synapses)")
    print(f"  fabric_topology.json: metadata")

    return files


def export_kitten_from_snns_fabric(
    fabric: "SNNFabric",
    out_dir: str | Path,
    quantize_8bit: bool = True,
) -> Dict[str, Path]:
    """Export an existing SNNFabric to Kitten binary format.

    This is a convenience wrapper that converts SNNFabric to KittenFabricData
    and then exports.

    Args:
        fabric: SNNFabric instance
        out_dir: Output directory
        quantize_8bit: Use 8-bit weight quantization

    Returns:
        Dictionary mapping file names to paths
    """
    import torch
    from .from_model import (
        KittenFabricData,
        quantize_q5_10,
        quantize_q1_14,
        quantize_q1_6,
        quantize_q1_14_weights,
    )

    # Calculate total neurons
    total_neurons = sum(pop.N for pop in fabric.populations.values())

    # Create neuron parameter arrays
    # Default values - in real use these would come from population params
    v_init = torch.zeros(total_neurons)
    v_th = torch.ones(total_neurons) * 0.5
    alpha = torch.ones(total_neurons) * 0.9

    # Apply per-population parameters if available
    offset = 0
    for name, pop in fabric.populations.items():
        N = pop.N
        if hasattr(pop, "params"):
            v_th[offset:offset + N] = pop.params.v_th
            alpha[offset:offset + N] = pop.params.alpha
        offset += N

    # Quantize
    v_init_fp = quantize_q5_10(v_init).numpy()
    v_th_fp = quantize_q5_10(v_th).numpy()
    alpha_fp = quantize_q1_14(alpha).numpy()

    fabric_data = KittenFabricData(
        name=fabric.name,
        num_neurons=total_neurons,
        v_init_fp=v_init_fp,
        v_th_fp=v_th_fp,
        alpha_fp=alpha_fp,
    )

    # Build population ID mappings
    pop_offsets = {}
    offset = 0
    for name, pop in fabric.populations.items():
        pop_offsets[name] = offset
        offset += pop.N

    # Export projections
    for proj in fabric.projections:
        pre_offset = pop_offsets[proj.pre]
        post_offset = pop_offsets[proj.post]
        pre_N = fabric.populations[proj.pre].N
        post_N = fabric.populations[proj.post].N

        # Get CSR data from projection synapse
        if hasattr(proj, "synapse"):
            syn = proj.synapse
            row_ptr = syn.indptr.detach().cpu().numpy().astype(np.int32)
            col_idx = syn.indices.detach().cpu().numpy().astype(np.int32)

            # Get effective weights
            values = syn.effective_weight_sparse()

            if quantize_8bit:
                w_fp = quantize_q1_6(values).numpy()
            else:
                w_fp = quantize_q1_14_weights(values).numpy()

            fabric_data.add_projection(
                name=proj.name,
                pre_start=pre_offset,
                pre_end=pre_offset + pre_N,
                post_start=post_offset,
                post_end=post_offset + post_N,
                row_ptr=row_ptr,
                col_idx=col_idx,
                weights_fp=w_fp,
            )

    return export_kitten_fabric(fabric_data, out_dir)


__all__ = [
    "quantize_weights",
    "export_fabric_to_dict",
    "export_fabric_to_binary",
    "load_fabric_from_export",
    "generate_hls_structs",
    "FPGAExporter",
    # Kitten-specific
    "export_kitten_fabric",
    "export_kitten_from_snns_fabric",
]
