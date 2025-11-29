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


__all__ = [
    "quantize_weights",
    "export_fabric_to_dict",
    "export_fabric_to_binary",
    "load_fabric_from_export",
    "FPGAExporter",
]
