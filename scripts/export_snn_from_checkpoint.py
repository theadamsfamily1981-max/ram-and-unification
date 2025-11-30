#!/usr/bin/env python3
"""Export TF-A-N SNN head to Kitten FPGA fabric files.

This script bridges the TF-A-N 7B model → SNN fabric → Kitten FPGA pipeline.

Usage:
    python scripts/export_snn_from_checkpoint.py \
        --checkpoint path/to/tfan7b_snn \
        --out-dir artifacts/fabric/kitten_run1 \
        --num-neurons 4096

Then point the FPGA host at artifacts/fabric/kitten_run1/:
    ./run_snn_fpga --xclbin snn_kernel.xclbin \
                   --topology artifacts/fabric/kitten_run1/fabric_topology.json \
                   --weights artifacts/fabric/kitten_run1/weights.bin \
                   --neurons artifacts/fabric/kitten_run1/neurons.bin

Environment:
    PYTHONPATH should include the src/ directory:
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# Add src to path if needed
src_path = Path(__file__).parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="Export TF-A-N SNN head to Kitten FPGA fabric files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    p.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to TF-A-N checkpoint (HF-style directory or .pt file)"
    )
    p.add_argument(
        "--state-dict-key", type=str, default=None,
        help="Key to extract state dict from checkpoint (if nested)"
    )

    # Output
    p.add_argument(
        "--out-dir", type=str, required=True,
        help="Output directory for fabric files"
    )

    # Fabric configuration
    p.add_argument(
        "--num-neurons", type=int, default=4096,
        help="Number of neurons to export (default: 4096)"
    )
    p.add_argument(
        "--alpha", type=float, default=0.9,
        help="Leak factor if not in model (default: 0.9)"
    )
    p.add_argument(
        "--v-th", type=float, default=0.5,
        help="Threshold voltage if not in model (default: 0.5)"
    )

    # Quantization
    p.add_argument(
        "--quantize-16bit", action="store_true",
        help="Use 16-bit weight quantization (default: 8-bit)"
    )

    # Model loading
    p.add_argument(
        "--device", type=str, default="cpu",
        help="Device to load model on (default: cpu)"
    )
    p.add_argument(
        "--dtype", type=str, default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for loading (default: float32)"
    )
    p.add_argument(
        "--snn-head-attr", type=str, default="snn_head",
        help="Attribute name for SNN head module (default: snn_head)"
    )
    p.add_argument(
        "--state-dict-prefix", type=str, default="snn_head.",
        help="Prefix for SNN head keys in state dict (default: snn_head.)"
    )

    # Mode
    p.add_argument(
        "--from-state-dict", action="store_true",
        help="Load from raw state dict instead of full model"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Parse checkpoint and show info without exporting"
    )

    return p.parse_args()


def str_to_dtype(s: str) -> torch.dtype:
    """Convert string to torch dtype."""
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(s.lower(), torch.float32)


def load_checkpoint(path: str, key: str = None) -> dict:
    """Load checkpoint from path.

    Args:
        path: Path to checkpoint file or directory
        key: Optional key to extract from checkpoint

    Returns:
        State dictionary
    """
    path = Path(path)

    if path.is_dir():
        # HuggingFace-style checkpoint directory
        pt_files = list(path.glob("*.bin")) + list(path.glob("*.pt")) + list(path.glob("*.safetensors"))
        if not pt_files:
            raise FileNotFoundError(f"No checkpoint files found in {path}")

        # Try to load model.safetensors or pytorch_model.bin
        for name in ["model.safetensors", "pytorch_model.bin", "model.pt"]:
            check_path = path / name
            if check_path.exists():
                if name.endswith(".safetensors"):
                    try:
                        from safetensors.torch import load_file
                        return load_file(check_path)
                    except ImportError:
                        continue
                else:
                    state_dict = torch.load(check_path, map_location="cpu")
                    break
        else:
            # Load first available file
            state_dict = torch.load(pt_files[0], map_location="cpu")
    else:
        # Single file
        if str(path).endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(path)
        else:
            state_dict = torch.load(path, map_location="cpu")

    # Extract nested state dict if key provided
    if key:
        if key in state_dict:
            state_dict = state_dict[key]
        else:
            raise KeyError(f"Key '{key}' not found in checkpoint")

    return state_dict


def analyze_state_dict(state_dict: dict, prefix: str) -> dict:
    """Analyze SNN-related tensors in state dict.

    Args:
        state_dict: Full state dictionary
        prefix: Prefix for SNN head keys

    Returns:
        Analysis results
    """
    results = {
        "total_keys": len(state_dict),
        "snn_keys": [],
        "snn_shapes": {},
    }

    for key, val in state_dict.items():
        if key.startswith(prefix) or "snn" in key.lower():
            results["snn_keys"].append(key)
            if isinstance(val, torch.Tensor):
                results["snn_shapes"][key] = list(val.shape)

    return results


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    quantize_8bit = not args.quantize_16bit

    print("=" * 80)
    print("TF-A-N → SNN Fabric → Kitten FPGA Export")
    print("=" * 80)
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Output dir:     {out_dir}")
    print(f"Num neurons:    {args.num_neurons}")
    print(f"Alpha:          {args.alpha}")
    print(f"Threshold:      {args.v_th}")
    print(f"Quantization:   {'16-bit' if args.quantize_16bit else '8-bit'}")
    print("-" * 80)

    # Load checkpoint
    print("\n[1] Loading checkpoint...")
    try:
        state_dict = load_checkpoint(args.checkpoint, args.state_dict_key)
        print(f"    Loaded {len(state_dict)} keys")
    except Exception as e:
        print(f"    ERROR: Failed to load checkpoint: {e}")
        return 1

    # Analyze state dict
    print("\n[2] Analyzing SNN-related tensors...")
    analysis = analyze_state_dict(state_dict, args.state_dict_prefix)
    print(f"    Found {len(analysis['snn_keys'])} SNN-related keys:")
    for key in analysis["snn_keys"][:10]:
        shape = analysis["snn_shapes"].get(key, "?")
        print(f"      - {key}: {shape}")
    if len(analysis["snn_keys"]) > 10:
        print(f"      ... and {len(analysis['snn_keys']) - 10} more")

    if args.dry_run:
        print("\n[dry-run] Skipping export")
        return 0

    # Build fabric
    print("\n[3] Building Kitten fabric data...")
    try:
        from models.tfan.snn.fabric import (
            build_kitten_fabric_from_state_dict,
            build_kitten_fabric_from_model,
            export_kitten_fabric,
        )

        if args.from_state_dict:
            # Build from state dict directly
            fabric_data = build_kitten_fabric_from_state_dict(
                state_dict=state_dict,
                num_neurons=args.num_neurons,
                alpha=args.alpha,
                v_th=args.v_th,
                quantize_8bit=quantize_8bit,
                prefix=args.state_dict_prefix,
            )
        else:
            # Try to load full model
            try:
                from models.tfan import TFANConfig, TFANForCausalLM

                print("    Loading full TF-A-N model...")
                config = TFANConfig.from_pretrained(args.checkpoint)
                model = TFANForCausalLM.from_pretrained(
                    args.checkpoint,
                    torch_dtype=str_to_dtype(args.dtype),
                    low_cpu_mem_usage=True,
                )
                model.to(args.device)
                model.eval()

                fabric_data = build_kitten_fabric_from_model(
                    model=model,
                    num_neurons=args.num_neurons,
                    alpha=args.alpha,
                    v_th=args.v_th,
                    quantize_8bit=quantize_8bit,
                    snn_head_attr=args.snn_head_attr,
                )
            except ImportError:
                print("    WARNING: Could not import TF-A-N model classes")
                print("    Falling back to state dict mode...")
                fabric_data = build_kitten_fabric_from_state_dict(
                    state_dict=state_dict,
                    num_neurons=args.num_neurons,
                    alpha=args.alpha,
                    v_th=args.v_th,
                    quantize_8bit=quantize_8bit,
                    prefix=args.state_dict_prefix,
                )
            except Exception as e:
                print(f"    WARNING: Failed to load model: {e}")
                print("    Falling back to state dict mode...")
                fabric_data = build_kitten_fabric_from_state_dict(
                    state_dict=state_dict,
                    num_neurons=args.num_neurons,
                    alpha=args.alpha,
                    v_th=args.v_th,
                    quantize_8bit=quantize_8bit,
                    prefix=args.state_dict_prefix,
                )

        print(f"    Fabric: {fabric_data.name}")
        print(f"    Neurons: {fabric_data.num_neurons}")
        print(f"    Projections: {len(fabric_data.projections)}")
        print(f"    Total synapses: {fabric_data.total_synapses:,}")

    except Exception as e:
        print(f"    ERROR: Failed to build fabric: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Export
    print("\n[4] Exporting to Kitten format...")
    try:
        files = export_kitten_fabric(fabric_data, out_dir)
        print(f"\n    Exported files:")
        for name, path in files.items():
            size = path.stat().st_size
            print(f"      {name}: {path.name} ({size:,} bytes)")
    except Exception as e:
        print(f"    ERROR: Failed to export: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Summary
    print("\n" + "=" * 80)
    print("Export Complete!")
    print("=" * 80)
    print(f"\nArtifacts in: {out_dir}/")
    print("  - neurons.bin          (neuron states)")
    print("  - weights.bin          (CSR projections)")
    print("  - fabric_topology.json (metadata)")
    print("\nNext steps:")
    print("  1. Build FPGA bitstream:")
    print("     cd hls && make TARGET=hw")
    print("  2. Run on FPGA:")
    print(f"     ./host/run_snn_fpga --xclbin snn_core.xclbin \\")
    print(f"         --topology {out_dir}/fabric_topology.json \\")
    print(f"         --weights {out_dir}/weights.bin \\")
    print(f"         --neurons {out_dir}/neurons.bin")

    return 0


if __name__ == "__main__":
    sys.exit(main())
