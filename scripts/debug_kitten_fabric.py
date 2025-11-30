#!/usr/bin/env python3
"""Debug script for Kitten SNN fabric.

Quick sanity check to verify the fabric loads and runs correctly.

Usage:
    python scripts/debug_kitten_fabric.py
    python scripts/debug_kitten_fabric.py --time-steps 64 --batch 4
    python scripts/debug_kitten_fabric.py --export /tmp/kitten_export
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.tfan.snn import (
    load_fabric_config,
    build_fabric_from_config,
    SNNFabricModel,
)
from models.tfan.snn.fabric import FPGAExporter


def parse_args():
    parser = argparse.ArgumentParser(description="Debug Kitten SNN fabric")
    parser.add_argument(
        "--config",
        default="configs/snn/kitten_fabric_4pop.yaml",
        help="Path to fabric config YAML",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=32,
        help="Number of simulation timesteps",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=2,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="If set, export fabric to this directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed info",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("KITTEN SNN FABRIC DEBUG")
    print("=" * 60)

    # Load config
    print(f"\n[1] Loading config: {args.config}")
    config = load_fabric_config(args.config)
    print(f"    Fabric name: {config.name}")
    print(f"    Populations: {len(config.populations)}")
    print(f"    Projections: {len(config.projections)}")

    # Build fabric
    print(f"\n[2] Building fabric on {args.device}...")
    dtype = torch.float32
    fabric = build_fabric_from_config(config, device=args.device, dtype=dtype)

    # Print summary
    print("\n[3] Fabric summary:")
    print(fabric.summary())

    # CI audit
    print("\n[4] CI compliance audit:")
    audit = fabric.ci_audit()
    print(f"    Compliant: {audit['compliant']}")
    for proj_name, proj_audit in audit.get("projections", {}).items():
        status = "OK" if proj_audit.get("ci_compliant", True) else "FAIL"
        print(f"    {proj_name}: rank={proj_audit.get('rank', '?')}, "
              f"k={proj_audit.get('max_nnz_per_row', '?')} [{status}]")

    # Create model
    print(f"\n[5] Creating SNNFabricModel (time_steps={args.time_steps})...")
    model = SNNFabricModel(
        fabric=fabric,
        time_steps=args.time_steps,
        input_pop="input",
        output_pop="output",
    )
    model.eval()

    # Run forward pass
    print(f"\n[6] Running forward pass (batch={args.batch})...")
    input_N = fabric.populations["input"].N
    x = torch.rand(args.batch, input_N, device=args.device, dtype=dtype)

    start_time = time.time()
    with torch.no_grad():
        out, aux = model(x)
    elapsed = time.time() - start_time

    print(f"    Input shape: {x.shape}")
    print(f"    Output shape: {out.shape}")
    print(f"    Elapsed: {elapsed * 1000:.2f} ms")
    print(f"    Steps/sec: {args.time_steps / elapsed:.1f}")

    # Print aux metrics
    print(f"\n[7] Spike metrics:")
    print(f"    Spike rate: {aux['spike_rate']:.4f}")
    print(f"    Sparsity: {aux['spike_sparsity']:.4f}")
    print(f"    Active events: {aux['active_events']}")

    # Population-wise spike counts
    if args.verbose and "spike_accumulators" in aux:
        print("\n    Per-population spike counts:")
        for name, spikes in aux["spike_accumulators"].items():
            total = spikes.sum().item()
            mean = spikes.mean().item()
            print(f"      {name}: total={total:.0f}, mean={mean:.3f}")

    # Export if requested
    if args.export:
        print(f"\n[8] Exporting to {args.export}...")
        exporter = FPGAExporter(
            fabric=fabric,
            output_dir=args.export,
            quantize=True,
            num_bits=16,
        )
        print(exporter.summary())
        files = exporter.export()
        print(f"    Exported files: {list(files.keys())}")

        # Validate
        validation = exporter.validate()
        print(f"    Validation: {'PASS' if validation['valid'] else 'FAIL'}")
        if "quantization_error" in validation:
            print(f"    Max quant error: {validation['quantization_error']:.6f}")

    print("\n" + "=" * 60)
    print("KITTEN FABRIC DEBUG COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
