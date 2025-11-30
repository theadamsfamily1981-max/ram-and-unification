#!/usr/bin/env python3
"""Export Kitten fabric for FPGA deployment.

Generates the export bundle (config.json + binary files) that the
Ara-SYNERGY FPGA loader consumes.

Usage:
    python scripts/export_kitten_fpga.py
    python scripts/export_kitten_fpga.py --output artifacts/kitten_fpga
    python scripts/export_kitten_fpga.py --no-quantize
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.tfan.snn import (
    load_fabric_config,
    build_fabric_from_config,
)
from models.tfan.snn.fabric import (
    export_fabric_to_binary,
    export_fabric_to_dict,
    FPGAExporter,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export Kitten fabric for FPGA")
    parser.add_argument(
        "--config",
        default="configs/snn/kitten_fabric_4pop.yaml",
        help="Path to fabric config YAML",
    )
    parser.add_argument(
        "--output",
        default="artifacts/kitten_fpga",
        help="Output directory for export bundle",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable weight quantization (export as float32)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=16,
        help="Quantization bits (default: 16)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate export after generation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("KITTEN FPGA EXPORT")
    print("=" * 60)

    output_dir = Path(args.output)
    quantize = not args.no_quantize

    # Load config
    print(f"\n[1] Loading config: {args.config}")
    config = load_fabric_config(args.config)
    print(f"    Fabric: {config.name}")

    # Build fabric
    print(f"\n[2] Building fabric...")
    fabric = build_fabric_from_config(config, device="cpu", dtype=torch.float32)
    print(fabric.summary())

    # Create exporter
    print(f"\n[3] Creating exporter...")
    exporter = FPGAExporter(
        fabric=fabric,
        output_dir=output_dir,
        quantize=quantize,
        num_bits=args.bits,
    )
    print(exporter.summary())

    # Export
    print(f"\n[4] Exporting to {output_dir}...")
    files = exporter.export()
    print(f"    Generated files:")
    for name, path in files.items():
        size = path.stat().st_size if path.exists() else 0
        print(f"      {name}: {path.name} ({size / 1024:.1f} KB)")

    # Validate if requested
    if args.validate:
        print(f"\n[5] Validating export...")
        validation = exporter.validate()
        print(f"    Valid: {validation['valid']}")

        for check, result in validation.get("checks", {}).items():
            status = "PASS" if result else "FAIL"
            print(f"      {check}: {status}")

        if "quantization_error" in validation:
            print(f"    Max quantization error: {validation['quantization_error']:.6f}")

    # Also save a human-readable summary
    summary_path = output_dir / "export_summary.txt"
    with open(summary_path, "w") as f:
        f.write(exporter.summary())
        f.write("\n\nFabric Summary:\n")
        f.write(fabric.summary())

    print(f"\n[6] Export complete!")
    print(f"    Bundle: {output_dir}")
    print(f"    Summary: {summary_path}")

    print("\n" + "=" * 60)
    print("READY FOR FPGA DEPLOYMENT")
    print("=" * 60)


if __name__ == "__main__":
    main()
