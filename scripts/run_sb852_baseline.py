#!/usr/bin/env python3
"""Baseline script for SB-852 / FWDNXT accelerator comparison.

Runs a reference CNN/DL model on the Micron SB-852 via FWDNXT SDK,
producing embeddings and latency measurements for comparison with
the SNN fabric and TF-A-N pipelines.

This script serves as:
  1. Board health verification (can we talk to the SB-852?)
  2. Baseline latency/throughput measurement
  3. Embedding extraction for hybrid pipelines (SB-852 CNN → SNN fabric)

Prerequisites:
  - FWDNXT SDK installed and configured
  - SB-852 board connected via PCIe
  - Model compiled via FWDNXT compiler (ONNX → .bin)

Usage:
    python scripts/run_sb852_baseline.py --model resnet18.bin --input data/test_images/
    python scripts/run_sb852_baseline.py --model efficientnet_b0.bin --batch 8 --warmup 10
    python scripts/run_sb852_baseline.py --list-devices
    python scripts/run_sb852_baseline.py --benchmark --iterations 100

Example hybrid pipeline:
    # 1. Extract embeddings from SB-852
    python scripts/run_sb852_baseline.py --model cnn_frontend.bin \\
        --input data/frames/ --output-embeddings /tmp/embeddings.pt

    # 2. Feed embeddings to SNN fabric
    python scripts/debug_kitten_fabric.py --input-embeddings /tmp/embeddings.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# FWDNXT SDK Stub (replace with actual import when SDK is installed)
# =============================================================================

FWDNXT_AVAILABLE = False
try:
    import fwdnxt  # type: ignore
    FWDNXT_AVAILABLE = True
except ImportError:
    fwdnxt = None


@dataclass
class FWDNXTStub:
    """Stub for FWDNXT SDK when not available.

    Replace this with actual SDK calls when FWDNXT is installed.
    The interface mirrors the expected FWDNXT API.
    """

    model_path: Optional[str] = None
    device_id: int = 0
    initialized: bool = False

    def init(self, device_id: int = 0) -> bool:
        """Initialize the FWDNXT runtime and connect to device."""
        print(f"[STUB] Would initialize FWDNXT device {device_id}")
        self.device_id = device_id
        self.initialized = True
        return True

    def load_model(self, model_path: str) -> bool:
        """Load a compiled model binary."""
        print(f"[STUB] Would load model from {model_path}")
        self.model_path = model_path
        return True

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Run inference on input tensor."""
        # Stub: return random output matching expected shape
        batch_size = inputs.shape[0]
        print(f"[STUB] Would run inference on batch of {batch_size}")
        # Simulate some latency
        time.sleep(0.001 * batch_size)
        # Return fake embeddings (1024-dim)
        return np.random.randn(batch_size, 1024).astype(np.float32)

    def get_info(self) -> Dict[str, Any]:
        """Get device info."""
        return {
            "device_id": self.device_id,
            "name": "SB-852 (STUB)",
            "memory_gb": 64,
            "pcie_gen": 3,
            "pcie_lanes": 16,
            "firmware_version": "stub",
        }

    def close(self) -> None:
        """Release device resources."""
        print("[STUB] Would close FWDNXT device")
        self.initialized = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    model_name: str
    device_info: Dict[str, Any]
    batch_size: int
    num_iterations: int
    warmup_iterations: int

    # Timing (milliseconds)
    latencies_ms: List[float] = field(default_factory=list)

    @property
    def mean_latency_ms(self) -> float:
        return np.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def std_latency_ms(self) -> float:
        return np.std(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p50_latency_ms(self) -> float:
        return np.percentile(self.latencies_ms, 50) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        return np.percentile(self.latencies_ms, 95) if self.latencies_ms else 0.0

    @property
    def p99_latency_ms(self) -> float:
        return np.percentile(self.latencies_ms, 99) if self.latencies_ms else 0.0

    @property
    def throughput_samples_per_sec(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return 1000.0 * self.batch_size / self.mean_latency_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device_info": self.device_info,
            "batch_size": self.batch_size,
            "num_iterations": self.num_iterations,
            "warmup_iterations": self.warmup_iterations,
            "mean_latency_ms": self.mean_latency_ms,
            "std_latency_ms": self.std_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
        }

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("SB-852 BASELINE BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Model:            {self.model_name}")
        print(f"Device:           {self.device_info.get('name', 'unknown')}")
        print(f"Batch size:       {self.batch_size}")
        print(f"Iterations:       {self.num_iterations} (warmup: {self.warmup_iterations})")
        print("-" * 60)
        print(f"Mean latency:     {self.mean_latency_ms:.3f} ms")
        print(f"Std latency:      {self.std_latency_ms:.3f} ms")
        print(f"P50 latency:      {self.p50_latency_ms:.3f} ms")
        print(f"P95 latency:      {self.p95_latency_ms:.3f} ms")
        print(f"P99 latency:      {self.p99_latency_ms:.3f} ms")
        print(f"Throughput:       {self.throughput_samples_per_sec:.1f} samples/sec")
        print("=" * 60)


# =============================================================================
# Core Functions
# =============================================================================

def get_fwdnxt_handle() -> FWDNXTStub:
    """Get FWDNXT handle (real or stub)."""
    if FWDNXT_AVAILABLE:
        # Return actual FWDNXT handle when SDK is available
        # return fwdnxt.FWDNXT()
        pass
    return FWDNXTStub()


def list_devices() -> None:
    """List available FWDNXT devices."""
    print("\n=== FWDNXT Devices ===")

    if not FWDNXT_AVAILABLE:
        print("[WARNING] FWDNXT SDK not installed. Using stub.")
        print("  Device 0: SB-852 (STUB) - 64 GB DDR4, PCIe Gen3 x16")
        return

    # TODO: Replace with actual device enumeration
    # devices = fwdnxt.list_devices()
    # for i, dev in enumerate(devices):
    #     print(f"  Device {i}: {dev.name} - {dev.memory_gb} GB, {dev.pcie_info}")
    print("  [Implement actual device listing when SDK available]")


def load_input_data(
    input_path: Optional[str],
    batch_size: int,
    input_shape: tuple = (3, 224, 224),
) -> np.ndarray:
    """Load or generate input data.

    Args:
        input_path: Path to input images/tensors. If None, generates random data.
        batch_size: Number of samples in batch.
        input_shape: Shape of each input sample (C, H, W).

    Returns:
        Input tensor as numpy array with shape (batch_size, *input_shape).
    """
    if input_path is None:
        # Generate random input for benchmarking
        print(f"[INFO] Generating random input: batch={batch_size}, shape={input_shape}")
        return np.random.randn(batch_size, *input_shape).astype(np.float32)

    input_path = Path(input_path)

    if input_path.suffix == ".pt":
        # Load PyTorch tensor
        import torch
        tensor = torch.load(input_path)
        return tensor.numpy()

    elif input_path.suffix == ".npy":
        # Load numpy array
        return np.load(input_path)

    elif input_path.is_dir():
        # Load images from directory
        try:
            from PIL import Image
            import torchvision.transforms as T
        except ImportError:
            print("[ERROR] PIL/torchvision required for image loading")
            sys.exit(1)

        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        images = []
        for img_path in sorted(input_path.glob("*.jpg"))[:batch_size]:
            img = Image.open(img_path).convert("RGB")
            images.append(transform(img).numpy())

        if not images:
            print(f"[WARNING] No images found in {input_path}, using random data")
            return np.random.randn(batch_size, *input_shape).astype(np.float32)

        return np.stack(images, axis=0)

    else:
        print(f"[ERROR] Unknown input format: {input_path}")
        sys.exit(1)


def run_benchmark(
    handle: FWDNXTStub,
    model_path: str,
    batch_size: int,
    iterations: int,
    warmup: int,
    input_shape: tuple = (3, 224, 224),
) -> BenchmarkResult:
    """Run latency benchmark on SB-852.

    Args:
        handle: FWDNXT handle.
        model_path: Path to compiled model.
        batch_size: Batch size for inference.
        iterations: Number of timed iterations.
        warmup: Number of warmup iterations (not timed).
        input_shape: Input tensor shape (C, H, W).

    Returns:
        BenchmarkResult with timing statistics.
    """
    print(f"\n[INFO] Running benchmark: model={model_path}, batch={batch_size}")
    print(f"[INFO] Iterations: {iterations} (warmup: {warmup})")

    # Load model
    handle.load_model(model_path)

    # Generate random input
    inputs = np.random.randn(batch_size, *input_shape).astype(np.float32)

    # Warmup
    print(f"[INFO] Running {warmup} warmup iterations...")
    for _ in range(warmup):
        _ = handle.run(inputs)

    # Timed iterations
    print(f"[INFO] Running {iterations} timed iterations...")
    latencies = []
    for i in range(iterations):
        start = time.perf_counter()
        _ = handle.run(inputs)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{iterations}] latency = {latencies[-1]:.3f} ms")

    return BenchmarkResult(
        model_name=Path(model_path).stem,
        device_info=handle.get_info(),
        batch_size=batch_size,
        num_iterations=iterations,
        warmup_iterations=warmup,
        latencies_ms=latencies,
    )


def extract_embeddings(
    handle: FWDNXTStub,
    model_path: str,
    input_data: np.ndarray,
    output_path: str,
) -> None:
    """Extract embeddings from SB-852 and save for downstream use.

    This enables hybrid pipelines:
      SB-852 (CNN) → embeddings → SNN fabric → output

    Args:
        handle: FWDNXT handle.
        model_path: Path to compiled model (should output embeddings, not logits).
        input_data: Input tensor.
        output_path: Path to save embeddings (.pt or .npy).
    """
    print(f"\n[INFO] Extracting embeddings: {input_data.shape[0]} samples")

    handle.load_model(model_path)
    embeddings = handle.run(input_data)

    print(f"[INFO] Embedding shape: {embeddings.shape}")

    output_path = Path(output_path)
    if output_path.suffix == ".pt":
        import torch
        torch.save(torch.from_numpy(embeddings), output_path)
    else:
        np.save(output_path, embeddings)

    print(f"[INFO] Saved embeddings to {output_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SB-852 / FWDNXT baseline for SNN comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available FWDNXT devices and exit",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency benchmark",
    )

    # Model / data
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to compiled FWDNXT model (.bin)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input data (directory of images, .pt, or .npy)",
    )
    parser.add_argument(
        "--output-embeddings",
        type=str,
        default=None,
        help="Path to save extracted embeddings (.pt or .npy)",
    )

    # Benchmark params
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )

    # Device
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="FWDNXT device ID",
    )

    # Output
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save benchmark results as JSON",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # List devices mode
    if args.list_devices:
        list_devices()
        return

    # Check SDK availability
    if not FWDNXT_AVAILABLE:
        print("[WARNING] FWDNXT SDK not installed. Running in stub mode.")
        print("         Install the SDK and update imports in this script.")
        print("")

    # Initialize handle
    handle = get_fwdnxt_handle()
    handle.init(device_id=args.device_id)

    try:
        # Benchmark mode
        if args.benchmark:
            if args.model is None:
                print("[ERROR] --model required for benchmark mode")
                sys.exit(1)

            result = run_benchmark(
                handle=handle,
                model_path=args.model,
                batch_size=args.batch,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            result.print_summary()

            if args.output_json:
                with open(args.output_json, "w") as f:
                    json.dump(result.to_dict(), f, indent=2)
                print(f"[INFO] Saved results to {args.output_json}")

        # Embedding extraction mode
        elif args.output_embeddings:
            if args.model is None:
                print("[ERROR] --model required for embedding extraction")
                sys.exit(1)

            input_data = load_input_data(args.input, args.batch)
            extract_embeddings(
                handle=handle,
                model_path=args.model,
                input_data=input_data,
                output_path=args.output_embeddings,
            )

        # Default: just run inference and print output shape
        else:
            if args.model is None:
                print("[INFO] No model specified. Use --list-devices, --benchmark, or --model.")
                print("[INFO] Run with --help for usage.")
                return

            handle.load_model(args.model)
            input_data = load_input_data(args.input, args.batch)

            print(f"\n[INFO] Running inference: input shape = {input_data.shape}")
            start = time.perf_counter()
            output = handle.run(input_data)
            elapsed = (time.perf_counter() - start) * 1000

            print(f"[INFO] Output shape: {output.shape}")
            print(f"[INFO] Latency: {elapsed:.3f} ms")

    finally:
        handle.close()

    print("\n[DONE]")


if __name__ == "__main__":
    main()
