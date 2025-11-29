"""Device detection and GPU acceleration utilities."""

import os
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


def get_optimal_device() -> str:
    """Automatically detect and select the best available device.

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    # Try CUDA (NVIDIA GPU)
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA available: {device_count} GPU(s), using {device_name}")
            return "cuda"
    except ImportError:
        pass

    # Try MPS (Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) available")
            return "mps"
    except (ImportError, AttributeError):
        pass

    # Fallback to CPU
    logger.info("No GPU acceleration available, using CPU")
    return "cpu"


def get_device_info() -> Dict[str, Any]:
    """Get detailed device information.

    Returns:
        Dictionary with device information
    """
    info = {
        "optimal_device": "cpu",
        "cuda_available": False,
        "mps_available": False,
        "cpu_count": os.cpu_count() or 1,
        "gpu_devices": []
    }

    try:
        import torch
        info["pytorch_version"] = torch.__version__

        # CUDA info
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["optimal_device"] = "cuda"
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()

            # GPU devices
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": torch.cuda.get_device_properties(i).total_memory / 1024**3,
                    "capability": torch.cuda.get_device_capability(i)
                }
                info["gpu_devices"].append(gpu_info)

        # MPS info
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info["mps_available"] = True
            info["optimal_device"] = "mps"

    except ImportError:
        logger.warning("PyTorch not installed, device detection limited")

    # CPU info
    try:
        import psutil
        info["cpu_percent"] = psutil.cpu_percent()
        info["memory_available_gb"] = psutil.virtual_memory().available / 1024**3
        info["memory_total_gb"] = psutil.virtual_memory().total / 1024**3
    except ImportError:
        pass

    return info


def estimate_optimal_batch_size(device: str, resolution: int = 512) -> Tuple[int, int]:
    """Estimate optimal batch sizes for avatar generation.

    Args:
        device: Device type (cuda, mps, cpu)
        resolution: Output resolution

    Returns:
        Tuple of (face_detection_batch_size, wav2lip_batch_size)
    """
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                # Get GPU memory
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

                # Estimate batch sizes based on memory
                if gpu_memory_gb >= 16:
                    # High-end GPU (RTX 3090, RTX 4090, etc.)
                    face_batch = 16
                    wav2lip_batch = 256
                elif gpu_memory_gb >= 8:
                    # Mid-range GPU (RTX 3070, RTX 4070, etc.)
                    face_batch = 8
                    wav2lip_batch = 128
                elif gpu_memory_gb >= 6:
                    # Entry-level GPU (RTX 3060, etc.)
                    face_batch = 4
                    wav2lip_batch = 64
                else:
                    # Low VRAM GPU
                    face_batch = 2
                    wav2lip_batch = 32

                logger.info(
                    f"GPU memory: {gpu_memory_gb:.1f}GB, "
                    f"batch sizes: face={face_batch}, wav2lip={wav2lip_batch}"
                )
                return face_batch, wav2lip_batch
        except:
            pass

    elif device == "mps":
        # Apple Silicon - moderate batch sizes
        face_batch = 8
        wav2lip_batch = 128
        return face_batch, wav2lip_batch

    # CPU - small batch sizes
    face_batch = 2
    wav2lip_batch = 32
    return face_batch, wav2lip_batch


def get_recommended_workers(device: str) -> int:
    """Get recommended number of worker threads.

    Args:
        device: Device type

    Returns:
        Recommended worker count
    """
    cpu_count = os.cpu_count() or 1

    if device == "cuda":
        # GPU: fewer workers since GPU does heavy lifting
        return min(cpu_count // 2, 4)
    elif device == "mps":
        # Apple Silicon: moderate workers
        return min(cpu_count // 2, 3)
    else:
        # CPU: more workers for parallelism
        return min(cpu_count, 8)


def check_gpu_requirements() -> Dict[str, Any]:
    """Check if system meets GPU requirements for avatar generation.

    Returns:
        Dictionary with requirement check results
    """
    results = {
        "meets_requirements": False,
        "warnings": [],
        "recommendations": []
    }

    device_info = get_device_info()

    # Check for GPU
    if device_info["cuda_available"]:
        gpu = device_info["gpu_devices"][0]
        memory_gb = gpu["total_memory_gb"]

        if memory_gb >= 6:
            results["meets_requirements"] = True
        elif memory_gb >= 4:
            results["meets_requirements"] = True
            results["warnings"].append(
                f"GPU memory ({memory_gb:.1f}GB) is low. Consider using lower resolution."
            )
        else:
            results["warnings"].append(
                f"GPU memory ({memory_gb:.1f}GB) is insufficient. Recommend 6GB+ for good performance."
            )
            results["recommendations"].append("Upgrade to GPU with more VRAM or use CPU mode")

        # Check CUDA compute capability
        capability = gpu["capability"]
        if capability[0] < 6:  # Older than Pascal
            results["warnings"].append(
                f"GPU compute capability {capability[0]}.{capability[1]} is outdated. "
                "May have compatibility issues."
            )

    elif device_info["mps_available"]:
        results["meets_requirements"] = True
        results["recommendations"].append(
            "Apple Silicon detected. Performance will be good but slower than high-end NVIDIA GPUs."
        )

    else:
        # CPU only
        results["warnings"].append("No GPU detected. Avatar generation will be slow on CPU.")
        results["recommendations"].append(
            "Install NVIDIA GPU for 10-100x faster generation"
        )

        # Check CPU and RAM
        if device_info.get("memory_total_gb", 0) < 8:
            results["warnings"].append(
                f"Low RAM ({device_info.get('memory_total_gb', 0):.1f}GB). Recommend 16GB+ for CPU mode."
            )

    return results


def optimize_torch_settings(device: str):
    """Apply PyTorch optimizations for device.

    Args:
        device: Device type
    """
    try:
        import torch

        if device == "cuda":
            # CUDA optimizations
            torch.backends.cudnn.benchmark = True  # Auto-tune algorithms
            torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere+
            logger.info("Applied CUDA optimizations")

        elif device == "cpu":
            # CPU optimizations
            torch.set_num_threads(os.cpu_count() or 1)
            logger.info(f"Set PyTorch threads: {os.cpu_count()}")

    except ImportError:
        pass
