#!/usr/bin/env python3
"""
Download required models for Phase 3 (Wav2Lip talking head generation).

This script downloads:
- Wav2Lip base model
- Wav2Lip GAN model (higher quality)
- S3FD face detection model
- GFPGAN face enhancement model (optional)
"""

import argparse
import sys
from pathlib import Path
import requests
from tqdm import tqdm
import hashlib


# Model URLs and metadata
MODELS = {
    "wav2lip": {
        "url": "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?download=1",
        "path": "models/wav2lip/wav2lip.pth",
        "size_mb": 350,
        "description": "Wav2Lip base model"
    },
    "wav2lip_gan": {
        "url": "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?download=1",
        "path": "models/wav2lip/wav2lip_gan.pth",
        "size_mb": 350,
        "description": "Wav2Lip GAN model (higher quality)"
    },
    "s3fd": {
        "url": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
        "path": "models/face_detection/s3fd.pth",
        "size_mb": 90,
        "description": "S3FD face detection model"
    },
    "gfpgan": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        "path": "models/gfpgan/GFPGANv1.3.pth",
        "size_mb": 350,
        "description": "GFPGAN face enhancement model (optional)"
    }
}


def download_file(url: str, destination: Path, description: str = "Downloading"):
    """Download a file with progress bar.

    Args:
        url: URL to download from
        destination: Path to save file
        description: Description for progress bar
    """
    # Create parent directory
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Download with streaming and progress bar
    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as f, tqdm(
        desc=description,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)


def verify_file_size(path: Path, expected_mb: int, tolerance: float = 0.1):
    """Verify file size is approximately correct.

    Args:
        path: Path to file
        expected_mb: Expected size in MB
        tolerance: Tolerance (0.1 = 10%)

    Returns:
        True if size is within tolerance
    """
    actual_mb = path.stat().st_size / (1024 * 1024)
    lower = expected_mb * (1 - tolerance)
    upper = expected_mb * (1 + tolerance)

    return lower <= actual_mb <= upper


def main():
    parser = argparse.ArgumentParser(description="Download Wav2Lip models")
    parser.add_argument(
        "--phase3",
        action="store_true",
        help="Download all Phase 3 models"
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Download specific model"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models including optional ones"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file exists"
    )
    parser.add_argument(
        "--skip-gfpgan",
        action="store_true",
        help="Skip optional GFPGAN model"
    )

    args = parser.parse_args()

    # Determine which models to download
    models_to_download = []

    if args.model:
        models_to_download = [args.model]
    elif args.all:
        models_to_download = list(MODELS.keys())
    elif args.phase3:
        # Download essential Phase 3 models (skip GFPGAN unless --all)
        models_to_download = ["wav2lip", "wav2lip_gan", "s3fd"]
        if not args.skip_gfpgan:
            models_to_download.append("gfpgan")
    else:
        parser.print_help()
        sys.exit(1)

    print("=" * 60)
    print("Phase 3 Model Download")
    print("=" * 60)

    # Download each model
    success_count = 0
    total_count = len(models_to_download)

    for model_name in models_to_download:
        model_info = MODELS[model_name]
        destination = Path(model_info["path"])

        print(f"\n[{success_count + 1}/{total_count}] {model_info['description']}")
        print(f"  Path: {destination}")
        print(f"  Size: ~{model_info['size_mb']} MB")

        # Check if already exists
        if destination.exists() and not args.force:
            # Verify size
            if verify_file_size(destination, model_info['size_mb']):
                print(f"  ✅ Already downloaded (verified)")
                success_count += 1
                continue
            else:
                print(f"  ⚠️  File exists but size mismatch, re-downloading...")

        # Download
        try:
            download_file(
                model_info["url"],
                destination,
                f"  Downloading {model_name}"
            )

            # Verify
            if verify_file_size(destination, model_info['size_mb']):
                print(f"  ✅ Downloaded and verified")
                success_count += 1
            else:
                print(f"  ⚠️  Downloaded but size mismatch")
                print(f"     Expected: ~{model_info['size_mb']} MB")
                actual_mb = destination.stat().st_size / (1024 * 1024)
                print(f"     Actual: {actual_mb:.1f} MB")

        except Exception as e:
            print(f"  ❌ Failed to download: {e}")
            continue

    print("\n" + "=" * 60)
    print(f"Downloaded {success_count}/{total_count} models successfully")
    print("=" * 60)

    if success_count == total_count:
        print("\n✅ All models downloaded successfully!")
        print("\nNext steps:")
        print("1. Prepare an avatar image: assets/avatars/default.jpg")
        print("2. Update config/config.yaml with your settings")
        print("3. Run: python main.py")
    else:
        print("\n⚠️  Some models failed to download.")
        print("Please try again or download manually.")
        print("See INSTALL_PHASE3.md for manual download instructions.")
        sys.exit(1)


if __name__ == "__main__":
    main()
