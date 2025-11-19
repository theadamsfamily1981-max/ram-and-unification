#!/usr/bin/env python3
"""CLI tool for managing models."""

import argparse
from pathlib import Path
from src.models import ModelManager
from src.config import settings


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Manage talking avatar models")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument(
        "model",
        choices=["wav2lip", "wav2lip_gan", "all"],
        help="Model to download"
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if exists"
    )

    # List command
    subparsers.add_parser("list", help="List available models")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a model")
    delete_parser.add_argument("model", help="Model to delete")

    args = parser.parse_args()

    # Initialize model manager
    manager = ModelManager(settings.model_cache_dir)

    if args.command == "download":
        if args.model == "all":
            for model_name in ["wav2lip", "wav2lip_gan"]:
                try:
                    manager.download_model(model_name, force=args.force)
                except Exception as e:
                    print(f"Error downloading {model_name}: {e}")
        else:
            manager.download_model(args.model, force=args.force)

    elif args.command == "list":
        status = manager.list_models()
        print("\nAvailable Models:")
        print("-" * 60)
        for name, info in status.items():
            status_str = "✓ Downloaded" if info["exists"] else "✗ Not downloaded"
            print(f"{name:20} {status_str}")
            if info["exists"]:
                print(f"{'':20} Path: {info['path']}")
        print("-" * 60)

    elif args.command == "delete":
        manager.delete_model(args.model)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
