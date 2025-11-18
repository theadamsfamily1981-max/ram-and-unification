#!/usr/bin/env python3
"""Test video playback functionality."""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.player import MediaPlayer
from src.utils.logger import setup_logger

setup_logger("INFO")


def main():
    parser = argparse.ArgumentParser(description="Test video playback")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--fullscreen", action="store_true", help="Play in fullscreen")
    parser.add_argument("--engine", choices=["opencv", "ffplay", "auto"], default="auto",
                       help="Playback engine")

    args = parser.parse_args()

    video_path = Path(args.video)

    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        sys.exit(1)

    print("=" * 60)
    print("Video Playback Test")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Engine: {args.engine}")
    print(f"Fullscreen: {args.fullscreen}")
    print("\nPress ESC to stop playback\n")

    try:
        player = MediaPlayer(engine=args.engine, fullscreen=args.fullscreen)
        player.play_video(video_path)
        print("\n✅ Playback complete")

    except Exception as e:
        print(f"\n❌ Playback failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
