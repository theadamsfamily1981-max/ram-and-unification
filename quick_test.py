#!/usr/bin/env python3
"""
Quick Test - Generate talking avatar with ara3.png and audio file
"""

import asyncio
import sys
from pathlib import Path
import logging
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def find_files():
    """Find ara3.png and mp3 files."""
    repo_root = Path(__file__).parent

    # Find image
    image_candidates = [
        repo_root / "ara3.png",
        repo_root / "assets" / "avatars" / "ara3.png",
        repo_root / "assets" / "avatars" / "test_avatar.jpg",
    ]

    # Search for ara3.png
    for candidate in image_candidates:
        if candidate.exists():
            image_file = candidate
            break
    else:
        # Try find command
        try:
            result = subprocess.run(
                ["find", str(repo_root), "-name", "ara3.png"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.stdout.strip():
                image_file = Path(result.stdout.strip().split('\n')[0])
            else:
                image_file = None
        except:
            image_file = None

    # Find audio
    audio_candidates = [
        repo_root / "outputs" / "test_audio.wav",
    ]

    # Search for mp3
    try:
        result = subprocess.run(
            ["find", str(repo_root), "-name", "*.mp3", "-type", "f"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout.strip():
            mp3_file = Path(result.stdout.strip().split('\n')[0])
        else:
            mp3_file = None
    except:
        mp3_file = None

    # Check for existing wav
    audio_file = None
    for candidate in audio_candidates:
        if candidate.exists():
            audio_file = candidate
            break

    return image_file, audio_file, mp3_file


def setup_test_files(image_file, audio_file, mp3_file):
    """Set up test files in expected locations."""

    # Ensure directories exist
    (Path(__file__).parent / "assets" / "avatars").mkdir(parents=True, exist_ok=True)
    (Path(__file__).parent / "outputs").mkdir(parents=True, exist_ok=True)

    test_image = Path(__file__).parent / "assets" / "avatars" / "test_avatar.jpg"
    test_audio = Path(__file__).parent / "outputs" / "test_audio.wav"

    # Copy image
    if image_file and image_file.exists():
        logger.info(f"✅ Found image: {image_file}")
        if image_file != test_image:
            import shutil
            shutil.copy2(image_file, test_image)
            logger.info(f"   Copied to: {test_image}")
        image_ready = True
    else:
        logger.warning("⚠️  Image file (ara3.png) not found")
        logger.info("\nPlease place ara3.png in the repository directory:")
        logger.info("  cp /path/to/ara3.png .")
        image_ready = False

    # Handle audio
    if audio_file and audio_file.exists():
        logger.info(f"✅ Found audio: {audio_file}")
        audio_ready = True
    elif mp3_file and mp3_file.exists():
        logger.info(f"✅ Found MP3: {mp3_file}")
        logger.info("   Converting MP3 to WAV...")

        try:
            # Convert mp3 to wav
            result = subprocess.run([
                "ffmpeg", "-i", str(mp3_file),
                "-ar", "22050",  # 22050 Hz sample rate
                "-ac", "1",       # Mono
                str(test_audio),
                "-y"
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                logger.info(f"✅ Converted to: {test_audio}")
                audio_ready = True
            else:
                logger.error(f"❌ FFmpeg conversion failed: {result.stderr}")
                audio_ready = False
        except FileNotFoundError:
            logger.error("❌ FFmpeg not found")
            logger.info("Install with: sudo apt install ffmpeg")
            audio_ready = False
        except Exception as e:
            logger.error(f"❌ Conversion error: {e}")
            audio_ready = False
    else:
        logger.warning("⚠️  Audio file (.mp3) not found")
        logger.info("\nPlease place your mp3 file in the repository directory:")
        logger.info("  cp /path/to/audio.mp3 .")
        audio_ready = False

    return image_ready and audio_ready, test_image, test_audio


async def test_generation(image_path, audio_path):
    """Test avatar generation."""

    logger.info("\n" + "=" * 60)
    logger.info("Testing Avatar Generation")
    logger.info("=" * 60)

    try:
        # Import backend
        from multi_ai_workspace.src.integrations.ara_avatar_backend import AraAvatarBackend

        logger.info("Initializing Ara Avatar Backend...")
        backend = AraAvatarBackend()

        # Output path
        output_path = Path(__file__).parent / "outputs" / "test_generation.mp4"

        logger.info(f"\n📸 Image: {image_path}")
        logger.info(f"🎵 Audio: {audio_path}")
        logger.info(f"🎬 Output: {output_path}")
        logger.info("\n⏳ Generating avatar (this may take 2-3 minutes in CPU mode)...")
        logger.info("   Press Ctrl+C to cancel\n")

        # Generate
        result = await backend.generate_talking_avatar(
            text="Test generation from quick_test.py",
            avatar_image=str(image_path),
            audio_path=str(audio_path),
            output_path=str(output_path)
        )

        if result.get('success'):
            logger.info("\n" + "=" * 60)
            logger.info("✅ SUCCESS! Avatar generated!")
            logger.info("=" * 60)
            logger.info(f"\n📹 Video saved to: {output_path}")
            logger.info(f"⏱️  Duration: {result.get('duration_seconds', 'unknown')}s")
            logger.info(f"💾 File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

            logger.info("\n🎬 Play your video:")
            logger.info(f"   mpv {output_path}")
            logger.info(f"   vlc {output_path}")
            logger.info(f"   firefox {output_path}")

            return True
        else:
            logger.error(f"\n❌ Generation failed: {result.get('error', 'Unknown error')}")
            return False

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("\nMake sure you're in the virtual environment:")
        logger.info("  source venv/bin/activate")
        return False
    except KeyboardInterrupt:
        logger.info("\n\n✋ Cancelled by user")
        return False
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return False


async def main():
    """Main entry point."""

    logger.info("\n" + "=" * 60)
    logger.info("🤖 Ara Avatar System - Quick Test")
    logger.info("=" * 60)
    logger.info("")

    # Find files
    logger.info("🔍 Looking for test files...")
    image_file, audio_file, mp3_file = find_files()

    # Setup
    ready, test_image, test_audio = setup_test_files(image_file, audio_file, mp3_file)

    if not ready:
        logger.info("\n" + "=" * 60)
        logger.info("⚠️  Test files not ready")
        logger.info("=" * 60)
        logger.info("\nOnce you have the files in place, run:")
        logger.info("  python quick_test.py")
        return

    # Generate
    success = await test_generation(test_image, test_audio)

    if success:
        logger.info("\n✅ All tests passed!")
        logger.info("\nNext steps:")
        logger.info("  1. Enable caching in config/avatar_config.yaml")
        logger.info("  2. Start API server: python run_ara.py --mode api")
        logger.info("  3. Test with different images/audio")
    else:
        logger.info("\n⚠️  Generation failed - see errors above")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n✋ Interrupted by user")
        sys.exit(0)
