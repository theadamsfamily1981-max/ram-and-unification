#!/usr/bin/env python3
"""
Simple test script for Ara Avatar API
Tests the complete pipeline: upload image → upload audio → generate avatar
"""

import asyncio
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_api_programmatically():
    """Test the API by directly calling the backend (no HTTP server needed)."""

    logger.info("=" * 60)
    logger.info("Testing Ara Avatar Generation Pipeline")
    logger.info("=" * 60)

    # Check for test files
    test_image = Path("assets/avatars/test_avatar.jpg")
    test_audio = Path("outputs/test_audio.wav")

    if not test_image.exists():
        logger.warning(f"⚠️  Test image not found: {test_image}")
        logger.info("\nTo test avatar generation, you need:")
        logger.info("  1. A portrait image at: assets/avatars/test_avatar.jpg")
        logger.info("     (Any front-facing photo, AI-generated or real)")
        logger.info("  2. An audio file at: outputs/test_audio.wav")
        logger.info("     (TTS output or recorded speech)")
        logger.info("\nYou can:")
        logger.info("  - Copy any existing avatar image to test_avatar.jpg")
        logger.info("  - Generate TTS audio with your text-to-speech system")
        logger.info("  - Or use the API server to upload files")
        return False

    if not test_audio.exists():
        logger.warning(f"⚠️  Test audio not found: {test_audio}")
        logger.info("\nCreating a simple test audio file...")

        # Try to create a simple beep for testing
        try:
            import numpy as np
            import soundfile as sf

            # Generate 2-second 440Hz tone (A note)
            sample_rate = 22050
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * 440 * t) * 0.3

            test_audio.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(test_audio), audio, sample_rate)

            logger.info(f"✅ Created test audio: {test_audio}")
        except ImportError:
            logger.warning("⚠️  Cannot create test audio (missing soundfile)")
            logger.info("Install with: pip install soundfile numpy")
            return False

    # Test the avatar generation
    try:
        logger.info("\n" + "=" * 60)
        logger.info("Testing Avatar Generation Backend")
        logger.info("=" * 60)

        # Import avatar backend
        from multi_ai_workspace.src.integrations.ara_avatar_backend import AraAvatarBackend

        # Initialize backend
        logger.info("Initializing Ara Avatar Backend...")
        backend = AraAvatarBackend()

        # Health check
        logger.info("Running health check...")
        healthy = await backend.health_check()

        if healthy:
            logger.info("✅ Backend health check passed")
        else:
            logger.warning("⚠️  Backend health check failed")
            logger.info("Note: This may be expected if Ollama isn't running")

        # Try generating avatar
        logger.info("\nGenerating talking avatar...")
        logger.info(f"  Image: {test_image}")
        logger.info(f"  Audio: {test_audio}")

        output_path = Path("outputs/test_generation.mp4")

        result = await backend.generate_talking_avatar(
            text="Test generation",  # Not used directly, but required param
            avatar_image=str(test_image),
            audio_path=str(test_audio),
            output_path=str(output_path)
        )

        if result.get('success'):
            logger.info(f"\n✅ Avatar generated successfully!")
            logger.info(f"   Output: {result.get('output_path', output_path)}")
            logger.info(f"   Duration: {result.get('duration_seconds', 'unknown')}s")
            logger.info("\nYou can play it with:")
            logger.info(f"   mpv {output_path}")
            logger.info(f"   vlc {output_path}")
            return True
        else:
            logger.error(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
            return False

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("\nMake sure all dependencies are installed:")
        logger.info("  pip install -r requirements.txt")
        return False
    except Exception as e:
        logger.error(f"❌ Generation error: {e}", exc_info=True)
        return False


async def test_api_server():
    """Test the API server (if running) via HTTP."""

    logger.info("\n" + "=" * 60)
    logger.info("Testing API Server (HTTP)")
    logger.info("=" * 60)

    try:
        import httpx

        # Check if server is running
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get("http://localhost:8000/health", timeout=2.0)

                if response.status_code == 200:
                    logger.info("✅ API server is running")
                    logger.info(f"   Status: {response.json()}")

                    # Try detailed health check
                    try:
                        detail_response = await client.get("http://localhost:8000/health/detailed", timeout=2.0)
                        if detail_response.status_code == 200:
                            detail = detail_response.json()
                            logger.info("\n📊 Detailed Status:")
                            logger.info(f"   Device: {detail.get('device', {}).get('optimal_device', 'unknown')}")
                            logger.info(f"   CUDA: {detail.get('device', {}).get('cuda_available', False)}")

                            if 'cache' in detail:
                                cache = detail['cache']
                                logger.info(f"   Cache entries: {cache.get('total_entries', 0)}")
                                logger.info(f"   Cache hit rate: {cache.get('hit_rate_percent', 0):.1f}%")
                    except:
                        pass

                    logger.info("\n📚 API Endpoints:")
                    logger.info("   GET  /health - Basic health check")
                    logger.info("   GET  /health/detailed - Full system status")
                    logger.info("   POST /upload/image - Upload avatar image")
                    logger.info("   POST /upload/audio - Upload audio file")
                    logger.info("   POST /generate/async - Generate avatar (async)")
                    logger.info("   GET  /status/{job_id} - Check job status")
                    logger.info("   WS   /ws/progress/{job_id} - Real-time progress")

                    return True
                else:
                    logger.warning(f"⚠️  API server returned {response.status_code}")
                    return False

            except httpx.ConnectError:
                logger.info("ℹ️  API server not running")
                logger.info("\nTo start the API server:")
                logger.info("  python run_ara.py --mode api")
                logger.info("  # Or interactively:")
                logger.info("  python run_ara.py")
                logger.info("  # Then select option 5")
                return False

    except ImportError:
        logger.warning("⚠️  httpx not installed (optional for HTTP testing)")
        logger.info("Install with: pip install httpx")
        return False


async def main():
    """Run all tests."""

    logger.info("\n🤖 Ara Avatar System - Test Suite\n")

    # Test 1: Programmatic backend test
    backend_ok = await test_api_programmatically()

    # Test 2: HTTP API test (if server running)
    server_ok = await test_api_server()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    logger.info(f"Backend Test: {'✅ PASS' if backend_ok else '❌ FAIL (see above for setup)'}")
    logger.info(f"API Server Test: {'✅ PASS' if server_ok else 'ℹ️  NOT RUNNING (start with run_ara.py)'}")

    if backend_ok or server_ok:
        logger.info("\n✅ System is operational!")
        logger.info("\nNext steps:")
        logger.info("  1. Start API server: python run_ara.py --mode api")
        logger.info("  2. Test with real image/audio files")
        logger.info("  3. Check cache performance with repeated requests")
        logger.info("  4. (Optional) Train RVC voice model - see docs/RVC_VOICE_SETUP.md")
    else:
        logger.info("\n⚠️  Setup needed - see messages above")

    logger.info("")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n✋ Interrupted by user")
        sys.exit(0)
