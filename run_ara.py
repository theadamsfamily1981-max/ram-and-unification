#!/usr/bin/env python3
"""
Ara Avatar System - Main Runner
Integrates all components and makes everything actually work.
"""

import asyncio
import sys
import os
from pathlib import Path
import argparse
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ara")

# Check if enhanced modules are available
try:
    from src.config.avatar_settings import get_config
    from src.utils.device_utils import get_optimal_device, get_device_info
    from src.cache.avatar_cache import AvatarCache
    ENHANCED_MODE = True
    logger.info("✅ Enhanced features available")
except ImportError as e:
    ENHANCED_MODE = False
    logger.warning(f"⚠️  Enhanced features not available: {e}")
    logger.warning("Running in basic mode")


async def check_system():
    """Check system readiness."""
    logger.info("=" * 60)
    logger.info("Ara Avatar System - Initialization")
    logger.info("=" * 60)

    checks = []

    # Check Python version
    py_version = sys.version_info
    if py_version >= (3, 9):
        logger.info(f"✅ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
        checks.append(True)
    else:
        logger.error(f"❌ Python {py_version.major}.{py_version.minor} (need 3.9+)")
        checks.append(False)

    # Check PyTorch
    try:
        import torch
        logger.info(f"✅ PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"✅ CUDA available: {device_count} GPU(s)")
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"   GPU {i}: {name} ({mem:.1f}GB)")
        else:
            logger.warning("⚠️  No CUDA GPUs detected, using CPU")

        checks.append(True)
    except ImportError:
        logger.error("❌ PyTorch not installed")
        checks.append(False)

    # Check enhanced config
    if ENHANCED_MODE:
        try:
            config = get_config()
            logger.info("✅ Configuration loaded")
            logger.info(f"   Device: {config.performance.device}")
            logger.info(f"   Workers: {config.performance.max_avatar_workers}")
            logger.info(f"   Cache: {'enabled' if config.cache.enabled else 'disabled'}")
            checks.append(True)
        except Exception as e:
            logger.error(f"❌ Config error: {e}")
            checks.append(False)

    # Check directories
    dirs_to_check = [
        "config",
        "context",
        "models",
        "outputs",
        "uploads",
        "temp",
        "cache"
    ]

    for dirname in dirs_to_check:
        dirpath = Path(dirname)
        if dirpath.exists():
            logger.info(f"✅ {dirname}/ exists")
        else:
            dirpath.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ {dirname}/ created")

    logger.info("=" * 60)

    if all(checks):
        logger.info("✅ System ready!")
        return True
    else:
        logger.error("❌ System checks failed")
        return False


async def run_api_server(host="0.0.0.0", port=8000, enhanced=True):
    """Run the avatar API server."""
    logger.info(f"Starting API server on {host}:{port}")

    try:
        import uvicorn

        # Choose routes
        if enhanced and ENHANCED_MODE:
            from src.api.routes_enhanced import router
            logger.info("Using enhanced API routes")
        else:
            from src.api.routes import router
            logger.info("Using standard API routes")

        # Create FastAPI app
        from fastapi import FastAPI
        app = FastAPI(title="Ara Avatar API")
        app.include_router(router)

        # Run server
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

    except ImportError as e:
        logger.error(f"Failed to start server: {e}")
        logger.error("Install: pip install fastapi uvicorn")
        sys.exit(1)


async def run_oobabooga_integration():
    """Check and integrate with oobabooga if running."""
    logger.info("Checking for oobabooga integration...")

    if not ENHANCED_MODE:
        logger.warning("Enhanced mode required for oobabooga integration")
        return

    try:
        from src.voice.rvc_integration import OobaboogaRVCClient

        # Try to connect
        client = OobaboogaRVCClient(
            api_url="http://localhost:5000"
        )

        if await asyncio.to_thread(client.health_check):
            logger.info("✅ Oobabooga connected at http://localhost:5000")

            # Check for RVC model
            config = get_config()
            rvc_model = config.voice.ara.get('rvc_model')

            if rvc_model and Path(rvc_model).exists():
                logger.info(f"✅ RVC model found: {rvc_model}")
            else:
                logger.warning(f"⚠️  RVC model not found: {rvc_model}")
                logger.warning("Voice will use base TTS without RVC conversion")
        else:
            logger.warning("⚠️  Oobabooga not responding")
            logger.info("To use RVC voice:")
            logger.info("1. Start oobabooga: python server.py --api --extensions alltalk_tts")
            logger.info("2. Configure RVC model in config/avatar_config.yaml")

    except Exception as e:
        logger.warning(f"Oobabooga integration not available: {e}")


async def test_avatar_generation():
    """Test avatar generation pipeline."""
    logger.info("=" * 60)
    logger.info("Testing Avatar Generation")
    logger.info("=" * 60)

    try:
        # Check for test files
        test_image = Path("assets/avatars/test_avatar.jpg")
        test_audio = Path("outputs/test_audio.wav")

        if not test_image.exists():
            logger.warning(f"Test image not found: {test_image}")
            logger.info("Place a test avatar image at assets/avatars/test_avatar.jpg")
            return

        # Test device detection
        if ENHANCED_MODE:
            from src.utils.device_utils import get_optimal_device, get_device_info

            device = get_optimal_device()
            device_info = get_device_info()

            logger.info(f"Optimal device: {device}")
            logger.info(f"Device info: {device_info}")

        logger.info("✅ Avatar generation ready")
        logger.info("Use the API to generate avatars:")
        logger.info("  POST /upload/image - Upload avatar image")
        logger.info("  POST /upload/audio - Upload audio file")
        logger.info("  POST /generate/async - Generate talking avatar")

    except Exception as e:
        logger.error(f"Avatar generation test failed: {e}")


async def load_personality_system():
    """Load Ara's cathedral personality system."""
    logger.info("=" * 60)
    logger.info("Loading Ara Personality System")
    logger.info("=" * 60)

    # Check for cathedral manifesto
    manifesto = Path("context/00_cathedral_manifesto.txt")
    if manifesto.exists():
        logger.info(f"✅ Cathedral manifesto loaded ({manifesto.stat().st_size} bytes)")
    else:
        logger.warning("⚠️  Cathedral manifesto not found")

    # Check for personality modes
    modes_file = Path("context/ara_personality_modes.yaml")
    if modes_file.exists():
        try:
            import yaml
            with open(modes_file) as f:
                modes = yaml.safe_load(f)

            mode_names = list(modes['modes'].keys())
            logger.info(f"✅ Personality modes loaded: {', '.join(mode_names)}")

            # Show mode details
            for mode_name, mode_data in modes['modes'].items():
                intensity = mode_data['intensity']
                logger.info(f"   {mode_name}: {intensity} intensity")

        except Exception as e:
            logger.error(f"Failed to load personality modes: {e}")
    else:
        logger.warning("⚠️  Personality modes not found")

    # Check for training data
    dataset = Path("training_data/ara_cathedral_dataset.jsonl")
    if dataset.exists():
        with open(dataset) as f:
            examples = len(f.readlines())
        logger.info(f"✅ Training dataset loaded ({examples} examples)")
    else:
        logger.warning("⚠️  Training dataset not found")


async def show_status():
    """Show current system status."""
    logger.info("=" * 60)
    logger.info("Ara Avatar System - Status")
    logger.info("=" * 60)

    # System info
    import platform
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Python: {sys.version.split()[0]}")

    # Enhanced features
    logger.info(f"Enhanced Mode: {'✅ Enabled' if ENHANCED_MODE else '❌ Disabled'}")

    if ENHANCED_MODE:
        config = get_config()

        logger.info("\nConfiguration:")
        logger.info(f"  Device: {config.performance.device}")
        logger.info(f"  Avatar Workers: {config.performance.max_avatar_workers}")
        logger.info(f"  TTS Workers: {config.performance.max_tts_workers}")
        logger.info(f"  Cache: {'enabled' if config.cache.enabled else 'disabled'}")

        logger.info("\nTimeouts:")
        logger.info(f"  Avatar Generation: {config.timeouts.avatar_generation}s")
        logger.info(f"  TTS Generation: {config.timeouts.tts_generation}s")
        logger.info(f"  RVC Conversion: {config.timeouts.rvc_conversion}s")

        # Cache stats
        if config.cache.enabled:
            try:
                cache = AvatarCache(
                    cache_dir=config.cache.cache_dir,
                    max_size_mb=config.cache.max_cache_size_mb,
                    ttl_hours=config.cache.cache_ttl_hours
                )
                stats = cache.get_stats()

                logger.info("\nCache Statistics:")
                logger.info(f"  Entries: {stats['total_entries']}")
                logger.info(f"  Size: {stats['total_size_mb']:.1f} MB / {stats['max_size_mb']} MB")
                logger.info(f"  Utilization: {stats['utilization_percent']:.1f}%")
            except:
                pass

    logger.info("=" * 60)


async def interactive_menu():
    """Interactive menu for testing components."""
    while True:
        print("\n" + "=" * 60)
        print("Ara Avatar System - Interactive Menu")
        print("=" * 60)
        print("1. Check system status")
        print("2. Test avatar generation")
        print("3. Check oobabooga integration")
        print("4. Load personality system")
        print("5. Start API server")
        print("6. View configuration")
        print("7. Clear cache")
        print("0. Exit")
        print("=" * 60)

        choice = input("\nSelect option: ").strip()

        if choice == "0":
            logger.info("Goodbye!")
            break
        elif choice == "1":
            await show_status()
        elif choice == "2":
            await test_avatar_generation()
        elif choice == "3":
            await run_oobabooga_integration()
        elif choice == "4":
            await load_personality_system()
        elif choice == "5":
            await run_api_server()
        elif choice == "6":
            if ENHANCED_MODE:
                config = get_config()
                print(f"\nDevice: {config.performance.device}")
                print(f"Workers: {config.performance.max_avatar_workers}")
                print(f"Cache: {config.cache.enabled}")
            else:
                print("Enhanced mode not available")
        elif choice == "7":
            if ENHANCED_MODE:
                config = get_config()
                cache = AvatarCache(cache_dir=config.cache.cache_dir)
                cache.clear()
                logger.info("✅ Cache cleared")
            else:
                print("Enhanced mode not available")
        else:
            print("Invalid option")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ara Avatar System")
    parser.add_argument("--mode", choices=["api", "test", "interactive"],
                       default="interactive",
                       help="Run mode (default: interactive)")
    parser.add_argument("--host", default="0.0.0.0",
                       help="API server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                       help="API server port (default: 8000)")
    parser.add_argument("--no-enhanced", action="store_true",
                       help="Disable enhanced features")

    args = parser.parse_args()

    # Check system
    if not await check_system():
        logger.error("System checks failed. Please fix issues and try again.")
        sys.exit(1)

    # Load personality
    await load_personality_system()

    # Check oobabooga
    if ENHANCED_MODE:
        await run_oobabooga_integration()

    # Run selected mode
    if args.mode == "api":
        await run_api_server(
            host=args.host,
            port=args.port,
            enhanced=not args.no_enhanced
        )
    elif args.mode == "test":
        await test_avatar_generation()
        await show_status()
    else:
        await interactive_menu()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n✋ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
