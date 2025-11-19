#!/usr/bin/env python3
"""
Demo: Generate talking avatar video from text input.

This script demonstrates the talking avatar pipeline without requiring
microphone input or LLM. It:
1. Takes text input
2. Generates speech with TTS
3. Creates talking head video
4. Plays the result

Usage:
    python scripts/demo_talking_avatar_from_text.py \
        --text "Hello, this is a demonstration of the talking avatar system." \
        --output outputs/demos/demo.mp4
"""

import sys
import argparse
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tts import CoquiTTS
from src.talking_head import Wav2LipTalkingHead
from src.player import MediaPlayer
from src.utils.logger import setup_logger, get_logger

setup_logger("INFO")
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate talking avatar from text")
    parser.add_argument("--text", required=True, help="Text to speak")
    parser.add_argument("--output", default="outputs/demos/demo.mp4", help="Output video path")
    parser.add_argument("--avatar", help="Path to avatar image (overrides config)")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--quality", choices=["standard", "high"], help="Quality mode (overrides config)")
    parser.add_argument("--play", action="store_true", help="Play video after generation")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep intermediate audio file")

    args = parser.parse_args()

    print("=" * 60)
    print("Talking Avatar Demo - Text to Video")
    print("=" * 60)
    print(f"\nText: {args.text}")
    print(f"Output: {args.output}\n")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    try:
        # 1. Initialize TTS
        logger.info("Initializing TTS...")
        print("🔊 Initializing TTS...")

        tts_cfg = config['tts']['coqui']
        device = config['system']['device']

        tts = CoquiTTS(
            model_name=tts_cfg['model'],
            device=device,
            speaker_wav=tts_cfg.get('speaker_wav'),
            language=tts_cfg.get('language', 'en')
        )

        # 2. Generate speech
        logger.info("Generating speech...")
        print("🗣️  Generating speech...")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio_path = output_path.parent / f"{output_path.stem}.wav"
        tts.synthesize(text=args.text, output_path=audio_path)

        print(f"✅ Audio generated: {audio_path}")

        # 3. Initialize talking head
        logger.info("Initializing talking head...")
        print("🎭 Initializing talking head...")

        talking_head_cfg = config['talking_head']
        quality_mode = args.quality or talking_head_cfg.get('quality_mode', 'standard')
        quality_cfg = talking_head_cfg.get(quality_mode, {})

        # Get avatar image
        avatar_image = args.avatar or talking_head_cfg.get('avatar_image')
        if not Path(avatar_image).exists():
            print(f"❌ Avatar image not found: {avatar_image}")
            print("Please provide a valid avatar image path with --avatar")
            sys.exit(1)

        # Get model path
        models_cfg = talking_head_cfg.get('models', {})
        model_name = quality_cfg.get('model', 'wav2lip')
        model_path = models_cfg.get(model_name)

        talking_head = Wav2LipTalkingHead(
            avatar_image_path=avatar_image,
            device=talking_head_cfg.get('device', device),
            quality_mode=quality_mode,
            model_path=model_path if Path(model_path).exists() else None,
            face_det_batch_size=quality_cfg.get('face_det_batch_size', 4),
            wav2lip_batch_size=quality_cfg.get('wav2lip_batch_size', 128)
        )

        info = talking_head.get_model_info()
        print(f"✅ Talking head ready ({info['quality_mode']} mode, {info['output_resolution']})")

        # 4. Generate video
        logger.info("Generating talking head video...")
        print("🎬 Generating video...")
        print("   (This may take a few seconds...)")

        import time
        start_time = time.time()

        video_path = talking_head.generate(
            audio_path=audio_path,
            output_path=output_path
        )

        elapsed = time.time() - start_time
        print(f"✅ Video generated in {elapsed:.1f}s: {video_path}")

        # 5. Clean up audio if requested
        if not args.no_cleanup:
            audio_path.unlink()
            logger.info("Intermediate audio file deleted")

        # 6. Play video if requested
        if args.play:
            print("\n📺 Playing video...")
            print("   (Press ESC to stop)\n")

            player = MediaPlayer(engine="auto")
            player.play_video(video_path)

        print("\n" + "=" * 60)
        print("✅ Demo complete!")
        print("=" * 60)
        print(f"\nGenerated video: {video_path}")
        print(f"Duration: {elapsed:.1f}s")
        print(f"\nTo play: python scripts/play_video_test.py --video {video_path}")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
