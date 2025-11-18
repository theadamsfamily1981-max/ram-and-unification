"""Minimal prototype orchestrator for voice conversation."""

import yaml
from pathlib import Path
from typing import Optional
import time

from ..audio_input import AudioRecorder, VoiceActivityDetector
from ..asr import WhisperASR
from ..dialogue import DialogueManager
from ..tts import CoquiTTS
from ..utils.logger import get_logger, setup_logger

logger = get_logger(__name__)


class VoiceAssistantPrototype:
    """Minimal voice assistant prototype (audio only)."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the voice assistant.

        Args:
            config_path: Path to config file (default: config/config.yaml)
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Setup logging
        log_level = self.config['system'].get('log_level', 'INFO')
        setup_logger(log_level)

        logger.info("=" * 60)
        logger.info("Voice Assistant Prototype Starting")
        logger.info("=" * 60)

        # Initialize components
        self._init_components()

        self.is_running = False
        self.turn_count = 0

    def _init_components(self):
        """Initialize all components."""
        device = self.config['system']['device']

        # Audio Input
        logger.info("Initializing audio input...")
        audio_cfg = self.config['audio_input']
        self.recorder = AudioRecorder(
            sample_rate=audio_cfg['sample_rate'],
            channels=audio_cfg['channels'],
            silence_duration=audio_cfg['silence_duration']
        )

        # VAD
        if audio_cfg.get('vad_enabled', True):
            self.vad = VoiceActivityDetector(
                sample_rate=audio_cfg['sample_rate'],
                aggressiveness=audio_cfg.get('vad_aggressiveness', 2)
            )
        else:
            self.vad = None

        # ASR
        logger.info("Initializing ASR...")
        asr_cfg = self.config['asr']
        self.asr = WhisperASR(
            model_name=asr_cfg['model'],
            device=device,
            language=asr_cfg.get('language'),
            fp16=asr_cfg.get('fp16', True)
        )

        # Dialogue Manager
        logger.info("Initializing dialogue manager...")
        dialogue_cfg = self.config['dialogue']
        engine = dialogue_cfg['engine']
        engine_cfg = dialogue_cfg.get(engine, {})

        self.dialogue = DialogueManager(
            engine=engine,
            model=engine_cfg.get('model', 'llama3.2'),
            system_prompt=dialogue_cfg.get('system_prompt'),
            max_history=dialogue_cfg.get('memory', {}).get('max_history', 10),
            **engine_cfg
        )

        # TTS
        logger.info("Initializing TTS...")
        tts_cfg = self.config['tts']
        engine = tts_cfg['engine']

        if engine == 'coqui':
            coqui_cfg = tts_cfg['coqui']
            self.tts = CoquiTTS(
                model_name=coqui_cfg['model'],
                device=device,
                speaker_wav=coqui_cfg.get('speaker_wav'),
                language=coqui_cfg.get('language', 'en')
            )
        else:
            logger.warning(f"TTS engine '{engine}' not fully implemented, using coqui")
            self.tts = CoquiTTS(device=device)

        logger.info("All components initialized successfully")

    def run_conversation_turn(self) -> bool:
        """Run a single conversation turn.

        Returns:
            True if turn completed successfully, False otherwise
        """
        self.turn_count += 1
        logger.info("=" * 60)
        logger.info(f"TURN {self.turn_count}")
        logger.info("=" * 60)

        try:
            # 1. Record audio
            logger.info("Listening... (speak now, pause when done)")
            print("\n🎤 Listening... (speak now)")

            vad_callback = self.vad.is_speech if self.vad else None
            audio_data = self.recorder.record_until_silence(
                vad_callback=vad_callback,
                max_duration=30.0
            )

            if audio_data is None or len(audio_data) == 0:
                logger.warning("No audio recorded")
                print("⚠️  No audio detected. Please try again.\n")
                return False

            # Save recording if configured
            if self.config['privacy'].get('save_recordings', False):
                save_path = Path(f"outputs/recordings/turn_{self.turn_count:03d}.wav")
                self.recorder.save_audio(audio_data, save_path)

            # 2. Transcribe
            logger.info("Transcribing...")
            print("🔄 Transcribing...")

            start_time = time.time()
            result = self.asr.transcribe(audio_data)
            asr_time = time.time() - start_time

            user_text = result['text']
            if not user_text:
                logger.warning("No text transcribed")
                print("⚠️  Could not understand audio. Please try again.\n")
                return False

            print(f"👤 You: {user_text}")
            logger.info(f"ASR Time: {asr_time:.2f}s")

            # Check for exit commands
            if user_text.lower() in ['exit', 'quit', 'goodbye', 'bye']:
                print("\n👋 Goodbye!")
                return False

            # 3. Get LLM response
            logger.info("Generating response...")
            print("🤔 Thinking...")

            start_time = time.time()
            assistant_text = self.dialogue.get_response(user_text)
            llm_time = time.time() - start_time

            print(f"🤖 Assistant: {assistant_text}")
            logger.info(f"LLM Time: {llm_time:.2f}s")

            # 4. Synthesize speech
            logger.info("Synthesizing speech...")
            print("🔊 Speaking...")

            start_time = time.time()
            audio_path = self.tts.synthesize(
                text=assistant_text,
                output_path=Path(f"outputs/audio/turn_{self.turn_count:03d}.wav")
            )
            tts_time = time.time() - start_time

            logger.info(f"TTS Time: {tts_time:.2f}s")

            # 5. Play audio
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            self.recorder.play_audio(audio)

            # Print timing summary
            total_time = asr_time + llm_time + tts_time
            print(f"\n⏱️  Timing: ASR={asr_time:.1f}s | LLM={llm_time:.1f}s | TTS={tts_time:.1f}s | Total={total_time:.1f}s\n")

            return True

        except KeyboardInterrupt:
            logger.info("Turn interrupted by user")
            return False

        except Exception as e:
            logger.error(f"Error in conversation turn: {e}", exc_info=True)
            print(f"\n❌ Error: {e}\n")
            return False

    def run(self):
        """Run the voice assistant main loop."""
        self.is_running = True

        print("\n" + "=" * 60)
        print("🎭 VOICE ASSISTANT PROTOTYPE")
        print("=" * 60)
        print("\nPress Ctrl+C to exit")
        print("Say 'exit', 'quit', or 'goodbye' to end conversation\n")

        try:
            while self.is_running:
                success = self.run_conversation_turn()

                if not success:
                    # Ask if user wants to continue
                    try:
                        response = input("\n▶️  Continue? (y/n): ").strip().lower()
                        if response not in ['y', 'yes', '']:
                            break
                    except KeyboardInterrupt:
                        break

                # Check max turns
                max_turns = self.config['system'].get('max_conversation_turns', 50)
                if self.turn_count >= max_turns:
                    logger.info(f"Reached max conversation turns ({max_turns})")
                    break

        except KeyboardInterrupt:
            logger.info("Assistant stopped by user")

        finally:
            self.shutdown()

    def shutdown(self):
        """Clean up and shut down."""
        logger.info("Shutting down voice assistant...")

        # Save conversation if configured
        if self.config['privacy'].get('save_conversations', False):
            output_path = Path(f"outputs/conversations/conversation_{int(time.time())}.txt")
            self.dialogue.export_conversation(output_path)

        # Cleanup temp files if configured
        if self.config['privacy'].get('auto_cleanup', True):
            logger.info("Cleaning up temporary files...")
            # Add cleanup logic here

        self.is_running = False
        logger.info("Voice assistant shut down complete")
        print("\n" + "=" * 60)
        print("👋 Thank you for using Voice Assistant Prototype!")
        print("=" * 60 + "\n")
