"""Text-to-speech using Coqui TTS."""

import torch
from TTS.api import TTS
from pathlib import Path
from typing import Optional
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class CoquiTTS:
    """Text-to-speech engine using Coqui TTS."""

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "cuda",
        speaker_wav: Optional[str] = None,
        language: str = "en"
    ):
        """Initialize Coqui TTS.

        Args:
            model_name: TTS model name
            device: Device to use (cuda, cpu, mps)
            speaker_wav: Path to speaker audio for voice cloning
            language: Language code
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.speaker_wav = speaker_wav
        self.language = language

        logger.info(f"Loading TTS model '{model_name}' on {self.device}...")

        try:
            self.tts = TTS(model_name).to(self.device)
            logger.info("TTS model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise

    def synthesize(
        self,
        text: str,
        output_path: Optional[Path] = None,
        speaker_wav: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0
    ) -> Path:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize
            output_path: Output audio file path
            speaker_wav: Speaker audio for voice cloning (overrides default)
            language: Language code (overrides default)
            speed: Speech speed multiplier

        Returns:
            Path to generated audio file
        """
        if not text or text.strip() == "":
            logger.warning("Empty text provided for synthesis")
            return None

        # Use defaults if not specified
        speaker = speaker_wav or self.speaker_wav
        lang = language or self.language

        # Generate output path if not provided
        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"outputs/audio/tts_{timestamp}.wav")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Synthesizing: '{text[:50]}...'")

            # Check if model supports voice cloning
            if "xtts" in self.model_name.lower():
                # XTTS model - supports voice cloning
                self.tts.tts_to_file(
                    text=text,
                    file_path=str(output_path),
                    speaker_wav=speaker,
                    language=lang,
                    speed=speed
                )
            else:
                # Standard model - no voice cloning
                self.tts.tts_to_file(
                    text=text,
                    file_path=str(output_path),
                    speaker=speaker,  # May be ignored by some models
                    language=lang
                )

            logger.info(f"Audio synthesized successfully: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise

    def synthesize_to_array(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        language: Optional[str] = None
    ) -> np.ndarray:
        """Synthesize speech and return as numpy array.

        Args:
            text: Text to synthesize
            speaker_wav: Speaker audio for voice cloning
            language: Language code

        Returns:
            Audio as numpy array
        """
        speaker = speaker_wav or self.speaker_wav
        lang = language or self.language

        try:
            logger.info(f"Synthesizing to array: '{text[:50]}...'")

            if "xtts" in self.model_name.lower():
                audio = self.tts.tts(
                    text=text,
                    speaker_wav=speaker,
                    language=lang
                )
            else:
                audio = self.tts.tts(
                    text=text,
                    speaker=speaker,
                    language=lang
                )

            return np.array(audio)

        except Exception as e:
            logger.error(f"TTS synthesis to array failed: {e}")
            raise

    def list_languages(self) -> list:
        """List supported languages.

        Returns:
            List of language codes
        """
        try:
            if hasattr(self.tts, 'languages'):
                return self.tts.languages
            else:
                return ["en"]
        except:
            return ["en"]

    def list_speakers(self) -> list:
        """List available speakers (if model supports it).

        Returns:
            List of speaker names
        """
        try:
            if hasattr(self.tts, 'speakers'):
                return self.tts.speakers
            else:
                return []
        except:
            return []

    def clone_voice(
        self,
        text: str,
        voice_sample: str,
        output_path: Optional[Path] = None,
        language: str = "en"
    ) -> Path:
        """Clone a voice and synthesize speech.

        Args:
            text: Text to synthesize
            voice_sample: Path to voice sample audio (3-10 seconds recommended)
            output_path: Output path
            language: Language code

        Returns:
            Path to generated audio
        """
        logger.info(f"Cloning voice from {voice_sample}")

        return self.synthesize(
            text=text,
            output_path=output_path,
            speaker_wav=voice_sample,
            language=language
        )

    @staticmethod
    def list_available_models() -> list:
        """List all available TTS models.

        Returns:
            List of model names
        """
        return TTS.list_models()
