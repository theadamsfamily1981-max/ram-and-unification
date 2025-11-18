"""Speech-to-text using OpenAI Whisper."""

import whisper
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict

from ..utils.logger import get_logger

logger = get_logger(__name__)


class WhisperASR:
    """Automatic Speech Recognition using Whisper."""

    def __init__(
        self,
        model_name: str = "small",
        device: str = "cuda",
        language: Optional[str] = "en",
        fp16: bool = True
    ):
        """Initialize Whisper ASR.

        Args:
            model_name: Model size (tiny, base, small, medium, large)
            device: Device to use (cuda, cpu, mps)
            language: Language code (None for auto-detect)
            fp16: Use FP16 precision for faster inference
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.language = language
        self.fp16 = fp16 and self.device == "cuda"

        logger.info(f"Loading Whisper model '{model_name}' on {self.device}...")

        try:
            self.model = whisper.load_model(model_name, device=self.device)
            logger.info(f"Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(
        self,
        audio: np.ndarray | str | Path,
        beam_size: int = 5,
        best_of: int = 5,
        temperature: float = 0.0
    ) -> Dict[str, any]:
        """Transcribe audio to text.

        Args:
            audio: Audio data (numpy array), file path (str/Path)
            beam_size: Beam size for beam search
            best_of: Number of candidates when sampling
            temperature: Temperature for sampling

        Returns:
            Dictionary with 'text', 'language', 'segments', etc.
        """
        try:
            # Load audio if path provided
            if isinstance(audio, (str, Path)):
                logger.info(f"Loading audio from {audio}")
                audio = whisper.load_audio(str(audio))

            # Ensure audio is in correct format
            if isinstance(audio, np.ndarray):
                # Whisper expects float32 in range [-1, 1]
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                # Normalize if needed
                if np.abs(audio).max() > 1.0:
                    audio = audio / np.abs(audio).max()

            # Transcribe
            logger.info("Transcribing audio...")
            result = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=beam_size,
                best_of=best_of,
                temperature=temperature,
                fp16=self.fp16,
                verbose=False
            )

            text = result["text"].strip()
            language = result.get("language", self.language)

            logger.info(f"Transcription: '{text}' (language: {language})")

            return {
                "text": text,
                "language": language,
                "segments": result.get("segments", []),
                "full_result": result
            }

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "text": "",
                "language": None,
                "segments": [],
                "error": str(e)
            }

    def transcribe_with_timestamps(
        self,
        audio: np.ndarray | str | Path
    ) -> list[dict]:
        """Transcribe with word-level timestamps.

        Args:
            audio: Audio data or file path

        Returns:
            List of segments with timestamps
        """
        result = self.transcribe(audio)
        segments = result.get("segments", [])

        formatted_segments = []
        for seg in segments:
            formatted_segments.append({
                "text": seg["text"].strip(),
                "start": seg["start"],
                "end": seg["end"],
                "confidence": seg.get("confidence", 1.0)
            })

        return formatted_segments

    def detect_language(self, audio: np.ndarray) -> tuple[str, float]:
        """Detect the language of audio.

        Args:
            audio: Audio data

        Returns:
            Tuple of (language_code, probability)
        """
        try:
            # Prepare audio
            audio = whisper.pad_or_trim(audio)

            # Make log-mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

            # Detect language
            _, probs = self.model.detect_language(mel)

            # Get most likely language
            language = max(probs, key=probs.get)
            probability = probs[language]

            logger.info(f"Detected language: {language} ({probability:.2%})")

            return language, probability

        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "en", 0.0

    def is_english(self, audio: np.ndarray, threshold: float = 0.5) -> bool:
        """Check if audio is in English.

        Args:
            audio: Audio data
            threshold: Minimum probability threshold

        Returns:
            True if English is detected above threshold
        """
        lang, prob = self.detect_language(audio)
        return lang == "en" and prob >= threshold
