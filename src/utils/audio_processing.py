"""Audio processing utilities."""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, TYPE_CHECKING

# Lazy imports for heavy ML dependencies
try:
    import numpy as np
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError as e:
    AUDIO_AVAILABLE = False
    AUDIO_IMPORT_ERROR = str(e)
    class np:
        ndarray = None


class AudioProcessor:
    """Audio processing for talking avatar."""

    def __init__(self, sample_rate: int = 16000):
        """Initialize audio processor.

        Args:
            sample_rate: Target sample rate
        """
        if not AUDIO_AVAILABLE:
            raise ImportError(
                f"Audio dependencies not installed: {AUDIO_IMPORT_ERROR}. "
                "Please install with: pip install numpy librosa soundfile"
            )
        self.sample_rate = sample_rate

    def load_audio(
        self,
        audio_path: Path | str,
        normalize: bool = True
    ) -> Tuple[np.ndarray, int]:
        """Load audio file.

        Args:
            audio_path: Path to audio file
            normalize: Whether to normalize audio

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio, sr = librosa.load(
            str(audio_path),
            sr=self.sample_rate,
            mono=True
        )

        if normalize:
            audio = audio / np.max(np.abs(audio) + 1e-8)

        return audio, sr

    def save_audio(
        self,
        audio: np.ndarray,
        output_path: Path | str,
        sample_rate: Optional[int] = None
    ):
        """Save audio to file.

        Args:
            audio: Audio data
            output_path: Output file path
            sample_rate: Sample rate (uses default if None)
        """
        sr = sample_rate or self.sample_rate
        sf.write(str(output_path), audio, sr)

    def get_mel_spectrogram(
        self,
        audio: np.ndarray,
        n_mels: int = 80,
        hop_length: int = 512
    ) -> np.ndarray:
        """Compute mel spectrogram.

        Args:
            audio: Audio signal
            n_mels: Number of mel bands
            hop_length: Hop length for STFT

        Returns:
            Mel spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=n_mels,
            hop_length=hop_length
        )

        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db

    def extract_audio_features(
        self,
        audio_path: Path | str
    ) -> dict:
        """Extract audio features for lip-sync.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary of audio features
        """
        audio, sr = self.load_audio(audio_path)

        # Extract features
        mel_spec = self.get_mel_spectrogram(audio)
        duration = librosa.get_duration(y=audio, sr=sr)

        # Extract MFCC for more detailed features
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=13
        )

        return {
            "mel_spectrogram": mel_spec,
            "mfcc": mfcc,
            "duration": duration,
            "sample_rate": sr,
            "audio": audio
        }

    def validate_audio(
        self,
        audio_path: Path | str,
        max_duration: Optional[float] = None
    ) -> Tuple[bool, str]:
        """Validate audio file.

        Args:
            audio_path: Path to audio file
            max_duration: Maximum allowed duration in seconds

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            audio, sr = self.load_audio(audio_path)
            duration = len(audio) / sr

            if max_duration and duration > max_duration:
                return False, f"Audio too long: {duration:.1f}s (max: {max_duration}s)"

            if duration < 0.1:
                return False, "Audio too short (min: 0.1s)"

            return True, "Valid"

        except Exception as e:
            return False, f"Error loading audio: {str(e)}"
