"""Voice Activity Detection using WebRTC VAD."""

import numpy as np
import webrtcvad
import struct
from typing import List

from ..utils.logger import get_logger

logger = get_logger(__name__)


class VoiceActivityDetector:
    """Detects voice activity in audio streams."""

    def __init__(
        self,
        sample_rate: int = 16000,
        aggressiveness: int = 2,
        frame_duration_ms: int = 30
    ):
        """Initialize VAD.

        Args:
            sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000)
            aggressiveness: VAD aggressiveness (0-3, higher = more aggressive)
            frame_duration_ms: Frame duration in ms (10, 20, or 30)
        """
        if sample_rate not in [8000, 16000, 32000, 48000]:
            logger.warning(f"Sample rate {sample_rate} not supported by WebRTC VAD, using 16000")
            sample_rate = 16000

        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        self.frame_duration_ms = frame_duration_ms
        self.frame_length = int(sample_rate * frame_duration_ms / 1000)

        self.vad = webrtcvad.Vad(aggressiveness)

        logger.info(f"VAD initialized: {sample_rate}Hz, aggressiveness={aggressiveness}")

    def is_speech(self, audio_data: np.ndarray) -> bool:
        """Detect if audio contains speech.

        Args:
            audio_data: Audio data as numpy array (float32 or int16)

        Returns:
            True if speech is detected
        """
        try:
            # Convert to int16 if needed
            if audio_data.dtype == np.float32:
                audio_data = (audio_data * 32767).astype(np.int16)

            # Ensure correct length (must be 10, 20, or 30ms)
            if len(audio_data) < self.frame_length:
                # Pad if too short
                audio_data = np.pad(audio_data, (0, self.frame_length - len(audio_data)))
            elif len(audio_data) > self.frame_length:
                # Truncate if too long
                audio_data = audio_data[:self.frame_length]

            # Convert to bytes
            audio_bytes = audio_data.tobytes()

            # Run VAD
            return self.vad.is_speech(audio_bytes, self.sample_rate)

        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False

    def detect_speech_segments(
        self,
        audio_data: np.ndarray,
        padding_ms: int = 300
    ) -> List[tuple]:
        """Detect speech segments in audio.

        Args:
            audio_data: Audio data
            padding_ms: Padding around speech segments in ms

        Returns:
            List of (start_sample, end_sample) tuples for speech segments
        """
        # Convert to int16 if needed
        if audio_data.dtype == np.float32:
            audio_data = (audio_data * 32767).astype(np.int16)

        # Split into frames
        num_frames = len(audio_data) // self.frame_length
        frames = [
            audio_data[i * self.frame_length:(i + 1) * self.frame_length]
            for i in range(num_frames)
        ]

        # Detect speech in each frame
        speech_frames = []
        for i, frame in enumerate(frames):
            try:
                is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
                speech_frames.append(is_speech)
            except:
                speech_frames.append(False)

        # Find segments
        segments = []
        in_segment = False
        segment_start = 0

        padding_frames = int(padding_ms / self.frame_duration_ms)

        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_segment:
                # Start of speech segment
                segment_start = max(0, i - padding_frames)
                in_segment = True
            elif not is_speech and in_segment:
                # Check if this is real silence or just a gap
                # Look ahead for more speech
                lookahead = speech_frames[i:i + padding_frames]
                if not any(lookahead):
                    # Real end of segment
                    segment_end = min(num_frames, i + padding_frames)
                    segments.append((
                        segment_start * self.frame_length,
                        segment_end * self.frame_length
                    ))
                    in_segment = False

        # Close last segment if still open
        if in_segment:
            segments.append((
                segment_start * self.frame_length,
                len(audio_data)
            ))

        logger.info(f"Detected {len(segments)} speech segments")
        return segments

    def filter_speech(self, audio_data: np.ndarray) -> np.ndarray:
        """Filter audio to keep only speech segments.

        Args:
            audio_data: Input audio

        Returns:
            Audio with only speech segments
        """
        segments = self.detect_speech_segments(audio_data)

        if not segments:
            logger.warning("No speech segments detected")
            return np.array([])

        # Concatenate all speech segments
        speech_audio = []
        for start, end in segments:
            speech_audio.append(audio_data[start:end])

        return np.concatenate(speech_audio)
