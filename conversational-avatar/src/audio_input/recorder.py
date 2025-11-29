"""Audio recording module with voice activity detection."""

import numpy as np
import sounddevice as sd
import wave
from pathlib import Path
from typing import Optional, Callable
from collections import deque
import time

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AudioRecorder:
    """Records audio from microphone with voice activity detection."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        device_index: Optional[int] = None,
        silence_duration: float = 1.5
    ):
        """Initialize audio recorder.

        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1=mono, 2=stereo)
            chunk_size: Number of frames per buffer
            device_index: Input device index (None for default)
            silence_duration: Seconds of silence before stopping
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index
        self.silence_duration = silence_duration

        self.is_recording = False
        self.audio_buffer = deque()
        self.stream = None

        logger.info(f"AudioRecorder initialized: {sample_rate}Hz, {channels}ch")

    def list_devices(self):
        """List available audio input devices."""
        devices = sd.query_devices()
        logger.info("Available audio devices:")
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                logger.info(f"  [{i}] {device['name']} - {device['max_input_channels']} channels")
        return devices

    def start_recording(self):
        """Start recording audio from microphone."""
        if self.is_recording:
            logger.warning("Already recording")
            return

        self.audio_buffer.clear()
        self.is_recording = True

        def audio_callback(indata, frames, time_info, status):
            """Callback for audio stream."""
            if status:
                logger.warning(f"Audio callback status: {status}")
            if self.is_recording:
                self.audio_buffer.append(indata.copy())

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self.chunk_size,
                device=self.device_index,
                callback=audio_callback
            )
            self.stream.start()
            logger.info("Recording started")

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.is_recording = False
            raise

    def stop_recording(self) -> Optional[np.ndarray]:
        """Stop recording and return captured audio.

        Returns:
            Audio data as numpy array or None if no audio recorded
        """
        if not self.is_recording:
            logger.warning("Not currently recording")
            return None

        self.is_recording = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if not self.audio_buffer:
            logger.warning("No audio data recorded")
            return None

        # Concatenate all chunks
        audio_data = np.concatenate(list(self.audio_buffer), axis=0)
        logger.info(f"Recording stopped. Captured {len(audio_data) / self.sample_rate:.2f}s of audio")

        return audio_data.flatten()

    def record_until_silence(
        self,
        vad_callback: Optional[Callable[[np.ndarray], bool]] = None,
        max_duration: float = 30.0
    ) -> Optional[np.ndarray]:
        """Record audio until silence is detected.

        Args:
            vad_callback: Function that returns True if speech is detected
            max_duration: Maximum recording duration in seconds

        Returns:
            Recorded audio or None
        """
        self.start_recording()

        silence_start = None
        start_time = time.time()
        speech_detected = False

        try:
            while self.is_recording:
                # Check max duration
                if time.time() - start_time > max_duration:
                    logger.info(f"Max duration ({max_duration}s) reached")
                    break

                # Wait for next chunk
                time.sleep(0.1)

                if not self.audio_buffer:
                    continue

                # Get recent audio for VAD
                recent_audio = list(self.audio_buffer)[-5:]  # Last ~0.5s
                if len(recent_audio) == 0:
                    continue

                recent_audio = np.concatenate(recent_audio, axis=0).flatten()

                # Check for speech using VAD callback
                if vad_callback:
                    is_speech = vad_callback(recent_audio)
                else:
                    # Simple energy-based detection
                    energy = np.sqrt(np.mean(recent_audio ** 2))
                    is_speech = energy > 0.01

                if is_speech:
                    speech_detected = True
                    silence_start = None
                elif speech_detected:  # Silence after speech
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > self.silence_duration:
                        logger.info(f"Silence detected for {self.silence_duration}s")
                        break

        except KeyboardInterrupt:
            logger.info("Recording interrupted by user")

        return self.stop_recording()

    def save_audio(self, audio_data: np.ndarray, output_path: Path):
        """Save audio data to WAV file.

        Args:
            audio_data: Audio data as numpy array
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure audio is in correct format
        if audio_data.dtype != np.int16:
            # Convert float32 to int16
            audio_data = (audio_data * 32767).astype(np.int16)

        with wave.open(str(output_path), 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

        logger.info(f"Audio saved to {output_path}")

    def play_audio(self, audio_data: np.ndarray, timeout: float = 60.0):
        """Play audio through speakers with timeout protection.

        Args:
            audio_data: Audio data to play
            timeout: Maximum playback time in seconds (default: 60s)
        """
        try:
            sd.play(audio_data, self.sample_rate)
            # Calculate expected duration
            expected_duration = len(audio_data) / self.sample_rate
            # Use the shorter of expected duration + 1s buffer or the timeout
            wait_time = min(expected_duration + 1.0, timeout)

            # Wait with timeout - sd.wait() doesn't have timeout so we poll
            import time as time_module
            start = time_module.time()
            while sd.get_stream() and sd.get_stream().active:
                if time_module.time() - start > wait_time:
                    sd.stop()
                    logger.warning(f"Audio playback timed out after {wait_time:.1f}s")
                    return
                time_module.sleep(0.1)

            logger.info("Audio playback complete")
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")

    def __del__(self):
        """Cleanup on deletion."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
