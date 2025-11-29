"""RVC (Retrieval-based Voice Conversion) integration for custom voice synthesis.

This module handles:
- Loading RVC models (Ara voice)
- Converting TTS output to custom voice
- Integration with oobabooga TTS extensions
- Voice parameter tuning (pitch, index rate, etc.)
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import hashlib

logger = logging.getLogger(__name__)


class RVCVoiceConverter:
    """RVC voice conversion engine."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        index_path: Optional[Path] = None,
        pitch: float = 0.12,
        index_rate: float = 0.65,
        device: str = "cpu"
    ):
        """Initialize RVC converter.

        Args:
            model_path: Path to RVC model file (.pth)
            index_path: Path to RVC index file (.index)
            pitch: Pitch adjustment in semitones
            index_rate: Index rate (0.0-1.0)
            device: Device to use (cpu, cuda)
        """
        self.model_path = Path(model_path) if model_path else None
        self.index_path = Path(index_path) if index_path else None
        self.pitch = pitch
        self.index_rate = index_rate
        self.device = device

        # RVC model instance (lazy loaded)
        self._model = None
        self._rvc_available = False

        # Check if RVC is available
        self._check_rvc_availability()

    def _check_rvc_availability(self):
        """Check if RVC is available."""
        try:
            # Try to import RVC dependencies
            # Note: This requires the RVC library to be installed
            # pip install rvc-python or use Mangio-RVC-Fork
            import torch
            self._rvc_available = True
            logger.info("RVC dependencies available")
        except ImportError as e:
            logger.warning(f"RVC not available: {e}")
            self._rvc_available = False

    def is_available(self) -> bool:
        """Check if RVC is available and model is loaded."""
        return self._rvc_available and self.model_path and self.model_path.exists()

    def load_model(self):
        """Load RVC model."""
        if not self._rvc_available:
            raise RuntimeError("RVC dependencies not installed")

        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError(f"RVC model not found: {self.model_path}")

        logger.info(f"Loading RVC model: {self.model_path}")

        try:
            # Load model using RVC library
            # This is a placeholder - actual implementation depends on RVC library version
            import torch
            self._model = torch.load(self.model_path, map_location=self.device)
            logger.info("RVC model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load RVC model: {e}")
            raise

    def convert(
        self,
        input_audio: Path,
        output_audio: Path,
        pitch_override: Optional[float] = None,
        index_rate_override: Optional[float] = None
    ) -> Path:
        """Convert audio using RVC model.

        Args:
            input_audio: Input audio file
            output_audio: Output audio file
            pitch_override: Override pitch setting
            index_rate_override: Override index rate

        Returns:
            Path to converted audio
        """
        if not self.is_available():
            logger.warning("RVC not available, returning original audio")
            return input_audio

        pitch = pitch_override if pitch_override is not None else self.pitch
        index_rate = index_rate_override if index_rate_override is not None else self.index_rate

        logger.info(f"Converting voice with RVC (pitch={pitch}, index_rate={index_rate})")

        try:
            # Use RVC CLI or Python API to convert
            # This uses the infer_cli.py from Mangio-RVC-Fork or similar
            self._rvc_convert_cli(input_audio, output_audio, pitch, index_rate)
            return output_audio

        except Exception as e:
            logger.error(f"RVC conversion failed: {e}")
            # Return original audio as fallback
            return input_audio

    def _rvc_convert_cli(
        self,
        input_audio: Path,
        output_audio: Path,
        pitch: float,
        index_rate: float
    ):
        """Convert using RVC CLI tool.

        This assumes you have Mangio-RVC-Fork or similar installed.
        """
        # Check for RVC CLI tools
        rvc_dirs = [
            Path.home() / "Mangio-RVC-Fork",
            Path.home() / "RVC",
            Path("/opt/RVC"),
            Path("./RVC")
        ]

        rvc_dir = None
        for d in rvc_dirs:
            if d.exists():
                rvc_dir = d
                break

        if not rvc_dir:
            logger.warning("RVC installation not found, using fallback")
            # Try using subprocess with generic RVC command
            self._rvc_convert_subprocess(input_audio, output_audio, pitch, index_rate)
            return

        # Use RVC infer_cli.py
        infer_script = rvc_dir / "infer_cli.py"
        if not infer_script.exists():
            raise FileNotFoundError(f"RVC infer script not found: {infer_script}")

        cmd = [
            sys.executable,
            str(infer_script),
            "--input", str(input_audio),
            "--output", str(output_audio),
            "--model", str(self.model_path),
            "--pitch", str(pitch),
            "--index_rate", str(index_rate),
            "--device", self.device
        ]

        if self.index_path and self.index_path.exists():
            cmd.extend(["--index_path", str(self.index_path)])

        logger.debug(f"Running RVC command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            raise RuntimeError(f"RVC conversion failed: {result.stderr}")

        logger.info("RVC conversion completed successfully")

    def _rvc_convert_subprocess(
        self,
        input_audio: Path,
        output_audio: Path,
        pitch: float,
        index_rate: float
    ):
        """Fallback RVC conversion using subprocess."""
        # This is a fallback - try common RVC command patterns
        cmd = [
            "rvc-convert",
            "--input", str(input_audio),
            "--output", str(output_audio),
            "--model", str(self.model_path),
            "--pitch", str(pitch),
            "--index-rate", str(index_rate)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                logger.info("RVC conversion completed (subprocess)")
            else:
                # If this fails too, just copy the file
                import shutil
                shutil.copy(input_audio, output_audio)
                logger.warning("RVC conversion failed, using original audio")

        except (subprocess.SubprocessError, FileNotFoundError):
            # Command not found, just copy
            import shutil
            shutil.copy(input_audio, output_audio)
            logger.warning("RVC not available, using original audio")

    def get_info(self) -> Dict[str, Any]:
        """Get RVC converter information."""
        return {
            "available": self.is_available(),
            "model_path": str(self.model_path) if self.model_path else None,
            "index_path": str(self.index_path) if self.index_path else None,
            "pitch": self.pitch,
            "index_rate": self.index_rate,
            "device": self.device,
            "model_loaded": self._model is not None
        }


class OobaboogaRVCClient:
    """Client for oobabooga text-generation-webui with RVC TTS extension."""

    def __init__(
        self,
        api_url: str = "http://localhost:5000",
        rvc_model: Optional[str] = None,
        pitch: float = 0.12,
        index_rate: float = 0.65
    ):
        """Initialize oobabooga RVC client.

        Args:
            api_url: Oobabooga API URL
            rvc_model: RVC model name/path
            pitch: Pitch adjustment
            index_rate: Index rate
        """
        self.api_url = api_url.rstrip('/')
        self.rvc_model = rvc_model
        self.pitch = pitch
        self.index_rate = index_rate

    def generate_tts(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None
    ) -> bool:
        """Generate TTS using oobabooga with RVC.

        Args:
            text: Text to synthesize
            output_path: Output audio file path
            voice: Voice name (optional)

        Returns:
            True if successful
        """
        try:
            import requests

            # Call oobabooga TTS API
            endpoint = f"{self.api_url}/api/v1/audio/speech"

            payload = {
                "input": text,
                "voice": voice or "en-US-JennyNeural",
                "extensions": ["rvc"],
                "rvc": {
                    "model": self.rvc_model,
                    "pitch": self.pitch,
                    "index_rate": self.index_rate
                }
            }

            response = requests.post(
                endpoint,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                # Save audio file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"TTS generated: {output_path}")
                return True
            else:
                logger.error(f"Oobabooga TTS failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Oobabooga TTS error: {e}")
            return False

    def health_check(self) -> bool:
        """Check if oobabooga API is available."""
        try:
            import requests
            response = requests.get(f"{self.api_url}/api/v1/model", timeout=5)
            return response.status_code == 200
        except:
            return False
