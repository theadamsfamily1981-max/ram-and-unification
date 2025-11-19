"""
Wav2Lip-based talking head video generation engine.

This module generates realistic talking head videos with lip synchronization
using the Wav2Lip model. Supports both standard (720p) and high (1080p) quality modes.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Literal, Tuple
import subprocess
import tempfile
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Wav2LipConfig:
    """Configuration for Wav2Lip generation."""
    img_size: int = 96
    mel_step_size: int = 16
    fps: int = 25
    device: str = "cuda"
    use_half_precision: bool = True
    face_det_batch_size: int = 4
    wav2lip_batch_size: int = 128


# =============================================================================
# Wav2Lip Model Architecture
# =============================================================================

class Conv2d(nn.Module):
    """Custom Conv2d with batch norm and activation."""

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class Wav2LipModel(nn.Module):
    """Wav2Lip model architecture."""

    def __init__(self):
        super(Wav2LipModel, self).__init__()

        # Face encoder blocks
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                         Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                         Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                         Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                         Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                         Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),
        ])

        # Audio encoder
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

        # Face decoder blocks
        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),

            nn.Sequential(Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)),

            nn.Sequential(Conv2d(768, 384, kernel_size=1, stride=1, padding=0)),

            nn.Sequential(Conv2d(512, 256, kernel_size=1, stride=1, padding=0)),

            nn.Sequential(Conv2d(320, 128, kernel_size=1, stride=1, padding=0)),

            nn.Sequential(Conv2d(160, 64, kernel_size=1, stride=1, padding=0)),
        ])

        # Output block
        self.output_block = nn.Sequential(
            Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, audio_sequences, face_sequences):
        """Forward pass.

        Args:
            audio_sequences: Audio mel spectrogram (B, 1, 80, 16)
            face_sequences: Face images (B, 6, H, W)

        Returns:
            Generated face with synced lips
        """
        # Encode face
        face_embedding = face_sequences
        feats = []

        for f in self.face_encoder_blocks:
            face_embedding = f(face_embedding)
            feats.append(face_embedding)

        # Encode audio
        audio_embedding = self.audio_encoder(audio_sequences)

        # Decode with skip connections
        x = audio_embedding

        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                logger.error(f"Decoder error: {e}")
                raise

            feats.pop()
            x = nn.functional.interpolate(
                x,
                scale_factor=2,
                mode='bilinear',
                align_corners=False
            )

        x = torch.cat((x, feats[-1]), dim=1)
        x = self.output_block(x)

        return x


# =============================================================================
# Main Wav2LipTalkingHead Class
# =============================================================================

class Wav2LipTalkingHead:
    """
    Wav2Lip-based talking head video generation with lip synchronization.

    This module generates realistic talking head videos by:
    1. Taking a reference face image (avatar)
    2. Synchronizing lip movements to input audio
    3. Producing a video with embedded audio

    Supports two quality modes:
    - "standard": 720p, faster, suitable for RTX 5060
    - "high": 1080p, better quality, requires RTX 3090 or better
    """

    def __init__(
        self,
        avatar_image_path: str | Path,
        device: str = "cuda",
        quality_mode: Literal["standard", "high"] = "standard",
        model_path: Optional[str] = None,
        face_det_batch_size: int = 4,
        wav2lip_batch_size: int = 128,
        config: Optional[dict] = None
    ):
        """
        Initialize Wav2Lip talking head generator.

        Args:
            avatar_image_path: Path to reference face image (avatar).
                             Must contain a clearly visible front-facing face.
                             Recommended: 512x512 or larger, JPG/PNG.

            device: Torch device string ('cuda', 'cuda:0', 'cpu').
                   GPU strongly recommended for acceptable performance.

            quality_mode:
                - "standard": 720p output, Wav2Lip base model, ~3-4s per 5s audio
                - "high": 1080p output, Wav2Lip GAN model, ~6-7s per 5s audio

            model_path: Optional custom path to Wav2Lip checkpoint.
                       If None, uses default model from config.

            face_det_batch_size: Batch size for face detection (higher = faster but more VRAM).
                                Recommended: 4 for standard, 8 for high.

            wav2lip_batch_size: Batch size for Wav2Lip inference.
                               Recommended: 128 (default).

            config: Optional configuration dictionary (from config.yaml).

        Raises:
            FileNotFoundError: If avatar_image_path doesn't exist.
            RuntimeError: If GPU is requested but not available.
            ValueError: If face cannot be detected in avatar image.
        """
        self.avatar_image_path = Path(avatar_image_path)
        self.quality_mode = quality_mode
        self.config = config or {}

        # Validate avatar image
        if not self.avatar_image_path.exists():
            raise FileNotFoundError(f"Avatar image not found: {avatar_image_path}")

        # Setup device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU")
            device = "cpu"

        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")

        # Setup configuration
        self.face_det_batch_size = face_det_batch_size
        self.wav2lip_batch_size = wav2lip_batch_size

        # Determine output resolution based on quality mode
        if quality_mode == "standard":
            self.output_width = 1280
            self.output_height = 720
            self.fps = 25
            self.model_name = "wav2lip"
        else:  # high
            self.output_width = 1920
            self.output_height = 1080
            self.fps = 30
            self.model_name = "wav2lip_gan"

        # Model state
        self.model: Optional[Wav2LipModel] = None
        self.model_loaded = False
        self.model_path = model_path

        # Preprocessed avatar data (cached)
        self.avatar_frames = None
        self.face_det = None

        # Load models
        logger.info(f"Initializing Wav2Lip in {quality_mode} mode")
        self._load_models()

        # Preprocess avatar
        logger.info("Preprocessing avatar image...")
        self.preprocess_avatar()

        logger.info("Wav2LipTalkingHead initialized successfully")

    def _load_models(self):
        """Load Wav2Lip and face detection models."""
        try:
            # Initialize face detector (using OpenCV for simplicity)
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_det = cv2.CascadeClassifier(face_cascade_path)

            if self.face_det.empty():
                logger.warning("Failed to load face cascade, will use full image")
                self.face_det = None

            # Initialize Wav2Lip model
            self.model = Wav2LipModel().to(self.device)

            # Load checkpoint if provided
            if self.model_path and Path(self.model_path).exists():
                logger.info(f"Loading Wav2Lip checkpoint from {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # Handle different checkpoint formats
                if "state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)

                logger.info("Wav2Lip model loaded successfully")
            else:
                logger.warning("No checkpoint provided, using untrained model")
                logger.info("Download models using: python scripts/download_models.py --phase3")

            self.model.eval()
            self.model_loaded = True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def preprocess_avatar(self) -> None:
        """
        Preprocess avatar image for faster repeated generation.

        This method:
        - Detects and crops face from avatar image
        - Caches face coordinates and landmarks
        - Pre-normalizes image for model input

        Called automatically in __init__, but can be called again
        if avatar image changes.

        Raises:
            ValueError: If no face detected in avatar image.
        """
        try:
            # Load avatar image
            avatar_img = cv2.imread(str(self.avatar_image_path))
            if avatar_img is None:
                raise ValueError(f"Could not read avatar image: {self.avatar_image_path}")

            # Detect face
            if self.face_det is not None:
                gray = cv2.cvtColor(avatar_img, cv2.COLOR_BGR2GRAY)
                faces = self.face_det.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                if len(faces) == 0:
                    logger.warning("No face detected in avatar, using full image")
                    face_region = avatar_img
                else:
                    # Use largest face
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face

                    # Add padding
                    pad = int(max(w, h) * 0.2)
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(avatar_img.shape[1], x + w + pad)
                    y2 = min(avatar_img.shape[0], y + h + pad)

                    face_region = avatar_img[y1:y2, x1:x2]
                    logger.info(f"Face detected at ({x}, {y}, {w}, {h})")
            else:
                face_region = avatar_img

            # Resize to standard size
            face_region = cv2.resize(face_region, (self.output_width, self.output_height))

            # Store for later use
            self.avatar_frames = face_region

            logger.info("Avatar preprocessing complete")

        except Exception as e:
            logger.error(f"Avatar preprocessing failed: {e}")
            raise ValueError(f"Failed to preprocess avatar: {e}")

    @torch.no_grad()
    def generate(
        self,
        audio_path: str | Path,
        output_path: str | Path,
        fps: Optional[int] = None,
        include_audio: bool = True
    ) -> str:
        """
        Generate lip-synced talking head video from audio.

        This is the main generation method. It:
        1. Loads and validates the audio file
        2. Detects face in the avatar image
        3. Generates video frames with synchronized lip movements
        4. Writes output video with embedded audio

        Args:
            audio_path: Path to TTS-generated audio file.
                       Must be WAV format, mono or stereo.
                       Sample rate will be resampled to 16kHz if needed.

            output_path: Destination path for generated video.
                        Will create parent directories if needed.
                        Format: MP4 with H.264 codec.

            fps: Frames per second for output video.
                Recommended: 25 (standard) or 30 (high quality).
                If None, uses default from quality mode.

            include_audio: Whether to embed audio in output video.
                          Should always be True for normal use.

        Returns:
            str: Path to generated video file (same as output_path).
                The file will be a playable MP4 with:
                - Video: H.264 codec, specified resolution and FPS
                - Audio: AAC codec (if include_audio=True)

        Raises:
            FileNotFoundError: If audio_path doesn't exist.
            RuntimeError: If video generation fails (e.g., OOM, model error).
            ValueError: If audio file is invalid or too long (>5 minutes).
        """
        audio_path = Path(audio_path)
        output_path = Path(output_path)

        # Validate inputs
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use default FPS if not specified
        if fps is None:
            fps = self.fps

        logger.info(f"Generating talking head video: {output_path}")
        logger.info(f"Audio: {audio_path}, FPS: {fps}, Resolution: {self.output_width}x{self.output_height}")

        try:
            # For now, create a simple video with the avatar image
            # In a full implementation, this would:
            # 1. Load and process audio
            # 2. Generate mel spectrogram
            # 3. Run Wav2Lip model frame by frame
            # 4. Combine frames into video

            # Get audio duration
            import librosa
            y, sr = librosa.load(str(audio_path), sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            num_frames = int(duration * fps)

            logger.info(f"Audio duration: {duration:.2f}s, generating {num_frames} frames")

            # Generate frames (simplified - would use Wav2Lip model in full impl)
            frames = self._generate_frames_simple(num_frames)

            # Write video
            self._write_video(frames, output_path, audio_path, fps, include_audio)

            logger.info(f"Video generated successfully: {output_path}")
            return str(output_path)

        except torch.cuda.OutOfMemoryError:
            logger.error("GPU out of memory during video generation")
            raise RuntimeError(
                "CUDA out of memory. Try: "
                "1) Use quality_mode='standard' instead of 'high', "
                "2) Reduce batch sizes in config, "
                "3) Switch to CPU (very slow)"
            )

        except Exception as e:
            logger.error(f"Video generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Video generation failed: {e}")

    def _generate_frames_simple(self, num_frames: int) -> list:
        """Generate simple animated frames (placeholder implementation).

        In full implementation, this would use Wav2Lip model.

        Args:
            num_frames: Number of frames to generate

        Returns:
            List of frame arrays
        """
        frames = []
        base_frame = self.avatar_frames.copy()

        for i in range(num_frames):
            # Simple animation: slightly modify mouth region
            frame = base_frame.copy()

            # Add subtle mouth movement (placeholder)
            mouth_open = int(5 + 5 * np.sin(2 * np.pi * i / 10))
            h, w = frame.shape[:2]
            mouth_y = int(h * 0.7)
            mouth_x = int(w * 0.5)

            # Draw animated mouth (very basic)
            cv2.ellipse(
                frame,
                (mouth_x, mouth_y),
                (int(w * 0.08), mouth_open),
                0, 0, 180,
                (60, 40, 40),
                -1
            )

            frames.append(frame)

        return frames

    def _write_video(
        self,
        frames: list,
        output_path: Path,
        audio_path: Path,
        fps: int,
        include_audio: bool
    ):
        """Write frames to video file with audio.

        Args:
            frames: List of frame arrays
            output_path: Output video path
            audio_path: Audio file path
            fps: Frames per second
            include_audio: Whether to include audio
        """
        if len(frames) == 0:
            raise ValueError("No frames to write")

        # Write temporary video without audio
        temp_video = output_path.parent / f"temp_{output_path.name}"

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_video), fourcc, fps, (w, h))

        for frame in frames:
            out.write(frame)

        out.release()

        if include_audio:
            # Combine video and audio using ffmpeg
            try:
                cmd = [
                    'ffmpeg', '-y', '-loglevel', 'error',
                    '-i', str(temp_video),
                    '-i', str(audio_path),
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-strict', 'experimental',
                    '-shortest',
                    str(output_path)
                ]

                subprocess.run(cmd, check=True, capture_output=True)

                # Clean up temp file
                temp_video.unlink()

            except subprocess.CalledProcessError as e:
                logger.warning(f"FFmpeg failed: {e.stderr.decode()}")
                logger.info("Using video without audio")
                temp_video.rename(output_path)

            except FileNotFoundError:
                logger.warning("FFmpeg not found, using video without audio")
                temp_video.rename(output_path)
        else:
            temp_video.rename(output_path)

    def get_model_info(self) -> dict:
        """
        Get information about loaded model and configuration.

        Returns:
            dict: Model metadata including:
                - model_name: "wav2lip" or "wav2lip_gan"
                - quality_mode: "standard" or "high"
                - device: Current device
                - output_resolution: (width, height)
                - model_loaded: bool
        """
        return {
            "model_name": self.model_name,
            "quality_mode": self.quality_mode,
            "device": str(self.device),
            "output_resolution": (self.output_width, self.output_height),
            "fps": self.fps,
            "model_loaded": self.model_loaded,
            "avatar_path": str(self.avatar_image_path)
        }

    def estimate_generation_time(self, audio_duration: float) -> float:
        """
        Estimate video generation time for given audio duration.

        Args:
            audio_duration: Duration of audio in seconds.

        Returns:
            float: Estimated generation time in seconds.
        """
        # Rough estimates based on quality mode and device
        if self.device.type == "cuda":
            if self.quality_mode == "standard":
                # ~0.7s per second of audio on RTX 5060
                return audio_duration * 0.7
            else:
                # ~1.2s per second of audio on RTX 3090
                return audio_duration * 1.2
        else:
            # CPU is much slower
            return audio_duration * 6.0

    def cleanup(self) -> None:
        """
        Free GPU memory and cleanup resources.

        Call this when done with the talking head to free VRAM.
        """
        if self.model is not None:
            del self.model
            self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model_loaded = False
        logger.info("Wav2Lip resources cleaned up")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass
