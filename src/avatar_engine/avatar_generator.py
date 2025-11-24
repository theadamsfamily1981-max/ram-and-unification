"""Main avatar generator combining all components."""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, List, TYPE_CHECKING
from dataclasses import dataclass
import uuid

# Lazy imports for heavy ML dependencies
try:
    import cv2
    import numpy as np
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    ML_IMPORT_ERROR = str(e)
    # Create stub for numpy
    class np:
        ndarray = None

from .lip_sync import LipSyncEngine, LipSyncConfig
from ..utils.face_detection import FaceDetector
from ..utils.audio_processing import AudioProcessor


@dataclass
class GenerationResult:
    """Result of avatar generation."""
    video_path: Path
    duration: float
    fps: int
    resolution: tuple
    frames_generated: int
    success: bool
    error_message: Optional[str] = None


class AvatarGenerator:
    """Main avatar generator class."""

    def __init__(
        self,
        device: str = "cpu",
        output_fps: int = 25,
        output_resolution: int = 512
    ):
        """Initialize avatar generator.

        Args:
            device: Device to use (cuda or cpu)
            output_fps: Output video FPS
            output_resolution: Output video resolution
        """
        self.device = device
        self.output_fps = output_fps
        self.output_resolution = output_resolution

        # Initialize components
        self.face_detector = FaceDetector(device=device)
        self.audio_processor = AudioProcessor()
        self.lip_sync_engine = LipSyncEngine(
            LipSyncConfig(device=device, fps=output_fps)
        )

    def load_models(self, model_path: Optional[Path] = None):
        """Load required models.

        Args:
            model_path: Path to model checkpoint (optional)
        """
        if model_path and model_path.exists():
            self.lip_sync_engine.load_model(model_path)

    def preprocess_image(
        self,
        image_path: Union[Path, str, np.ndarray]
    ) -> Optional[np.ndarray]:
        """Preprocess input image.

        Args:
            image_path: Path to image or numpy array

        Returns:
            Preprocessed image or None
        """
        # Load image
        if isinstance(image_path, np.ndarray):
            image = image_path
        else:
            image = cv2.imread(str(image_path))
            if image is None:
                return None

        # Detect and crop face
        face = self.face_detector.crop_face(
            image,
            target_size=(self.output_resolution, self.output_resolution)
        )

        if face is None:
            # If no face detected, resize the whole image
            face = cv2.resize(
                image,
                (self.output_resolution, self.output_resolution)
            )

        return face

    def generate(
        self,
        image_input: Union[Path, str, np.ndarray],
        audio_input: Union[Path, str],
        output_path: Optional[Path] = None,
        temp_dir: Path = Path("./temp")
    ) -> GenerationResult:
        """Generate talking avatar video.

        Args:
            image_input: Input image (path or array)
            audio_input: Input audio path
            output_path: Output video path (auto-generated if None)
            temp_dir: Temporary directory for processing

        Returns:
            GenerationResult with video path and metadata
        """
        try:
            # Validate and process audio
            audio_path = Path(audio_input)
            is_valid, msg = self.audio_processor.validate_audio(audio_path)
            if not is_valid:
                return GenerationResult(
                    video_path=Path(""),
                    duration=0,
                    fps=self.output_fps,
                    resolution=(self.output_resolution, self.output_resolution),
                    frames_generated=0,
                    success=False,
                    error_message=msg
                )

            # Extract audio features
            audio_features = self.audio_processor.extract_audio_features(audio_path)
            duration = audio_features["duration"]
            num_frames = int(duration * self.output_fps)

            # Preprocess image
            face_image = self.preprocess_image(image_input)
            if face_image is None:
                return GenerationResult(
                    video_path=Path(""),
                    duration=0,
                    fps=self.output_fps,
                    resolution=(self.output_resolution, self.output_resolution),
                    frames_generated=0,
                    success=False,
                    error_message="Could not process input image"
                )

            # Prepare mel spectrogram for lip sync
            mel_spec = audio_features["mel_spectrogram"]

            # Generate talking frames
            frames = self.lip_sync_engine.generate_talking_face(
                face_image,
                mel_spec,
                num_frames
            )

            # Generate output path if not provided
            if output_path is None:
                temp_dir.mkdir(parents=True, exist_ok=True)
                output_path = temp_dir / f"avatar_{uuid.uuid4().hex}.mp4"

            # Write video
            self._write_video(
                frames,
                output_path,
                audio_path,
                self.output_fps
            )

            return GenerationResult(
                video_path=output_path,
                duration=duration,
                fps=self.output_fps,
                resolution=(self.output_resolution, self.output_resolution),
                frames_generated=len(frames),
                success=True
            )

        except Exception as e:
            return GenerationResult(
                video_path=Path(""),
                duration=0,
                fps=self.output_fps,
                resolution=(self.output_resolution, self.output_resolution),
                frames_generated=0,
                success=False,
                error_message=f"Generation failed: {str(e)}"
            )

    def _write_video(
        self,
        frames: List[np.ndarray],
        output_path: Path,
        audio_path: Path,
        fps: int
    ):
        """Write video with audio.

        Args:
            frames: List of video frames
            output_path: Output video path
            audio_path: Audio file path
            fps: Frames per second
        """
        if len(frames) == 0:
            raise ValueError("No frames to write")

        # Create temporary video without audio
        temp_video = output_path.parent / f"temp_{output_path.name}"

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_video), fourcc, fps, (w, h))

        for frame in frames:
            out.write(frame)

        out.release()

        # Combine video and audio using ffmpeg
        import subprocess
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', str(temp_video),
                '-i', str(audio_path),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-shortest',
                str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            # If ffmpeg fails, use video without audio
            temp_video.rename(output_path)
        finally:
            if temp_video.exists():
                temp_video.unlink()

    def generate_batch(
        self,
        image_inputs: List[Union[Path, str, np.ndarray]],
        audio_inputs: List[Union[Path, str]],
        output_dir: Path
    ) -> List[GenerationResult]:
        """Generate multiple talking avatars.

        Args:
            image_inputs: List of input images
            audio_inputs: List of input audio files
            output_dir: Output directory

        Returns:
            List of GenerationResults
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, (img, aud) in enumerate(zip(image_inputs, audio_inputs)):
            output_path = output_dir / f"avatar_{i:04d}.mp4"
            result = self.generate(img, aud, output_path)
            results.append(result)

        return results
