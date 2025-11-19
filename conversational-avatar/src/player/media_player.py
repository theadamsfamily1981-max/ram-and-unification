"""Media player for video and audio playback."""

import cv2
import subprocess
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from typing import Optional, Literal
import platform

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MediaPlayer:
    """
    Unified media player for audio and video playback.

    Supports multiple playback backends:
    - opencv: Simple CV2-based player (cross-platform)
    - ffplay: FFmpeg player (better quality, if available)
    - sounddevice: For audio-only playback
    """

    def __init__(
        self,
        engine: Literal["opencv", "ffplay", "auto"] = "auto",
        fullscreen: bool = False,
        window_size: Optional[tuple] = None,
        config: Optional[dict] = None
    ):
        """
        Initialize media player.

        Args:
            engine: Playback engine to use:
                - "opencv": OpenCV-based (always available)
                - "ffplay": FFmpeg player (better quality)
                - "auto": Auto-detect best available
            fullscreen: Whether to play in fullscreen
            window_size: Window size (width, height) for non-fullscreen
            config: Optional configuration dict
        """
        self.config = config or {}
        self.fullscreen = fullscreen
        self.window_size = window_size or (1280, 720)

        # Detect and set engine
        if engine == "auto":
            self.engine = self._detect_best_engine()
        else:
            self.engine = engine

        logger.info(f"MediaPlayer initialized with {self.engine} engine")

    def _detect_best_engine(self) -> str:
        """Detect best available playback engine.

        Returns:
            Engine name
        """
        # Check for ffplay
        try:
            subprocess.run(
                ["ffplay", "-version"],
                capture_output=True,
                check=True
            )
            logger.info("FFplay detected, using ffplay engine")
            return "ffplay"
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # Fall back to OpenCV
        logger.info("Using OpenCV engine")
        return "opencv"

    def play_video(
        self,
        video_path: str | Path,
        fullscreen: Optional[bool] = None,
        blocking: bool = True
    ) -> None:
        """
        Play video file.

        Args:
            video_path: Path to video file (MP4, AVI, etc.)
            fullscreen: Override default fullscreen setting
            blocking: Whether to block until playback completes

        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If playback fails
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        use_fullscreen = fullscreen if fullscreen is not None else self.fullscreen

        logger.info(f"Playing video: {video_path}")

        try:
            if self.engine == "ffplay":
                self._play_video_ffplay(video_path, use_fullscreen, blocking)
            else:
                self._play_video_opencv(video_path, use_fullscreen, blocking)

            logger.info("Video playback complete")

        except Exception as e:
            logger.error(f"Video playback failed: {e}")
            raise RuntimeError(f"Failed to play video: {e}")

    def _play_video_opencv(
        self,
        video_path: Path,
        fullscreen: bool,
        blocking: bool
    ):
        """Play video using OpenCV.

        Args:
            video_path: Video file path
            fullscreen: Fullscreen mode
            blocking: Block until complete
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 25  # Default fallback

        frame_delay = int(1000 / fps)  # Delay in ms

        # Create window
        window_name = "Talking Avatar"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        if fullscreen:
            cv2.setWindowProperty(
                window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )
        else:
            cv2.resizeWindow(window_name, *self.window_size)

        try:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                cv2.imshow(window_name, frame)

                # Wait for next frame or ESC key
                key = cv2.waitKey(frame_delay) & 0xFF
                if key == 27:  # ESC key
                    logger.info("Playback stopped by user")
                    break

        finally:
            cap.release()
            cv2.destroyWindow(window_name)

    def _play_video_ffplay(
        self,
        video_path: Path,
        fullscreen: bool,
        blocking: bool
    ):
        """Play video using FFplay.

        Args:
            video_path: Video file path
            fullscreen: Fullscreen mode
            blocking: Block until complete
        """
        cmd = ["ffplay", "-autoexit", "-loglevel", "error"]

        if fullscreen:
            cmd.append("-fs")
        else:
            w, h = self.window_size
            cmd.extend(["-x", str(w), "-y", str(h)])

        cmd.append(str(video_path))

        if blocking:
            subprocess.run(cmd, check=True)
        else:
            subprocess.Popen(cmd)

    def play_audio(
        self,
        audio_path: str | Path
    ) -> None:
        """
        Play audio file.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)

        Raises:
            FileNotFoundError: If audio file doesn't exist
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Playing audio: {audio_path}")

        try:
            # Load and play audio
            data, samplerate = sf.read(str(audio_path))
            sd.play(data, samplerate)
            sd.wait()  # Wait for playback to complete

            logger.info("Audio playback complete")

        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
            raise RuntimeError(f"Failed to play audio: {e}")

    def is_playing(self) -> bool:
        """Check if audio is currently playing.

        Returns:
            True if playing
        """
        return sd.get_stream().active if sd.get_stream() else False

    def stop(self):
        """Stop current playback."""
        try:
            sd.stop()
            cv2.destroyAllWindows()
            logger.info("Playback stopped")
        except Exception as e:
            logger.warning(f"Error stopping playback: {e}")

    def test_playback(self):
        """Test if playback is working.

        Returns:
            bool: True if playback capabilities are available
        """
        logger.info("Testing media playback...")

        # Test video
        try:
            import cv2
            logger.info("✅ OpenCV video playback available")
        except ImportError:
            logger.warning("❌ OpenCV not available")
            return False

        # Test audio
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            logger.info(f"✅ Audio playback available ({len(devices)} devices)")
        except Exception as e:
            logger.warning(f"❌ Audio playback not available: {e}")

        # Test ffplay
        try:
            subprocess.run(
                ["ffplay", "-version"],
                capture_output=True,
                check=True
            )
            logger.info("✅ FFplay available")
        except:
            logger.info("ℹ️  FFplay not available (optional)")

        logger.info("Media playback test complete")
        return True
