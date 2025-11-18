# Phase 3: Video Generation Pipeline Design

## Overview

Phase 3 adds realistic talking head video generation to the voice assistant, transforming it into a complete conversational avatar system with synchronized lip movements and natural facial animation.

## Data Flow

```
User Speech
    ↓
┌─────────────────┐
│ Audio Recorder  │ ← Voice Activity Detection
│  + VAD          │
└────────┬────────┘
         │ [audio_chunk: np.ndarray]
         ↓
┌─────────────────┐
│ Whisper ASR     │ ← Speech-to-Text
└────────┬────────┘
         │ [text: str]
         ↓
┌─────────────────┐
│ Dialogue        │ ← LLM Response Generation
│ Manager         │
└────────┬────────┘
         │ [reply_text: str]
         ↓
┌─────────────────┐
│ Coqui TTS       │ ← Text-to-Speech
└────────┬────────┘
         │ [audio_path: str] (.wav file)
         ↓
┌─────────────────┐
│ Wav2Lip         │ ← NEW: Lip-sync Video Generation
│ Talking Head    │
└────────┬────────┘
         │ [video_path: str] (.mp4 file with audio)
         ↓
┌─────────────────┐
│ Media Player    │ ← NEW: Video + Audio Playback
└─────────────────┘
```

## Module Artifacts

### Between TTS and Talking Head

**Input to Wav2Lip:**
- `audio_path: str` - Path to TTS-generated audio file (WAV format, 22050Hz)
- `avatar_image_path: str` - Path to reference face image (from config)
- `output_path: str` - Destination for generated video

**Output from Wav2Lip:**
- `video_path: str` - MP4 file with:
  - Lip-synced video frames
  - Original audio track embedded
  - Specified resolution and FPS

### Between Talking Head and Player

**Input to Player:**
- `video_path: str` - Complete MP4 file ready for playback

**Output:**
- Visual display + audio playback to user

## Latency Expectations

### Target Hardware Profiles

#### Profile 1: RTX 5060 (Mid-Range) - "Standard Mode"

**Configuration:**
- Resolution: 1280x720 (720p)
- FPS: 25
- Model: Wav2Lip (base, not GAN)
- Face detection batch size: 4

**Expected Latency (per 5-second audio):**
- Face preprocessing: 0.2-0.3s
- Wav2Lip inference: 2.5-3.5s
- Post-processing (mux audio): 0.3-0.5s
- **Total**: ~3-4 seconds

**Memory Usage:**
- Model: ~500MB VRAM
- Processing buffer: ~1GB VRAM
- **Total**: ~1.5GB VRAM

#### Profile 2: RTX 3090 (High-End) - "High Quality Mode"

**Configuration:**
- Resolution: 1920x1080 (1080p)
- FPS: 25 or 30
- Model: Wav2Lip_GAN (enhanced quality)
- Face detection batch size: 8
- Optional: GFPGAN face enhancement

**Expected Latency (per 5-second audio):**
- Face preprocessing: 0.2-0.3s
- Wav2Lip_GAN inference: 3-5s
- Upscaling (720p→1080p): 0.5-1s
- Optional GFPGAN enhancement: 1-2s
- Post-processing: 0.3-0.5s
- **Total**: ~5-9 seconds (without GFPGAN: ~4-7s)

**Memory Usage:**
- Model: ~800MB VRAM
- Processing buffer: ~2GB VRAM
- GFPGAN (optional): +1.5GB VRAM
- **Total**: ~3-4.5GB VRAM

### Latency Trade-offs

| Factor | Impact on Latency | Quality Impact |
|--------|------------------|----------------|
| Resolution (720p vs 1080p) | +30-50% time | Higher detail |
| FPS (25 vs 30) | +20% time | Smoother motion |
| Model (base vs GAN) | +20-30% time | Better visuals |
| Face enhancement (GFPGAN) | +100% time | Refined faces |
| Batch size (4 vs 8) | -10-20% time | No visual change |

### Full Conversation Turn Latency

**Standard Mode (RTX 5060):**
```
Component          Time      Cumulative
------------------------------------------
ASR (Whisper)      0.5s      0.5s
LLM (Llama 3.2)    1.5s      2.0s
TTS (Coqui)        2.0s      4.0s
Wav2Lip (720p)     3.5s      7.5s
------------------------------------------
Total              7.5s
```

**High Quality Mode (RTX 3090):**
```
Component          Time      Cumulative
------------------------------------------
ASR (Whisper)      0.3s      0.3s
LLM (Llama 3.2)    1.0s      1.3s
TTS (Coqui)        1.5s      2.8s
Wav2Lip (1080p)    6.0s      8.8s
------------------------------------------
Total              8.8s
```

**Target for "Near Real-Time":** < 10 seconds total per turn

## Wav2Lip Module API Specification

### Class: `Wav2LipTalkingHead`

**Location:** `src/talking_head/wav2lip_engine.py`

```python
from pathlib import Path
from typing import Optional, Literal
import torch

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
        wav2lip_batch_size: int = 128
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
                       If None, uses default model from config/models.

            face_det_batch_size: Batch size for face detection (higher = faster but more VRAM).
                                Recommended: 4 for standard, 8 for high.

            wav2lip_batch_size: Batch size for Wav2Lip inference.
                               Recommended: 128 (default).

        Raises:
            FileNotFoundError: If avatar_image_path doesn't exist.
            RuntimeError: If GPU is requested but not available.
            ValueError: If face cannot be detected in avatar image.
        """

    def generate(
        self,
        audio_path: str | Path,
        output_path: str | Path,
        fps: int = 25,
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

        Example:
            >>> talking_head = Wav2LipTalkingHead(
            ...     avatar_image_path="avatars/avatar.jpg",
            ...     device="cuda",
            ...     quality_mode="standard"
            ... )
            >>> video_path = talking_head.generate(
            ...     audio_path="outputs/audio/response.wav",
            ...     output_path="outputs/videos/response.mp4"
            ... )
            >>> print(f"Video generated: {video_path}")
        """

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

    def estimate_generation_time(self, audio_duration: float) -> float:
        """
        Estimate video generation time for given audio duration.

        Args:
            audio_duration: Duration of audio in seconds.

        Returns:
            float: Estimated generation time in seconds.
        """

    def cleanup(self) -> None:
        """
        Free GPU memory and cleanup resources.

        Call this when done with the talking head to free VRAM.
        """
```

### Error Handling Strategy

**Graceful Degradation:**

```python
try:
    video_path = talking_head.generate(audio_path, output_path)
    player.play_video(video_path)
except torch.cuda.OutOfMemoryError:
    logger.error("GPU out of memory during video generation")
    logger.info("Falling back to audio-only playback")
    logger.info("Suggestion: Use quality_mode='standard' or switch to CPU")
    # Fallback to audio-only
    player.play_audio(audio_path)
except RuntimeError as e:
    logger.error(f"Video generation failed: {e}")
    logger.info("Continuing with audio-only playback")
    player.play_audio(audio_path)
```

## Configuration Schema

### New Section in `config/config.yaml`

```yaml
talking_head:
  enabled: true
  engine: "wav2lip"

  # Avatar image
  avatar_image: "assets/avatars/default.jpg"

  # Device and quality
  device: "cuda"  # cuda, cpu, or cuda:0, cuda:1, etc.
  quality_mode: "standard"  # "standard" or "high"

  # Standard profile (720p, RTX 5060)
  standard:
    resolution:
      width: 1280
      height: 720
    fps: 25
    model: "wav2lip"  # base model
    face_det_batch_size: 4
    wav2lip_batch_size: 128

  # High quality profile (1080p, RTX 3090)
  high:
    resolution:
      width: 1920
      height: 1080
    fps: 30
    model: "wav2lip_gan"  # GAN-enhanced model
    face_det_batch_size: 8
    wav2lip_batch_size: 128
    enhance_face: false  # Enable GFPGAN (adds latency)

  # Output settings
  output:
    format: "mp4"
    codec: "h264"
    audio_codec: "aac"
    audio_bitrate: "192k"

  # Performance
  cache_preprocessed_face: true
  use_half_precision: true  # FP16 for faster inference (GPU only)

  # Model paths
  models:
    wav2lip: "models/wav2lip/wav2lip.pth"
    wav2lip_gan: "models/wav2lip/wav2lip_gan.pth"
    face_detection: "models/face_detection/s3fd.pth"
    gfpgan: "models/gfpgan/GFPGANv1.3.pth"
```

### GPU Profile Auto-Configuration

```yaml
# Global GPU profile that affects multiple modules
gpu_profile: "medium"  # "low", "medium", "high"

# Auto-configured based on gpu_profile:
# low:
#   - ASR: tiny
#   - TTS: fast model
#   - Talking Head: disabled or audio-only
# medium:
#   - ASR: small
#   - TTS: standard
#   - Talking Head: standard mode (720p)
# high:
#   - ASR: medium
#   - TTS: best quality
#   - Talking Head: high mode (1080p)
```

## Integration Points

### Orchestrator Changes

**Before (Phase 2):**
```python
# Generate response
reply_text = self.dialogue.get_response(user_text)

# Synthesize speech
audio_path = self.tts.synthesize(reply_text)

# Play audio
self.recorder.play_audio(audio_path)
```

**After (Phase 3):**
```python
# Generate response
reply_text = self.dialogue.get_response(user_text)

# Synthesize speech
audio_path = self.tts.synthesize(reply_text)

# Generate video or play audio
if self.config['talking_head']['enabled']:
    try:
        video_path = self.talking_head.generate(
            audio_path=audio_path,
            output_path=self._get_video_output_path()
        )
        self.player.play_video(video_path)
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        self.player.play_audio(audio_path)  # Fallback
else:
    self.player.play_audio(audio_path)
```

## Performance Optimization Strategies

### 1. Model Caching
- Load Wav2Lip model once at startup
- Keep in GPU memory across turns
- Preprocess avatar image once

### 2. Batch Processing
- Process face detection in batches
- Use optimal batch sizes for GPU

### 3. Half Precision (FP16)
- Enable for 30-50% speedup on modern GPUs
- Negligible quality impact
- Automatic fallback to FP32 if unsupported

### 4. Async Generation (Future)
- Start video generation while TTS is still running
- Stream partial results
- Reduce perceived latency

### 5. Resolution Trade-offs
- Start with 512x512 internal generation
- Upscale to target resolution only if needed
- Use efficient interpolation (Lanczos)

## Testing Strategy

### Unit Tests
1. **Test model loading**: Verify Wav2Lip loads without errors
2. **Test avatar preprocessing**: Ensure face detection works
3. **Test video generation**: Generate short test video (1s audio)
4. **Test error handling**: Verify graceful degradation

### Integration Tests
1. **Full pipeline**: ASR → LLM → TTS → Video → Player
2. **Quality modes**: Test both standard and high
3. **Fallback paths**: Test audio-only fallback

### Performance Tests
1. **Latency measurement**: Track generation time vs audio duration
2. **Memory usage**: Monitor VRAM usage
3. **Batch size optimization**: Find optimal settings

## Next Steps

With this design complete, we can now proceed to:

1. **Step 2**: Update dependencies and configuration
2. **Step 3**: Implement `Wav2LipTalkingHead` class
3. **Step 4**: Implement media player
4. **Step 5**: Integrate into orchestrator
5. **Step 6**: Performance tuning
6. **Step 7**: Documentation and examples
