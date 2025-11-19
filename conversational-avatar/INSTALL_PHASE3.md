# Phase 3 Installation Guide - Video Generation

Additional installation steps for Phase 3 (Talking Head Video Generation).

## Prerequisites

**Phase 2 must be installed and working** before proceeding with Phase 3.

If you haven't completed Phase 2 installation, see `INSTALL.md` first.

## System Requirements for Phase 3

### Minimum (Standard Mode - 720p)
- **GPU**: NVIDIA RTX 5060 or equivalent (6GB+ VRAM)
- **CUDA**: 11.8 or 12.1
- **RAM**: 16GB system RAM
- **Storage**: Additional 5GB for Wav2Lip models

### Recommended (High Quality Mode - 1080p)
- **GPU**: NVIDIA RTX 3090 or better (12GB+ VRAM)
- **CUDA**: 11.8 or 12.1
- **RAM**: 32GB system RAM
- **Storage**: Additional 10GB for models + GFPGAN

## Installation Steps

### 1. Install System Dependencies

#### Ubuntu/Debian

```bash
# FFmpeg (if not already installed)
sudo apt-get update
sudo apt-get install -y ffmpeg

# Additional video processing libraries
sudo apt-get install -y \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libfdk-aac-dev \
    libmp3lame-dev \
    libopus-dev

# For face detection/enhancement
sudo apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
```

#### macOS

```bash
# FFmpeg
brew install ffmpeg

# Additional codecs
brew install x264 x265 libvpx
```

#### Windows

1. **FFmpeg**: Download from https://ffmpeg.org/download.html
2. Add FFmpeg to system PATH
3. Install Visual C++ Redistributable if needed

### 2. Install Python Dependencies

```bash
# Activate your virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install Phase 3 dependencies
pip install -r requirements.txt

# Verify GPU is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

### 3. Download Wav2Lip Models

#### Option A: Automatic Download (Recommended)

We'll create a model download script:

```bash
python scripts/download_models.py --phase3
```

This will download:
- Wav2Lip base model (~350MB)
- Wav2Lip GAN model (~350MB)
- S3FD face detection model (~90MB)
- GFPGAN model (optional, ~350MB)

#### Option B: Manual Download

**Wav2Lip Base Model:**
```bash
mkdir -p models/wav2lip
cd models/wav2lip

# Download from GitHub releases or Google Drive
wget https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?download=1 -O wav2lip.pth

cd ../..
```

**Wav2Lip GAN Model (for high quality):**
```bash
cd models/wav2lip

wget https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?download=1 -O wav2lip_gan.pth

cd ../..
```

**S3FD Face Detection:**
```bash
mkdir -p models/face_detection

cd models/face_detection

wget https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth -O s3fd.pth

cd ../..
```

**GFPGAN (Optional - for face enhancement):**
```bash
mkdir -p models/gfpgan

cd models/gfpgan

wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth

cd ../..
```

### 4. Prepare Avatar Image

```bash
# Create avatars directory
mkdir -p assets/avatars

# Copy your avatar image
# Requirements:
# - Front-facing portrait
# - Clear, visible face
# - Good lighting
# - Recommended: 512x512 or larger
# - Format: JPG or PNG

cp /path/to/your/avatar.jpg assets/avatars/default.jpg

# Or use a provided example
# cp examples/avatars/example_avatar.jpg assets/avatars/default.jpg
```

**Avatar Image Tips:**
- Use a high-quality photo
- Face should fill at least 30% of the image
- Neutral or slight smile works best
- Avoid extreme angles or occlusions
- Good lighting is crucial

### 5. Configure Phase 3 Settings

Edit `config/config.yaml`:

```yaml
# Enable video generation
talking_head:
  enabled: true
  avatar_image: "assets/avatars/default.jpg"

  # For RTX 5060 or similar
  device: "cuda"
  quality_mode: "standard"  # 720p, ~3-4s per turn

  # For RTX 3090 or better
  # quality_mode: "high"  # 1080p, ~6-7s per turn

# Set global GPU profile
system:
  gpu_profile: "medium"  # or "high" for RTX 3090+
```

### 6. Verify Installation

#### Test Imports

```bash
python << EOF
import torch
import cv2
from gfpgan import GFPGANer
print("✅ All Phase 3 imports successful")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
EOF
```

#### Test Face Detection

```bash
python << EOF
from src.talking_head.wav2lip_engine import Wav2LipTalkingHead

# This will test if face can be detected in avatar
talking_head = Wav2LipTalkingHead(
    avatar_image_path="assets/avatars/default.jpg",
    device="cuda",
    quality_mode="standard"
)

info = talking_head.get_model_info()
print(f"Model info: {info}")
print("✅ Wav2Lip initialized successfully")
EOF
```

#### Test Video Generation

```bash
# Run the demo script (creates a test video from text)
python scripts/demo_talking_avatar_from_text.py \
    --text "Hello, this is a test of the talking avatar system." \
    --output outputs/demos/test.mp4
```

This should generate a short video at `outputs/demos/test.mp4`.

### 7. Test Full System

```bash
# Run the full voice + video assistant
python main.py
```

You should now see video playback in addition to audio!

## Troubleshooting Phase 3

### Issue: "CUDA out of memory"

**Solution:**
```yaml
# In config/config.yaml:
talking_head:
  quality_mode: "standard"  # Use standard instead of high

  standard:
    face_det_batch_size: 2  # Reduce from 4
    wav2lip_batch_size: 64  # Reduce from 128
```

Or switch to CPU (very slow):
```yaml
talking_head:
  device: "cpu"
```

### Issue: "No face detected in avatar image"

**Solution:**
1. Check your avatar image:
   ```bash
   python -c "from PIL import Image; img = Image.open('assets/avatars/default.jpg'); print(f'Size: {img.size}')"
   ```

2. Try a different image with a clearer, front-facing face

3. Adjust face detection settings:
   ```yaml
   talking_head:
     standard:
       face_det_batch_size: 1  # More lenient detection
   ```

### Issue: "Wav2Lip model not found"

**Solution:**
```bash
# Check model paths
ls -lh models/wav2lip/
ls -lh models/face_detection/

# If files are missing, re-download
python scripts/download_models.py --phase3 --force
```

### Issue: "Video generation is very slow (>30s per turn)"

**Solutions:**

1. **Use GPU** instead of CPU:
   ```yaml
   talking_head:
     device: "cuda"
   ```

2. **Reduce quality**:
   ```yaml
   talking_head:
     quality_mode: "standard"
   ```

3. **Optimize batch sizes** for your GPU:
   ```yaml
   # For 6GB VRAM (RTX 5060):
   standard:
     face_det_batch_size: 4
     wav2lip_batch_size: 128

   # For 8GB VRAM:
   standard:
     face_det_batch_size: 6
     wav2lip_batch_size: 160

   # For 12GB+ VRAM (RTX 3090):
   high:
     face_det_batch_size: 8
     wav2lip_batch_size: 256
   ```

4. **Enable FP16** (if not already):
   ```yaml
   talking_head:
     use_half_precision: true
   ```

### Issue: "Video has no audio" or "Audio is out of sync"

**Solution:**
```bash
# Check FFmpeg installation
ffmpeg -version

# Reinstall ffmpeg-python
pip uninstall ffmpeg-python
pip install ffmpeg-python==0.2.0
```

### Issue: "Video quality is poor/blurry"

**Solutions:**

1. **Use high quality mode**:
   ```yaml
   talking_head:
     quality_mode: "high"
   ```

2. **Enable face enhancement**:
   ```yaml
   high:
     enhance_face: true  # Adds 1-2s but improves quality
   ```

3. **Use GAN model**:
   ```yaml
   high:
     model: "wav2lip_gan"  # Better than base wav2lip
   ```

4. **Use higher resolution avatar image**:
   - Minimum: 512x512
   - Recommended: 1024x1024 or higher

### Issue: "ModuleNotFoundError: No module named 'gfpgan'"

**Solution:**
```bash
pip install gfpgan==1.3.8 basicsr==1.4.2 facexlib==0.3.0
```

## Performance Benchmarks

Expected performance on different hardware:

### RTX 5060 (8GB VRAM) - Standard Mode
```
Audio duration: 5 seconds
Video generation time: 3-4 seconds
Total turn latency: 7-8 seconds
Resolution: 1280x720 @ 25fps
```

### RTX 3090 (24GB VRAM) - High Mode
```
Audio duration: 5 seconds
Video generation time: 5-7 seconds
Total turn latency: 9-11 seconds
Resolution: 1920x1080 @ 30fps
```

### RTX 4090 (24GB VRAM) - High Mode + Enhancement
```
Audio duration: 5 seconds
Video generation time: 6-8 seconds (with GFPGAN)
Total turn latency: 10-12 seconds
Resolution: 1920x1080 @ 30fps
Enhanced quality: Yes
```

## Model Storage Requirements

```
models/
├── wav2lip/
│   ├── wav2lip.pth           (~350MB)
│   └── wav2lip_gan.pth       (~350MB)
├── face_detection/
│   └── s3fd.pth              (~90MB)
├── gfpgan/ (optional)
│   └── GFPGANv1.3.pth        (~350MB)
└── whisper/ (from Phase 2)
    └── small.pt              (~466MB)

Total: ~1.6GB (without GFPGAN)
Total: ~2GB (with GFPGAN)
```

## Next Steps

After successful Phase 3 installation:

1. **Test the system**: Run `python main.py` and have a conversation
2. **Adjust quality settings**: Tune `config/config.yaml` for your hardware
3. **Try different avatars**: Experiment with different face images
4. **Measure performance**: Check turn latency with different settings

See `README.md` for usage examples and `docs/PHASE3_VIDEO_PIPELINE.md` for technical details.

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review logs in `logs/` directory
3. Test each component separately using test scripts
4. Open an issue on GitHub with:
   - Your GPU model and VRAM
   - Full error message
   - Config file settings
   - Output of `python quick_test.py`
