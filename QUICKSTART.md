# Quick Start Guide - Talking Avatar API

Get your talking avatar running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed
- 4GB+ RAM
- (Optional) NVIDIA GPU with CUDA for faster processing

## Installation (3 steps)

### 1. Install Dependencies

```bash
# Clone and navigate to directory
cd ram-and-unification

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy example config
cp .env.example .env

# Edit .env if needed (optional)
# Default settings work fine for most users
```

### 3. Run

```bash
# Option A: Using quick start script
chmod +x run.sh
./run.sh

# Option B: Manual start
python -m src.main
```

The API will start at http://localhost:8000

Visit http://localhost:8000/docs for interactive API documentation.

## First Avatar (2 minutes)

### Using Python

```python
# Save this as test.py
import requests

API_URL = "http://localhost:8000/api/v1"

# Upload image
with open("your_photo.jpg", "rb") as f:
    img_resp = requests.post(f"{API_URL}/upload/image", files={"file": f})
    img_file = img_resp.json()["filename"]

# Upload audio
with open("your_voice.wav", "rb") as f:
    aud_resp = requests.post(f"{API_URL}/upload/audio", files={"file": f})
    aud_file = aud_resp.json()["filename"]

# Generate avatar
gen_resp = requests.post(f"{API_URL}/generate", json={
    "image_filename": img_file,
    "audio_filename": aud_file
})

# Download result
if gen_resp.json()["success"]:
    video_url = gen_resp.json()["video_url"]
    video = requests.get(f"{API_URL}{video_url}")
    with open("output.mp4", "wb") as f:
        f.write(video.content)
    print("✓ Avatar saved to output.mp4")
```

Run: `python test.py`

### Using Example Script

```bash
python examples/simple_client.py your_photo.jpg your_voice.wav
```

## Using AI-Generated Content

### With AI Photos

1. Generate a portrait using:
   - Stable Diffusion
   - Midjourney
   - DALL-E
   - Any AI image generator

2. Tips for best results:
   - Front-facing portrait
   - Clear facial features
   - Good lighting
   - 512x512 or higher resolution

### With AI Voices

1. Generate speech using:
   - ElevenLabs
   - Play.ht
   - Bark
   - Coqui TTS
   - Any text-to-speech system

2. Export as WAV or MP3

3. Use with the API!

## Docker (Alternative)

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t talking-avatar .
docker run -p 8000:8000 talking-avatar
```

## Troubleshooting

**"Module not found" error:**
```bash
pip install -r requirements.txt
```

**"FFmpeg not found":**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
Download from https://ffmpeg.org
```

**Server won't start:**
- Check if port 8000 is available
- Try: `PORT=8001 python -m src.main`

**Generation is slow:**
- Set `DEVICE=cuda` in `.env` if you have a GPU
- Reduce `OUTPUT_RESOLUTION` to 256
- Use lower FPS (20 instead of 25)

## Next Steps

- Read [API_GUIDE.md](API_GUIDE.md) for detailed API documentation
- Check [README.md](README.md) for advanced features
- Explore example scripts in `examples/` directory
- Download models: `python manage_models.py download wav2lip`

## Need Help?

- API Docs: http://localhost:8000/docs
- Check examples in `examples/` folder
- Open an issue on GitHub

Happy avatar creating! 🎭
