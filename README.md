# Talking Avatar API 🎭

A powerful, locally-running AI-powered talking avatar system that generates realistic talking videos from static images and audio. Perfect for creating AI avatars, virtual presenters, and animated spokespersons.

## ✨ Features

- **Realistic Lip Sync** - Advanced lip-sync technology for natural mouth movements
- **Local Deployment** - Run entirely on your machine, no cloud dependencies
- **AI-Generated Content Support** - Works with AI-generated photos and voices
- **REST API** - Easy-to-use API endpoints for integration
- **Async Processing** - Support for both synchronous and asynchronous generation
- **Multiple Formats** - Support for various image and audio formats
- **Batch Processing** - Generate multiple avatars in one go
- **Model Caching** - Efficient model management and caching

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ram-and-unification
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Download models (optional)**
```bash
python manage_models.py download wav2lip
```

### Running the Server

```bash
# Development mode
python -m src.main

# Or using uvicorn directly
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## 📖 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Core Endpoints

#### 1. Health Check
```bash
GET /api/v1/health
```

#### 2. Upload Image
```bash
POST /api/v1/upload/image
Content-Type: multipart/form-data

# Using curl
curl -X POST -F "file=@avatar.jpg" http://localhost:8000/api/v1/upload/image
```

#### 3. Upload Audio
```bash
POST /api/v1/upload/audio
Content-Type: multipart/form-data

# Using curl
curl -X POST -F "file=@speech.wav" http://localhost:8000/api/v1/upload/audio
```

#### 4. Generate Avatar (Synchronous)
```bash
POST /api/v1/generate
Content-Type: application/json

{
  "image_filename": "uploaded_image.jpg",
  "audio_filename": "uploaded_audio.wav",
  "output_fps": 25,
  "output_resolution": 512
}
```

#### 5. Generate Avatar (Asynchronous)
```bash
POST /api/v1/generate/async
# Returns job_id for tracking

GET /api/v1/status/{job_id}
# Check generation status
```

#### 6. Download Video
```bash
GET /api/v1/download/{filename}
```

## 💻 Usage Examples

### Python Client Example

```python
import requests
from pathlib import Path

API_URL = "http://localhost:8000/api/v1"

# 1. Upload image
with open("avatar.jpg", "rb") as f:
    response = requests.post(
        f"{API_URL}/upload/image",
        files={"file": f}
    )
    image_filename = response.json()["filename"]

# 2. Upload audio
with open("speech.wav", "rb") as f:
    response = requests.post(
        f"{API_URL}/upload/audio",
        files={"file": f}
    )
    audio_filename = response.json()["filename"]

# 3. Generate avatar
response = requests.post(
    f"{API_URL}/generate",
    json={
        "image_filename": image_filename,
        "audio_filename": audio_filename,
        "output_fps": 25,
        "output_resolution": 512
    }
)

result = response.json()
if result["success"]:
    video_url = result["video_url"]
    print(f"Video generated: {API_URL}{video_url}")

    # 4. Download video
    video_response = requests.get(f"{API_URL}{video_url}")
    with open("output.mp4", "wb") as f:
        f.write(video_response.content)
else:
    print(f"Error: {result['error_message']}")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const API_URL = 'http://localhost:8000/api/v1';

async function generateAvatar() {
  // 1. Upload image
  const imageForm = new FormData();
  imageForm.append('file', fs.createReadStream('avatar.jpg'));

  const imageResponse = await axios.post(
    `${API_URL}/upload/image`,
    imageForm,
    { headers: imageForm.getHeaders() }
  );
  const imageFilename = imageResponse.data.filename;

  // 2. Upload audio
  const audioForm = new FormData();
  audioForm.append('file', fs.createReadStream('speech.wav'));

  const audioResponse = await axios.post(
    `${API_URL}/upload/audio`,
    audioForm,
    { headers: audioForm.getHeaders() }
  );
  const audioFilename = audioResponse.data.filename;

  // 3. Generate avatar
  const generateResponse = await axios.post(`${API_URL}/generate`, {
    image_filename: imageFilename,
    audio_filename: audioFilename,
    output_fps: 25,
    output_resolution: 512
  });

  if (generateResponse.data.success) {
    const videoUrl = generateResponse.data.video_url;
    console.log(`Video generated: ${API_URL}${videoUrl}`);

    // 4. Download video
    const videoResponse = await axios.get(
      `${API_URL}${videoUrl}`,
      { responseType: 'stream' }
    );

    videoResponse.data.pipe(fs.createWriteStream('output.mp4'));
  } else {
    console.error(`Error: ${generateResponse.data.error_message}`);
  }
}

generateAvatar();
```

### cURL Example

```bash
#!/bin/bash

API_URL="http://localhost:8000/api/v1"

# 1. Upload image
IMAGE_RESPONSE=$(curl -X POST -F "file=@avatar.jpg" "$API_URL/upload/image")
IMAGE_FILENAME=$(echo $IMAGE_RESPONSE | jq -r '.filename')

# 2. Upload audio
AUDIO_RESPONSE=$(curl -X POST -F "file=@speech.wav" "$API_URL/upload/audio")
AUDIO_FILENAME=$(echo $AUDIO_RESPONSE | jq -r '.filename')

# 3. Generate avatar
GENERATE_RESPONSE=$(curl -X POST "$API_URL/generate" \
  -H "Content-Type: application/json" \
  -d "{
    \"image_filename\": \"$IMAGE_FILENAME\",
    \"audio_filename\": \"$AUDIO_FILENAME\",
    \"output_fps\": 25,
    \"output_resolution\": 512
  }")

# 4. Download video
VIDEO_URL=$(echo $GENERATE_RESPONSE | jq -r '.video_url')
curl -o output.mp4 "$API_URL$VIDEO_URL"

echo "Video saved to output.mp4"
```

## 🎬 Using AI-Generated Content

This system works great with AI-generated images and voices!

### AI-Generated Images

You can use images from:
- **Stable Diffusion** - Generate portraits
- **Midjourney** - Create realistic faces
- **DALL-E** - Generate avatar images
- **Custom AI models** - Any AI portrait generator

Tips for best results:
- Use front-facing portraits
- Ensure good lighting in the generated image
- Higher resolution images work better (512x512 or higher)
- Clear facial features improve lip-sync quality

### AI-Generated Voices

Compatible with voices from:
- **ElevenLabs** - Realistic voice cloning
- **Play.ht** - Text-to-speech with emotions
- **Bark** - Open-source TTS
- **Coqui TTS** - Local voice synthesis
- **Any TTS system** - As long as it outputs WAV/MP3

Audio requirements:
- Format: WAV, MP3
- Sample rate: 16kHz or higher
- Mono or stereo
- Clear speech without heavy background noise

## 🛠️ Configuration

Edit `.env` file to customize:

```bash
# Server
HOST=0.0.0.0
PORT=8000

# Processing
DEVICE=cuda          # Use 'cuda' for GPU, 'cpu' for CPU
OUTPUT_FPS=25        # Video frame rate
OUTPUT_RESOLUTION=512 # Output size (256, 512, or 1024)

# Directories
MODEL_CACHE_DIR=./models
UPLOAD_DIR=./uploads
OUTPUT_DIR=./outputs
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t talking-avatar .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  talking-avatar
```

## 📊 Model Management

```bash
# List available models
python manage_models.py list

# Download specific model
python manage_models.py download wav2lip

# Download all models
python manage_models.py download all

# Delete a model
python manage_models.py delete wav2lip
```

## 🎯 Performance Tips

1. **GPU Acceleration**: Set `DEVICE=cuda` for much faster processing
2. **Resolution**: Lower resolution (256) is faster, higher (1024) is better quality
3. **FPS**: 25 FPS is standard, lower FPS reduces processing time
4. **Batch Processing**: Use async endpoint for multiple avatars

## 🔧 Troubleshooting

### No GPU detected
- Make sure PyTorch with CUDA is installed
- Set `DEVICE=cpu` in `.env` if no GPU available

### Poor lip sync quality
- Ensure audio is clear and loud enough
- Use front-facing images with visible mouth
- Try higher resolution output

### Out of memory errors
- Reduce `OUTPUT_RESOLUTION`
- Set `DEVICE=cpu` if GPU memory is limited
- Process shorter audio clips

## 📝 Requirements

- Python 3.8+
- FFmpeg (for video encoding)
- 4GB+ RAM
- (Optional) NVIDIA GPU with CUDA for faster processing

## 🤝 Contributing

Contributions welcome! Please feel free to submit pull requests.

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Wav2Lip paper and implementation
- FastAPI framework
- PyTorch team
- Open source AI community

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation at `/docs`
- Review example scripts in `examples/`

---

**Made with ❤️ for the AI community**
