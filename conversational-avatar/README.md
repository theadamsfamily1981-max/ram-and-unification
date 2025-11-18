# Conversational Talking Avatar AI 🎭🗣️

A modular, real-time conversational AI system that combines speech recognition, natural language understanding, text-to-speech, and realistic talking head video generation to create an interactive virtual assistant.

## ✨ Features

### ✅ Completed Features (Phases 1-3)
- ✅ **Voice Input** - Microphone capture with voice activity detection
- ✅ **Speech Recognition** - OpenAI Whisper (state-of-the-art ASR)
- ✅ **AI Dialogue** - Local LLM via Ollama (Llama 3.2) or cloud APIs
- ✅ **Natural TTS** - Coqui TTS with voice cloning
- ✅ **Talking Head Video** - Wav2Lip lip-sync avatar generation (NEW!)
- ✅ **Video Playback** - OpenCV/FFplay media player (NEW!)
- ✅ **Conversational Memory** - Context-aware responses
- ✅ **Multiple Quality Modes** - Standard (720p) and High (1080p)
- ✅ **100% Local** - No cloud required (if using local models)

### Coming Soon
- 🔜 **Web UI** - Browser-based interface (Phase 6)
- 🔜 **Real-time Streaming** - Lower latency video (Phase 5)
- 🔜 **Enhanced Face Quality** - GFPGAN integration (Phase 5)

## 🎯 Current Status: Phase 3 Complete

**What Works:**
- Full voice + video conversation loop:
  - Listen (microphone + VAD)
  - Transcribe (Whisper ASR)
  - Think (Ollama/GPT)
  - Speak (Coqui TTS)
  - **Generate Video (Wav2Lip)** ← NEW!
  - **Play Video (synchronized audio + video)** ← NEW!
- 720p @ 25fps (standard mode) or 1080p @ 30fps (high quality mode)
- Graceful degradation (falls back to audio if video fails)
- Performance metrics and timing

## 🚀 Quick Start

### Prerequisites

```bash
# System dependencies
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y ffmpeg portaudio19-dev espeak-ng

# macOS:
brew install ffmpeg portaudio espeak-ng

# Windows:
# Download FFmpeg from https://ffmpeg.org
# Install Python 3.9+
```

### Installation

```bash
# 1. Clone repository
cd conversational-avatar

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# For GPU support (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install and start Ollama (for local LLM)
# Visit: https://ollama.ai
# Then: ollama pull llama3.2

# 5. Copy and configure settings
cp config/config.yaml config/config.yaml.local
# Edit config/config.yaml.local with your preferences
```

### Running the Voice Assistant

```bash
# Run the prototype
python main.py

# The assistant will:
# 1. Listen when you speak
# 2. Transcribe your speech
# 3. Generate a response
# 4. Speak the response back to you
```

## 📖 Usage Example

### With Video (Phase 3)

```
🎤 Listening... (speak now)
👤 You: What is machine learning?

🤔 Thinking...
🤖 Assistant: Machine learning is a branch of AI where computers learn from data
              to make predictions or decisions without being explicitly programmed.
              It powers things like recommendation systems and voice assistants.

🔊 Speaking...
🎬 Generating video...
📺 Playing video...
⏱️  Timing: ASR=0.5s | LLM=1.2s | TTS=1.8s | Video=3.2s | Total=6.7s

▶️  Continue? (y/n):
```

### Audio-Only Mode (Phase 2)

Set `talking_head.enabled: false` in config for audio-only:

```
🎤 Listening... (speak now)
👤 You: What is machine learning?

🤔 Thinking...
🤖 Assistant: Machine learning is a branch of AI...

🔊 Speaking...
⏱️  Timing: ASR=0.5s | LLM=1.2s | TTS=1.8s | Total=3.5s
```

## 🎬 Phase 3: Video Generation

### Quick Setup for Video

```bash
# 1. Download Wav2Lip models
python scripts/download_models.py --phase3

# 2. Prepare avatar image
cp /path/to/your/photo.jpg assets/avatars/default.jpg

# 3. Enable video in config
# Edit config/config.yaml:
talking_head:
  enabled: true
  avatar_image: "assets/avatars/default.jpg"
  quality_mode: "standard"  # or "high" for 1080p

# 4. Run!
python main.py
```

### Demo Scripts

**Text-to-Video Demo** (no mic needed):
```bash
python scripts/demo_talking_avatar_from_text.py \
    --text "Hello, welcome to the talking avatar system!" \
    --output outputs/demos/demo.mp4 \
    --play
```

**Test Video Playback**:
```bash
python scripts/play_video_test.py --video outputs/demos/demo.mp4
```

### Video Quality Modes

#### Standard Mode (720p) - RTX 5060
- Resolution: 1280x720 @ 25fps
- Generation time: ~3-4s per 5s audio
- VRAM: ~1.5GB
- Best for: Real-time conversation

#### High Quality Mode (1080p) - RTX 3090+
- Resolution: 1920x1080 @ 30fps
- Generation time: ~6-7s per 5s audio
- VRAM: ~3-4GB
- Best for: Recording, demos, presentations

Configure in `config/config.yaml`:
```yaml
talking_head:
  quality_mode: "high"  # or "standard"
```

### Troubleshooting Video

**Issue: "No face detected in avatar"**
- Use a front-facing portrait with clear facial features
- Ensure good lighting
- Recommended: 512x512 or larger

**Issue: "CUDA out of memory"**
```yaml
# In config.yaml:
talking_head:
  quality_mode: "standard"  # Use standard instead of high
  standard:
    face_det_batch_size: 2  # Reduce batch size
```

**Issue: "Video generation is slow"**
- Use GPU (cuda) not CPU
- Use `quality_mode: "standard"`
- Enable FP16: `use_half_precision: true`

See `INSTALL_PHASE3.md` for detailed troubleshooting.

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

### Quick Settings

```yaml
system:
  device: "cuda"  # cuda, cpu, or mps (Mac M1/M2)

asr:
  model: "small"  # tiny, base, small (RECOMMENDED), medium, large

dialogue:
  engine: "ollama"  # ollama (local) or openai (cloud)
  ollama:
    model: "llama3.2"  # llama3.2, phi3, mistral

tts:
  engine: "coqui"
  coqui:
    speaker_wav: null  # Path to voice sample for cloning
```

### Performance vs Quality

**Fast Mode (CPU-friendly):**
```yaml
asr:
  model: "tiny"  # 1s transcription
dialogue:
  ollama:
    model: "phi3"  # Smaller, faster model
```

**Quality Mode (GPU recommended):**
```yaml
asr:
  model: "medium"  # More accurate
dialogue:
  ollama:
    model: "llama3.2"  # Better responses
```

## 🎭 Voice Cloning

To clone a specific voice:

1. Record a 3-10 second audio sample of the target voice
2. Save as `assets/voices/my_voice.wav`
3. Update config:

```yaml
tts:
  coqui:
    speaker_wav: "assets/voices/my_voice.wav"
```

## 📊 Architecture

```
User Speech
    ↓
┌─────────────────┐
│ Audio Recorder  │ ← Voice Activity Detection
│  + VAD          │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Whisper ASR     │ ← Speech-to-Text
└────────┬────────┘
         ↓
┌─────────────────┐
│ Dialogue        │ ← LLM Response Generation
│ Manager         │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Coqui TTS       │ ← Text-to-Speech
└────────┬────────┘
         ↓
    Audio Output
```

## 🔧 Troubleshooting

### "Ollama connection refused"
```bash
# Make sure Ollama is running:
ollama serve

# In another terminal:
ollama pull llama3.2
```

### "No audio input device found"
```bash
# List available devices:
python -c "import sounddevice as sd; print(sd.query_devices())"

# Set device index in config.yaml:
audio_input:
  device_index: X  # Replace X with your device number
```

### "CUDA out of memory"
```yaml
# Switch to CPU or use smaller models:
system:
  device: "cpu"

asr:
  model: "tiny"  # or "base"
```

### "TTS is too slow"
```yaml
# Use faster TTS (less quality):
tts:
  engine: "pyttsx3"  # Instant but robotic

# Or use smaller Coqui model:
tts:
  coqui:
    model: "tts_models/en/ljspeech/tacotron2-DDC"
```

## 📁 Project Structure

```
conversational-avatar/
├── config/
│   ├── config.yaml        # Main configuration
│   └── prompts.yaml       # System prompts for different personas
├── src/
│   ├── audio_input/       # Audio recording + VAD
│   ├── asr/               # Speech recognition (Whisper)
│   ├── dialogue/          # LLM dialogue management
│   ├── tts/               # Text-to-speech (Coqui)
│   ├── orchestrator/      # Main conversation loop
│   └── utils/             # Logging and utilities
├── outputs/
│   ├── audio/             # Generated TTS audio
│   ├── recordings/        # Recorded user speech (optional)
│   └── conversations/     # Conversation logs (optional)
├── main.py                # Entry point
└── requirements.txt       # Python dependencies
```

## 🛣️ Roadmap

- [x] **Phase 1**: Architecture design ✅
- [x] **Phase 2**: Minimal voice assistant prototype ✅
- [x] **Phase 3**: Talking head/avatar video generation ✅ **← YOU ARE HERE**
- [ ] **Phase 4**: Full voice + video integration (complete)
- [ ] **Phase 5**: Quality improvements & optimization
  - [ ] GFPGAN face enhancement
  - [ ] Real-time streaming
  - [ ] Performance optimizations
  - [ ] Batch processing
- [ ] **Phase 6**: Web UI and advanced controls
  - [ ] Browser-based interface
  - [ ] Real-time webcam avatar
  - [ ] Multiple avatar switching
  - [ ] Session recording

## 🔒 Privacy & Ethics

### Privacy
- ✅ **100% Local Option** - Use local models (no data leaves your machine)
- ⚠️ **Cloud APIs** - OpenAI, ElevenLabs send data externally
- ✅ **No Recording** - Disable audio saving in config
- ✅ **Auto Cleanup** - Temporary files removed after session

### Ethics & Responsible AI
- ⚠️ **Deepfakes** - This technology can be misused
- ✅ **Consent** - Only clone voices with permission
- ✅ **Disclosure** - Be transparent about AI-generated content
- ✅ **Use Cases** - Designed for accessibility, education, and creative projects

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- OpenAI Whisper - Speech recognition
- Ollama - Local LLM deployment
- Coqui TTS - Text-to-speech
- WebRTC VAD - Voice activity detection

## 📞 Support

- Check `/docs/ARCHITECTURE.md` for detailed design docs
- Review `config/config.yaml` for all options
- Open an issue for bugs or feature requests

---

**Made with ❤️ for creating accessible, privacy-respecting AI assistants**
