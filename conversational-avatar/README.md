# Conversational Talking Avatar AI 🎭🗣️

A modular, real-time conversational AI system that combines speech recognition, natural language understanding, text-to-speech, and realistic talking head video generation to create an interactive virtual assistant.

## ✨ Features

### Current (Phase 2 - Audio Prototype)
- ✅ **Voice Input** - Microphone capture with voice activity detection
- ✅ **Speech Recognition** - OpenAI Whisper (state-of-the-art ASR)
- ✅ **AI Dialogue** - Local LLM via Ollama (Llama 3.2) or cloud APIs
- ✅ **Natural TTS** - Coqui TTS with voice cloning
- ✅ **Conversational Memory** - Context-aware responses
- ✅ **100% Local** - No cloud required (if using local models)

### Coming Soon
- 🔜 **Talking Head Video** - Realistic lip-sync avatars (Phase 3)
- 🔜 **1080p Output** - High-quality video generation (Phase 4)
- 🔜 **Web UI** - Browser-based interface (Phase 6)
- 🔜 **Real-time Streaming** - Lower latency video (Phase 5)

## 🎯 Current Status: Phase 2 Complete

**What Works:**
- Full voice conversation loop (listen → transcribe → think → speak)
- Modular architecture ready for video integration
- Multiple configuration options
- Voice activity detection for natural conversation flow

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

```
🎤 Listening... (speak now)
👤 You: What is machine learning?

🤔 Thinking...
🤖 Assistant: Machine learning is a branch of AI where computers learn from data
              to make predictions or decisions without being explicitly programmed.
              It powers things like recommendation systems and voice assistants.

🔊 Speaking...
⏱️  Timing: ASR=0.5s | LLM=1.2s | TTS=1.8s | Total=3.5s

▶️  Continue? (y/n):
```

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

- [x] Phase 1: Architecture design
- [x] Phase 2: Minimal voice assistant prototype
- [ ] Phase 3: Talking head/avatar video generation
- [ ] Phase 4: Full voice + video integration
- [ ] Phase 5: Quality improvements & optimization
- [ ] Phase 6: Web UI and advanced controls

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
