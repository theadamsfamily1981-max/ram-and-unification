# Installation Guide - Conversational Avatar AI

Complete installation instructions for all platforms.

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation by Platform](#installation-by-platform)
- [Model Setup](#model-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum
- **OS**: Linux, macOS, or Windows 10+
- **CPU**: 4+ cores (Intel i5 or AMD Ryzen 5)
- **RAM**: 8GB
- **Storage**: 10GB free space
- **Python**: 3.9 - 3.11 (3.10 recommended)
- **Internet**: For initial model downloads

### Recommended
- **OS**: Ubuntu 22.04 LTS or macOS 13+
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **Storage**: 20GB SSD
- **CUDA**: 11.8 or 12.1 (if using GPU)

## Installation by Platform

### 🐧 Linux (Ubuntu/Debian)

```bash
# 1. Update system
sudo apt-get update
sudo apt-get upgrade -y

# 2. Install system dependencies
sudo apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    portaudio19-dev \
    espeak-ng \
    git \
    build-essential

# 3. Clone repository (if not already done)
git clone <your-repo-url>
cd conversational-avatar

# 4. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 5. Upgrade pip
pip install --upgrade pip setuptools wheel

# 6. Install PyTorch (choose one):

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 7. Install other dependencies
pip install -r requirements.txt

# 8. Verify installation
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### 🍎 macOS

```bash
# 1. Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install system dependencies
brew install python@3.10 ffmpeg portaudio espeak-ng

# 3. Clone repository
git clone <your-repo-url>
cd conversational-avatar

# 4. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 5. Upgrade pip
pip install --upgrade pip setuptools wheel

# 6. Install PyTorch (MPS for M1/M2 Macs)
pip install torch torchvision torchaudio

# 7. Install other dependencies
pip install -r requirements.txt

# 8. Verify installation
python -c "import torch; print('PyTorch:', torch.__version__); print('MPS available:', torch.backends.mps.is_available())"
```

### 🪟 Windows

```powershell
# 1. Install Python 3.10
# Download from: https://www.python.org/downloads/

# 2. Install FFmpeg
# Download from: https://ffmpeg.org/download.html
# Add to PATH

# 3. Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/
# Select "Desktop development with C++"

# 4. Clone repository
git clone <your-repo-url>
cd conversational-avatar

# 5. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 6. Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# 7. Install PyTorch (choose one):

# For CPU only:
pip install torch torchvision torchaudio

# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 8. Install other dependencies
pip install -r requirements.txt

# 9. Verify installation
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Model Setup

### 1. Install Ollama (for local LLM)

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

**Windows:**
Download from https://ollama.ai/download

**Start Ollama and download model:**
```bash
# Start Ollama server (if not auto-started)
ollama serve

# In another terminal, pull the model
ollama pull llama3.2

# Verify
ollama list
```

### 2. Download Whisper Model

Whisper models are downloaded automatically on first use, but you can pre-download:

```python
python -c "import whisper; whisper.load_model('small')"
```

### 3. Download TTS Model

Coqui TTS models are downloaded on first use. To pre-download:

```python
python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
```

## Configuration

```bash
# 1. Copy example config
cp config/config.yaml config/config.yaml.local

# 2. Edit configuration
nano config/config.yaml.local  # or use your preferred editor

# 3. Key settings to check:

system:
  device: "cuda"  # Change to "cpu" if no GPU, or "mps" for Mac M1/M2

asr:
  model: "small"  # Adjust based on your hardware

dialogue:
  engine: "ollama"  # Use "openai" if you have API key
  ollama:
    model: "llama3.2"

# 4. Test audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# 5. Set your microphone device index if needed
audio_input:
  device_index: null  # null for default, or specific number
```

## Verification

### Test Each Component

```bash
# Activate environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Test imports
python << EOF
from src.audio_input import AudioRecorder, VoiceActivityDetector
from src.asr import WhisperASR
from src.dialogue import DialogueManager
from src.tts import CoquiTTS
print("✅ All imports successful!")
EOF

# Test audio recording
python << EOF
from src.audio_input import AudioRecorder
recorder = AudioRecorder()
recorder.list_devices()
EOF

# Test ASR
python << EOF
from src.asr import WhisperASR
asr = WhisperASR(model_name="tiny", device="cpu")
print("✅ ASR initialized")
EOF

# Test Dialogue (requires Ollama running)
python << EOF
from src.dialogue import DialogueManager
dm = DialogueManager(engine="ollama", model="llama3.2")
response = dm.get_response("Hello!")
print(f"Response: {response}")
EOF

# Test TTS
python << EOF
from src.tts import CoquiTTS
tts = CoquiTTS(device="cpu")
print("✅ TTS initialized")
EOF
```

### Run the Full System

```bash
python main.py
```

You should see:
```
============================================================
Voice Assistant Prototype Starting
============================================================
Initializing audio input...
Initializing ASR...
Initializing dialogue manager...
Initializing TTS...
All components initialized successfully

============================================================
🎭 VOICE ASSISTANT PROTOTYPE
============================================================

Press Ctrl+C to exit
Say 'exit', 'quit', or 'goodbye' to end conversation

🎤 Listening... (speak now)
```

## Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError: No module named 'X'"
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### 2. "CUDA out of memory"
```yaml
# Edit config/config.yaml:
system:
  device: "cpu"

# Or use smaller models:
asr:
  model: "tiny"
```

#### 3. "Ollama connection refused"
```bash
# Make sure Ollama is running
ollama serve

# Check it's working
ollama list
```

#### 4. "No audio input device"
```bash
# List devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Update config with correct device_index
```

#### 5. "TTS model download fails"
```bash
# Try manual download
python -c "from TTS.api import TTS; print(TTS.list_models())"

# Use a different model
tts:
  coqui:
    model: "tts_models/en/ljspeech/tacotron2-DDC"
```

#### 6. "ImportError: libportaudio"
```bash
# Linux
sudo apt-get install portaudio19-dev

# Mac
brew install portaudio

# Windows
# Reinstall pyaudio: pip install pyaudio
```

#### 7. "Whisper is very slow"
```yaml
# Use smaller model
asr:
  model: "tiny"  # or "base"

# Or install faster-whisper
pip install faster-whisper
```

### Getting Help

If you encounter issues:
1. Check the error message carefully
2. Search existing issues on GitHub
3. Review the troubleshooting section in README.md
4. Create a new issue with:
   - Your OS and Python version
   - Full error message
   - Steps to reproduce

## Next Steps

After successful installation:

1. **Test the voice assistant**: Run `python main.py`
2. **Customize prompts**: Edit `config/prompts.yaml`
3. **Try voice cloning**: Record a voice sample and configure in config.yaml
4. **Read the docs**: Check `/docs/ARCHITECTURE.md` for system design

## Performance Optimization

### For Faster Response

1. **Use GPU** if available
2. **Use smaller models**:
   - ASR: `tiny` or `base`
   - LLM: `phi3` instead of `llama3.2`
3. **Reduce max_tokens** in dialogue config
4. **Use faster TTS**: pyttsx3 (though less natural)

### For Better Quality

1. **Use larger models**:
   - ASR: `medium` or `large`
   - LLM: `llama3.2` or cloud APIs
2. **Use XTTS-v2** for TTS with voice cloning
3. **Enable GPU** acceleration

---

**Installation complete! 🎉**

Run `python main.py` to start your conversational AI assistant!
