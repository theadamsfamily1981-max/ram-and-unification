# RVC Voice Setup Guide

Complete guide for setting up custom voice synthesis using RVC (Retrieval-based Voice Conversion) models with the Avatar API.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start - Pre-trained Models](#quick-start)
4. [Training Your Own Voice Model](#training-custom-voice)
5. [Integration with Avatar API](#integration)
6. [Oobabooga Integration](#oobabooga)
7. [Voice Tuning](#tuning)
8. [Troubleshooting](#troubleshooting)

---

## Overview

RVC (Retrieval-based Voice Conversion) allows you to convert any TTS voice into a custom voice using a trained model. This means:

- **Custom voice character**: Convert generic TTS to your specific voice/character
- **Emotional range**: Preserve emotions and expressiveness
- **Fast conversion**: Real-time or near real-time voice conversion
- **High quality**: Natural-sounding results

---

## Prerequisites

### System Requirements

- **RAM**: 8GB+ (16GB recommended)
- **GPU**: NVIDIA GPU with 6GB+ VRAM (recommended, but CPU works)
- **Storage**: 5GB+ for models and training data
- **OS**: Windows, Linux, or macOS

### Software Requirements

```bash
# Python 3.9-3.11
python --version

# PyTorch with CUDA (for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or for CPU only
pip install torch torchvision torchaudio
```

---

## Quick Start - Pre-trained Models

### Option 1: Using Existing RVC Models

If you already have a `.pth` RVC model file:

1. **Place model files**:
   ```bash
   mkdir -p models/rvc
   cp YourModel.pth models/rvc/Ara.pth
   cp YourModel.index models/rvc/Ara.index  # Optional but recommended
   ```

2. **Configure**:
   ```yaml
   # config/avatar_config.yaml
   voice:
     engine: "rvc"
     default_voice: "ara"
     ara:
       rvc_model: "models/rvc/Ara.pth"
       rvc_index: "models/rvc/Ara.index"
       pitch: 0.12
       index_rate: 0.65
       speed: 0.95
   ```

3. **Test**:
   ```bash
   python test_rvc_voice.py --text "Hello, this is a test" --output test.wav
   ```

### Option 2: Download Pre-trained Models

Popular RVC model sources:

- **Hugging Face**: https://huggingface.co/models?search=rvc
- **AI Hub**: https://www.aihub.wtf/
- **Voice Models Discord**: Various communities share trained models

```bash
# Example: Download from Hugging Face
git lfs install
git clone https://huggingface.co/USERNAME/MODEL_NAME
cp MODEL_NAME/*.pth models/rvc/
```

---

## Training Your Own Voice Model

### Step 1: Prepare Training Data

Collect **high-quality audio samples** of the target voice:

- **Format**: WAV files, 16-bit, 44.1kHz or 48kHz
- **Duration**: 10-30 minutes total (more is better)
- **Quality**: Clean, no background noise, consistent volume
- **Content**: Varied speech (different emotions, words, tones)

```bash
mkdir training_data/ara
# Place your audio files here
```

**Tips for best results**:
- Remove silence/noise with Audacity
- Normalize audio levels
- Include emotional range (happy, sad, angry, etc.)
- Include various phonemes and sounds

### Step 2: Install Mangio-RVC-Fork

**Mangio-RVC-Fork** is the most popular and easiest RVC trainer:

```bash
# Clone repository
git clone https://github.com/Mangio621/Mangio-RVC-Fork.git
cd Mangio-RVC-Fork

# Install dependencies
pip install -r requirements.txt

# Download required models
python download_models.py
```

### Step 3: Preprocess Audio

```bash
# In Mangio-RVC-Fork directory
python infer/modules/train/preprocess.py \
    --exp_dir "ara_voice" \
    --input_dir "../training_data/ara" \
    --sr 40000 \
    --n_p 4
```

### Step 4: Extract Features

```bash
python infer/modules/train/extract/extract_f0_print.py \
    --exp_dir "ara_voice" \
    --f0method "rmvpe" \
    --hop_length 128

python infer/modules/train/extract_feature_print.py \
    --exp_dir "ara_voice" \
    --version "v2" \
    --f0method "rmvpe"
```

### Step 5: Train the Model

```bash
python infer/modules/train/train.py \
    --exp_dir "ara_voice" \
    --sr 40000 \
    --batch_size 8 \
    --total_epoch 500 \
    --save_every_epoch 50 \
    --if_latest 1 \
    --save_only_latest 1 \
    --if_cache_data_in_gpu 1
```

**Training time**: 30 minutes to 2 hours depending on data and GPU

**Monitor training**:
- Check `logs/ara_voice/` for training logs
- Listen to checkpoints every 50 epochs
- Training typically converges around 300-500 epochs

### Step 6: Generate Index File

```bash
python infer/modules/train/train_index.py \
    --exp_dir "ara_voice" \
    --version "v2"
```

### Step 7: Export Model

```bash
# Your trained model:
cp logs/ara_voice/ara_voice.pth ../models/rvc/Ara.pth
cp logs/ara_voice/added_*.index ../models/rvc/Ara.index
```

---

## Integration with Avatar API

### Update Configuration

```yaml
# config/avatar_config.yaml
voice:
  engine: "rvc"
  default_voice: "ara"
  ara:
    rvc_model: "models/rvc/Ara.pth"
    rvc_index: "models/rvc/Ara.index"
    pitch: 0.12          # Adjust based on voice (negative = lower, positive = higher)
    index_rate: 0.65     # 0.0-1.0, higher = more like training data
    speed: 0.95          # Speech speed multiplier
    base_tts_voice: "jenny"  # Base TTS before RVC conversion
```

### Use in Code

```python
from src.voice.rvc_integration import RVCVoiceConverter
from pathlib import Path

# Initialize converter
rvc = RVCVoiceConverter(
    model_path=Path("models/rvc/Ara.pth"),
    index_path=Path("models/rvc/Ara.index"),
    pitch=0.12,
    index_rate=0.65
)

# Convert TTS audio to custom voice
rvc.convert(
    input_audio=Path("outputs/tts_jenny.wav"),
    output_audio=Path("outputs/tts_ara.wav")
)
```

### Full Pipeline Example

```python
from src.integrations.ara_avatar_backend import AraAvatarBackend

# Initialize backend with RVC
ara = AraAvatarBackend(
    ollama_model="ara",
    config={
        "voice_engine": "rvc",
        "rvc_model": "models/rvc/Ara.pth"
    }
)

# Generate avatar with custom voice
result = await ara.generate_avatar_response(
    prompt="Tell me about yourself",
    use_tts=True
)

# Result contains video with custom RVC voice
print(f"Video: {result['video_path']}")
```

---

## Oobabooga Integration

### Setup Oobabooga with RVC Extension

1. **Install oobabooga text-generation-webui**:
   ```bash
   git clone https://github.com/oobabooga/text-generation-webui
   cd text-generation-webui
   ./start_linux.sh  # or start_windows.bat
   ```

2. **Install TTS extensions**:
   ```bash
   cd extensions
   git clone https://github.com/erew123/alltalk_tts
   git clone https://github.com/Continuation/silero-tts
   ```

3. **Configure RVC in extensions**:
   - Open oobabooga web UI: http://localhost:7860
   - Go to **Session** tab → **Extensions**
   - Enable: `alltalk_tts` and `silero_tts`
   - Configure RVC settings:
     - Model: `Ara.pth`
     - Pitch: `0.12`
     - Index Rate: `0.65`

4. **Start with API**:
   ```bash
   python server.py --api --extensions alltalk_tts silero_tts --listen
   ```

### Use Oobabooga API

```python
from src.voice.rvc_integration import OobaboogaRVCClient

# Initialize client
client = OobaboogaRVCClient(
    api_url="http://localhost:5000",
    rvc_model="Ara",
    pitch=0.12,
    index_rate=0.65
)

# Generate TTS with RVC
client.generate_tts(
    text="Hello, I'm Ara, your AI assistant",
    output_path=Path("output.wav"),
    voice="en-US-JennyNeural"
)
```

### Avatar API Configuration

```yaml
# config/avatar_config.yaml
oobabooga:
  enabled: true
  api_url: "http://localhost:5000"
  tts_extensions:
    - "silero_tts"
    - "alltalk_tts"
  rvc_enabled: true
  rvc_pitch: 0.12
  rvc_index_rate: 0.65
  streaming: true
```

---

## Voice Tuning

### Pitch Adjustment

- **Positive values** (0.1 - 12.0): Higher/feminine voice
- **Negative values** (-12.0 - -0.1): Lower/masculine voice
- **Recommended range**: -2.0 to +2.0 for natural sound

```yaml
ara:
  pitch: 0.12  # Slightly higher, feminine
  # pitch: -0.5  # Slightly lower, masculine
  # pitch: 1.2   # Noticeably higher
```

### Index Rate

Controls how much the model follows the training data:

- **0.0**: Completely ignores training data (sounds like original)
- **0.5**: Balanced blend
- **1.0**: Strictly follows training data

```yaml
ara:
  index_rate: 0.65  # Good balance
  # index_rate: 0.4  # More original TTS character
  # index_rate: 0.9  # More like training voice
```

### Speed

Speech rate multiplier:

```yaml
ara:
  speed: 0.95  # 5% slower (sultry, deliberate)
  # speed: 1.0   # Normal speed
  # speed: 1.1   # 10% faster (energetic)
```

### Finding Your Perfect Settings

```python
# Test different settings
test_phrases = [
    "Hello, how can I help you today?",
    "I'm feeling happy and excited!",
    "This is a more serious, thoughtful message."
]

for pitch in [-0.5, 0.0, 0.12, 0.5, 1.0]:
    for index_rate in [0.4, 0.5, 0.65, 0.8]:
        # Generate with settings
        output = f"test_p{pitch}_i{index_rate}.wav"
        # Listen and compare
```

---

## Troubleshooting

### Model Not Loading

**Error**: `FileNotFoundError: RVC model not found`

**Solution**:
- Check file path in config
- Ensure `.pth` file exists
- Check file permissions

```bash
ls -lh models/rvc/Ara.pth
# Should show file with read permissions
```

### Voice Sounds Robotic/Distorted

**Causes**:
1. **Index rate too high**: Try lowering to 0.4-0.6
2. **Poor training data**: Retrain with better audio
3. **Pitch too extreme**: Keep within ±2.0

**Fix**:
```yaml
ara:
  index_rate: 0.5  # Lower from 0.65
  pitch: 0.0       # Reset to neutral
```

### Voice Doesn't Sound Like Training Data

**Causes**:
1. **Index rate too low**: Increase to 0.7-0.9
2. **No index file**: Make sure `.index` file is present
3. **Undertrained model**: Train longer (500+ epochs)

**Fix**:
```yaml
ara:
  index_rate: 0.8  # Increase
  rvc_index: "models/rvc/Ara.index"  # Ensure index is set
```

### Slow Conversion Time

**Solutions**:
1. **Use GPU**: Ensure CUDA is enabled
2. **Check device**: Should auto-detect, but verify:
   ```python
   from src.utils.device_utils import get_device_info
   print(get_device_info())
   ```
3. **Reduce quality**: Use lower sample rates

### Out of Memory (OOM)

**For GPU**:
```python
# Reduce batch size
os.environ['RVC_BATCH_SIZE'] = '1'
```

**For CPU**:
- Close other applications
- Use smaller models
- Process shorter audio clips

### Oobabooga API Not Responding

**Check**:
1. **Server running**: http://localhost:5000
2. **Extensions enabled**: Check UI
3. **Firewall**: Allow port 5000

```bash
# Test API
curl http://localhost:5000/api/v1/model
```

---

## Advanced Tips

### Multi-Voice Setup

Support multiple voice models:

```yaml
voice:
  voices:
    ara:
      rvc_model: "models/rvc/Ara.pth"
      pitch: 0.12
    morgan:
      rvc_model: "models/rvc/Morgan.pth"
      pitch: -0.5
    zara:
      rvc_model: "models/rvc/Zara.pth"
      pitch: 0.8
```

### Voice Blending

Mix two voices:

```python
# Convert with first voice
rvc1.convert(input, temp1)
# Convert with second voice
rvc2.convert(input, temp2)
# Blend outputs
blend_audio(temp1, temp2, output, ratio=0.5)
```

### Real-time Streaming

For low-latency voice chat:

```python
# Use smaller chunks
rvc.convert(
    audio_chunk,
    output_chunk,
    use_streaming=True
)
```

---

## Resources

- **Mangio-RVC-Fork**: https://github.com/Mangio621/Mangio-RVC-Fork
- **RVC Training Guide**: https://docs.aihub.wtf/rvc/
- **Voice Model Library**: https://voice-models.com/
- **Discord Communities**: RVC Voice Models, AI Hub

---

## Support

For issues specific to this integration:
- Open an issue on GitHub
- Check logs in `logs/avatar_api.log`
- Enable detailed logging:
  ```yaml
  monitoring:
    detailed_logging: true
  ```

Happy voice synthesis! 🎙️✨
