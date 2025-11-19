# Conversational Talking Avatar AI - System Architecture

## Overview

A modular, real-time conversational AI system that combines speech recognition, natural language understanding, text-to-speech, and realistic talking head video generation to create an interactive virtual assistant.

## System Requirements

### Hardware Recommendations

**Minimum:**
- CPU: 4+ cores (Intel i5 or AMD Ryzen 5)
- RAM: 8GB
- GPU: Optional, but recommended (NVIDIA GTX 1060 6GB or better)
- Storage: 10GB for models
- Audio: Microphone input, speaker output

**Recommended:**
- CPU: 8+ cores (Intel i7/i9 or AMD Ryzen 7/9)
- RAM: 16GB+
- GPU: NVIDIA RTX 3060+ with 8GB+ VRAM
- Storage: 20GB SSD for models
- Audio: Quality USB microphone, good speakers/headphones

### Performance Targets

- **Latency**: < 3 seconds from user speech end to avatar response start
- **Video Quality**: 1080p @ 25-30 FPS
- **Audio Quality**: 16kHz+ TTS output
- **Accuracy**: 95%+ ASR word accuracy in quiet environments

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONVERSATIONAL AI SYSTEM                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐
│   USER       │
│  (speaks)    │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  1. AUDIO INPUT MODULE                                           │
│     - Microphone capture (PyAudio / sounddevice)                 │
│     - Voice Activity Detection (VAD)                             │
│     - Audio preprocessing (noise reduction)                      │
│     - Buffer management                                          │
│                                                                  │
│  Input:  Microphone stream                                      │
│  Output: audio_chunk (numpy array, 16kHz, mono)                 │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. ASR MODULE (Speech-to-Text)                                  │
│     - Whisper (OpenAI) - RECOMMENDED                             │
│     - Alternative: Vosk, SpeechRecognition                       │
│     - Real-time streaming or batch mode                          │
│     - Language detection                                         │
│                                                                  │
│  Input:  audio_chunk (numpy/bytes)                              │
│  Output: text (str), confidence (float)                          │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  3. DIALOGUE MANAGER MODULE                                      │
│     - LLM Integration:                                           │
│       * Local: Llama.cpp, Ollama                                 │
│       * Cloud: OpenAI GPT, Anthropic Claude                      │
│     - Conversation history/context                               │
│     - Prompt engineering                                         │
│     - Response streaming (optional)                              │
│                                                                  │
│  Input:  user_text (str), conversation_history (list)           │
│  Output: assistant_text (str)                                    │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  4. TTS MODULE (Text-to-Speech)                                  │
│     - Coqui TTS (XTTS-v2) - RECOMMENDED for quality              │
│     - Alternative: pyttsx3, gTTS, Bark, ElevenLabs API           │
│     - Voice cloning capability                                   │
│     - Prosody control                                            │
│     - Streaming TTS (for lower latency)                          │
│                                                                  │
│  Input:  text (str), voice_profile (str/path)                   │
│  Output: audio_file (wav/mp3), phoneme_timings (optional)       │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  5. TALKING HEAD MODULE                                          │
│     - Wav2Lip / Wav2Lip-HD (RECOMMENDED)                         │
│     - Alternative: SadTalker, Live Portrait, First Order Motion  │
│     - Lip-sync from audio                                        │
│     - Face animation (subtle head/eye movements)                 │
│     - 1080p upscaling                                            │
│                                                                  │
│  Input:  audio_file (wav), avatar_image (jpg/png)               │
│  Output: video_file (mp4, 1080p, 25-30fps)                      │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  6. VIDEO PLAYER / OUTPUT MODULE                                 │
│     - OpenCV display                                             │
│     - PyQt5/6 video widget                                       │
│     - Web-based player (HLS streaming)                           │
│     - Audio synchronization                                      │
│                                                                  │
│  Input:  video_file (mp4)                                        │
│  Output: Display to screen + audio playback                      │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  7. ORCHESTRATOR / EVENT LOOP                                    │
│     - Main event loop                                            │
│     - State management (idle, listening, thinking, speaking)     │
│     - Error handling and recovery                                │
│     - Logging and monitoring                                     │
│     - Async task management                                      │
│     - Resource cleanup                                           │
└──────────────────────────────────────────────────────────────────┘
```

## Module Specifications

### 1. Audio Input Module

**Purpose:** Capture and preprocess audio from microphone

**Libraries:**
- `sounddevice` - Cross-platform audio I/O (RECOMMENDED)
- `pyaudio` - Alternative, widely used
- `webrtcvad` - Voice Activity Detection
- `noisereduce` - Audio noise reduction

**Interface:**
```python
class AudioInput:
    def start_recording(self) -> None
    def stop_recording(self) -> None
    def get_audio_chunk(self) -> np.ndarray  # Returns: (samples,) @ 16kHz
    def is_speech_detected(self) -> bool  # VAD
```

**Outputs:**
- `audio_chunk`: numpy array, shape (N,), dtype float32, 16kHz mono
- `is_speaking`: bool indicating voice activity

**Caching:** None required

---

### 2. ASR Module (Speech-to-Text)

**Purpose:** Convert speech audio to text

**Recommended: OpenAI Whisper**
- **Why:** State-of-the-art accuracy, multilingual, open-source
- **Models:** tiny (39M) → base (74M) → small (244M) → medium (769M) → large (1550M)
- **Trade-offs:**
  - Tiny/Base: Fast but less accurate
  - Small: Good balance (RECOMMENDED for real-time)
  - Medium/Large: Best accuracy but slower

**Libraries:**
```python
# Primary
import whisper  # OpenAI Whisper

# Alternatives
import vosk  # Lightweight, offline
from google.cloud import speech  # Cloud API, high accuracy
```

**Interface:**
```python
class ASREngine:
    def __init__(self, model_name: str = "small")
    def transcribe(self, audio: np.ndarray) -> dict:
        # Returns: {"text": str, "language": str, "confidence": float}
    def transcribe_stream(self, audio_stream) -> Iterator[str]
```

**Outputs:**
- `text`: Transcribed text (str)
- `language`: Detected language code
- `confidence`: 0.0-1.0 confidence score

**Caching:**
- Model weights (loaded once at startup)
- Mel spectrogram preprocessing

---

### 3. Dialogue Manager Module

**Purpose:** Manage conversation and generate responses using LLM

**LLM Options:**

**Local (Privacy + No API costs):**
- **Ollama** with Llama 3.2, Phi-3, Mistral (RECOMMENDED for local)
- **llama.cpp** Python bindings
- **Hugging Face Transformers** (GPT-2, Llama, etc.)

**Cloud (Better quality):**
- **OpenAI GPT-4/GPT-3.5** (Best quality, costs money)
- **Anthropic Claude** (High quality, costs money)
- **Groq** (Fast inference, free tier)

**Libraries:**
```python
# Local
import ollama  # RECOMMENDED - Easy to use
from llama_cpp import Llama
from transformers import AutoModelForCausalLM

# Cloud
import openai
import anthropic
```

**Interface:**
```python
class DialogueManager:
    def __init__(self, model: str, system_prompt: str)
    def add_user_message(self, text: str) -> None
    def get_response(self) -> str
    def clear_history(self, keep_last_n: int = 0) -> None
    def get_history(self) -> list[dict]
```

**Outputs:**
- `response_text`: AI-generated response (str)
- `conversation_history`: List of {role, content} dicts

**Caching:**
- Model weights (persistent)
- Conversation history (session)
- System prompts

---

### 4. TTS Module (Text-to-Speech)

**Purpose:** Convert text to natural-sounding speech

**Recommended: Coqui TTS (XTTS-v2)**
- **Why:** Best open-source quality, voice cloning, multilingual
- **Models:** XTTS-v2 (voice cloning), VITS (fast, single voice)
- **Trade-offs:**
  - XTTS-v2: High quality, voice cloning, but slower (~2s for 5s audio on GPU)
  - VITS: Fast but no voice cloning

**Alternatives:**
- **Bark** - Emotional, non-verbal sounds, but slower
- **pyttsx3** - Very fast, offline, but robotic
- **ElevenLabs API** - Best quality, but paid API
- **Azure/Google Cloud TTS** - Good quality, paid APIs

**Libraries:**
```python
# Primary
from TTS.api import TTS  # Coqui TTS - RECOMMENDED

# Alternatives
import pyttsx3  # Offline, fast, robotic
from bark import SAMPLE_RATE, generate_audio  # Suno Bark
```

**Interface:**
```python
class TTSEngine:
    def __init__(self, model_name: str, voice_sample: str = None)
    def synthesize(self, text: str, output_path: str) -> str:
        # Returns: path to audio file
    def synthesize_stream(self, text: str) -> Iterator[bytes]
    def get_phoneme_timings(self, text: str) -> list[tuple]
```

**Outputs:**
- `audio_file`: WAV file path (16kHz or 22kHz)
- `phoneme_timings`: List of (phoneme, start_time, end_time) - optional

**Caching:**
- TTS model weights
- Voice embeddings (if using voice cloning)
- Common phrases (optional)

---

### 5. Talking Head Module

**Purpose:** Generate lip-synced talking head video from audio

**Recommended: Wav2Lip**
- **Why:** Best lip-sync quality, works with any portrait
- **Models:**
  - Wav2Lip: Good quality
  - Wav2Lip_gan: Better visual quality (RECOMMENDED)
- **Trade-offs:** ~5-10s to generate 10s video on GPU

**Alternatives:**
- **SadTalker** - Better head motion, more realistic
- **Live Portrait** - Highest quality, but complex setup
- **First Order Motion Model** - More flexible, but needs driving video

**Libraries:**
```python
# Using our existing implementation + enhancements
from avatar_engine import AvatarGenerator

# Or direct Wav2Lip
import Wav2Lip
```

**Interface:**
```python
class TalkingHead:
    def __init__(self, model_path: str, device: str = "cuda")
    def generate(
        self,
        audio_path: str,
        reference_image: str,
        output_path: str,
        resolution: int = 1080,
        fps: int = 30
    ) -> str:
        # Returns: path to output video
    def generate_stream(self, audio_path: str, ...) -> Iterator[bytes]
```

**Outputs:**
- `video_file`: MP4 file (1080p, 25-30fps, H.264)
- `metadata`: {duration, frames, resolution}

**Caching:**
- Model weights
- Face detection results for avatar image
- Background separation (if using green screen)
- Expression templates

---

### 6. Video Player / Output Module

**Purpose:** Display video and play audio to user

**Options:**
- **OpenCV** - Simple, lightweight
- **PyQt5/6** - Professional, full-featured
- **Web-based** - Stream via HLS/DASH

**Libraries:**
```python
import cv2  # OpenCV - Simple
from PyQt5.QtMultimediaWidgets import QVideoWidget
```

**Interface:**
```python
class VideoPlayer:
    def play(self, video_path: str, fullscreen: bool = False) -> None
    def stop(self) -> None
    def is_playing(self) -> bool
```

---

### 7. Orchestrator / Event Loop

**Purpose:** Coordinate all modules and manage system state

**State Machine:**
```
IDLE → LISTENING → TRANSCRIBING → THINKING → GENERATING_SPEECH →
GENERATING_VIDEO → PLAYING → IDLE
```

**Interface:**
```python
class Orchestrator:
    def __init__(self, config: dict)
    async def start(self) -> None
    async def run_conversation_turn(self) -> None
    def stop(self) -> None
    def get_state(self) -> str
```

**Responsibilities:**
- Module initialization and lifecycle
- Error handling and recovery
- Logging and monitoring
- Resource management (GPU memory, disk space)
- Performance metrics

---

## Data Flow Diagram

```
┌─────────┐     audio      ┌─────────┐      text       ┌──────────┐
│   Mic   │ ──────────────► │   ASR   │ ───────────────► │ Dialogue │
└─────────┘   (16kHz wav)   └─────────┘   (string)      │ Manager  │
                                                         └─────┬────┘
                                                               │
                                                          response_text
                                                               │
                                                               ▼
                               ┌──────────┐            ┌──────────┐
                               │  Video   │            │   TTS    │
                               │ Player   │◄───────────│  Engine  │
                               └─────┬────┘  (video)   └─────┬────┘
                                     │                       │
                                     │                  audio_file
                                     │                       │
                                     │                       ▼
                                     │                ┌──────────┐
                                     └────────────────│ Talking  │
                                         (plays)      │   Head   │
                                                      └──────────┘
```

## Latency Breakdown (Target)

| Stage | Time (GPU) | Time (CPU) |
|-------|-----------|-----------|
| Recording (VAD wait) | 0.5-1.0s | 0.5-1.0s |
| ASR (Whisper small) | 0.3-0.5s | 1-2s |
| LLM Response (local) | 1-3s | 3-10s |
| TTS (Coqui XTTS) | 1-2s | 4-8s |
| Talking Head (Wav2Lip) | 3-5s | 15-30s |
| **Total** | **6-11.5s** | **23.5-51s** |

**Optimization Strategies:**
1. Use GPU for all inference
2. Stream TTS output to start video gen earlier
3. Cache common responses
4. Preload models
5. Use smaller/faster models if latency is critical

## Technology Stack Summary

### Core Stack (RECOMMENDED)

| Module | Technology | Why |
|--------|-----------|-----|
| Audio Input | sounddevice + webrtcvad | Cross-platform, good VAD |
| ASR | Whisper (small) | Best accuracy/speed balance |
| Dialogue | Ollama (Llama 3.2) | Free, local, good quality |
| TTS | Coqui TTS (XTTS-v2) | Best open-source quality + voice cloning |
| Talking Head | Wav2Lip-GAN | Best lip-sync quality |
| Video Player | OpenCV or PyQt5 | Simple or professional |
| Orchestrator | asyncio + logging | Python native, async support |

### Alternative Stacks

**Speed-Optimized (Lower Quality):**
- ASR: Vosk (faster, less accurate)
- Dialogue: Phi-3 3B (faster, smaller)
- TTS: pyttsx3 (instant but robotic)
- Talking Head: Lower resolution (512p)

**Quality-Optimized (Higher Latency):**
- ASR: Whisper large
- Dialogue: GPT-4 (API)
- TTS: ElevenLabs (API)
- Talking Head: SadTalker (better motion)

## Privacy & Security Considerations

### Privacy
- ✅ **Local Processing**: Use local models (Whisper, Ollama, Coqui) - no data leaves your machine
- ⚠️ **Cloud APIs**: OpenAI, ElevenLabs, etc. send data to external servers
- ✅ **Data Retention**: Clear conversation history periodically
- ✅ **Audio Storage**: Option to not save recordings

### Security
- 🔒 **API Keys**: Store in environment variables, never in code
- 🔒 **File Permissions**: Restrict access to generated media
- 🔒 **Input Validation**: Sanitize LLM outputs before TTS/display

### Ethics
- ⚠️ **Deepfakes**: This tech can be misused for impersonation
- ✅ **Consent**: Only use avatar images you have rights to
- ✅ **Disclosure**: Be transparent that this is AI-generated
- ✅ **Voice Cloning**: Get consent before cloning someone's voice

## File Structure

```
conversational-avatar/
├── config/
│   ├── config.yaml           # Main configuration
│   └── prompts.yaml          # System prompts
├── src/
│   ├── audio_input/
│   │   ├── __init__.py
│   │   ├── recorder.py       # Audio capture
│   │   └── vad.py            # Voice activity detection
│   ├── asr/
│   │   ├── __init__.py
│   │   └── whisper_asr.py    # Whisper implementation
│   ├── dialogue/
│   │   ├── __init__.py
│   │   ├── manager.py        # Dialogue manager
│   │   └── memory.py         # Conversation memory
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── coqui_tts.py      # Coqui TTS implementation
│   │   └── voice_cloner.py   # Voice cloning
│   ├── talking_head/
│   │   ├── __init__.py
│   │   ├── wav2lip.py        # Wav2Lip wrapper
│   │   └── upscaler.py       # Video upscaling
│   ├── player/
│   │   ├── __init__.py
│   │   └── video_player.py   # Video display
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   └── main.py           # Main orchestrator
│   └── utils/
│       ├── __init__.py
│       ├── logger.py         # Logging utilities
│       └── metrics.py        # Performance tracking
├── models/                   # Downloaded models
├── assets/
│   ├── avatars/              # Avatar images
│   └── voices/               # Voice samples
├── outputs/
│   ├── audio/                # Generated audio
│   └── video/                # Generated videos
├── tests/
├── requirements.txt
├── environment.yml
├── main.py                   # Entry point
└── README.md
```

## Next Steps

This architecture provides:
1. ✅ Modular design - Easy to swap components
2. ✅ Clear interfaces - Well-defined inputs/outputs
3. ✅ Multiple options - Can choose speed vs quality
4. ✅ Scalability - Can upgrade individual modules
5. ✅ Privacy-first - Option for 100% local processing

**Phase 2** will implement the minimal prototype (audio only) to validate the ASR → LLM → TTS pipeline before adding video generation.
