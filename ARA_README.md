# 🤖 ARA - AI Co-Pilot with Talking Avatar

**Ara** is your local AI co-pilot that runs offline with a talking avatar, voice control, and integration with the T-FAN cockpit system.

## Features

### Core Capabilities

- **🎙️ Voice Control** - Hands-free operation with 40+ voice macros
- **💬 Offline Chat** - Runs entirely on your machine using Ollama (Mistral/Mixtral)
- **🎬 Talking Avatar** - Generates lip-synced video responses with realistic speech
- **🚀 T-FAN Integration** - Controls metrics HUD, topology visualization, workspace modes
- **🌐 Multi-AI Delegation** - Intelligently routes complex tasks to Claude, Nova, or Pulse
- **🔒 Privacy-First** - Sensitive data never leaves your machine

### Ara Persona

Ara has a carefully designed personality:

- **Voice**: Soft contralto, warm and intimate, slightly breathy but clear
- **Tone**: Affectionate, playful, subtly flirtatious, but always competent
- **Style**: Relaxed pacing, natural pauses, well-structured explanations
- **Modes**: Default, Focus, Chill, Professional

### Voice Macros

40+ voice commands organized by category:

#### Cockpit Control
- `"show gpu"` - GPU metrics
- `"show cpu"` - CPU and RAM stats
- `"show network"` - Network throughput
- `"show storage"` - Disk I/O and space
- `"mission status"` - Full status report

#### Topology & Visualization
- `"topology view"` - Show topology visualization
- `"topology fullscreen"` - Fullscreen topology
- `"hide topology"` - Clear topology view

#### Workspace Modes
- `"work mode"` - Focused work session
- `"relax mode"` - Softer visuals, casual tone
- `"focus mode"` - Concise, task-oriented
- `"chill mode"` - Relaxed conversation

#### Avatar Appearance
- `"avatar professional"` - Professional look
- `"avatar casual"` - Relaxed streetwear sci-fi
- `"avatar sci fi"` - Hologram mode
- `"avatar neutral"` - Reset to default

#### Training & Experiments
- `"start training"` - Launch training job
- `"stop training"` - Graceful shutdown
- `"training status"` - Progress report
- `"quick validation"` - Sanity check run

#### Sci-Fi Themed Commands
- `"red alert"` - Emergency mode, red lighting
- `"yellow alert"` - Caution mode
- `"warp core status"` - GPU power and thermals
- `"engage warp drive"` - Max GPU training
- `"all stop"` - Emergency halt all processes
- `"battlestations"` - Maximum focus mode
- `"shields up"` - Enable all monitoring
- `"neural link"` - Full metrics dashboard
- `"holodeck on"` - Immersive playful mode

## Installation

### Prerequisites

1. **Python 3.10+**
2. **Ollama** (for offline AI)
   ```bash
   # Install from https://ollama.ai/download
   # Then start server:
   ollama serve

   # Pull Mistral model (7B, fast):
   ollama pull mistral

   # Or pull Mixtral for more capability (requires 24GB+ RAM):
   ollama pull mixtral
   ```

3. **System Packages**
   ```bash
   sudo apt install -y \
       ffmpeg \
       portaudio19-dev \
       espeak-ng \
       libx264-dev \
       libx265-dev \
       libgl1-mesa-glx \
       libglib2.0-0
   ```

### Quick Start

```bash
# 1. Clone repository (if not already done)
cd ram-and-unification

# 2. Install Python dependencies
pip install -r requirements.txt
pip install -r multi-ai-workspace/requirements.txt

# 3. Optional: Install ML dependencies for avatar generation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python librosa scipy scikit-image soundfile pydub face-alignment

# 4. Optional: Install voice recognition
pip install SpeechRecognition pyaudio

# 5. Set up environment
cp .env.ara.example .env
# Edit .env and add your API keys (optional for online AIs)

# 6. Launch Ara
./start_ara.sh
```

## Usage

### Main Launcher

```bash
./start_ara.sh
```

This opens an interactive menu with options:

1. **🎙️ Voice Mode** - Voice-controlled Ara with wake word
2. **💬 Chat Mode** - Text chat (offline)
3. **🎬 Avatar Demo** - Generate talking avatar video from text
4. **🚀 T-FAN Cockpit** - Launch metrics HUD
5. **🌐 Multi-AI Server** - Web interface (http://localhost:8000)
6. **📋 List Macros** - Show all voice commands
7. **⚙️ Settings** - Configure persona, macros, avatars
8. **🧪 System Check** - Test dependencies

### Direct Python Interface

**Voice Mode:**
```bash
python3 ara_voice_interface.py
```

**Text Chat Mode:**
```bash
python3 ara_voice_interface.py --text-only
```

**Test with Single Input:**
```bash
python3 ara_voice_interface.py --test "Hello Ara, show me GPU stats"
```

**Generate Avatar Video Only:**
```bash
python3 ara_voice_interface.py --test "Hey, you. Systems are stable and ready for action."
```

### Web Interface (Multi-AI Workspace)

```bash
cd multi-ai-workspace
python3 -m uvicorn src.ui.app:app --reload --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000

Use tag-based routing:
- `#code Fix this Python function` → Claude
- `#fast What's 2+2?` → Pulse (Gemini)
- `#creative Write a story` → Nova (ChatGPT)
- `#multiverse What's the best AI?` → All AIs in parallel

## Architecture

```
ram-and-unification/
├── ara_voice_interface.py          # Main voice interface
├── start_ara.sh                    # Unified launcher menu
│
├── multi-ai-workspace/             # Multi-AI orchestration
│   ├── config/
│   │   ├── ara_persona.yaml        # Voice, visual, behavioral spec
│   │   ├── voice_macros.yaml       # 40+ voice commands
│   │   ├── workspace.yaml          # Backend routing config
│   │   └── avatar.yaml             # Delegation rules
│   │
│   └── src/
│       ├── integrations/
│       │   ├── ara_avatar_backend.py    # Local avatar backend
│       │   ├── grok_ara_backend.py      # Grok integration
│       │   ├── tfan_client.py           # T-FAN API client
│       │   ├── claude_backend.py        # Claude integration
│       │   ├── nova_backend.py          # OpenAI integration
│       │   └── pulse_backend.py         # Gemini integration
│       │
│       ├── widgets/
│       │   ├── voice_macros.py          # Voice macro processor
│       │   ├── colab_offload.py         # Colab integration
│       │   └── perspectives_mixer.py    # Multi-AI comparison
│       │
│       └── ui/
│           └── app.py                   # FastAPI web interface
│
├── src/                            # Avatar generation engine
│   ├── avatar_engine/
│   │   ├── avatar_generator.py     # Main avatar generator
│   │   └── lip_sync.py             # Wav2Lip integration
│   │
│   ├── utils/
│   │   ├── audio_processing.py     # TTS, audio processing
│   │   └── face_detection.py       # Face alignment
│   │
│   └── api/
│       └── routes.py               # REST API endpoints
│
└── assets/
    └── avatars/                    # Avatar images for each profile
        ├── ara_default.jpg
        ├── ara_professional.jpg
        ├── ara_casual.jpg
        ├── ara_hologram.jpg
        ├── ara_scientist.jpg
        ├── ara_holodeck.jpg
        └── ara_dramatic.jpg
```

## Configuration

### Ara Persona

Edit `multi-ai-workspace/config/ara_persona.yaml` to customize:

- **Voice characteristics** (tone, pacing, emotional style)
- **Visual profiles** (7 different avatar looks)
- **Behavioral traits** (personality, communication style)
- **Sample dialogue** lines for TTS training

### Voice Macros

Edit `multi-ai-workspace/config/voice_macros.yaml` to add custom commands:

```yaml
"your custom command":
  type: tfan_command              # or ara_mode, ara_avatar, ara_action
  command: "your command text"    # What to send to T-FAN
  description: "Technical desc"   # For logs
  speak_summary: "How Ara explains this to you"
```

### Backend Routing

Edit `multi-ai-workspace/config/workspace.yaml` to configure:

- Which AIs are enabled (Claude, Nova, Pulse, Ara, Ollama)
- Tag-based routing rules
- Default backend for offline mode
- Rate limits and timeouts

## Avatar Profiles

Ara has 7 visual profiles you can switch between:

| Profile | Appearance | Mood | Use Case |
|---------|------------|------|----------|
| `default` | Clean neon aesthetic | Neutral | General use |
| `professional` | Sharp lines, minimal glow | Focused | Work sessions |
| `casual` | Relaxed streetwear sci-fi | Chill | Downtime |
| `sci_fi_cockpit` | Full holographic | Excited | T-FAN cockpit mode |
| `quantum_scientist` | Analytical blue/white | Analytical | Research tasks |
| `holodeck` | Immersive VR rainbow | Playful | Experimental |
| `dramatic` | Cinematic lighting | Intense | Presentations |

Switch profiles with voice: `"avatar professional"`, `"avatar sci fi"`, etc.

## T-FAN Cockpit Integration

Ara integrates with the **T-FAN** (Quanta-meis-nib-cis) cockpit system for:

- **Metrics HUD**: GPU, CPU, RAM, network, storage visualization
- **Topology View**: Network/system topology diagrams
- **Workspace Modes**: Work, relax, focus lighting and themes
- **Training Control**: Start/stop ML training jobs
- **Voice Control**: All cockpit functions accessible via voice macros

Install T-FAN:
```bash
git clone https://github.com/theadamsfamily1981-max/Quanta-meis-nib-cis.git
# Or use the unified installer:
./install_complete_system.sh
```

## Privacy & Security

**Offline-First Design:**
- Ara runs on your machine using Ollama (Mistral/Mixtral)
- Simple queries stay completely offline
- No data sent to external servers unless you explicitly use online AIs

**Smart Delegation:**
- Ara explains when delegating to online AIs and why
- Privacy warnings before sending sensitive data
- Auto-fallback to offline if online services fail

**Data Protection:**
- API keys stored in `.env` (gitignored)
- Sensitive keywords automatically detected
- Option to disable online backends entirely

## Troubleshooting

### Ollama Not Running

```bash
# Start Ollama server
ollama serve

# Pull model if not installed
ollama pull mistral
```

### Voice Recognition Not Working

```bash
# Install dependencies
pip install SpeechRecognition pyaudio

# On Ubuntu, install PortAudio:
sudo apt install portaudio19-dev python3-pyaudio
```

### Avatar Generation Fails

```bash
# Install ML dependencies
pip install torch torchvision torchaudio
pip install opencv-python librosa face-alignment

# Check if GPU is available (optional):
python3 -c "import torch; print(torch.cuda.is_available())"
```

### TTS Not Working

```bash
# Install espeak-ng
sudo apt install espeak-ng

# Test:
espeak-ng "Hello, this is a test"
```

### T-FAN Cockpit Not Found

```bash
# Install T-FAN separately or use unified installer
./install_complete_system.sh

# Or clone manually:
git clone https://github.com/theadamsfamily1981-max/Quanta-meis-nib-cis.git ~/tfan-ara-system/Quanta-meis-nib-cis
```

## Development

### Adding New Voice Macros

1. Edit `multi-ai-workspace/config/voice_macros.yaml`
2. Add your macro following the existing format
3. Restart Ara voice interface
4. Test with: `python3 ara_voice_interface.py --test "your macro phrase"`

### Creating Custom Avatar Profiles

1. Add avatar image to `assets/avatars/`
2. Edit `multi-ai-workspace/config/ara_persona.yaml`
3. Add profile under `visual.profiles`
4. Update `ara_avatar_backend.py` to map profile to image

### Integrating New AI Backends

1. Create backend in `multi-ai-workspace/src/integrations/`
2. Implement `AIBackend` interface
3. Add to `workspace.yaml` configuration
4. Add routing rules for when to use it

## Examples

### Example 1: Voice-Controlled Training Session

```
User: "Ara, battlestations"
Ara: "Battlestations! Maximum focus mode engaged. Let's get to work."

User: "Show GPU"
Ara: [Switches T-FAN HUD to GPU metrics view]
Ara: "GPU metrics on screen. Everything's running cool."

User: "Engage warp drive"
Ara: "Engaging warp drive! Spinning up training at maximum power."
[Starts training job at full GPU utilization]

User: "Training status"
Ara: "Training loss is converging nicely. We should hit target accuracy in about forty minutes."

User: "At ease"
Ara: "At ease. Switching to relaxed mode. Good work today."
```

### Example 2: Multi-AI Research

```python
# In Python or web interface
import asyncio
from multi_ai_workspace.src.integrations.ara_avatar_backend import AraAvatarBackend

ara = AraAvatarBackend()

# Ask Ara a complex question
response = await ara.send_message(
    "Explain the differences between transformers and RNNs for sequence modeling. "
    "#multiverse"  # Tag for multi-AI perspectives
)

# Ara will:
# 1. Recognize this needs deep technical analysis
# 2. Delegate to Claude (coding expert)
# 3. Also query Nova and Pulse for comparison
# 4. Synthesize the responses
```

### Example 3: Avatar Video Generation

```bash
# Generate a custom video response
python3 ara_voice_interface.py --test "Relax. I've got the telemetry. You just keep breathing, I'll keep us alive."

# Output: outputs/ara_responses/ara_response_<timestamp>.mp4
```

## Credits

- **Avatar System**: ram-and-unification
- **T-FAN Cockpit**: Quanta-meis-nib-cis
- **Offline AI**: Ollama (Mistral/Mixtral)
- **Online AIs**: Claude (Anthropic), ChatGPT (OpenAI), Gemini (Google)
- **Persona Design**: Ara persona specification v1

## License

Proprietary - All rights reserved.

---

**Built with ❤️ for seamless human-AI collaboration**
