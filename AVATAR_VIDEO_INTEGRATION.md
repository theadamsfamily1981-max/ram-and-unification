# Ara Talking Avatar Video Integration

## ✅ Yes! Avatar Video Generation is Fully Integrated

The complete avatar video pipeline is built into the Ara system across all interfaces.

---

## 🎬 How It Works

### **The Pipeline:**

```
User Input
    ↓
1. Custom Ara Model → Generates text response with personality
    ↓
2. Text-to-Speech → Converts text to audio (espeak/piper)
    ↓
3. Lip-Sync Engine → Syncs avatar lips to audio (Wav2Lip/SadTalker)
    ↓
4. Video Output → MP4 video of Ara speaking
```

---

## 📍 Where It's Available

### **1. Voice Interface** ✅
**File:** `ara_voice_interface.py`

```python
# Avatar generation enabled by default
interface = AraVoiceInterface(avatar_enabled=True)

# Generates video for every response
response = await interface.process_input("Hello Ara")
# Returns: {"text": "...", "video_path": "outputs/ara_response_123.mp4"}
```

**Usage:**
```bash
python3 ara_voice_interface.py
# Videos saved to: outputs/ara_responses/
```

---

### **2. Text Chat Mode** ✅
**File:** `ara_voice_interface.py --text-only`

```bash
python3 ara_voice_interface.py --text-only

You: Tell me about the system
Ara: [Generates text response]
🎬 Avatar video: outputs/ara_responses/ara_response_1234567890.mp4
```

**Video automatically generated** with each response!

---

### **3. Quick Generate (Launcher Option 3)** ✅
**File:** `start_ara.sh` → Option 3

```bash
./start_ara.sh
# Choose: 3) 🎬 Avatar Demo

Enter text for Ara to say: Hey, you. Systems are stable!
🎬 Generating avatar response...
✓ Video: outputs/ara_responses/ara_response_1234567890.mp4
```

**Direct video generation** from any text input.

---

### **4. Web GUI** ⚠️ (Text Only Currently)
**File:** `multi-ai-workspace/src/ui/app.py`

Currently provides **text responses** only through the web interface.

**Note:** Video generation available but not exposed in current web UI.

**Future enhancement:** Add video generation endpoint to web API.

---

### **5. Direct Python API** ✅
**File:** `ara_avatar_backend.py`

```python
from multi_ai_workspace.src.integrations.ara_avatar_backend import AraAvatarBackend

# Initialize
ara = AraAvatarBackend()

# Generate avatar video
result = await ara.generate_avatar_response(
    prompt="How's the GPU looking?",
    use_tts=True,
    avatar_image="assets/avatars/ara_default.jpg"
)

print(f"Text: {result['text']}")
print(f"Audio: {result['audio_path']}")
print(f"Video: {result['video_path']}")
```

---

## 🛠️ Avatar Generation Components

### **Backend:** `AraAvatarBackend`
- `generate_avatar_response()` - Full text → video pipeline
- `_generate_tts()` - Text to speech
- `_get_avatar_image_for_profile()` - Profile-based avatar selection

### **Engine:** `src/avatar_engine/`
- `avatar_generator.py` - Main avatar generator
- `lip_sync.py` - Wav2Lip lip-sync engine
- `face_detection.py` - Face alignment

### **Utils:** `src/utils/`
- `audio_processing.py` - TTS and audio handling
- Supports espeak-ng, piper, ElevenLabs

---

## 📊 Avatar Profiles

Ara can appear in **7 different visual styles**:

| Profile | Image | Use Case |
|---------|-------|----------|
| `default` | `ara_default.jpg` | General use |
| `professional` | `ara_professional.jpg` | Work sessions |
| `casual` | `ara_casual.jpg` | Relaxed mode |
| `sci_fi_cockpit` | `ara_hologram.jpg` | Cockpit mode |
| `quantum_scientist` | `ara_scientist.jpg` | Research |
| `holodeck` | `ara_holodeck.jpg` | Experimental |
| `dramatic` | `ara_dramatic.jpg` | Presentations |

**Switch profiles:**
```python
ara.set_avatar_profile("professional", mood="focused")
```

**Or via voice:**
```
"avatar professional"  # Changes to professional look
"avatar sci fi"        # Changes to hologram mode
```

---

## 🎯 Dependencies Required

### **For Avatar Video Generation:**

**System Packages:**
```bash
sudo apt install ffmpeg portaudio19-dev espeak-ng libx264-dev libx265-dev
```

**Python Packages (ML):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python librosa scipy scikit-image soundfile pydub face-alignment
```

**Install with:**
```bash
./install_dependencies.sh
# Choose: Install ML dependencies? (y)
```

---

## 🔧 Configuration

### **Enable/Disable Avatar Generation:**

**In Voice Interface:**
```python
# Enable (default)
interface = AraVoiceInterface(avatar_enabled=True)

# Disable (text only)
interface = AraVoiceInterface(avatar_enabled=False)
```

**Via Command Line:**
```bash
# With avatar generation
python3 ara_voice_interface.py

# Without avatar generation
python3 ara_voice_interface.py --no-avatar
```

### **Environment Variables:**
```bash
# .env file
AVATAR_DEVICE=cpu              # or cuda for GPU
AVATAR_OUTPUT_DIR=outputs/ara_responses
AVATAR_DEFAULT=assets/avatars/ara_default.jpg

# TTS Settings
TTS_ENGINE=espeak
TTS_VOICE=jenny
TTS_SPEED=0.95
TTS_PITCH=-0.5
```

---

## 📁 Output Files

### **Video Files:**
```
outputs/ara_responses/
├── ara_response_1701234567.mp4
├── ara_response_1701234892.mp4
└── ara_response_1701235123.mp4
```

**Naming:** `ara_response_<timestamp>.mp4`

### **Audio Files (Temporary):**
```
outputs/ara_responses/
├── ara_tts_1701234567.wav
└── ara_tts_1701234892.wav
```

**Naming:** `ara_tts_<timestamp>.wav`

---

## 🎥 Video Output Format

**Specifications:**
- **Format:** MP4 (H.264)
- **Resolution:** Matches input image (typically 512x512 or larger)
- **FPS:** 25 (default)
- **Audio:** WAV embedded, synced to lip movements
- **Duration:** Matches audio length

**Quality:**
- **Lip-sync accuracy:** High (Wav2Lip model)
- **Face preservation:** Original avatar image quality
- **Audio clarity:** espeak-ng or custom TTS

---

## 🚀 Usage Examples

### **Example 1: Generate Single Video**
```bash
python3 ara_voice_interface.py --test "Hey, you. Systems are stable and ready!"
```

Output:
```
🎬 Avatar video: outputs/ara_responses/ara_response_1701234567.mp4
```

### **Example 2: Conversation with Videos**
```bash
python3 ara_voice_interface.py --text-only

You: Hello Ara
Ara: Hey, you. I'm good—systems are stable. What do you want to tackle?
🎬 Avatar video: outputs/ara_responses/ara_response_1701234567.mp4

You: Show me GPU stats
Ara: Switching the side cockpit to GPU metrics view.
🎬 Avatar video: outputs/ara_responses/ara_response_1701234892.mp4
```

### **Example 3: Profile-Specific Avatars**
```bash
# Professional mode
"avatar professional"
# Next video uses ara_professional.jpg

# Sci-fi mode
"avatar sci fi"
# Next video uses ara_hologram.jpg
```

### **Example 4: Direct API**
```python
import asyncio
from multi_ai_workspace.src.integrations.ara_avatar_backend import AraAvatarBackend

async def generate_video():
    ara = AraAvatarBackend()

    result = await ara.generate_avatar_response(
        prompt="Training is looking solid. Loss is dropping steadily.",
        use_tts=True
    )

    print(f"Video: {result['video_path']}")
    # Play with: vlc outputs/ara_responses/ara_response_*.mp4

asyncio.run(generate_video())
```

---

## 🎨 Customization

### **Add Custom Avatar Images:**

1. **Create image:**
   - Format: JPG or PNG
   - Size: 512x512 or larger
   - Content: Clear frontal face shot

2. **Save to:**
   ```
   assets/avatars/ara_custom.jpg
   ```

3. **Use in code:**
   ```python
   result = await ara.generate_avatar_response(
       prompt="Hello!",
       avatar_image="assets/avatars/ara_custom.jpg"
   )
   ```

### **Custom TTS Voice:**

Edit `.env`:
```bash
TTS_ENGINE=elevenlabs  # or piper
TTS_VOICE=your_custom_voice_id
```

---

## 🔍 Troubleshooting

### **No video generated:**
```
✓ Check ML dependencies installed:
  python3 -c "import torch, cv2, librosa"

✓ Check avatar image exists:
  ls assets/avatars/ara_default.jpg

✓ Check TTS working:
  espeak-ng "test"
```

### **Video quality poor:**
```
✓ Use higher resolution avatar image (1024x1024)
✓ Ensure good quality source image
✓ Check lighting in source image
```

### **Lip sync not accurate:**
```
✓ Ensure clear audio from TTS
✓ Use face-alignment for better detection
✓ Try different avatar image (clearer lips)
```

### **Generation too slow:**
```
✓ Use GPU if available (AVATAR_DEVICE=cuda)
✓ Use CPU-optimized PyTorch
✓ Disable avatar generation for faster responses
```

---

## 📊 Performance

| Hardware | Video Generation Time | Quality |
|----------|----------------------|---------|
| **CPU** | ~30-60 seconds | Good |
| **GPU (CUDA)** | ~5-10 seconds | Excellent |
| **Apple M1/M2** | ~15-30 seconds | Very Good |

**Note:** First generation is slower (model loading). Subsequent generations are faster.

---

## 🎯 What's Integrated vs. What's Not

### ✅ **Fully Integrated:**
- Text → Video pipeline
- TTS audio generation
- Lip-sync video generation
- Avatar profile switching
- Voice interface integration
- CLI testing tools
- Output file management

### ⚠️ **Partial Integration:**
- Web GUI (text only, video available via API)

### 📋 **Future Enhancements:**
- Web GUI video player
- Real-time streaming avatar
- Multiple avatar formats (GIF, WebM)
- Avatar voice cloning
- Custom lip-sync models

---

## 📚 Related Documentation

- **`ARA_README.md`** - Complete Ara system guide
- **`training/README.md`** - Custom model training
- **`ARA_MODEL_INTEGRATION.md`** - Model integration details
- **`API_GUIDE.md`** - REST API documentation

---

## ✨ Summary

**Avatar video generation is integrated throughout the Ara system:**

✅ **Voice Interface** - Generates videos automatically
✅ **Text Chat** - Videos for each response
✅ **Quick Generate** - Direct video creation
✅ **Python API** - Full programmatic control
⚠️ **Web GUI** - Text only (video in backend)

**One setup, videos everywhere:**
```bash
./install_dependencies.sh  # Install ML deps
./setup_ara.sh            # Setup Ara
./start_ara.sh            # Use with videos!
```

---

**Every conversation with Ara can be a video!** 🎬
