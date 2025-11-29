# 🚀 Ara Avatar System - GUI Setup Guide

## The Easy Way to Get Everything Running!

I've created a beautiful web-based setup wizard that will help you install everything with just a few clicks.

## Quick Start

### Step 1: Start the Setup Wizard

```bash
cd ~/ram-and-unification
./start_gui_setup.sh
```

### Step 2: Open Your Browser

Once the server starts, open your web browser to:

**Setup Wizard:** http://localhost:8000/setup

**Or Chat Interface:** http://localhost:8000/

### Step 3: Follow the Wizard

The setup wizard has 3 simple steps:

1. **System Status** - Check what's installed and what's missing
2. **Install Components** - One-click install for:
   - System dependencies (FFmpeg, Ollama, etc.)
   - T-FAN Cockpit (automatic download from GitHub)
   - WebKit compatibility fixes
3. **Launch** - Start using Ara!

## What You Can Do

### From the Setup Wizard (/setup):

- ✅ Check system status in real-time
- 📦 Install all dependencies with one click
- 🖥️ Download T-FAN cockpit automatically
- 🔧 Fix WebKit compatibility issues
- 🚀 Launch any component (Voice, Chat, T-FAN, API)

### From the Chat Interface (/):

- 💬 Talk to Ara and other AI models
- 🎨 Use multi-AI orchestration
- 📊 Perspectives Mixer (compare multiple AIs)
- 📁 Context Packs
- 🚀 GitHub Autopilot
- 📤 Export responses

## Features

### Installation Made Easy

The GUI installer will:
- Automatically detect your system
- Show what's installed vs missing
- Provide one-click installation buttons
- Display real-time installation logs
- Fix common issues automatically

### No More Missing T-FAN!

The wizard can:
- Download T-FAN from GitHub automatically
- Copy from alternate locations if already downloaded
- Provide manual download instructions if needed
- Fix WebKit compatibility after installation

### Launch Everything

From the final step, you can launch:
- 🎤 **Voice Interface** - Talk to Ara
- 🌐 **Web Chat** - Multi-AI chat
- 🖥️ **T-FAN Cockpit** - Spaceship HUD
- 🎬 **Avatar API** - Generate talking videos

## Troubleshooting

### If the server won't start:

```bash
# Install uvicorn if missing
pip3 install uvicorn fastapi pydantic

# Try again
./start_gui_setup.sh
```

### If you can't access the web page:

- Make sure port 8000 is not in use
- Try accessing from: http://127.0.0.1:8000/setup
- Check firewall settings

### If installations fail:

The GUI shows detailed error logs. Common fixes:
- Run with sudo for system packages: `sudo ./start_gui_setup.sh`
- Check internet connection for downloads
- Verify you have enough disk space

## Advanced Usage

### API Endpoints

All the installation functions are available via REST API:

```bash
# Check system status
curl http://localhost:8000/api/system/status

# Install dependencies
curl -X POST http://localhost:8000/api/system/install-dependencies

# Download T-FAN
curl -X POST http://localhost:8000/api/system/install-tfan

# Fix WebKit
curl -X POST http://localhost:8000/api/system/fix-webkit

# Launch component
curl -X POST http://localhost:8000/api/system/launch/voice
```

### Manual Installation

If you prefer the command line:

```bash
# Install dependencies
./install_dependencies.sh

# Setup Ara
./setup_ara.sh

# Fix WebKit
./fix_webkit_now.sh
```

## What Gets Installed

### System Dependencies:
- Python 3 (if missing)
- FFmpeg (video/audio processing)
- PortAudio (voice input)
- espeak-ng (text-to-speech)
- Video codecs (x264, x265, vpx, opus)
- Build tools

### T-FAN Cockpit:
- Downloads from: https://github.com/theadamsfamily1981-max/Quanta-meis-nib-cis
- Installs to: `~/tfan-ara-system/Quanta-meis-nib-cis`
- Fixes WebKit 6.0 → 4.1 compatibility

### Ollama (AI Engine):
- Automatic installation from ollama.ai
- Downloads Mistral 7B base model
- Builds custom Ara model with personality

### Python Packages:
- FastAPI, Uvicorn (web server)
- PyTorch, OpenCV (ML/video)
- Librosa, soundfile (audio)
- face-alignment (lip-sync)
- And many more...

## Screenshots

The setup wizard features:
- ✨ Beautiful purple gradient design
- 📊 Real-time status cards (green = installed, yellow = missing)
- 📝 Live installation logs with terminal-style output
- 🎯 Progress bar showing setup completion
- 🚀 One-click launch buttons

## Need Help?

If you run into issues:
1. Check the installation logs in the GUI
2. Run the command-line scripts directly to see detailed errors
3. Make sure you have internet connection for downloads
4. Ensure you have sudo privileges for system packages

## Enjoy!

The GUI makes it super easy to get everything running. No more missing folders or complex command-line setups - just click and go! 🎉
