#!/bin/bash
##############################################################################
# Ara Quick Setup Script
# Sets up Ara avatar system with all dependencies
##############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║                 🤖  ARA QUICK SETUP  🤖                       ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}❌ Please do not run this script as root${NC}"
    exit 1
fi

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python 3 found${NC}"

# Check Ollama
echo -e "\n${CYAN}Checking Ollama...${NC}"
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}⚠ Ollama not installed${NC}"
    echo -e "${CYAN}Install from: https://ollama.ai/download${NC}"
    echo ""
    read -p "Continue without Ollama? (y/n): " choice
    if [ "$choice" != "y" ]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ Ollama installed${NC}"

    # Check if Ollama is running
    if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo -e "${YELLOW}⚠ Ollama server not running${NC}"
        echo -e "${CYAN}Starting Ollama server in background...${NC}"
        ollama serve &
        sleep 3
    fi

    # Check if Mistral is installed
    if ! ollama list | grep -q mistral; then
        echo -e "${YELLOW}⚠ Mistral model not found${NC}"
        echo -e "${CYAN}Pulling Mistral model (this may take a few minutes)...${NC}"
        ollama pull mistral
    fi

    echo -e "${GREEN}✓ Ollama ready with Mistral model${NC}"
fi

# Install system dependencies (optional)
echo -e "\n${CYAN}Installing system dependencies (requires sudo)...${NC}"
read -p "Install system packages? (y/n): " install_sys

if [ "$install_sys" = "y" ]; then
    sudo apt update
    sudo apt install -y \
        ffmpeg \
        portaudio19-dev \
        espeak-ng \
        libx264-dev \
        libx265-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        python3-pyaudio

    echo -e "${GREEN}✓ System dependencies installed${NC}"
fi

# Create virtual environment
echo -e "\n${CYAN}Setting up Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate venv
source venv/bin/activate

# Install core dependencies
echo -e "\n${CYAN}Installing core Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
pip install -r multi-ai-workspace/requirements.txt

echo -e "${GREEN}✓ Core dependencies installed${NC}"

# Optional: ML dependencies for avatar generation
echo -e "\n${CYAN}ML dependencies for avatar generation${NC}"
read -p "Install ML dependencies (PyTorch, OpenCV)? (y/n): " install_ml

if [ "$install_ml" = "y" ]; then
    echo -e "${CYAN}Installing ML dependencies (this may take a while)...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install opencv-python librosa scipy scikit-image soundfile pydub face-alignment

    echo -e "${GREEN}✓ ML dependencies installed${NC}"
fi

# Optional: Voice recognition
echo -e "\n${CYAN}Voice recognition dependencies${NC}"
read -p "Install voice recognition (SpeechRecognition)? (y/n): " install_voice

if [ "$install_voice" = "y" ]; then
    pip install SpeechRecognition
    echo -e "${GREEN}✓ Voice recognition installed${NC}"
fi

# Set up environment file
echo -e "\n${CYAN}Setting up environment configuration...${NC}"
if [ ! -f ".env" ]; then
    cp .env.ara.example .env
    echo -e "${GREEN}✓ Created .env file${NC}"
    echo -e "${YELLOW}⚠ Edit .env to add your API keys (optional)${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Create output directories
mkdir -p outputs/ara_responses
mkdir -p assets/avatars
mkdir -p uploads
mkdir -p temp
mkdir -p models

echo -e "${GREEN}✓ Output directories created${NC}"

# Check for avatar images
echo -e "\n${CYAN}Checking avatar images...${NC}"
if [ ! -f "assets/avatars/ara_default.jpg" ]; then
    echo -e "${YELLOW}⚠ No avatar images found in assets/avatars/${NC}"
    echo -e "${CYAN}You'll need to add at least one avatar image:${NC}"
    echo "  - ara_default.jpg (required)"
    echo "  - ara_professional.jpg, ara_casual.jpg, etc. (optional)"
    echo ""
    echo -e "${CYAN}Avatar images should be:${NC}"
    echo "  - JPG or PNG format"
    echo "  - 512x512 or larger recommended"
    echo "  - Clear frontal face shot"
fi

# Test Ara
echo -e "\n${CYAN}Testing Ara backend...${NC}"
python3 -c "
from multi_ai_workspace.src.integrations.ara_avatar_backend import AraAvatarBackend
import asyncio

async def test():
    ara = AraAvatarBackend()
    healthy = await ara.health_check()
    if healthy:
        print('${GREEN}✓ Ara backend healthy${NC}')
        return True
    else:
        print('${YELLOW}⚠ Ara backend health check failed${NC}')
        print('Make sure Ollama is running: ollama serve')
        return False

asyncio.run(test())
" || echo -e "${YELLOW}⚠ Backend test skipped${NC}"

# Summary
echo -e "\n${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                    SETUP COMPLETE! 🎉                         ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Ara is ready to use!${NC}"
echo ""
echo -e "${CYAN}Quick Start:${NC}"
echo "  1. Make sure Ollama is running:"
echo "     ${YELLOW}ollama serve${NC}"
echo ""
echo "  2. Launch Ara:"
echo "     ${YELLOW}./start_ara.sh${NC}"
echo ""
echo "  3. Or use directly:"
echo "     ${YELLOW}python3 ara_voice_interface.py${NC}          (voice mode)"
echo "     ${YELLOW}python3 ara_voice_interface.py --text-only${NC}  (text chat)"
echo ""
echo -e "${CYAN}Documentation:${NC}"
echo "  - README: ${YELLOW}ARA_README.md${NC}"
echo "  - Persona spec: ${YELLOW}multi-ai-workspace/config/ara_persona.yaml${NC}"
echo "  - Voice macros: ${YELLOW}multi-ai-workspace/config/voice_macros.yaml${NC}"
echo ""
echo -e "${GREEN}Have fun with Ara! 🤖${NC}"
