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
echo ""
echo -e "${YELLOW}Choose installation type:${NC}"
echo ""
echo "  1) Ara Only        - AI co-pilot with voice control"
echo "  2) Ara + T-FAN     - Complete system with cockpit HUD"
echo ""
read -p "Select (1 or 2): " install_choice

if [ "$install_choice" = "2" ]; then
    echo -e "\n${CYAN}Launching complete system installer (Ara + T-FAN)...${NC}"
    echo -e "${YELLOW}This will install both the avatar system and T-FAN cockpit${NC}"
    echo ""
    read -p "Continue? (y/n): " confirm
    if [ "$confirm" = "y" ]; then
        exec ./install_complete_system.sh
    else
        echo "Installation cancelled"
        exit 0
    fi
fi

echo -e "\n${CYAN}Installing Ara only (lightweight)${NC}"
echo -e "${YELLOW}Note: You can install T-FAN later with ./install_complete_system.sh${NC}"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}❌ Please do not run this script as root${NC}"
    exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    OS="unknown"
fi

echo -e "${CYAN}Detected OS: ${YELLOW}$OS${NC}"
echo ""

##############################################################################
# INSTALL OLLAMA
##############################################################################

echo -e "${CYAN}Checking Ollama...${NC}"
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}⚠ Ollama not installed${NC}"
    echo -e "${CYAN}Ollama is required for Ara to work offline${NC}"
    echo ""
    read -p "Install Ollama automatically? (y/n): " install_ollama

    if [ "$install_ollama" = "y" ]; then
        echo -e "${CYAN}Installing Ollama...${NC}"
        curl -fsSL https://ollama.ai/install.sh | sh

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Ollama installed successfully${NC}"
        else
            echo -e "${RED}❌ Ollama installation failed${NC}"
            echo -e "${YELLOW}Install manually from: https://ollama.ai/download${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}⚠ Skipping Ollama installation${NC}"
        echo -e "${YELLOW}Ara will not work without Ollama${NC}"
        echo -e "${YELLOW}Install from: https://ollama.ai/download${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Ollama installed${NC}"
fi

# Start Ollama service
echo -e "${CYAN}Starting Ollama service...${NC}"
if ! pgrep -x "ollama" > /dev/null; then
    if systemctl is-enabled ollama &> /dev/null; then
        sudo systemctl start ollama
    else
        nohup ollama serve > /dev/null 2>&1 &
        sleep 3
    fi
    echo -e "${GREEN}✓ Ollama started${NC}"
else
    echo -e "${GREEN}✓ Ollama already running${NC}"
fi

# Pull Mistral base model
echo -e "${CYAN}Checking for Mistral base model...${NC}"
if ollama list | grep -q "mistral"; then
    echo -e "${GREEN}✓ Mistral model already downloaded${NC}"
else
    echo -e "${CYAN}Downloading Mistral 7B model (this may take a few minutes)...${NC}"
    ollama pull mistral
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Mistral model downloaded${NC}"
    else
        echo -e "${YELLOW}⚠ Could not download Mistral model${NC}"
    fi
fi

echo ""

##############################################################################
# INSTALL SYSTEM DEPENDENCIES
##############################################################################

echo -e "${CYAN}System dependencies are needed for audio/video processing${NC}"
echo -e "${YELLOW}Packages: FFmpeg, PortAudio, espeak-ng, video codecs${NC}"
echo ""
read -p "Install system dependencies? (y/n): " install_sys

if [ "$install_sys" = "y" ]; then
    echo -e "${CYAN}Installing system packages...${NC}"

    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        sudo apt update
        sudo apt install -y \
            python3 \
            python3-pip \
            python3-venv \
            python3-dev \
            ffmpeg \
            portaudio19-dev \
            espeak-ng \
            libx264-dev \
            libx265-dev \
            libgl1-mesa-glx \
            libglib2.0-0 \
            python3-pyaudio \
            build-essential \
            git \
            curl \
            wget

    elif [ "$OS" = "fedora" ] || [ "$OS" = "rhel" ] || [ "$OS" = "centos" ]; then
        sudo dnf install -y \
            python3 \
            python3-pip \
            python3-devel \
            ffmpeg \
            portaudio-devel \
            espeak-ng \
            x264-devel \
            x265-devel \
            mesa-libGL \
            glib2 \
            gcc \
            gcc-c++ \
            git \
            curl \
            wget

    elif [ "$OS" = "arch" ] || [ "$OS" = "manjaro" ]; then
        sudo pacman -S --noconfirm \
            python \
            python-pip \
            ffmpeg \
            portaudio \
            espeak-ng \
            x264 \
            x265 \
            mesa \
            glib2 \
            base-devel \
            git \
            curl \
            wget

    else
        echo -e "${YELLOW}⚠ Unsupported OS: $OS${NC}"
        echo -e "${YELLOW}Please install dependencies manually${NC}"
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ System dependencies installed${NC}"
    else
        echo -e "${YELLOW}⚠ Some packages may have failed to install${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Skipping system dependencies${NC}"
    echo -e "${YELLOW}Some features may not work without them${NC}"
fi

echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    echo -e "${YELLOW}Please install Python 3.10+ and try again${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓ Python ${PYTHON_VERSION} found${NC}"
echo ""

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
mkdir -p training

echo -e "${GREEN}✓ Output directories created${NC}"

# Build custom Ara model
echo -e "\n${CYAN}Building custom Ara model with personality...${NC}"
if command -v ollama &> /dev/null; then
    # Generate dataset if not exists
    if [ ! -f "training/Modelfile.ara" ]; then
        echo -e "${CYAN}Generating Ara training dataset...${NC}"
        python3 training/generate_ara_dataset.py
    fi

    # Check if ara model already exists
    if ollama list | grep -q "^ara "; then
        echo -e "${GREEN}✓ Custom 'ara' model already exists${NC}"
    else
        echo -e "${CYAN}Creating custom 'ara' model (this may take a moment)...${NC}"
        if ollama create ara -f training/Modelfile.ara 2>&1; then
            echo -e "${GREEN}✓ Custom 'ara' model created successfully!${NC}"
            echo -e "${CYAN}Ara now has her personality baked into the model${NC}"
        else
            echo -e "${YELLOW}⚠ Could not create custom model, will fall back to base Mistral${NC}"
            echo -e "${YELLOW}You can build it later with: ./training/build_ara_model.sh${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠ Ollama not available, skipping custom model creation${NC}"
    echo -e "${YELLOW}Install Ollama and run: ./training/build_ara_model.sh${NC}"
fi

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
