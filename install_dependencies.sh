#!/bin/bash
##############################################################################
# Install Dependencies and Ollama for Ara Avatar System
# Comprehensive installer for all required system packages and Ollama
##############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
MAGENTA='\033[0;35m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║        📦 ARA DEPENDENCIES & OLLAMA INSTALLER 📦              ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}❌ Do not run this script as root or with sudo${NC}"
    echo -e "${YELLOW}The script will ask for sudo password when needed${NC}"
    exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VERSION=$VERSION_ID
else
    echo -e "${RED}❌ Cannot detect OS${NC}"
    exit 1
fi

echo -e "${CYAN}Detected OS: ${YELLOW}$OS $VERSION${NC}"
echo ""

##############################################################################
# 1. INSTALL OLLAMA
##############################################################################

echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║                    OLLAMA INSTALLATION                        ║${NC}"
echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama already installed${NC}"
    ollama --version
else
    echo -e "${CYAN}Installing Ollama...${NC}"
    echo -e "${YELLOW}This will download and install Ollama from ollama.ai${NC}"
    echo ""

    read -p "Install Ollama? (y/n): " install_ollama

    if [ "$install_ollama" = "y" ]; then
        # Download and run Ollama installer
        curl -fsSL https://ollama.ai/install.sh | sh

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Ollama installed successfully${NC}"
        else
            echo -e "${RED}❌ Ollama installation failed${NC}"
            echo -e "${YELLOW}Try manual installation: https://ollama.ai/download${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}⚠ Skipping Ollama installation${NC}"
        echo -e "${YELLOW}Install manually from: https://ollama.ai/download${NC}"
    fi
fi

echo ""

# Start Ollama service
echo -e "${CYAN}Starting Ollama service...${NC}"
if ! pgrep -x "ollama" > /dev/null; then
    # Try to start as systemd service
    if systemctl is-enabled ollama &> /dev/null; then
        sudo systemctl start ollama
        echo -e "${GREEN}✓ Ollama service started${NC}"
    else
        # Start in background
        nohup ollama serve > /dev/null 2>&1 &
        sleep 3
        echo -e "${GREEN}✓ Ollama started in background${NC}"
    fi
else
    echo -e "${GREEN}✓ Ollama already running${NC}"
fi

echo ""

# Pull base Mistral model
echo -e "${CYAN}Checking for Mistral base model...${NC}"
if ollama list | grep -q "mistral"; then
    echo -e "${GREEN}✓ Mistral model already downloaded${NC}"
else
    echo -e "${CYAN}Downloading Mistral 7B model (this may take a few minutes)...${NC}"
    ollama pull mistral

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Mistral model downloaded successfully${NC}"
    else
        echo -e "${YELLOW}⚠ Could not download Mistral model${NC}"
        echo -e "${YELLOW}You can try again later with: ollama pull mistral${NC}"
    fi
fi

echo ""

##############################################################################
# 2. SYSTEM DEPENDENCIES
##############################################################################

echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║                  SYSTEM DEPENDENCIES                          ║${NC}"
echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${CYAN}The following system packages will be installed:${NC}"
echo "  • Python 3 and pip"
echo "  • FFmpeg (audio/video processing)"
echo "  • PortAudio (audio I/O)"
echo "  • espeak-ng (text-to-speech)"
echo "  • Video codecs (x264, x265)"
echo "  • OpenGL libraries"
echo "  • Development headers"
echo ""

read -p "Install system dependencies? (y/n): " install_sys

if [ "$install_sys" = "y" ]; then
    echo ""
    echo -e "${CYAN}Updating package lists...${NC}"

    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        sudo apt update

        echo -e "${CYAN}Installing packages...${NC}"
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

        echo -e "${GREEN}✓ System packages installed${NC}"

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

        echo -e "${GREEN}✓ System packages installed${NC}"

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

        echo -e "${GREEN}✓ System packages installed${NC}"
    else
        echo -e "${YELLOW}⚠ Unsupported OS: $OS${NC}"
        echo -e "${YELLOW}Please install dependencies manually${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Skipping system dependencies${NC}"
fi

echo ""

##############################################################################
# 3. PYTHON DEPENDENCIES
##############################################################################

echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║                  PYTHON DEPENDENCIES                          ║${NC}"
echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}✓ Python ${PYTHON_VERSION} found${NC}"
else
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi

echo ""

# Create virtual environment
echo -e "${CYAN}Setting up Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate venv
source venv/bin/activate

echo ""
echo -e "${CYAN}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

echo ""
echo -e "${CYAN}Installing core Python packages...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Core packages installed${NC}"
else
    echo -e "${YELLOW}⚠ requirements.txt not found${NC}"
fi

if [ -f "multi-ai-workspace/requirements.txt" ]; then
    pip install -r multi-ai-workspace/requirements.txt
    echo -e "${GREEN}✓ Multi-AI workspace packages installed${NC}"
else
    echo -e "${YELLOW}⚠ multi-ai-workspace/requirements.txt not found${NC}"
fi

echo ""

##############################################################################
# 4. OPTIONAL: ML DEPENDENCIES
##############################################################################

echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║               ML DEPENDENCIES (OPTIONAL)                      ║${NC}"
echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${CYAN}ML dependencies enable avatar video generation:${NC}"
echo "  • PyTorch (deep learning framework)"
echo "  • OpenCV (computer vision)"
echo "  • Librosa (audio processing)"
echo "  • Face alignment tools"
echo ""
echo -e "${YELLOW}Note: This is a large download (~2-3GB) and takes time${NC}"
echo ""

read -p "Install ML dependencies? (y/n): " install_ml

if [ "$install_ml" = "y" ]; then
    echo ""
    echo -e "${CYAN}Installing PyTorch (CPU version)...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

    echo -e "${CYAN}Installing CV and audio packages...${NC}"
    pip install opencv-python librosa scipy scikit-image soundfile pydub face-alignment

    echo -e "${GREEN}✓ ML dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ Skipping ML dependencies${NC}"
    echo -e "${YELLOW}Avatar video generation will not be available${NC}"
    echo -e "${YELLOW}Install later with: pip install torch opencv-python librosa face-alignment${NC}"
fi

echo ""

##############################################################################
# 5. OPTIONAL: VOICE RECOGNITION
##############################################################################

echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║            VOICE RECOGNITION (OPTIONAL)                       ║${NC}"
echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${CYAN}Voice recognition enables voice control mode:${NC}"
echo "  • SpeechRecognition (Google Speech API)"
echo "  • PyAudio (microphone input)"
echo ""

read -p "Install voice recognition? (y/n): " install_voice

if [ "$install_voice" = "y" ]; then
    echo ""
    echo -e "${CYAN}Installing voice recognition packages...${NC}"
    pip install SpeechRecognition

    echo -e "${GREEN}✓ Voice recognition installed${NC}"
else
    echo -e "${YELLOW}⚠ Skipping voice recognition${NC}"
    echo -e "${YELLOW}Voice mode will not be available${NC}"
fi

echo ""

##############################################################################
# 6. VERIFICATION
##############################################################################

echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║                    VERIFICATION                               ║${NC}"
echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${CYAN}Checking installation...${NC}"
echo ""

# Check Ollama
if command -v ollama &> /dev/null && curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo -e "${GREEN}✓ Ollama installed and running${NC}"
    ollama list
else
    echo -e "${YELLOW}⚠ Ollama installed but not running${NC}"
    echo -e "${YELLOW}  Start with: ollama serve${NC}"
fi

echo ""

# Check Python packages
echo -e "${CYAN}Checking Python packages...${NC}"
python3 << 'EOFPY'
import sys

packages = {
    'yaml': 'PyYAML',
    'httpx': 'httpx',
    'fastapi': 'FastAPI',
}

optional = {
    'torch': 'PyTorch',
    'cv2': 'OpenCV',
    'librosa': 'Librosa',
    'speech_recognition': 'SpeechRecognition'
}

print("Core packages:")
all_ok = True
for module, name in packages.items():
    try:
        __import__(module)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name} (missing)")
        all_ok = False

print("\nOptional packages:")
for module, name in optional.items():
    try:
        __import__(module)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  - {name} (not installed)")

sys.exit(0 if all_ok else 1)
EOFPY

echo ""

##############################################################################
# 7. SUMMARY
##############################################################################

echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║                    INSTALLATION COMPLETE! 🎉                  ║${NC}"
echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${GREEN}✅ Dependencies and Ollama installation complete!${NC}"
echo ""

echo -e "${CYAN}What was installed:${NC}"
echo "  ✓ Ollama (local LLM server)"
echo "  ✓ Mistral 7B base model"
echo "  ✓ System dependencies (FFmpeg, PortAudio, espeak, etc.)"
echo "  ✓ Python packages (core + optional)"
echo ""

echo -e "${CYAN}Next steps:${NC}"
echo ""
echo "  1. ${YELLOW}Build custom Ara model:${NC}"
echo "     ${BLUE}./training/build_ara_model.sh${NC}"
echo ""
echo "  2. ${YELLOW}Run Ara setup:${NC}"
echo "     ${BLUE}./setup_ara.sh${NC}"
echo ""
echo "  3. ${YELLOW}Launch Ara:${NC}"
echo "     ${BLUE}./start_ara.sh${NC}"
echo ""

echo -e "${CYAN}Useful commands:${NC}"
echo "  • Start Ollama: ${BLUE}ollama serve${NC}"
echo "  • List models: ${BLUE}ollama list${NC}"
echo "  • Test Ara model: ${BLUE}ollama run ara${NC}"
echo "  • Pull more models: ${BLUE}ollama pull mixtral${NC}"
echo ""

echo -e "${GREEN}Ara is ready to be set up! 🚀${NC}"
