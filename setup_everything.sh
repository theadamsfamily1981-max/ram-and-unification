#!/bin/bash

##############################################################################
# 🚀 TALKING AVATAR SETUP SCRIPT - Does Everything Automatically!
##############################################################################
# This script:
# 1. Downloads missing repositories (T-FAN cockpit)
# 2. Installs all Python packages
# 3. Installs system dependencies (ffmpeg, audio, etc.)
# 4. Creates needed folders
# 5. Tests that everything works
##############################################################################

set -e  # Stop if anything fails

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Emoji-like markers
CHECKMARK="[OK]"
CROSS="[FAIL]"
ARROW="==>"

echo ""
echo "========================================================================"
echo "  🚀 TALKING AVATAR COMPLETE SETUP"
echo "========================================================================"
echo ""

##############################################################################
# Step 1: Check Python Version
##############################################################################
echo -e "${BLUE}${ARROW} Step 1: Checking Python...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}${CROSS} Python 3 not found!${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}${CHECKMARK} Python ${PYTHON_VERSION} found${NC}"
echo ""

##############################################################################
# Step 2: Download T-FAN Cockpit Repository
##############################################################################
echo -e "${BLUE}${ARROW} Step 2: Checking for T-FAN cockpit...${NC}"

TFAN_DIR="/home/user/Quanta-meis-nib-cis"

if [ -d "$TFAN_DIR" ]; then
    echo -e "${GREEN}${CHECKMARK} T-FAN already downloaded${NC}"
else
    echo -e "${YELLOW}Downloading T-FAN cockpit repository...${NC}"
    cd /home/user

    if git clone https://github.com/theadamsfamily1981-max/Quanta-meis-nib-cis 2>/dev/null; then
        echo -e "${GREEN}${CHECKMARK} T-FAN downloaded successfully${NC}"
    else
        echo -e "${YELLOW}Note: Could not download T-FAN (may need authentication)${NC}"
        echo -e "${YELLOW}You can download it manually later if needed${NC}"
    fi

    cd /home/user/ram-and-unification
fi
echo ""

##############################################################################
# Step 3: Install System Dependencies
##############################################################################
echo -e "${BLUE}${ARROW} Step 3: Installing system packages...${NC}"

echo "This needs sudo permissions. Installing:"
echo "  - ffmpeg (video encoding)"
echo "  - portaudio (microphone/speakers)"
echo "  - espeak-ng (text-to-speech)"
echo ""

if command -v apt-get &> /dev/null; then
    # Check if packages are already installed
    PACKAGES_TO_INSTALL=""

    for pkg in ffmpeg portaudio19-dev espeak-ng libx264-dev libx265-dev; do
        if ! dpkg -l | grep -q "^ii  $pkg"; then
            PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL $pkg"
        fi
    done

    if [ -n "$PACKAGES_TO_INSTALL" ]; then
        echo "Installing: $PACKAGES_TO_INSTALL"
        apt-get update -qq 2>/dev/null || true
        apt-get install -y $PACKAGES_TO_INSTALL 2>&1 | grep -E "(Setting up|already)" || true
        echo -e "${GREEN}${CHECKMARK} System packages installed${NC}"
    else
        echo -e "${GREEN}${CHECKMARK} All system packages already installed${NC}"
    fi
else
    echo -e "${YELLOW}Note: Not on Ubuntu/Debian, skipping apt-get packages${NC}"
fi
echo ""

##############################################################################
# Step 4: Install Python Packages
##############################################################################
echo -e "${BLUE}${ARROW} Step 4: Installing Python ML packages...${NC}"
echo "This might take a few minutes (downloading PyTorch, etc.)..."
echo ""

cd /home/user/ram-and-unification

# Install core packages first
pip install -q fastapi uvicorn pydantic pydantic-settings python-multipart aiofiles python-dotenv 2>&1 | tail -3

# Install ML packages
pip install -q numpy pillow pyyaml requests tqdm imageio imageio-ffmpeg 2>&1 | tail -3
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3 || \
    pip install -q torch torchvision torchaudio 2>&1 | tail -3

# Install image/video processing
pip install -q opencv-python scikit-image scipy pydub 2>&1 | tail -3

# Install audio processing
pip install -q librosa soundfile 2>&1 | tail -3

# Install face detection
pip install -q face-alignment 2>&1 | tail -3

echo -e "${GREEN}${CHECKMARK} Python packages installed${NC}"
echo ""

##############################################################################
# Step 5: Create Necessary Directories
##############################################################################
echo -e "${BLUE}${ARROW} Step 5: Creating folders...${NC}"

mkdir -p models
mkdir -p uploads
mkdir -p outputs
mkdir -p temp
mkdir -p assets/avatars
mkdir -p assets/backgrounds
mkdir -p assets/videos
mkdir -p conversational-avatar/assets/avatars

echo -e "${GREEN}${CHECKMARK} Folders created${NC}"
echo ""

##############################################################################
# Step 6: Create .env File if Missing
##############################################################################
echo -e "${BLUE}${ARROW} Step 6: Setting up configuration...${NC}"

if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}${CHECKMARK} Created .env from example${NC}"
    else
        # Create a basic .env file
        cat > .env << 'EOF'
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Model Configuration
DEVICE=cpu
BATCH_SIZE=1

# Processing Configuration
OUTPUT_FPS=25
OUTPUT_RESOLUTION=512
MAX_VIDEO_LENGTH=300

# API Security (optional)
# API_KEY=your-secret-key-here
EOF
        echo -e "${GREEN}${CHECKMARK} Created basic .env file${NC}"
    fi
else
    echo -e "${GREEN}${CHECKMARK} .env already exists${NC}"
fi
echo ""

##############################################################################
# Step 7: Test Installation
##############################################################################
echo -e "${BLUE}${ARROW} Step 7: Testing installation...${NC}"

python3 << 'PYTEST'
import sys

print("\nTesting imports...")

# Test basic imports
try:
    import torch
    print(f"  {chr(10004)} PyTorch: {torch.__version__}")
except Exception as e:
    print(f"  {chr(10008)} PyTorch: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"  {chr(10004)} NumPy: {np.__version__}")
except Exception as e:
    print(f"  {chr(10008)} NumPy: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"  {chr(10004)} OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"  {chr(10008)} OpenCV: {e}")
    sys.exit(1)

try:
    import librosa
    print(f"  {chr(10004)} Librosa: {librosa.__version__}")
except Exception as e:
    print(f"  {chr(10008)} Librosa: {e}")
    sys.exit(1)

# Test avatar engine
try:
    from src.avatar_engine import AvatarGenerator
    gen = AvatarGenerator(device='cpu')
    print(f"  {chr(10004)} AvatarGenerator works!")
except Exception as e:
    print(f"  {chr(10008)} AvatarGenerator: {e}")
    sys.exit(1)

# Test API
try:
    from src.main import app
    print(f"  {chr(10004)} API imports successfully!")
except Exception as e:
    print(f"  {chr(10008)} API: {e}")
    sys.exit(1)

print("\n✨ All tests passed!")
PYTEST

if [ $? -eq 0 ]; then
    echo -e "${GREEN}${CHECKMARK} All components working!${NC}"
else
    echo -e "${RED}${CROSS} Some tests failed${NC}"
    exit 1
fi
echo ""

##############################################################################
# Step 8: Summary
##############################################################################
echo ""
echo "========================================================================"
echo "  ✨ SETUP COMPLETE!"
echo "========================================================================"
echo ""
echo -e "${GREEN}Everything is installed and working!${NC}"
echo ""
echo "📁 Your directories:"
echo "  - Avatar API:  /home/user/ram-and-unification"
if [ -d "$TFAN_DIR" ]; then
    echo "  - T-FAN Cockpit: $TFAN_DIR"
fi
echo ""
echo "🚀 What you can do now:"
echo ""
echo "  1️⃣  Start the REST API server:"
echo "      cd /home/user/ram-and-unification"
echo "      python -m src.main"
echo "      # Then visit: http://localhost:8000/docs"
echo ""
echo "  2️⃣  Run the voice avatar:"
echo "      cd /home/user/ram-and-unification/conversational-avatar"
echo "      python main.py"
echo ""
echo "  3️⃣  Test video generation:"
echo "      cd /home/user/ram-and-unification"
echo "      python examples/simple_client.py"
echo ""
echo "📝 Next steps:"
echo "  - Add an avatar image to: assets/avatars/default.jpg"
echo "  - Get some test audio files"
echo "  - Try generating your first talking avatar!"
echo ""
echo "========================================================================"
echo ""
