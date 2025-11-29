#!/bin/bash
##############################################################################
# Quick Test Script - Uses ara3.png and audio file for avatar generation
##############################################################################

set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         Ara Avatar Generation - Quick Test                    ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Activate venv if not active
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
fi

# Check for files
IMAGE_FILE=""
AUDIO_FILE=""

# Look for ara3.png
if [ -f "ara3.png" ]; then
    IMAGE_FILE="ara3.png"
elif [ -f "assets/avatars/ara3.png" ]; then
    IMAGE_FILE="assets/avatars/ara3.png"
else
    echo -e "${YELLOW}Looking for ara3.png...${NC}"
    IMAGE_FILE=$(find . -name "ara3.png" 2>/dev/null | head -1)
fi

# Look for mp3 file
if [ -f "sssxio.mp3" ]; then
    AUDIO_FILE="sssxio.mp3"
else
    echo -e "${YELLOW}Looking for mp3 file...${NC}"
    AUDIO_FILE=$(find . -name "*.mp3" 2>/dev/null | head -1)
fi

if [ -z "$IMAGE_FILE" ]; then
    echo -e "${RED}❌ ara3.png not found${NC}"
    echo -e "${CYAN}Please place ara3.png in the repository directory${NC}"
    echo ""
    echo "If you have the file elsewhere, copy it:"
    echo "  cp /path/to/ara3.png ."
    exit 1
fi

if [ -z "$AUDIO_FILE" ]; then
    echo -e "${RED}❌ MP3 audio file not found${NC}"
    echo -e "${CYAN}Please place your mp3 file in the repository directory${NC}"
    echo ""
    echo "If you have the file elsewhere, copy it:"
    echo "  cp /path/to/audio.mp3 ."
    exit 1
fi

echo -e "${GREEN}✅ Found image: $IMAGE_FILE${NC}"
echo -e "${GREEN}✅ Found audio: $AUDIO_FILE${NC}"
echo ""

# Ensure directories exist
mkdir -p assets/avatars
mkdir -p outputs
mkdir -p temp

# Copy files to expected locations
echo -e "${CYAN}Setting up test files...${NC}"

cp "$IMAGE_FILE" assets/avatars/test_avatar.jpg
echo -e "${GREEN}✅ Image copied to assets/avatars/test_avatar.jpg${NC}"

# Convert mp3 to wav if needed (avatar system expects wav)
if command -v ffmpeg &> /dev/null; then
    echo -e "${CYAN}Converting MP3 to WAV...${NC}"
    ffmpeg -i "$AUDIO_FILE" -ar 22050 -ac 1 outputs/test_audio.wav -y 2>&1 | grep -E "(Duration|size=)" || true
    echo -e "${GREEN}✅ Audio converted to outputs/test_audio.wav${NC}"
else
    echo -e "${YELLOW}⚠️  ffmpeg not found, copying as-is${NC}"
    cp "$AUDIO_FILE" outputs/test_audio.wav
fi

echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Test files ready!${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${CYAN}Now running avatar generation test...${NC}"
echo ""

# Run the test
python3 test_avatar_api.py

echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}Alternative: Test via API server${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
echo ""
echo "1. Start the API server:"
echo -e "   ${YELLOW}python run_ara.py --mode api${NC}"
echo ""
echo "2. In another terminal, test upload + generation:"
echo -e "   ${YELLOW}curl -X POST http://localhost:8000/upload/image \\"
echo "     -F \"file=@assets/avatars/test_avatar.jpg\"${NC}"
echo ""
echo -e "   ${YELLOW}curl -X POST http://localhost:8000/upload/audio \\"
echo "     -F \"file=@outputs/test_audio.wav\"${NC}"
echo ""
echo -e "   ${YELLOW}curl -X POST http://localhost:8000/generate \\"
echo "     -H \"Content-Type: application/json\" \\"
echo "     -d '{\"image_filename\":\"test_avatar.jpg\",\"audio_filename\":\"test_audio.wav\"}'${NC}"
echo ""
