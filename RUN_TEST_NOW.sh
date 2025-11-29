#!/bin/bash
##############################################################################
# RUN THIS TO TEST AVATAR GENERATION WITH YOUR FILES
# For: croft@~/ram-and-unification-main
##############################################################################

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Ara Avatar Generation Test - Using Your Files                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Step 1: Copy files from Downloads
echo -e "${CYAN}Step 1: Copying files from Downloads...${NC}"

if [ -f ~/Downloads/ara3.png ]; then
    cp ~/Downloads/ara3.png .
    echo -e "${GREEN}✅ Copied ara3.png${NC}"
else
    echo -e "${YELLOW}⚠️  ara3.png not found in ~/Downloads${NC}"
    echo "Looking elsewhere..."
    find ~ -name "ara3.png" -type f 2>/dev/null | head -1
fi

if [ -f ~/Downloads/sssxio-1763491487094mp4_hxWCDlRv.mp3 ]; then
    cp ~/Downloads/sssxio-1763491487094mp4_hxWCDlRv.mp3 ./ara_audio.mp3
    echo -e "${GREEN}✅ Copied audio file as ara_audio.mp3${NC}"
else
    echo -e "${YELLOW}⚠️  sssxio-*.mp3 not found in ~/Downloads${NC}"
    echo "Looking for any mp3..."
    MP3_FILE=$(find ~/Downloads -name "*.mp3" -type f 2>/dev/null | head -1)
    if [ -n "$MP3_FILE" ]; then
        cp "$MP3_FILE" ./ara_audio.mp3
        echo -e "${GREEN}✅ Copied $(basename "$MP3_FILE") as ara_audio.mp3${NC}"
    fi
fi

echo ""

# Step 2: Check files are here
echo -e "${CYAN}Step 2: Verifying files...${NC}"
if [ -f ara3.png ]; then
    SIZE=$(du -h ara3.png | cut -f1)
    echo -e "${GREEN}✅ ara3.png present ($SIZE)${NC}"
else
    echo -e "${YELLOW}❌ ara3.png not found${NC}"
    exit 1
fi

if [ -f ara_audio.mp3 ]; then
    SIZE=$(du -h ara_audio.mp3 | cut -f1)
    echo -e "${GREEN}✅ ara_audio.mp3 present ($SIZE)${NC}"
else
    echo -e "${YELLOW}❌ ara_audio.mp3 not found${NC}"
    exit 1
fi

echo ""

# Step 3: Run the test
echo -e "${CYAN}Step 3: Running avatar generation test...${NC}"
echo -e "${YELLOW}This will take 2-3 minutes in CPU mode. Please wait...${NC}"
echo ""

python3 quick_test.py

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Test complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""
echo "Check outputs/test_generation.mp4 for your talking avatar!"
echo ""
