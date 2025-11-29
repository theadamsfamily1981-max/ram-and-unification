#!/bin/bash
##############################################################################
# QUICK START - Ara Avatar System Setup
# This script works from ANY directory and downloads the latest setup server
##############################################################################

set -e

echo "========================================================================"
echo "  🚀 ARA AVATAR SYSTEM - QUICK START"
echo "========================================================================"
echo ""

# Check if we're in a ram-and-unification directory
if [ -f "setup_server.py" ]; then
    echo "✅ Found setup_server.py in current directory"
    SETUP_SCRIPT="./setup_server.py"
else
    # Download the latest version
    echo "📥 Downloading latest setup server..."

    SETUP_URL="https://raw.githubusercontent.com/theadamsfamily1981-max/ram-and-unification/main/setup_server.py"
    SETUP_SCRIPT="/tmp/ara_setup_server.py"

    if command -v curl &> /dev/null; then
        curl -sL "$SETUP_URL" -o "$SETUP_SCRIPT" 2>/dev/null || {
            echo "⚠️  Download failed. Using local fallback..."
            SETUP_SCRIPT="setup_server.py"
        }
    elif command -v wget &> /dev/null; then
        wget -q "$SETUP_URL" -O "$SETUP_SCRIPT" 2>/dev/null || {
            echo "⚠️  Download failed. Using local fallback..."
            SETUP_SCRIPT="setup_server.py"
        }
    else
        echo "❌ Neither curl nor wget found. Please install one of them."
        exit 1
    fi
fi

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Installing..."
    sudo apt update && sudo apt install -y python3
fi

# Check if fastapi is installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing FastAPI and Uvicorn..."
    sudo apt install -y python3-fastapi python3-uvicorn 2>/dev/null || {
        pip3 install fastapi uvicorn --break-system-packages
    }
fi

echo ""
echo "========================================================================"
echo "  ✅ STARTING SETUP SERVER"
echo "========================================================================"
echo ""
echo "  Open your browser to:"
echo "  👉 http://localhost:8000"
echo ""
echo "  Press CTRL+C to stop the server"
echo ""
echo "========================================================================"
echo ""

# Run the setup server
python3 "$SETUP_SCRIPT"
