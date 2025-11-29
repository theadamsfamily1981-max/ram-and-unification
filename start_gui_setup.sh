#!/bin/bash

##############################################################################
# 🚀 ARA AVATAR SYSTEM - GUI SETUP LAUNCHER
##############################################################################
# Quick launcher for the web-based setup wizard
# This provides a beautiful interface to install everything with one click!
##############################################################################

echo "========================================================================"
echo "  🚀 Starting Ara Avatar System Setup Wizard"
echo "========================================================================"
echo ""
echo "The web-based setup wizard is starting..."
echo ""
echo "Once the server starts, open your browser to:"
echo "  👉 http://localhost:8000/setup"
echo ""
echo "Or use the chat interface at:"
echo "  👉 http://localhost:8000/"
echo ""
echo "========================================================================"
echo ""

# Change to multi-ai-workspace directory
cd "$(dirname "$0")/multi-ai-workspace"

# Start the server
echo "Starting web server..."
python3 -m uvicorn src.ui.app:app --host 0.0.0.0 --port 8000 --reload

