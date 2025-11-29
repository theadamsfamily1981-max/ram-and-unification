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
echo "  👉 http://localhost:8000"
echo ""
echo "========================================================================"
echo ""

# Change to project directory
cd "$(dirname "$0")"

# Start the standalone setup server
python3 setup_server.py

