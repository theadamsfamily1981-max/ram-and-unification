#!/bin/bash
# Quick start script for Talking Avatar API

echo "=========================================="
echo "Talking Avatar API - Quick Start"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
if [ ! -f "venv/installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch venv/installed
else
    echo "Dependencies already installed"
fi

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "Please edit .env file to configure your settings"
fi

# Create directories
mkdir -p models uploads outputs temp

# Start server
echo "=========================================="
echo "Starting Talking Avatar API..."
echo "API will be available at:"
echo "  - http://localhost:8000"
echo "  - Docs: http://localhost:8000/docs"
echo "=========================================="

python -m src.main
