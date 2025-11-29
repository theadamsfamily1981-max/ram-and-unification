#!/bin/bash
##############################################################################
# ARA - AI Co-Pilot Launcher
# Unified launcher for Ara avatar system with voice macros and T-FAN integration
##############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

##############################################################################
# Display Functions
##############################################################################

print_header() {
    clear
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                                                               ║"
    echo "║                 🤖  ARA - AI CO-PILOT  🤖                     ║"
    echo "║                                                               ║"
    echo "║         Your Local AI Avatar with Voice Control              ║"
    echo "║                                                               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_menu() {
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}MAIN MENU${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${CYAN}  1)${NC} 🎙️  Voice Mode      - Voice-controlled Ara with voice macros"
    echo -e "${CYAN}  2)${NC} 💬 Chat Mode       - Text chat with Ara (offline)"
    echo -e "${CYAN}  3)${NC} 🎬 Avatar Demo     - Generate talking avatar video"
    echo -e "${CYAN}  4)${NC} 🚀 T-FAN Cockpit   - Launch T-FAN HUD/metrics dashboard"
    echo -e "${CYAN}  5)${NC} 🌐 Multi-AI Server - Start web interface (all AIs)"
    echo ""
    echo -e "${GREEN}───────────────────────────────────────────────────────────────${NC}"
    echo -e "${CYAN}  6)${NC} 📋 List Macros     - Show all voice macros"
    echo -e "${CYAN}  7)${NC} ⚙️  Settings        - Configure Ara (mode, avatar, voice)"
    echo -e "${CYAN}  8)${NC} 🧪 System Check    - Test dependencies and health"
    echo ""
    echo -e "${GREEN}───────────────────────────────────────────────────────────────${NC}"
    echo -e "${CYAN}  0)${NC} 🚪 Exit"
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
}

##############################################################################
# Health Check Functions
##############################################################################

check_ollama() {
    if ! command -v ollama &> /dev/null; then
        echo -e "${RED}✗ Ollama not installed${NC}"
        echo -e "${YELLOW}Install from: https://ollama.ai/download${NC}"
        return 1
    fi

    if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo -e "${RED}✗ Ollama server not running${NC}"
        echo -e "${YELLOW}Start with: ollama serve${NC}"
        return 1
    fi

    echo -e "${GREEN}✓ Ollama running${NC}"
    return 0
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}✗ Python 3 not installed${NC}"
        return 1
    fi

    echo -e "${GREEN}✓ Python 3 installed${NC}"
    return 0
}

check_dependencies() {
    echo -e "\n${CYAN}Checking system dependencies...${NC}\n"

    check_python
    check_ollama

    # Check if venv exists
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}⚠ Virtual environment not found${NC}"
        echo -e "${YELLOW}Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt${NC}"
    else
        echo -e "${GREEN}✓ Virtual environment exists${NC}"
    fi

    # Check for ML dependencies (optional)
    if python3 -c "import torch; import cv2" 2>/dev/null; then
        echo -e "${GREEN}✓ ML dependencies (PyTorch, OpenCV) installed${NC}"
    else
        echo -e "${YELLOW}⚠ ML dependencies not fully installed (avatar generation limited)${NC}"
    fi

    # Check for espeak (TTS)
    if command -v espeak-ng &> /dev/null; then
        echo -e "${GREEN}✓ espeak-ng (TTS) installed${NC}"
    else
        echo -e "${YELLOW}⚠ espeak-ng not installed (voice output limited)${NC}"
    fi

    echo ""
}

##############################################################################
# Mode Functions
##############################################################################

voice_mode() {
    print_header
    echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║                    🎙️  VOICE MODE  🎙️                        ║${NC}"
    echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}Starting Ara in voice mode...${NC}"
    echo -e "${YELLOW}Say 'Ara' to wake up, then give your command.${NC}"
    echo -e "${YELLOW}Try voice macros like 'show gpu', 'red alert', 'warp core status'${NC}"
    echo ""

    # Check if speech recognition is available
    if ! python3 -c "import speech_recognition" 2>/dev/null; then
        echo -e "${YELLOW}⚠ speech_recognition not installed${NC}"
        echo -e "${YELLOW}Install with: pip install SpeechRecognition pyaudio${NC}"
        echo ""
        read -p "Continue anyway? (y/n): " choice
        if [ "$choice" != "y" ]; then
            return
        fi
    fi

    python3 ara_voice_interface.py
}

chat_mode() {
    print_header
    echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║                    💬 CHAT MODE  💬                          ║${NC}"
    echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}Starting Ara in text chat mode (offline)...${NC}"
    echo -e "${YELLOW}Type your messages and press Enter.${NC}"
    echo ""

    python3 ara_voice_interface.py --text-only
}

avatar_demo() {
    print_header
    echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║                   🎬 AVATAR DEMO  🎬                         ║${NC}"
    echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}Generate a talking avatar video from text...${NC}"
    echo ""

    read -p "Enter text for Ara to say: " text_input

    if [ -z "$text_input" ]; then
        echo -e "${RED}No text provided${NC}"
        return
    fi

    echo ""
    echo -e "${CYAN}Generating avatar response...${NC}"

    python3 ara_voice_interface.py --test "$text_input" --no-avatar=false

    echo ""
    read -p "Press Enter to continue..."
}

start_tfan() {
    print_header
    echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║                 🚀 T-FAN COCKPIT  🚀                         ║${NC}"
    echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    TFAN_DIR="$HOME/tfan-ara-system/Quanta-meis-nib-cis"

    if [ ! -d "$TFAN_DIR" ]; then
        echo -e "${YELLOW}⚠ T-FAN cockpit not found at $TFAN_DIR${NC}"
        echo ""
        echo -e "${CYAN}T-FAN provides:${NC}"
        echo "  - Spaceship-style HUD with metrics"
        echo "  - GPU, CPU, network, storage monitoring"
        echo "  - Topology visualization"
        echo "  - Workspace modes (work, relax, focus)"
        echo ""
        echo -e "${YELLOW}Installation:${NC}"
        echo "  1. Clone repository: git clone https://github.com/theadamsfamily1981-max/Quanta-meis-nib-cis.git"
        echo "  2. Or run the unified installer: ./install_complete_system.sh"
        echo ""
        read -p "Press Enter to continue..."
        return
    fi

    echo -e "${CYAN}Starting T-FAN cockpit...${NC}"
    cd "$TFAN_DIR"

    if [ -f "start_cockpit.sh" ]; then
        ./start_cockpit.sh
    else
        echo -e "${RED}start_cockpit.sh not found${NC}"
    fi

    cd "$SCRIPT_DIR"
}

start_multi_ai_server() {
    print_header
    echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║              🌐 MULTI-AI WEB SERVER  🌐                      ║${NC}"
    echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}Starting multi-AI workspace web interface...${NC}"
    echo -e "${YELLOW}Access at: http://localhost:8000${NC}"
    echo ""
    echo -e "${CYAN}This provides:${NC}"
    echo "  - Web UI for all AIs (Claude, Nova, Pulse, Ara)"
    echo "  - Tag-based routing (#code, #creative, #multiverse)"
    echo "  - Perspectives mixer (compare multiple AIs)"
    echo "  - Context packs and cross-posting"
    echo ""

    cd multi-ai-workspace

    # Check if .env exists
    if [ ! -f ".env" ]; then
        echo -e "${YELLOW}⚠ .env not found, creating from .env.example${NC}"
        cp .env.example .env
        echo -e "${YELLOW}Edit .env to add your API keys${NC}"
        echo ""
    fi

    python3 -m uvicorn src.ui.app:app --reload --host 0.0.0.0 --port 8000
}

list_macros() {
    print_header
    echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║                  📋 VOICE MACROS  📋                         ║${NC}"
    echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    python3 << 'EOFPY'
import yaml
from pathlib import Path

config_path = Path("multi-ai-workspace/config/voice_macros.yaml")

if not config_path.exists():
    print("❌ Voice macros config not found")
    exit(1)

with open(config_path) as f:
    config = yaml.safe_load(f)

macros = config.get("macros", {})

# Group by type
groups = {}
for name, macro in macros.items():
    macro_type = macro.get("type", "unknown")
    if macro_type not in groups:
        groups[macro_type] = []
    groups[macro_type].append((name, macro.get("speak_summary", "No description")))

# Display
for group_type, items in groups.items():
    print(f"\n\033[1;36m{group_type.replace('_', ' ').title()}\033[0m")
    print("─" * 60)
    for name, desc in items[:10]:  # Limit to 10 per group
        print(f"  \033[0;33m'{name}'\033[0m")
        print(f"    {desc}")
    if len(items) > 10:
        print(f"\n  ... and {len(items) - 10} more")

print(f"\n\033[1;32mTotal: {len(macros)} voice macros\033[0m")
EOFPY

    echo ""
    read -p "Press Enter to continue..."
}

settings_menu() {
    print_header
    echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║                   ⚙️  SETTINGS  ⚙️                           ║${NC}"
    echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}Configuration:${NC}"
    echo ""
    echo "  Ara Persona:     multi-ai-workspace/config/ara_persona.yaml"
    echo "  Voice Macros:    multi-ai-workspace/config/voice_macros.yaml"
    echo "  Workspace:       multi-ai-workspace/config/workspace.yaml"
    echo "  Avatar Images:   assets/avatars/"
    echo ""
    echo -e "${YELLOW}Quick Settings:${NC}"
    echo ""
    echo "  1) Edit persona configuration"
    echo "  2) Edit voice macros"
    echo "  3) View avatar profiles"
    echo "  0) Back to main menu"
    echo ""

    read -p "Select option: " choice

    case $choice in
        1)
            ${EDITOR:-nano} multi-ai-workspace/config/ara_persona.yaml
            ;;
        2)
            ${EDITOR:-nano} multi-ai-workspace/config/voice_macros.yaml
            ;;
        3)
            echo -e "\n${CYAN}Avatar Profiles:${NC}"
            echo "  - default:          Clean neon aesthetic"
            echo "  - professional:     Focused, competent"
            echo "  - casual:           Relaxed, streetwear sci-fi"
            echo "  - sci_fi_cockpit:   Hologram mode"
            echo "  - quantum_scientist: Analytical scientist"
            echo "  - holodeck:         Immersive VR aesthetic"
            echo "  - dramatic:         Cinematic lighting"
            echo ""
            echo "Use voice macros to switch: 'avatar professional', 'avatar sci fi', etc."
            echo ""
            read -p "Press Enter to continue..."
            ;;
    esac
}

system_check() {
    print_header
    echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║                  🧪 SYSTEM CHECK  🧪                         ║${NC}"
    echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════════╝${NC}"

    check_dependencies

    echo -e "${CYAN}Testing Ara backend...${NC}"
    python3 ara_voice_interface.py --test "Hello, this is a test" --text-only

    echo ""
    read -p "Press Enter to continue..."
}

##############################################################################
# Main Loop
##############################################################################

main_loop() {
    while true; do
        print_header
        print_menu

        read -p "Select option (0-8): " choice

        case $choice in
            1) voice_mode ;;
            2) chat_mode ;;
            3) avatar_demo ;;
            4) start_tfan ;;
            5) start_multi_ai_server ;;
            6) list_macros ;;
            7) settings_menu ;;
            8) system_check ;;
            0)
                print_header
                echo -e "${CYAN}Shutting down Ara. It's been good working with you.${NC}"
                echo ""
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option${NC}"
                sleep 1
                ;;
        esac
    done
}

# Start main loop
main_loop
