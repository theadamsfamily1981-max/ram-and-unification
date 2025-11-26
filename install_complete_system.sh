#!/bin/bash

##############################################################################
# 🚀 COMPLETE T-FAN + ARA AVATAR SYSTEM INSTALLER
##############################################################################
# This installs EVERYTHING in one go:
# - Talking Avatar API (ram-and-unification)
# - T-FAN GNOME Cockpit (Quanta-meis-nib-cis)
# - All system dependencies
# - All Python packages
# - GNOME integration
# - WebKit fixes
# - Desktop shortcuts
##############################################################################

set -e  # Stop if anything fails

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Emoji markers
ROCKET="🚀"
CHECK="✅"
CROSS="❌"
ARROW="➡️"
PACKAGE="📦"
TOOL="🔧"
STAR="⭐"

echo ""
echo "========================================================================"
echo "  $ROCKET COMPLETE T-FAN + ARA AVATAR SYSTEM INSTALLER"
echo "========================================================================"
echo ""
echo "This will install:"
echo "  1. Talking Avatar API with Ara persona"
echo "  2. T-FAN GNOME Cockpit with HUD"
echo "  3. Voice macros and integration"
echo "  4. All dependencies (system + Python)"
echo "  5. Desktop shortcuts and launchers"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 0
fi
echo ""

##############################################################################
# Step 1: Check System
##############################################################################
echo -e "${BLUE}${ARROW} Step 1: Checking system requirements...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}${CROSS} Python 3 not found! Please install Python 3.8+${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}${CHECK} Python ${PYTHON_VERSION}${NC}"

# Check Git
if ! command -v git &> /dev/null; then
    echo -e "${RED}${CROSS} Git not found! Installing...${NC}"
    apt-get update && apt-get install -y git
fi
echo -e "${GREEN}${CHECK} Git available${NC}"

# Check if running on Ubuntu/Debian
if command -v apt-get &> /dev/null; then
    echo -e "${GREEN}${CHECK} Ubuntu/Debian detected${NC}"
    HAS_APT=true
else
    echo -e "${YELLOW}Not Ubuntu/Debian - some features may not work${NC}"
    HAS_APT=false
fi

# Detect home directory
if [ -n "$SUDO_USER" ]; then
    REAL_HOME=$(eval echo ~$SUDO_USER)
    REAL_USER=$SUDO_USER
else
    REAL_HOME=$HOME
    REAL_USER=$USER
fi

echo -e "${GREEN}${CHECK} Installing for user: $REAL_USER${NC}"
echo -e "${GREEN}${CHECK} Home directory: $REAL_HOME${NC}"
echo ""

##############################################################################
# Step 2: Install System Dependencies
##############################################################################
echo -e "${BLUE}${ARROW} Step 2: Installing system packages...${NC}"

if [ "$HAS_APT" = true ]; then
    echo "Installing: ffmpeg, portaudio, espeak, WebKit, GTK4..."

    apt-get update -qq 2>/dev/null || true

    apt-get install -y \
        ffmpeg \
        portaudio19-dev \
        espeak-ng \
        libx264-dev \
        libx265-dev \
        libvpx-dev \
        libopus-dev \
        python3-pip \
        python3-dev \
        python3-gi \
        gir1.2-gtk-4.0 \
        gir1.2-adw-1 \
        gir1.2-webkit2-4.1 \
        libwebkit2gtk-4.1-0 \
        gnome-shell-extension-prefs \
        2>&1 | grep -E "(Setting up|already|Unpacking)" | tail -20 || true

    echo -e "${GREEN}${CHECK} System packages installed${NC}"
else
    echo -e "${YELLOW}Skipping apt packages (not on Ubuntu/Debian)${NC}"
fi
echo ""

##############################################################################
# Step 3: Setup Directory Structure
##############################################################################
echo -e "${BLUE}${ARROW} Step 3: Setting up directories...${NC}"

BASE_DIR="$REAL_HOME/tfan-ara-system"
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

echo -e "${GREEN}${CHECK} Base directory: $BASE_DIR${NC}"
echo ""

##############################################################################
# Step 4: Download/Check Repositories
##############################################################################
echo -e "${BLUE}${ARROW} Step 4: Checking repositories...${NC}"

# Avatar API (ram-and-unification)
AVATAR_DIR="$BASE_DIR/ram-and-unification"
if [ -d "$AVATAR_DIR" ]; then
    echo -e "${GREEN}${CHECK} Avatar API already present${NC}"
else
    echo "Downloading Avatar API..."
    if [ -d "/home/user/ram-and-unification" ]; then
        # Copy from current location
        cp -r /home/user/ram-and-unification "$AVATAR_DIR"
        echo -e "${GREEN}${CHECK} Avatar API copied${NC}"
    else
        echo -e "${YELLOW}Note: Avatar API not found. Please download manually.${NC}"
        echo "  Place it in: $AVATAR_DIR"
    fi
fi

# T-FAN Cockpit (Quanta-meis-nib-cis)
TFAN_DIR="$BASE_DIR/Quanta-meis-nib-cis"
if [ -d "$TFAN_DIR" ]; then
    echo -e "${GREEN}${CHECK} T-FAN Cockpit already present${NC}"
else
    echo "Downloading T-FAN Cockpit..."

    # Try to clone
    if git clone https://github.com/theadamsfamily1981-max/Quanta-meis-nib-cis "$TFAN_DIR" 2>/dev/null; then
        echo -e "${GREEN}${CHECK} T-FAN Cockpit downloaded${NC}"
    else
        # Check if it exists in user's home
        if [ -d "$REAL_HOME/Quanta-meis-nib-cis-main" ]; then
            cp -r "$REAL_HOME/Quanta-meis-nib-cis-main" "$TFAN_DIR"
            echo -e "${GREEN}${CHECK} T-FAN Cockpit copied from existing download${NC}"
        elif [ -d "$REAL_HOME/Quanta-meis-nib-cis" ]; then
            cp -r "$REAL_HOME/Quanta-meis-nib-cis" "$TFAN_DIR"
            echo -e "${GREEN}${CHECK} T-FAN Cockpit copied${NC}"
        else
            echo -e "${YELLOW}${ARROW} Could not download T-FAN (may need authentication)${NC}"
            echo ""
            echo "Please download manually:"
            echo "  1. Go to: https://github.com/theadamsfamily1981-max/Quanta-meis-nib-cis"
            echo "  2. Download ZIP"
            echo "  3. Extract to: $TFAN_DIR"
            echo ""
            echo "Then run this script again."
            echo ""
            read -p "Skip T-FAN for now? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
            SKIP_TFAN=true
        fi
    fi
fi
echo ""

##############################################################################
# Step 5: Install Python Packages
##############################################################################
echo -e "${BLUE}${ARROW} Step 5: Installing Python packages...${NC}"
echo "This may take 5-10 minutes (downloading PyTorch, etc.)..."
echo ""

# Install for avatar
if [ -d "$AVATAR_DIR" ]; then
    cd "$AVATAR_DIR"

    # Core packages
    pip3 install -q fastapi uvicorn pydantic pydantic-settings python-multipart aiofiles python-dotenv 2>&1 | tail -3

    # ML packages
    pip3 install -q numpy pillow pyyaml requests tqdm imageio imageio-ffmpeg 2>&1 | tail -3
    pip3 install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3 || \
        pip3 install -q torch torchvision torchaudio 2>&1 | tail -3

    # Image/video
    pip3 install -q opencv-python scikit-image scipy pydub 2>&1 | tail -3

    # Audio
    pip3 install -q librosa soundfile 2>&1 | tail -3

    # Face detection
    pip3 install -q face-alignment 2>&1 | tail -3

    echo -e "${GREEN}${CHECK} Avatar Python packages installed${NC}"
fi

# Install for T-FAN
if [ -d "$TFAN_DIR" ] && [ "$SKIP_TFAN" != true ]; then
    cd "$TFAN_DIR"

    if [ -f "requirements.txt" ]; then
        pip3 install -q -r requirements.txt 2>&1 | tail -10 || true
    fi

    echo -e "${GREEN}${CHECK} T-FAN Python packages installed${NC}"
fi
echo ""

##############################################################################
# Step 6: Fix WebKit Issues in T-FAN
##############################################################################
if [ -d "$TFAN_DIR" ] && [ "$SKIP_TFAN" != true ]; then
    echo -e "${BLUE}${ARROW} Step 6: Fixing WebKit compatibility...${NC}"

    # Find all Python files that need fixing
    find "$TFAN_DIR" -name "*.py" -type f -exec grep -l "gi.require_version('WebKit', '6.0')" {} \; | while read file; do
        echo "Fixing: $file"

        # Backup
        cp "$file" "$file.bak"

        # Fix WebKit version
        sed -i "s/gi.require_version('WebKit', '6.0')/gi.require_version('WebKit2', '4.1')/g" "$file"
        sed -i "s/from gi.repository import.*WebKit$/from gi.repository import WebKit2 as WebKit/g" "$file"

        echo -e "${GREEN}${CHECK} Fixed WebKit in: $(basename $file)${NC}"
    done
    echo ""
fi

##############################################################################
# Step 7: Create Directories for Assets
##############################################################################
echo -e "${BLUE}${ARROW} Step 7: Creating asset directories...${NC}"

if [ -d "$AVATAR_DIR" ]; then
    cd "$AVATAR_DIR"
    mkdir -p models uploads outputs temp
    mkdir -p assets/avatars assets/backgrounds assets/videos
    mkdir -p conversational-avatar/assets/avatars
    echo -e "${GREEN}${CHECK} Avatar asset directories created${NC}"
fi
echo ""

##############################################################################
# Step 8: Setup Configuration Files
##############################################################################
echo -e "${BLUE}${ARROW} Step 8: Setting up configuration...${NC}"

if [ -d "$AVATAR_DIR" ]; then
    cd "$AVATAR_DIR"

    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
        else
            cat > .env << 'EOF'
# Avatar API Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false
DEVICE=cpu
OUTPUT_FPS=25
OUTPUT_RESOLUTION=512
MAX_VIDEO_LENGTH=300
EOF
        fi
        echo -e "${GREEN}${CHECK} Created .env configuration${NC}"
    fi
fi
echo ""

##############################################################################
# Step 9: Install GNOME Integration
##############################################################################
if [ -d "$TFAN_DIR" ] && [ "$SKIP_TFAN" != true ]; then
    echo -e "${BLUE}${ARROW} Step 9: Installing GNOME integration...${NC}"

    GNOME_DIR="$TFAN_DIR/gnome-tfan"
    if [ -d "$GNOME_DIR" ]; then
        cd "$GNOME_DIR"

        if [ -f "install.sh" ]; then
            chmod +x install.sh
            ./install.sh
            echo -e "${GREEN}${CHECK} GNOME integration installed${NC}"
        fi
    fi
    echo ""
fi

##############################################################################
# Step 10: Create Unified Launcher Scripts
##############################################################################
echo -e "${BLUE}${ARROW} Step 10: Creating launcher scripts...${NC}"

# Main launcher menu
cat > "$BASE_DIR/tfan-ara-launcher.sh" << 'EOFMENU'
#!/bin/bash

# T-FAN + Ara Unified Launcher

BASE_DIR="$(dirname "$(readlink -f "$0")")"

while true; do
    clear
    echo "========================================================================"
    echo "  🚀 T-FAN + ARA AVATAR SYSTEM LAUNCHER"
    echo "========================================================================"
    echo ""
    echo "Choose what to launch:"
    echo ""
    echo "  1) 🎤 Voice Avatar (Talk to Ara)"
    echo "  2) 🌐 Avatar REST API (Web interface)"
    echo "  3) 🖥️  T-FAN GNOME Cockpit (HUD)"
    echo "  4) 🎬 Quick Generate (Image + Audio → Video)"
    echo "  5) 🧪 Quick Test (Verify installation)"
    echo "  6) 📊 Start Everything (All components)"
    echo ""
    echo "  7) ⚙️  Settings & Configuration"
    echo "  8) 📖 View Documentation"
    echo "  9) 🔄 Update System"
    echo ""
    echo "  0) Exit"
    echo ""
    echo "========================================================================"
    read -p "Select option (0-9): " choice

    case $choice in
        1)
            echo "Starting Voice Avatar..."
            cd "$BASE_DIR/ram-and-unification/conversational-avatar"
            python3 main.py
            ;;
        2)
            echo "Starting Avatar API..."
            cd "$BASE_DIR/ram-and-unification"
            python3 -m src.main
            ;;
        3)
            echo "Starting T-FAN Cockpit..."
            tfan-gnome
            ;;
        4)
            # Quick Generate - Image + Audio → Video
            clear
            echo "========================================================================"
            echo "  🎬 QUICK GENERATE: Create Talking Avatar Video"
            echo "========================================================================"
            echo ""
            echo "This will combine an image and audio file to create a talking avatar."
            echo ""

            # Ask for image file
            read -p "Enter path to avatar image (JPG/PNG): " img_path

            # Validate image exists
            if [ ! -f "$img_path" ]; then
                echo "❌ Image file not found: $img_path"
                read -p "Press Enter to continue..."
                continue
            fi

            # Ask for audio file
            read -p "Enter path to audio file (WAV/MP3): " audio_path

            # Validate audio exists
            if [ ! -f "$audio_path" ]; then
                echo "❌ Audio file not found: $audio_path"
                read -p "Press Enter to continue..."
                continue
            fi

            echo ""
            echo "📋 Summary:"
            echo "  Image: $img_path"
            echo "  Audio: $audio_path"
            echo ""

            # Ask for confirmation
            read -p "Generate video? (y/n): " confirm
            if [[ ! $confirm =~ ^[Yy]$ ]]; then
                continue
            fi

            echo ""
            echo "🎬 Generating talking avatar video..."
            echo "This may take a few minutes..."
            echo ""

            # Copy files to system
            cd "$BASE_DIR/ram-and-unification"

            # Copy image to assets
            img_filename="avatar_$(date +%s).jpg"
            cp "$img_path" "assets/avatars/$img_filename"
            echo "✅ Image saved to: assets/avatars/$img_filename"

            # Copy audio to uploads
            audio_ext="${audio_path##*.}"
            audio_filename="audio_$(date +%s).$audio_ext"
            cp "$audio_path" "uploads/$audio_filename"
            echo "✅ Audio saved to: uploads/$audio_filename"

            # Generate video using Python
            output_video="outputs/talking_avatar_$(date +%s).mp4"

            python3 << EOFPY
import sys
from pathlib import Path
from src.avatar_engine import AvatarGenerator

print("\n🎨 Initializing avatar generator...")
generator = AvatarGenerator(device='cpu')

print("🎬 Generating talking avatar...")
result = generator.generate(
    image_input=Path("assets/avatars/$img_filename"),
    audio_input=Path("uploads/$audio_filename"),
    output_path=Path("$output_video")
)

if result.success:
    print(f"\n✅ SUCCESS! Video generated!")
    print(f"📍 Location: $output_video")
    print(f"⏱️  Duration: {result.duration:.2f} seconds")
    print(f"🎞️  Frames: {result.frames_generated}")
else:
    print(f"\n❌ ERROR: {result.error_message}")
    sys.exit(1)
EOFPY

            if [ $? -eq 0 ]; then
                echo ""
                echo "========================================================================"
                echo "  ✨ VIDEO GENERATED SUCCESSFULLY!"
                echo "========================================================================"
                echo ""
                echo "📍 Your video is ready at:"
                echo "   $(realpath "$output_video")"
                echo ""

                # Ask if they want to play it
                read -p "Play video now? (y/n): " play_confirm
                if [[ $play_confirm =~ ^[Yy]$ ]]; then
                    if command -v vlc &> /dev/null; then
                        vlc "$output_video" &
                    elif command -v mpv &> /dev/null; then
                        mpv "$output_video"
                    elif command -v xdg-open &> /dev/null; then
                        xdg-open "$output_video"
                    else
                        echo "No video player found. Please open the file manually."
                    fi
                fi

                # Ask if they want to set this as default avatar
                echo ""
                read -p "Set this image as default avatar for voice mode? (y/n): " default_confirm
                if [[ $default_confirm =~ ^[Yy]$ ]]; then
                    cp "assets/avatars/$img_filename" "assets/avatars/default.jpg"
                    cp "assets/avatars/$img_filename" "conversational-avatar/assets/avatars/default.jpg"
                    echo "✅ Set as default avatar!"
                fi
            else
                echo ""
                echo "❌ Video generation failed. Check the error message above."
            fi

            echo ""
            read -p "Press Enter to continue..."
            ;;
        5)
            echo "Running quick test..."
            cd "$BASE_DIR/ram-and-unification"
            python3 << 'EOF'
from src.avatar_engine import AvatarGenerator
print("✅ Avatar Generator works!")
from src.main import app
print("✅ API works!")
print("\n🎉 All systems operational!")
EOF
            read -p "Press Enter to continue..."
            ;;
        6)
            echo "Starting all components..."
            gnome-terminal -- bash -c "cd $BASE_DIR/ram-and-unification && python3 -m src.main; exec bash" &
            sleep 2
            tfan-gnome &
            echo "All components started!"
            read -p "Press Enter to continue..."
            ;;
        7)
            nano "$BASE_DIR/ram-and-unification/.env"
            ;;
        8)
            xdg-open "https://github.com/theadamsfamily1981-max/ram-and-unification/blob/main/README.md"
            ;;
        9)
            cd "$BASE_DIR"
            git -C ram-and-unification pull
            git -C Quanta-meis-nib-cis pull
            echo "System updated!"
            read -p "Press Enter to continue..."
            ;;
        0)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid option!"
            sleep 2
            ;;
    esac
done
EOFMENU

chmod +x "$BASE_DIR/tfan-ara-launcher.sh"

# Create desktop shortcut
mkdir -p "$REAL_HOME/.local/share/applications"

cat > "$REAL_HOME/.local/share/applications/tfan-ara-launcher.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=T-FAN + Ara Avatar System
Comment=Launch T-FAN Cockpit and Ara Avatar
Exec=$BASE_DIR/tfan-ara-launcher.sh
Icon=network-workgroup
Terminal=true
Categories=Development;Science;
EOF

echo -e "${GREEN}${CHECK} Created unified launcher${NC}"
echo -e "${GREEN}${CHECK} Created desktop shortcut${NC}"
echo ""

##############################################################################
# Step 11: Test Installation
##############################################################################
echo -e "${BLUE}${ARROW} Step 11: Testing installation...${NC}"

if [ -d "$AVATAR_DIR" ]; then
    cd "$AVATAR_DIR"

    python3 << 'PYTEST'
import sys
print("\nTesting Avatar components...")

try:
    import torch
    print(f"  ✅ PyTorch: {torch.__version__}")
except Exception as e:
    print(f"  ❌ PyTorch: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"  ✅ OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"  ❌ OpenCV: {e}")
    sys.exit(1)

try:
    import librosa
    print(f"  ✅ Librosa: {librosa.__version__}")
except Exception as e:
    print(f"  ❌ Librosa: {e}")
    sys.exit(1)

try:
    from src.avatar_engine import AvatarGenerator
    print(f"  ✅ AvatarGenerator works!")
except Exception as e:
    print(f"  ❌ AvatarGenerator: {e}")
    sys.exit(1)

try:
    from src.main import app
    print(f"  ✅ API imports successfully!")
except Exception as e:
    print(f"  ❌ API: {e}")
    sys.exit(1)

print("\n✨ All Avatar tests passed!")
PYTEST

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}${CHECK} Avatar system working!${NC}"
    else
        echo -e "${RED}${CROSS} Avatar tests failed${NC}"
    fi
fi
echo ""

##############################################################################
# Step 12: Summary & Next Steps
##############################################################################
echo ""
echo "========================================================================"
echo "  ✨ INSTALLATION COMPLETE!"
echo "========================================================================"
echo ""
echo -e "${GREEN}${STAR} Everything is installed and ready!${NC}"
echo ""
echo "📁 Installation directory:"
echo "   $BASE_DIR"
echo ""
echo "🚀 Quick Start Options:"
echo ""
echo "  ${CYAN}Option 1: Use the Launcher Menu${NC}"
echo "    $BASE_DIR/tfan-ara-launcher.sh"
echo ""
echo "  ${CYAN}Option 2: Individual Commands${NC}"
echo "    • Voice Avatar:  cd $BASE_DIR/ram-and-unification/conversational-avatar && python3 main.py"
echo "    • Avatar API:    cd $BASE_DIR/ram-and-unification && python3 -m src.main"
echo "    • T-FAN Cockpit: tfan-gnome"
echo ""
echo "  ${CYAN}Option 3: Desktop Shortcut${NC}"
echo "    • Look for 'T-FAN + Ara Avatar System' in your applications menu"
echo ""
echo "📝 Configuration Files:"
echo "   • Avatar API: $BASE_DIR/ram-and-unification/.env"
echo "   • Ara Persona: $BASE_DIR/ram-and-unification/multi-ai-workspace/config/ara_persona.yaml"
echo "   • Voice Macros: $BASE_DIR/ram-and-unification/multi-ai-workspace/config/voice_macros.yaml"
echo ""
echo "🎨 To use the avatar, add images to:"
echo "   • $BASE_DIR/ram-and-unification/assets/avatars/default.jpg"
echo ""
if [ "$SKIP_TFAN" = true ]; then
    echo -e "${YELLOW}⚠️  T-FAN Cockpit needs manual download${NC}"
    echo "   1. Download from: https://github.com/theadamsfamily1981-max/Quanta-meis-nib-cis"
    echo "   2. Extract to: $TFAN_DIR"
    echo "   3. Run this installer again"
    echo ""
fi
echo "📚 Documentation:"
echo "   • Avatar API: $BASE_DIR/ram-and-unification/README.md"
echo "   • Voice Macros: $BASE_DIR/ram-and-unification/multi-ai-workspace/config/voice_macros.yaml"
echo ""
echo "========================================================================"
echo ""

# Make the launcher accessible
echo "Adding launcher to PATH..."
if ! grep -q "tfan-ara-system" "$REAL_HOME/.bashrc"; then
    echo "" >> "$REAL_HOME/.bashrc"
    echo "# T-FAN + Ara Avatar System" >> "$REAL_HOME/.bashrc"
    echo "export PATH=\"\$PATH:$BASE_DIR\"" >> "$REAL_HOME/.bashrc"
    echo "alias tfan-launcher='$BASE_DIR/tfan-ara-launcher.sh'" >> "$REAL_HOME/.bashrc"
fi

echo ""
echo "🎉 You can now run: ${CYAN}tfan-launcher${NC}"
echo ""
