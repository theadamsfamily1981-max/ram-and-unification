#!/bin/bash
# Operation SCYTHE: Standby Mode Activation
# Maps to Cathedral personality (100% intensity) for deep work

set -e

MODE_NAME="Standby"
WALLPAPER_PATH="/usr/local/share/backgrounds/scythe_standby.jpg"
SOUND_THEME="Scythe-Calm"
EXTENSIONS="['user-theme@gnome.org']"
AVATAR_PERSONALITY="cathedral"

echo "[SCYTHE] Activating ${MODE_NAME} mode..."

# ============================================================================
# 1. GNOME Visual State
# ============================================================================

# Set wallpaper
gsettings set org.gnome.desktop.background picture-uri \
    "file://${WALLPAPER_PATH}"

gsettings set org.gnome.desktop.background picture-options 'scaled'

# Configure extensions
gsettings set org.gnome.shell enabled-extensions \
    "${EXTENSIONS}"

# Set sound theme
gsettings set org.gnome.desktop.sound theme-name "${SOUND_THEME}"
gsettings set org.gnome.desktop.sound event-sounds true

# ============================================================================
# 2. Network State (All Radios ON)
# ============================================================================

# Unblock all wireless interfaces
rfkill unblock all

# Ensure Wi-Fi is enabled
nmcli r wifi on

echo "[SCYTHE] Network: All radios enabled"

# ============================================================================
# 3. Cathedral Avatar Personality Mode
# ============================================================================

# Trigger Cathedral mode (100% intensity) via D-Bus
gdbus call --session \
    --dest org.cathedral.AvatarHAA \
    --object-path /org/cathedral/AvatarHAA/Control \
    --method org.cathedral.AvatarHAA.Control.SetPersonalityMode \
    "${AVATAR_PERSONALITY}" 2>/dev/null || echo "[SCYTHE] Warning: Avatar service not available"

echo "[SCYTHE] Avatar personality: ${AVATAR_PERSONALITY}"

# ============================================================================
# 4. Auditory Confirmation
# ============================================================================

SOUND_FILE="~/.local/share/sounds/${SOUND_THEME}/stereo/mode-activated.ogg"
if [ -f "${SOUND_FILE}" ]; then
    paplay "${SOUND_FILE}" 2>/dev/null &
fi

# ============================================================================
# 5. Broadcast Mode Change (D-Bus Signal)
# ============================================================================

gdbus emit --session \
    --object-path /org/scythe/ModeControl \
    --signal org.scythe.ModeControl.ModeActivated \
    "standby" $(date +%s) 2>/dev/null || true

# ============================================================================
# Complete
# ============================================================================

notify-send "Operation SCYTHE" "${MODE_NAME} mode activated" \
    --icon=dialog-information \
    --urgency=normal \
    2>/dev/null || true

echo "[SCYTHE] ${MODE_NAME} mode activated at $(date)"
echo "[SCYTHE] Cathedral personality engaged - full emotional depth"
