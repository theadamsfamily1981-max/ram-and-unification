#!/bin/bash
# Operation SCYTHE: Flight Mode Activation
# Maps to Cockpit personality (40% intensity) for navigation/tactical ops

set -e

MODE_NAME="Flight"
WALLPAPER_PATH="/usr/local/share/backgrounds/scythe_flight.jpg"
SOUND_THEME="Scythe-Flight"
EXTENSIONS="['user-theme@gnome.org', 'window-overlay-hud@scythe']"
AVATAR_PERSONALITY="cockpit"

echo "[SCYTHE] Activating ${MODE_NAME} mode..."

# ============================================================================
# 1. GNOME Visual State
# ============================================================================

# Set wallpaper
gsettings set org.gnome.desktop.background picture-uri \
    "file://${WALLPAPER_PATH}"

gsettings set org.gnome.desktop.background picture-options 'scaled'

# Configure extensions (add Window Overlay HUD for flight instruments)
gsettings set org.gnome.shell enabled-extensions \
    "${EXTENSIONS}"

# Set sound theme
gsettings set org.gnome.desktop.sound theme-name "${SOUND_THEME}"
gsettings set org.gnome.desktop.sound event-sounds true

# ============================================================================
# 2. Network State (Evasive Signal Mode - Wi-Fi OFF)
# ============================================================================

# Block Wi-Fi (go dark)
rfkill block wifi

echo "[SCYTHE] Network: Wi-Fi blocked (Evasive Signal Mode)"

# Keep Bluetooth and other radios active for local communication
rfkill unblock bluetooth 2>/dev/null || true

# ============================================================================
# 3. Cathedral Avatar Personality Mode
# ============================================================================

# Trigger Cockpit mode (40% intensity - direct, efficient) via D-Bus
gdbus call --session \
    --dest org.cathedral.AvatarHAA \
    --object-path /org/cathedral/AvatarHAA/Control \
    --method org.cathedral.AvatarHAA.Control.SetPersonalityMode \
    "${AVATAR_PERSONALITY}" 2>/dev/null || echo "[SCYTHE] Warning: Avatar service not available"

echo "[SCYTHE] Avatar personality: ${AVATAR_PERSONALITY} (40% intensity)"

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
    "flight" $(date +%s) 2>/dev/null || true

# ============================================================================
# Complete
# ============================================================================

notify-send "Operation SCYTHE" "${MODE_NAME} mode activated" \
    --icon=airplane-mode \
    --urgency=normal \
    2>/dev/null || true

echo "[SCYTHE] ${MODE_NAME} mode activated at $(date)"
echo "[SCYTHE] Cockpit personality engaged - direct, efficient communication"
echo "[SCYTHE] Wi-Fi disabled - operating in evasive signal mode"
