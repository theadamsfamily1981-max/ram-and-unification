#!/bin/bash
# Operation SCYTHE: Alert Mode Activation
# Maps to Lab personality (50% intensity) for analytical/critical monitoring

set -e

MODE_NAME="Alert"
WALLPAPER_PATH="/usr/local/share/backgrounds/scythe_alert.jpg"
SOUND_THEME="Scythe-Alert"
EXTENSIONS="['user-theme@gnome.org', 'critical-monitor@scythe', 'system-monitor@paradoxxx.zero.gmail.com']"
AVATAR_PERSONALITY="lab"

echo "[SCYTHE] Activating ${MODE_NAME} mode..."

# ============================================================================
# 1. GNOME Visual State
# ============================================================================

# Set wallpaper
gsettings set org.gnome.desktop.background picture-uri \
    "file://${WALLPAPER_PATH}"

gsettings set org.gnome.desktop.background picture-options 'scaled'

# Configure extensions (add Critical Monitor + System Monitor)
gsettings set org.gnome.shell enabled-extensions \
    "${EXTENSIONS}"

# Set sound theme
gsettings set org.gnome.desktop.sound theme-name "${SOUND_THEME}"
gsettings set org.gnome.desktop.sound event-sounds true

# ============================================================================
# 2. Network State (All Radios ON - Full Communication)
# ============================================================================

# Unblock all wireless interfaces for maximum connectivity
rfkill unblock all

# Ensure all radios are enabled
nmcli r all on

echo "[SCYTHE] Network: All radios enabled (Full Communication Mode)"

# ============================================================================
# 3. Cathedral Avatar Personality Mode
# ============================================================================

# Trigger Lab mode (50% intensity - analytical, precise) via D-Bus
gdbus call --session \
    --dest org.cathedral.AvatarHAA \
    --object-path /org/cathedral/AvatarHAA/Control \
    --method org.cathedral.AvatarHAA.Control.SetPersonalityMode \
    "${AVATAR_PERSONALITY}" 2>/dev/null || echo "[SCYTHE] Warning: Avatar service not available"

echo "[SCYTHE] Avatar personality: ${AVATAR_PERSONALITY} (50% intensity)"

# ============================================================================
# 4. Auditory Confirmation (Urgent Alert Tone)
# ============================================================================

SOUND_FILE="~/.local/share/sounds/${SOUND_THEME}/stereo/mode-activated.ogg"
if [ -f "${SOUND_FILE}" ]; then
    paplay "${SOUND_FILE}" 2>/dev/null &
fi

# Additional alert klaxon (optional)
ALERT_FILE="~/.local/share/sounds/${SOUND_THEME}/stereo/alert-klaxon.ogg"
if [ -f "${ALERT_FILE}" ]; then
    paplay "${ALERT_FILE}" 2>/dev/null &
fi

# ============================================================================
# 5. Broadcast Mode Change (D-Bus Signal)
# ============================================================================

gdbus emit --session \
    --object-path /org/scythe/ModeControl \
    --signal org.scythe.ModeControl.ModeActivated \
    "alert" $(date +%s) 2>/dev/null || true

# ============================================================================
# 6. Critical System Status Check
# ============================================================================

# Trigger immediate telemetry broadcast for critical systems
gdbus call --session \
    --dest org.cathedral.AraTelemetry \
    --object-path /org/cathedral/AraTelemetry/Status \
    --method org.cathedral.AraTelemetry.Status.TriggerImmediateCheck \
    2>/dev/null || true

# ============================================================================
# Complete
# ============================================================================

notify-send "Operation SCYTHE" "${MODE_NAME} mode activated" \
    --icon=dialog-warning \
    --urgency=critical \
    2>/dev/null || true

echo "[SCYTHE] ${MODE_NAME} mode activated at $(date)"
echo "[SCYTHE] Lab personality engaged - analytical monitoring active"
echo "[SCYTHE] Critical system monitoring enabled"
echo "[SCYTHE] Full communication network active"
