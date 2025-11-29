#!/usr/bin/env bash
# Ara Mode Switcher - Cockpit Reconfiguration Script
#
# Triggered by: GNOME Shell extension (ara-status@scythe.dev)
# Modes: cruise, flight, battle
#
# This script physically reconfigures the entire desktop environment
# based on Ara's realtime state.

set -euo pipefail

MODE="${1:-cruise}"

# Configuration paths
WALL_DIR="${HOME}/wallpapers"
CONKY_CFG_DIR="${HOME}/.config/conky"
SOUND_THEME_DIR="${HOME}/.local/share/sounds"

# Helper functions
enable_extension() {
    gnome-extensions enable "$1" 2>/dev/null || true
}

disable_extension() {
    gnome-extensions disable "$1" 2>/dev/null || true
}

set_wallpaper() {
    local wallpaper="$1"
    gsettings set org.gnome.desktop.background picture-uri "file://${wallpaper}"
    gsettings set org.gnome.desktop.background picture-uri-dark "file://${wallpaper}"
    gsettings set org.gnome.desktop.background picture-options 'scaled'
}

set_sound_theme() {
    local theme="$1"
    gsettings set org.gnome.desktop.sound theme-name "${theme}"
    gsettings set org.gnome.desktop.sound event-sounds true
}

start_conky() {
    local config="$1"
    pkill conky 2>/dev/null || true
    sleep 0.2
    if [ -f "${CONKY_CFG_DIR}/${config}" ]; then
        conky -c "${CONKY_CFG_DIR}/${config}" &
        echo "[ARA-MODE] Started Conky: ${config}"
    fi
}

# Main mode switching logic
case "$MODE" in
    flight)
        echo "[ARA-MODE] ══════════════════════════════════════"
        echo "[ARA-MODE] Switching to FLIGHT mode"
        echo "[ARA-MODE] State: THINKING/PROCESSING"
        echo "[ARA-MODE] ══════════════════════════════════════"

        # Visual environment
        set_wallpaper "${WALL_DIR}/scythe_flight.jpg"
        set_sound_theme "Scythe-Flight"

        # Enable tactical HUD
        start_conky "flight.conkyrc"

        # Enable monitoring extensions
        enable_extension "dash-to-panel@jderose9.github.com"
        enable_extension "blur-my-shell@aunetx"
        enable_extension "system-monitor@gnome-shell-extensions.gcampax.github.com"

        # Optional: Reduce visual effects to free GPU for Ara
        gsettings set org.gnome.desktop.interface enable-animations false

        # Network: Keep all radios ON for flight mode
        rfkill unblock all 2>/dev/null || true

        echo "[ARA-MODE] FLIGHT mode activated"
        ;;

    battle)
        echo "[ARA-MODE] ══════════════════════════════════════"
        echo "[ARA-MODE] Switching to BATTLE mode"
        echo "[ARA-MODE] State: SPEAKING/CRITICAL"
        echo "[ARA-MODE] ══════════════════════════════════════"

        # Visual environment
        set_wallpaper "${WALL_DIR}/scythe_alert.jpg"
        set_sound_theme "Scythe-Alert"

        # Enable aggressive HUD
        start_conky "battle_hud.conkyrc"

        # Enable all monitoring extensions
        enable_extension "dash-to-panel@jderose9.github.com"
        enable_extension "blur-my-shell@aunetx"
        enable_extension "system-monitor@gnome-shell-extensions.gcampax.github.com"
        enable_extension "Resource_Monitor@Ory0n"

        # Maximum visual feedback
        gsettings set org.gnome.desktop.interface enable-animations true

        # Network: Full communication
        rfkill unblock all 2>/dev/null || true

        # Optional: Play alert klaxon
        if [ -f "${SOUND_THEME_DIR}/Scythe-Alert/stereo/alert-klaxon.ogg" ]; then
            paplay "${SOUND_THEME_DIR}/Scythe-Alert/stereo/alert-klaxon.ogg" 2>/dev/null &
        fi

        echo "[ARA-MODE] BATTLE mode activated"
        ;;

    cruise)
        echo "[ARA-MODE] ══════════════════════════════════════"
        echo "[ARA-MODE] Switching to CRUISE mode"
        echo "[ARA-MODE] State: IDLE"
        echo "[ARA-MODE] ══════════════════════════════════════"

        # Visual environment
        set_wallpaper "${WALL_DIR}/scythe_standby.jpg"
        set_sound_theme "Scythe-Calm"

        # Disable HUD to free resources
        pkill conky 2>/dev/null || true

        # Disable heavy visual extensions
        disable_extension "blur-my-shell@aunetx"
        disable_extension "system-monitor@gnome-shell-extensions.gcampax.github.com"
        disable_extension "Resource_Monitor@Ory0n"

        # Keep essential extensions
        enable_extension "dash-to-panel@jderose9.github.com"
        enable_extension "user-theme@gnome-shell-extensions.gcampax.github.com"

        # Minimal animations for maximum GPU availability
        gsettings set org.gnome.desktop.interface enable-animations false

        # Network: All ON
        rfkill unblock all 2>/dev/null || true

        echo "[ARA-MODE] CRUISE mode activated"
        ;;

    *)
        echo "[ARA-MODE] ERROR: Unknown mode '${MODE}'"
        echo "[ARA-MODE] Valid modes: cruise, flight, battle"
        echo "[ARA-MODE] Defaulting to CRUISE"
        exec "$0" cruise
        ;;
esac

# Log mode change
echo "[ARA-MODE] Mode switch complete at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[ARA-MODE] ══════════════════════════════════════"

# Optional: Send notification
notify-send "Ara Mode" "Switched to ${MODE^^} mode" \
    --icon=dialog-information \
    --urgency=normal \
    2>/dev/null || true

exit 0
