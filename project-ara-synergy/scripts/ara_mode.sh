#!/usr/bin/env bash
#
# Ara Mode Switcher
#
# Synchronizes GNOME environment with Cathedral Avatar state:
# - CRUISE: IDLE state, minimal HUD, calm aesthetics
# - FLIGHT: THINKING/PROCESSING state, tactical monitoring
# - BATTLE: SPEAKING/CRITICAL state, maximum awareness
#
# Called by GNOME Shell extension (ara-status@scythe.dev)
# based on D-Bus telemetry signals

set -euo pipefail

MODE="${1:-cruise}"

WALL_DIR="$HOME/wallpapers"
CONKY_CFG_DIR="$HOME/.config/conky"

enable_extension() {
  gnome-extensions enable "$1" 2>/dev/null || true
}

disable_extension() {
  gnome-extensions disable "$1" 2>/dev/null || true
}

case "$MODE" in
  flight)
    echo "[ARA-MODE] FLIGHT"
    gsettings set org.gnome.desktop.background picture-uri "file://$WALL_DIR/flight_tactical.png"
    gsettings set org.gnome.desktop.background picture-uri-dark "file://$WALL_DIR/flight_tactical.png"

    # Start tactical HUD
    pkill conky || true
    conky -c "$CONKY_CFG_DIR/flight.conkyrc" &

    # Enable monitoring extensions
    enable_extension "dash-to-panel@jderose9.github.com"
    enable_extension "blur-my-shell@aunetx"
    ;;

  battle)
    echo "[ARA-MODE] BATTLE"
    gsettings set org.gnome.desktop.background picture-uri "file://$WALL_DIR/battle_alert.png"
    gsettings set org.gnome.desktop.background picture-uri-dark "file://$WALL_DIR/battle_alert.png"

    # Start battle HUD
    pkill conky || true
    conky -c "$CONKY_CFG_DIR/battle_hud.conkyrc" &

    # Enable all monitoring
    enable_extension "dash-to-panel@jderose9.github.com"
    enable_extension "blur-my-shell@aunetx"

    # Switch to alert sound theme
    gsettings set org.gnome.desktop.sound theme-name "Scythe-Alert"
    ;;

  cruise)
    echo "[ARA-MODE] CRUISE"
    gsettings set org.gnome.desktop.background picture-uri "file://$WALL_DIR/cruise_nebula.png"
    gsettings set org.gnome.desktop.background picture-uri-dark "file://$WALL_DIR/cruise_nebula.png"

    # Disable HUD for minimal overhead
    pkill conky || true

    # Disable heavy extensions to free GPU
    disable_extension "blur-my-shell@aunetx"

    # Switch to calm sound theme
    gsettings set org.gnome.desktop.sound theme-name "Scythe-Calm"
    ;;

  *)
    echo "[ARA-MODE] unknown '$MODE', defaulting to CRUISE"
    exec "$0" cruise
    ;;
esac
