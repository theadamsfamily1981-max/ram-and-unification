# Operation SCYTHE

**Sci-Fi Cockpit Interface for Cathedral Avatar System**

---

## Overview

Operation SCYTHE is a tactical GNOME environment that integrates with the Cathedral Avatar HAA (Hybrid Accelerated Architecture) system. It provides a touch-optimized, sci-fi cockpit interface with synchronized state management across visual theming, network configuration, and avatar personality modes.

## Components

### 1. Mode Activation Scripts

**Location**: `mode-scripts/`

Three bash scripts that trigger synchronized state transitions:

- `activate_standby.sh` - Cathedral personality (100%), all radios ON
- `activate_flight.sh` - Cockpit personality (40%), Wi-Fi OFF (evasive signal mode)
- `activate_alert.sh` - Lab personality (50%), all radios ON, critical monitoring

**Installation**:
```bash
sudo cp mode-scripts/*.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/activate_*.sh
```

### 2. Macro Console (GTK4/LibAdwaita)

**Location**: `gtk-apps/macro-console/`

Touch-optimized mode switcher with large OSD-style buttons.

**Features**:
- Borderless, always-on-top window
- 3x mode buttons (Standby/Flight/Alert)
- Active mode highlighting
- D-Bus integration for mode feedback

**Installation**:
```bash
cp gtk-apps/macro-console/scythe_macro_console.py ~/.local/bin/
chmod +x ~/.local/bin/scythe_macro_console.py
```

**Autostart**:
```bash
cat > ~/.config/autostart/scythe-macro-console.desktop <<EOF
[Desktop Entry]
Type=Application
Name=SCYTHE Macro Console
Exec=~/.local/bin/scythe_macro_console.py
X-GNOME-Autostart-enabled=true
EOF
```

### 3. Ara Status Panel (GTK4/LibAdwaita)

**Location**: `gtk-apps/ara-status-panel/`

Real-time telemetry display for HAA system metrics.

**Features**:
- Borderless overlay panel
- D-Bus telemetry subscriber
- Conditional CSS styling (Normal/Warning/Critical)
- GNOME notification integration

**Installation**:
```bash
cp gtk-apps/ara-status-panel/ara_status_panel.py ~/.local/bin/
chmod +x ~/.local/bin/ara_status_panel.py
```

**Autostart**:
```bash
cat > ~/.config/autostart/ara-status-panel.desktop <<EOF
[Desktop Entry]
Type=Application
Name=Ara Status Panel
Exec=~/.local/bin/ara_status_panel.py
X-GNOME-Autostart-enabled=true
EOF
```

### 4. GNOME Shell Theme

**Location**: `gnome-shell-theme/` (to be created)

Custom CSS with glowing, transparent UI elements.

**Installation**: (After theme is created)
```bash
mkdir -p ~/.themes/Scythe-Theme/gnome-shell/
cp gnome-shell-theme/gnome-shell.css ~/.themes/Scythe-Theme/gnome-shell/
gsettings set org.gnome.shell.extensions.user-theme name 'Scythe-Theme'
```

### 5. Sound Themes

**Location**: `sounds/Scythe-{Calm,Flight,Alert}/`

Custom audio feedback for mode transitions.

**Structure**:
```
sounds/Scythe-Calm/
├── index.theme
└── stereo/
    ├── mode-activated.ogg
    ├── desktop-login.ogg
    └── dialog-error.ogg
```

**Installation**:
```bash
cp -r sounds/* ~/.local/share/sounds/
gsettings set org.gnome.desktop.sound theme-name 'Scythe-Calm'
gsettings set org.gnome.desktop.sound event-sounds true
```

### 6. Wallpapers

**Location**: `wallpapers/` (to be created)

Mode-specific backgrounds:
- `scythe_standby.jpg` - Calm, cathedral aesthetic
- `scythe_flight.jpg` - Tactical, navigation interface
- `scythe_alert.jpg` - Critical, red alert status

**Installation**:
```bash
sudo mkdir -p /usr/local/share/backgrounds/
sudo cp wallpapers/*.jpg /usr/local/share/backgrounds/
```

---

## Integration with HAA

SCYTHE integrates with the Cathedral Avatar HAA system via D-Bus:

### D-Bus Services

**Avatar Control** (`org.cathedral.AvatarHAA`):
- `SetPersonalityMode(mode)` - Switch avatar personality
- Signal: `ModeChanged(old, new)` - Personality change notification

**Ara Telemetry** (`org.cathedral.AraTelemetry`):
- Signal: `StatusUpdate(name, value, level, timestamp)` - System status
- Signal: `PerformanceMetrics(fpga_ms, gpu_ms, total_ms)` - Latency metrics

**SCYTHE Mode Control** (`org.scythe.ModeControl`):
- `ActivateMode(mode)` - Trigger mode transition
- Signal: `ModeActivated(mode, timestamp)` - Mode change confirmation

---

## Mode System

### Mode Mapping

| Mode | Avatar Personality | Intensity | Network | Visual Theme |
|------|-------------------|-----------|---------|--------------|
| **Standby** | Cathedral | 100% | All ON | Scythe-Calm |
| **Flight** | Cockpit | 40% | Wi-Fi OFF | Scythe-Flight |
| **Alert** | Lab | 50% | All ON | Scythe-Alert |

### Data Flow

```
Touch Input (Macro Console)
    ↓
Mode Script Execution
    ├─→ GNOME State (gsettings)
    ├─→ Network Toggle (rfkill/nmcli)
    ├─→ Avatar Personality (D-Bus)
    └─→ Audio Feedback (paplay)
    ↓
Synchronized Update
    ├─→ Wallpaper + Extensions
    ├─→ Sound Theme
    ├─→ HAA Rendering Parameters
    └─→ Ara Panel Configuration
```

---

## Prerequisites

### System Packages

```bash
# GTK4/LibAdwaita
sudo apt install \
    python3-gi \
    gir1.2-gtk-4.0 \
    gir1.2-adwaita-1 \
    libadwaita-1-0

# D-Bus Python bindings
pip install pydbus

# Window management
sudo apt install wmctrl

# Network control
sudo apt install network-manager rfkill

# Audio playback
sudo apt install pulseaudio-utils
```

### Optional

```bash
# Conky for advanced HUD overlays
sudo apt install conky-all

# D-Bus introspection tools
sudo apt install d-feet
```

---

## Quick Start

1. **Install mode scripts**:
```bash
sudo cp mode-scripts/*.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/activate_*.sh
```

2. **Install GTK apps**:
```bash
cp gtk-apps/macro-console/scythe_macro_console.py ~/.local/bin/
cp gtk-apps/ara-status-panel/ara_status_panel.py ~/.local/bin/
chmod +x ~/.local/bin/scythe_macro_console.py ~/.local/bin/ara_status_panel.py
```

3. **Test Macro Console**:
```bash
~/.local/bin/scythe_macro_console.py
```

4. **Test Ara Status Panel**:
```bash
~/.local/bin/ara_status_panel.py
```

5. **Test mode switching**:
```bash
/usr/local/bin/activate_flight.sh
```

---

## Troubleshooting

### Macro Console not Always-on-Top

**Cause**: `wmctrl` not installed

**Fix**:
```bash
sudo apt install wmctrl
```

### D-Bus signals not received

**Cause**: HAA orchestrator not running or not publishing D-Bus service

**Fix**:
```bash
# Check if service is published
gdbus introspect --session --dest org.cathedral.AvatarHAA \
    --object-path /org/cathedral/AvatarHAA/Control

# Start HAA orchestrator if needed
cd ../ara-haa-avatar/python
python orchestrator.py
```

### Mode scripts not found

**Cause**: Scripts not in `/usr/local/bin/` or not executable

**Fix**:
```bash
sudo cp mode-scripts/*.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/activate_*.sh
```

### No audio feedback

**Cause**: Sound files not installed or paplay not available

**Fix**:
```bash
# Install pulseaudio-utils
sudo apt install pulseaudio-utils

# Test audio playback
paplay /usr/share/sounds/freedesktop/stereo/bell.oga
```

---

## Development

### Testing D-Bus Integration

**Monitor D-Bus signals**:
```bash
dbus-monitor --session "interface='org.scythe.ModeControl'"
dbus-monitor --session "interface='org.cathedral.AraTelemetry.Status'"
```

**Manually emit test signal**:
```bash
gdbus emit --session \
    --object-path /org/scythe/ModeControl \
    --signal org.scythe.ModeControl.ModeActivated \
    "flight" $(date +%s)
```

### GTK Inspector

Debug GTK apps with live inspector:
```bash
# Press Ctrl+Shift+D while app is running
# Or set environment variable:
GTK_DEBUG=interactive ~/.local/bin/scythe_macro_console.py
```

---

## Architecture Documentation

See `docs/haa-scythe-integration.md` for detailed integration architecture, D-Bus interfaces, and deployment checklist.

---

**Built like a cathedral. Runs in your cathedral. Displayed in your cockpit.**
