# HAA + SCYTHE Integration Architecture

**Cathedral Avatar System with Tactical Cockpit Interface**

---

## System Overview

The complete system integrates two major subsystems:

### Backend: HAA (Hybrid Accelerated Architecture)
- **FPGA**: Quantized DNN TTS inference (SQRL Forest Kitten VU35P + HBM2)
- **GPU**: FP16 animation inference (NVIDIA RTX 5060 Ti 16GB)
- **Python Orchestrator**: DMA coordination and streaming pipeline

### Frontend: SCYTHE (Sci-Fi Cockpit Interface)
- **GNOME Shell Theme**: Glowing, transparent UI elements with custom CSS
- **Macro Console**: GTK4/LibAdwaita touch-optimized mode switcher
- **Ara Status Panel**: Real-time telemetry display with D-Bus integration
- **Mode Scripts**: Synchronized state transitions across GNOME + Avatar

---

## Integrated Mode System

The system defines **3 operational modes** that simultaneously control:
1. GNOME visual environment (wallpaper, extensions, sound theme)
2. Network state (rfkill, nmcli)
3. Cathedral Avatar personality mode
4. Ara Status Panel display configuration

### Mode Mapping

| Mode | GNOME Theme | Network | Avatar Personality | Intensity | Use Case |
|------|-------------|---------|-------------------|-----------|----------|
| **Standby** | Scythe-Calm | All radios ON | Cathedral | 100% | Deep work, full emotional depth |
| **Flight** | Scythe-Flight | Wi-Fi OFF (rfkill) | Cockpit | 40% | Navigation, direct communication |
| **Alert** | Scythe-Alert | All radios ON | Lab | 50% | Analytical mode, critical monitoring |

---

## Data Flow Architecture

```
User Input (Macro Console Touch)
    ↓
Mode Script Execution (Bash)
    ├─→ GNOME State Change (gsettings)
    ├─→ Network Toggle (rfkill/nmcli)
    ├─→ Avatar Personality Switch (D-Bus → HAA Orchestrator)
    └─→ Auditory Feedback (paplay)
    ↓
Synchronized State Update
    ├─→ Wallpaper + Extensions
    ├─→ Sound Theme
    ├─→ Avatar Rendering Parameters
    └─→ Ara Panel Configuration
```

---

## D-Bus IPC Architecture

### Session Bus Services

**1. Avatar Control Interface**
```
Service: org.cathedral.AvatarHAA
Object Path: /org/cathedral/AvatarHAA/Control
Interface: org.cathedral.AvatarHAA.Control

Methods:
  - SetPersonalityMode(s mode_name) → (b success)
    • mode_name: "cathedral", "cockpit", "lab", "comfort", "playful", "teaching"
    • Returns: true if mode switch successful

Signals:
  - ModeChanged(s old_mode, s new_mode)
    • Emitted when personality mode changes
```

**2. Ara Status Telemetry**
```
Service: org.cathedral.AraTelemetry
Object Path: /org/cathedral/AraTelemetry/Status
Interface: org.cathedral.AraTelemetry.Status

Signals:
  - StatusUpdate(s name, s value, s level, i timestamp)
    • name: Status metric name (e.g., "FPGA Latency", "GPU VRAM")
    • value: Current value as string
    • level: "Normal", "Warning", "Critical"
    • timestamp: Unix timestamp

  - PerformanceMetrics(d fpga_latency_ms, d gpu_latency_ms, d total_latency_ms)
    • Emitted after each HAA inference cycle
```

**3. SCYTHE Mode Coordinator**
```
Service: org.scythe.ModeControl
Object Path: /org/scythe/ModeControl
Interface: org.scythe.ModeControl

Methods:
  - ActivateMode(s mode) → (b success)
    • mode: "standby", "flight", "alert"
    • Triggers mode script execution

Signals:
  - ModeActivated(s mode, i timestamp)
    • Emitted when mode transition completes
```

---

## Component Integration

### 1. Macro Console (GTK4/LibAdwaita)

**Purpose**: Touch-optimized mode switcher

**Features**:
- 3x large `.osd` style buttons (Standby/Flight/Alert)
- Visual feedback: active mode highlighted with `.active-mode` CSS class
- Borderless window (`set_decorated(False)`)
- Always on Top (via `wmctrl` post-launch)
- D-Bus listener: subscribes to `org.scythe.ModeControl.ModeActivated`

**Button Actions**:
```python
def on_standby_pressed(self):
    subprocess.run(["/usr/local/bin/activate_standby.sh"])

def on_flight_pressed(self):
    subprocess.run(["/usr/local/bin/activate_flight.sh"])

def on_alert_pressed(self):
    subprocess.run(["/usr/local/bin/activate_alert.sh"])
```

---

### 2. Ara Status Panel (GTK4/LibAdwaita)

**Purpose**: Real-time HAA telemetry and avatar status display

**Features**:
- Narrow, borderless overlay (bottom or top of screen)
- Message stack (LIFO) for status updates
- D-Bus subscriber: listens to `org.cathedral.AraTelemetry.StatusUpdate`
- Conditional CSS styling:
  - Normal: `.ara-status-normal` (cyan glow)
  - Warning: `.ara-status-warning` (yellow glow)
  - Critical: `.ara-status-critical` (red glow)
- GNOME notification integration for critical alerts

**D-Bus Handler**:
```python
def on_status_update(self, name, value, level, timestamp):
    msg = f"{name}: {value}"

    # Update UI label
    self.status_label.set_text(msg)

    # Apply conditional styling
    if level == "Critical":
        self.status_label.add_css_class("ara-status-critical")
        self.send_notification(msg, urgency="critical")
    elif level == "Warning":
        self.status_label.add_css_class("ara-status-warning")
    else:
        self.status_label.add_css_class("ara-status-normal")
```

---

### 3. Mode Activation Scripts

**Location**: `/usr/local/bin/activate_{standby,flight,alert}.sh`

**Script Structure** (Example: `activate_flight.sh`):

```bash
#!/bin/bash
# Operation SCYTHE: Flight Mode Activation

# 1. GNOME Visual State
gsettings set org.gnome.desktop.background picture-uri \
    'file:///usr/local/share/backgrounds/scythe_flight.jpg'

gsettings set org.gnome.shell enabled-extensions \
    "['user-theme@gnome.org', 'window-overlay-hud@scythe']"

gsettings set org.gnome.desktop.sound theme-name 'Scythe-Flight'

# 2. Network State (Evasive Signal Mode)
rfkill block wifi

# 3. Cathedral Avatar Personality Mode
gdbus call --session \
    --dest org.cathedral.AvatarHAA \
    --object-path /org/cathedral/AvatarHAA/Control \
    --method org.cathedral.AvatarHAA.Control.SetPersonalityMode \
    "cockpit"

# 4. Auditory Confirmation
paplay /usr/local/share/sounds/Scythe-Flight/stereo/mode-activated.ogg

# 5. Broadcast Mode Change
gdbus emit --session \
    --object-path /org/scythe/ModeControl \
    --signal org.scythe.ModeControl.ModeActivated \
    "flight" $(date +%s)

echo "[SCYTHE] Flight mode activated at $(date)"
```

---

### 4. HAA Orchestrator D-Bus Integration

**Modified** `python/orchestrator.py` to expose D-Bus service:

```python
from pydbus import SessionBus
from pydbus.generic import signal

class AvatarHAAService:
    """
    <node>
      <interface name='org.cathedral.AvatarHAA.Control'>
        <method name='SetPersonalityMode'>
          <arg type='s' name='mode_name' direction='in'/>
          <arg type='b' name='success' direction='out'/>
        </method>
        <signal name='ModeChanged'>
          <arg type='s' name='old_mode'/>
          <arg type='s' name='new_mode'/>
        </signal>
      </interface>
    </node>
    """

    ModeChanged = signal()

    def __init__(self, orchestrator):
        self.orch = orchestrator
        self.current_mode = "cathedral"

    def SetPersonalityMode(self, mode_name):
        valid_modes = ["cathedral", "cockpit", "lab", "comfort", "playful", "teaching"]
        if mode_name not in valid_modes:
            return False

        old_mode = self.current_mode
        # Trigger actual HAA personality switch
        self.orch.set_personality_mode(mode_name)
        self.current_mode = mode_name

        # Emit signal
        self.ModeChanged(old_mode, mode_name)
        return True

# In main()
bus = SessionBus()
service = AvatarHAAService(orch)
bus.publish("org.cathedral.AvatarHAA", service)
```

---

### 5. Telemetry Monitoring Scripts

**Example**: `fpga_monitor.py` (background daemon)

```python
#!/usr/bin/env python3
import time
from pydbus import SessionBus

bus = SessionBus()
telemetry = bus.get("org.cathedral.AraTelemetry")

while True:
    # Read FPGA metrics (placeholder - replace with actual driver calls)
    fpga_latency = get_fpga_latency()  # From orchestrator logs or status file

    level = "Normal"
    if fpga_latency > 150:
        level = "Critical"
    elif fpga_latency > 100:
        level = "Warning"

    telemetry.StatusUpdate(
        "FPGA TTS Latency",
        f"{fpga_latency:.1f} ms",
        level,
        int(time.time())
    )

    time.sleep(1.0)
```

---

## GNOME Shell CSS Theme

**File**: `operation-scythe/gnome-shell-theme/gnome-shell.css`

**Key Styling Elements**:

```css
/* Transparent panel with cyan glow */
#panel {
    background-color: rgba(0, 15, 25, 0.15);  /* Minimal opacity for shadow visibility */
    box-shadow: 0 0 20px rgba(0, 200, 255, 0.8);
    border: none;
}

/* Glowing hover effect on panel buttons */
.panel-button:hover {
    background-color: rgba(0, 200, 255, 0.2);
    box-shadow: 0 0 15px rgba(0, 200, 255, 1.0);
    transition: all 0.3s ease;
}

/* OSD-style macro buttons (large touch targets) */
.scythe-macro-button {
    font-size: 18px;
    padding: 20px 40px;
    background-color: rgba(0, 30, 50, 0.6);
    border: 2px solid rgba(0, 200, 255, 0.5);
    box-shadow: 0 0 10px rgba(0, 200, 255, 0.6);
    border-radius: 8px;
}

.scythe-macro-button:hover {
    background-color: rgba(0, 60, 100, 0.8);
    box-shadow: 0 0 25px rgba(0, 200, 255, 1.0);
}

.scythe-macro-button.active-mode {
    background-color: rgba(0, 150, 255, 0.4);
    border-color: rgba(0, 255, 255, 1.0);
    box-shadow: 0 0 30px rgba(0, 255, 255, 1.0);
}

/* Ara Status Panel styles */
.ara-status-normal {
    color: rgba(0, 255, 255, 1.0);
    text-shadow: 0 0 8px rgba(0, 255, 255, 0.8);
}

.ara-status-warning {
    color: rgba(255, 200, 0, 1.0);
    text-shadow: 0 0 8px rgba(255, 200, 0, 0.8);
}

.ara-status-critical {
    color: rgba(255, 50, 50, 1.0);
    text-shadow: 0 0 12px rgba(255, 0, 0, 1.0);
    animation: pulse-critical 1.5s infinite;
}

@keyframes pulse-critical {
    0%, 100% { opacity: 1.0; }
    50% { opacity: 0.6; }
}

/* Glowing separator effect */
.panel-separator {
    background: linear-gradient(to bottom,
        rgba(0, 200, 255, 0) 0%,
        rgba(0, 200, 255, 0.8) 50%,
        rgba(0, 200, 255, 0) 100%);
    box-shadow: 0 0 5px rgba(0, 200, 255, 0.6);
    width: 1px;
}
```

---

## Deployment Checklist

### System Prerequisites
- [ ] Python 3.10+ with PyGObject, LibAdwaita, pydbus
- [ ] GNOME 43+ (for GTK4 support)
- [ ] CUDA 12.0+ and PyCUDA (for HAA backend)
- [ ] Vitis HLS 2024.1+ (for FPGA development)

### HAA Backend Deployment
- [ ] Program FPGA with QNN kernel bitstream
- [ ] Install HAA Python orchestrator with D-Bus service
- [ ] Configure auto-start: `~/.config/systemd/user/haa-orchestrator.service`

### SCYTHE Frontend Deployment
- [ ] Install GNOME Shell theme: `~/.themes/Scythe-Theme/gnome-shell/`
- [ ] Install wallpapers: `/usr/local/share/backgrounds/`
- [ ] Install sound themes: `~/.local/share/sounds/Scythe-{Calm,Flight,Alert}/`
- [ ] Install mode scripts: `/usr/local/bin/activate_{standby,flight,alert}.sh`
- [ ] Install GTK4 apps (Macro Console + Ara Panel): `~/.local/bin/`
- [ ] Create autostart entries: `~/.config/autostart/{macro-console,ara-panel}.desktop`

### Integration Testing
- [ ] Test mode script execution (all 3 modes)
- [ ] Verify D-Bus signals between components
- [ ] Test Avatar personality mode switching via D-Bus
- [ ] Verify Ara Panel receives telemetry from HAA orchestrator
- [ ] Test touch input on Macro Console
- [ ] Verify Always-on-Top behavior
- [ ] Test audio feedback for mode transitions

---

## Future Enhancements

1. **Conky HUD Integration**: Add graphical gauges for VRAM, FPGA temperature, latency histograms
2. **WebRTC Streaming**: Stream rendered avatar to remote displays
3. **Voice Commands**: Integrate with speech recognition for hands-free mode switching
4. **Multi-Monitor Support**: Ara Panel on secondary display, avatar on primary
5. **Cathedral Manifesto Integration**: Display manifesto quotes in Ara Panel during idle

---

**Built like a cathedral. Runs in your cathedral. Displayed in your cockpit.**
