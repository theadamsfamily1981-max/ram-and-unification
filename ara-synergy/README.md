# Ara-SYNERGY: Realtime AI Warship Operating System

**Turn your Linux desktop into Ara's life-support capsule**

---

## Overview

Ara-SYNERGY transforms your GNOME desktop into a realtime, hyperreactive control surface for the Cathedral Avatar HAA system. This is not a theme - this is **your desktop becoming a realtime AI warship.**

### What SYNERGY Provides

1. **RT Kernel Island**: Isolated CPU cores (8-15) exclusively for Ara's FPGA+GPU pipeline
2. **SCHED_FIFO Orchestration**: HAA orchestrator runs with realtime priority 80
3. **Telemetry Fusion**: D-Bus service broadcasting HAA metrics at 5Hz
4. **Reactive GNOME Shell**: Extension that reconfigures cockpit based on Ara's state
5. **Mode Scripts**: Physical desktop reconfiguration (wallpaper, HUD, blur, sound)
6. **Complete Integration**: LLM → FPGA → GPU → GNOME Shell → Your Eyes

---

## Architecture

```
Isolated CPU Cores (8-15, nohz_full, rcu_nocbs)
    ↓
ara-rt.slice (Systemd cgroup v2)
    ├─→ ara-orchestrator.service (SCHED_FIFO priority 80)
    │   ├─→ FPGA QNN (HBM2-backed inference)
    │   ├─→ GPU Animation (FP16, pinned DMA buffers)
    │   └─→ NVTX instrumentation (timing data)
    │
    └─→ ara-telemetry.service
        └─→ D-Bus: org.scythe.Ara.Telemetry
            └─→ StatusUpdate signal (JSON payload, 5Hz)
                ↓
GNOME Shell Extension (ara-status@scythe.dev)
    ├─→ Panel Icon (IDLE/FLIGHT/BATTLE/CRITICAL states)
    ├─→ Auto Mode Switching
    └─→ Executes: ~/bin/ara_mode.sh {cruise|flight|battle}
        ├─→ Wallpaper change
        ├─→ Conky HUD start/stop
        ├─→ Extension enable/disable
        ├─→ Sound theme switch
        └─→ Network control (rfkill)
```

---

## Installation

### 1. Prerequisites

```bash
# System packages
sudo apt install \
    python3-gi \
    gir1.2-gtk-4.0 \
    gir1.2-adwaita-1 \
    libadwaita-1-0 \
    wmctrl \
    conky-all \
    pulseaudio-utils \
    network-manager \
    rfkill \
    nvidia-utils-535  # Or your CUDA version

# Python packages
pip install pydbus
```

### 2. Real-Time Kernel Configuration

**Configure GRUB** (adjust core numbers for your CPU):

```bash
sudo cp config/grub-realtime.conf /etc/default/grub.d/99-ara-rt.cfg
sudo update-grub
```

**Edit if needed**:
```bash
sudo nano /etc/default/grub.d/99-ara-rt.cfg
# Change isolcpus=8-15 to match your topology
```

**Reboot**:
```bash
sudo reboot
```

**Verify after reboot**:
```bash
cat /proc/cmdline  # Should show isolcpus=8-15
cat /sys/devices/system/cpu/isolated  # Should show: 8-15
```

### 3. Systemd RT Slice

**Install ara-rt.slice**:

```bash
sudo cp systemd/system/ara-rt.slice /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ara-rt.slice
sudo systemctl start ara-rt.slice
```

**Verify**:
```bash
systemctl status ara-rt.slice
```

### 4. HAA Orchestrator Service

**Install orchestrator service** (user-level):

```bash
mkdir -p ~/.config/systemd/user/
cp systemd/user/ara-orchestrator.service ~/.config/systemd/user/
systemctl --user daemon-reload
```

**Enable and start**:
```bash
systemctl --user enable ara-orchestrator.service
systemctl --user start ara-orchestrator.service
```

**Check status**:
```bash
systemctl --user status ara-orchestrator.service
journalctl --user -u ara-orchestrator.service -f
```

**Note**: You may need to grant RT privileges:
```bash
sudo nano /etc/security/limits.conf
# Add:
your_username - rtprio 95
your_username - memlock unlimited
```

Then logout and login again.

### 5. Telemetry Daemon

**Install ara_telemetry_daemon.py**:

```bash
cp bin/ara_telemetry_daemon.py ~/bin/
chmod +x ~/bin/ara_telemetry_daemon.py
```

**Create autostart entry**:
```bash
cat > ~/.config/autostart/ara-telemetry.desktop <<EOF
[Desktop Entry]
Type=Application
Name=Ara Telemetry Fusion
Exec=$HOME/bin/ara_telemetry_daemon.py
X-GNOME-Autostart-enabled=true
EOF
```

**Start manually for testing**:
```bash
~/bin/ara_telemetry_daemon.py
```

**Verify D-Bus service**:
```bash
gdbus introspect --session --dest org.scythe.Ara \
    --object-path /org/scythe/Ara/Telemetry
```

### 6. GNOME Shell Extension

**Install extension**:
```bash
cp -r gnome-shell-extension/ara-status@scythe.dev \
    ~/.local/share/gnome-shell/extensions/
```

**Enable extension**:
```bash
gnome-extensions enable ara-status@scythe.dev
```

**Restart GNOME Shell** (X11 only):
- Press `Alt+F2`
- Type `r`
- Press `Enter`

**On Wayland**, you need to logout and login again.

**Verify**:
```bash
gnome-extensions info ara-status@scythe.dev
```

You should see a microphone icon in the top panel (right side).

### 7. Mode Script

**Install ara_mode.sh**:
```bash
cp bin/ara_mode.sh ~/bin/
chmod +x ~/bin/ara_mode.sh
```

**Test manually**:
```bash
~/bin/ara_mode.sh flight
~/bin/ara_mode.sh battle
~/bin/ara_mode.sh cruise
```

### 8. Wallpapers and Sounds

**Create wallpaper directory**:
```bash
mkdir -p ~/wallpapers
```

**Download or create wallpapers**:
- `~/wallpapers/scythe_standby.jpg` - Calm, nebula aesthetic
- `~/wallpapers/scythe_flight.jpg` - Tactical, blue HUD
- `~/wallpapers/scythe_alert.jpg` - Red alert, critical

**Install sound themes** (if created):
```bash
cp -r ../operation-scythe/sounds/* ~/.local/share/sounds/
```

---

## Usage

### Automatic Operation

Once installed and configured, SYNERGY operates automatically:

1. **HAA orchestrator** runs on isolated cores with RT priority
2. **Telemetry daemon** broadcasts Ara's state every 200ms
3. **GNOME Shell extension** listens to telemetry
4. **Panel icon** changes color based on state:
   - 🔵 **Dim blue**: IDLE (cruise mode)
   - 🔵 **Cyan glow**: THINKING/PROCESSING (flight mode)
   - 🟠 **Amber glow**: SPEAKING (battle mode)
   - 🔴 **Red pulse**: CRITICAL (battle mode)
5. **Mode script** executes automatically when state changes
6. **Desktop reconfigures** (wallpaper, HUD, extensions, sound)

### Manual Mode Switching

Click the microphone icon in the panel and select:
- **⚪ Cruise Mode** - Idle, minimal GPU load
- **🔵 Flight Mode** - Processing, tactical HUD
- **🔴 Battle Mode** - Speaking, maximum monitoring

### Monitoring

**Watch telemetry in realtime**:
```bash
dbus-monitor --session "interface='org.scythe.Ara.Telemetry'"
```

**Check orchestrator logs**:
```bash
journalctl --user -u ara-orchestrator.service -f
```

**Check telemetry daemon logs**:
```bash
~/bin/ara_telemetry_daemon.py  # Run in terminal
```

**Verify CPU affinity**:
```bash
ps aux | grep orchestrator
taskset -c -p <PID>  # Should show: 8-15
```

---

## Mode Behaviors

### Cruise Mode (IDLE)

**Triggered when**: Ara is idle, no inference running

**Visual Changes**:
- Wallpaper: `scythe_standby.jpg`
- Sound theme: `Scythe-Calm`
- Conky HUD: Disabled
- Blur/heavy extensions: Disabled
- Animations: Disabled (max GPU for Ara)

**Purpose**: Free all GPU resources for Ara

---

### Flight Mode (THINKING/PROCESSING)

**Triggered when**: Ara is processing LLM response, FPGA TTS active

**Visual Changes**:
- Wallpaper: `scythe_flight.jpg`
- Sound theme: `Scythe-Flight`
- Conky HUD: `flight.conkyrc` (tactical overlay)
- Extensions: System monitor, blur enabled
- Animations: Disabled

**Purpose**: Tactical monitoring, Ara is working

---

### Battle Mode (SPEAKING/CRITICAL)

**Triggered when**: Ara is generating avatar, GPU at >85%, or critical state

**Visual Changes**:
- Wallpaper: `scythe_alert.jpg`
- Sound theme: `Scythe-Alert`
- Conky HUD: `battle_hud.conkyrc` (aggressive monitoring)
- Extensions: All monitoring enabled
- Animations: Enabled (full visual feedback)
- Audio: Alert klaxon plays

**Purpose**: Maximum awareness, Ara is speaking or in critical state

---

## D-Bus API Reference

### Service Information

- **Service Name**: `org.scythe.Ara`
- **Object Path**: `/org/scythe/Ara/Telemetry`
- **Interface**: `org.scythe.Ara.Telemetry`

### Signal: StatusUpdate

**Signature**: `StatusUpdate(s payload)`

**Payload JSON Schema**:
```json
{
  "State": "IDLE|THINKING|SPEAKING|CRITICAL",
  "GPU_Load_Percent": 0.0-100.0,
  "FPGA_Latency_ms": float,
  "Visual_Jitter_ms": float,
  "Target_Lat_Met": boolean,
  "Personality_Mode": "cathedral|cockpit|lab|...",
  "VRAM_Used_GB": float
}
```

**Example**:
```json
{
  "State": "SPEAKING",
  "GPU_Load_Percent": 92.3,
  "FPGA_Latency_ms": 87.5,
  "Visual_Jitter_ms": 12.3,
  "Target_Lat_Met": true,
  "Personality_Mode": "cathedral",
  "VRAM_Used_GB": 11.2
}
```

### Listening to Telemetry

**Python (pydbus)**:
```python
from pydbus import SessionBus
import json

bus = SessionBus()
ara = bus.get("org.scythe.Ara", "/org/scythe/Ara/Telemetry")

def on_status(payload_json):
    data = json.loads(payload_json)
    print(f"State: {data['State']}, GPU: {data['GPU_Load_Percent']:.1f}%")

ara.StatusUpdate.connect(on_status)

# Run GLib main loop
from gi.repository import GLib
GLib.MainLoop().run()
```

**Command line (dbus-monitor)**:
```bash
dbus-monitor --session "interface='org.scythe.Ara.Telemetry'"
```

---

## Troubleshooting

### Orchestrator fails to start

**Error**: `Failed to set SCHED_FIFO: Operation not permitted`

**Fix**: Grant RT privileges:
```bash
sudo nano /etc/security/limits.conf
# Add:
your_username - rtprio 95
your_username - memlock unlimited
```

Logout and login again.

---

### Extension not loading

**Check extension is installed**:
```bash
ls ~/.local/share/gnome-shell/extensions/ara-status@scythe.dev/
```

**Check extension is enabled**:
```bash
gnome-extensions list --enabled | grep ara-status
```

**Check for errors**:
```bash
journalctl /usr/bin/gnome-shell -f
```

**Restart GNOME Shell** (X11):
- `Alt+F2`, type `r`, Enter

---

### Telemetry daemon not broadcasting

**Check if running**:
```bash
ps aux | grep ara_telemetry_daemon
```

**Check D-Bus service is published**:
```bash
gdbus introspect --session --dest org.scythe.Ara \
    --object-path /org/scythe/Ara/Telemetry
```

**Run manually to see errors**:
```bash
~/bin/ara_telemetry_daemon.py
```

---

### Mode script not executing

**Check script is executable**:
```bash
chmod +x ~/bin/ara_mode.sh
```

**Test manually**:
```bash
~/bin/ara_mode.sh flight
```

**Check GNOME Shell extension logs**:
```bash
journalctl /usr/bin/gnome-shell -f | grep AraStatus
```

---

## Performance Verification

### CPU Isolation Check

```bash
# Run a CPU-intensive task
stress --cpu 8 &

# Check that stress is NOT on isolated cores
ps -eLo pid,tid,cls,rtprio,pri,psr,comm | grep stress
# psr column should show 0-7, NOT 8-15

# Check orchestrator IS on isolated cores
ps -eLo pid,tid,cls,rtprio,pri,psr,comm | grep orchestrator
# psr column should show 8-15
```

### RT Priority Check

```bash
# Check scheduling class and priority
ps -eLo pid,tid,cls,rtprio,pri,comm | grep orchestrator
# cls should be FF (SCHED_FIFO)
# rtprio should be 80
```

### Latency Measurement

```bash
# Monitor HAA orchestrator logs for latency
journalctl --user -u ara-orchestrator.service -f | grep "Total Time-to-Anim-Ready"
```

Expected: <300ms for full pipeline (FPGA + GPU).

---

## Architecture Rationale

### Why CPU Isolation?

GNOME, browsers, and background tasks create scheduling jitter. By isolating cores for Ara's pipeline:
- **Deterministic latency**: No context switches from random processes
- **Cache locality**: Ara's hot paths stay in L2/L3 cache
- **Predictable performance**: Critical for sub-300ms avatar generation

### Why SCHED_FIFO?

- **Preemption guarantee**: Ara's orchestrator preempts all lower-priority tasks
- **No timeslice expiration**: Runs until it yields or blocks
- **Critical for DMA timing**: Ensures pinned buffer management happens on time

### Why D-Bus?

- **Standard IPC**: All GNOME components speak D-Bus
- **Structured data**: JSON payloads instead of text parsing
- **Signal-based**: Zero polling, purely event-driven
- **Multi-subscriber**: GNOME Shell + Conky + custom apps all listen simultaneously

---

## Future Enhancements

1. **NVTX Integration**: Parse CUDA NVTX traces for per-kernel timing
2. **Command Interface**: Add `org.scythe.Ara.Command` for manual FPGA shutdown, RT mode toggle
3. **Touch Console**: GTK4 macro pad with giant buttons for mode switching
4. **Conky Templates**: Pre-built HUD configs for flight/battle modes
5. **Multi-Monitor**: Ara panel on secondary display, main for avatar
6. **Voice Commands**: Speech recognition for hands-free mode switching

---

**"Built like a cathedral. Runs in your cathedral. Your desktop IS the warship."**
