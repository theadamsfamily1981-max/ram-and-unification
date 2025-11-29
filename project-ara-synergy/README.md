# Project Ara-SYNERGY

Fusion of a **real-time AI avatar pipeline (Ara)** with a **GNOME/Wayland sci-fi cockpit**.

Core idea: treat the avatar and desktop as **one organism**.
The Hybrid Accelerated Architecture (HAA) lives on its own real-time island (FPGA + RTX), and GNOME becomes a reactive control surface driven by Ara's telemetry.

---

## Components

### 1. Real-Time Core

- **FPGA (SQRL Forest Kitten + HBM2)**
  - Stage I: Quantized DNN TTS / acoustic feature generator (`fpga/tts_kernel_hls.cpp`).
- **GPU (RTX 5060 Ti 16 GB)**
  - Stage II: CUDA animation head (`cuda/avatar_inference.cu`).
- **Python Orchestrator**
  - Stage III: Manages streaming, latency, and device coordination (`python/orchestrator.py`).
- **Real-time enforcement**
  - Isolated CPU cores + `SCHED_FIFO` via systemd slice (`systemd/ara-rt.slice`, `systemd/ara-orchestrator.service`).

### 2. Bi-Directional Telemetry (ATF)

- `python/ara_telemetry_daemon.py` publishes:
  - `org.scythe.Ara.Telemetry /org/scythe/Ara/Telemetry`
  - Signal `StatusUpdate(s payload_json)`
- Payload includes:
  - `State` (`IDLE`, `THINKING`, `SPEAKING`, `CRITICAL`)
  - `GPU_Load_Percent`
  - `FPGA_Latency_ms`
  - `Visual_Jitter_ms`
  - `Target_Lat_Met`

### 3. GNOME Cockpit Integration

- GNOME Shell extension:
  - `gnome/ara-status@scythe.dev/`
  - Panel icon listens to ATF on D-Bus
  - Triggers `scripts/ara_mode.sh` to flip between:
    - `CRUISE` (quiet, nebula, low HUD)
    - `FLIGHT` (tactical monitoring)
    - `BATTLE` (alert HUD, heavy glow, aggressive visuals)
- Touch console:
  - `python/ara_touch_console.py` – GTK4 big-button macro pad on the touch display.

---

## Quickstart

> ⚠️ This assumes you're on a GNOME / Wayland system with NVIDIA + CUDA set up.

### 0. Dependencies

```bash
sudo apt install -y \
  python3 python3-pip python3-gi gir1.2-gtk-4.0 gir1.2-adw-1 \
  libdbus-1-dev libglib2.0-dev \
  gnome-shell-extensions \
  nvidia-cuda-toolkit conky

# Optional: PyFPGA for FPGA automation
pip install -r python/requirements.txt
```

### 1. Build the CUDA library

```bash
cd cuda
make
cd ..
```

This creates `build/libavatar_qnn.so`.

### 2. Install systemd units

```bash
sudo cp systemd/ara-rt.slice /etc/systemd/system/
sudo cp systemd/ara-orchestrator.service /etc/systemd/system/
sudo cp systemd/ara-telemetry.service /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable ara-orchestrator.service ara-telemetry.service
# Start manually first while you debug:
sudo systemctl start ara-telemetry.service
sudo systemctl start ara-orchestrator.service
```

### 3. Install GNOME Shell extension

```bash
mkdir -p ~/.local/share/gnome-shell/extensions
cp -r gnome/ara-status@scythe.dev ~/.local/share/gnome-shell/extensions/

gnome-extensions enable ara-status@scythe.dev
```

### 4. Mode script

```bash
mkdir -p ~/bin
cp scripts/ara_mode.sh ~/bin/
chmod +x ~/bin/ara_mode.sh
```

### 5. Touch console (optional)

```bash
python3 python/ara_touch_console.py
```

Pin it on the touchscreen workspace, full screen.

---

## Where to Put the Heavy Code

* Use the **HLS QNN kernel** code we wrote earlier in:
  * `fpga/tts_kernel_hls.cpp`
* Use the **CUDA Stage II** code with NVTX and C API in:
  * `cuda/avatar_inference.cu`
* Use the **orchestrator** + **DMA simulation** code in:
  * `python/orchestrator.py`

Each file in this repo is wired around those contracts (dimensions, function names, D-Bus APIs).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│ GNOME Cockpit Layer                                     │
│ - Shell Extension (ara-status@scythe.dev)              │
│ - Touch Console (GTK4/LibAdwaita)                       │
│ - Mode Switcher Script (ara_mode.sh)                    │
├─────────────────────────────────────────────────────────┤
│ D-Bus Telemetry Fusion (ATF)                            │
│ - org.scythe.Ara.Telemetry (StatusUpdate signal)       │
│ - 5Hz broadcast of HAA metrics                          │
├─────────────────────────────────────────────────────────┤
│ Python Orchestrator (Stage III)                         │
│ - PyFPGA deployment                                      │
│ - DMA buffer management                                  │
│ - CUDA stream coordination                               │
│ - Latency tracking (FPGA + GPU + Total)                │
├─────────────────────────────────────────────────────────┤
│ GPU Animation (Stage II)                                │
│ - RTX 5060 Ti 16GB                                      │
│ - uint16 → FP16/FP32 dequantization                     │
│ - Animation inference (blendshapes + landmarks)          │
│ - NVTX instrumentation                                   │
├─────────────────────────────────────────────────────────┤
│ FPGA QNN Inference (Stage I)                            │
│ - SQRL Forest Kitten (VU3P + HBM2)                      │
│ - Quantized DNN TTS                                      │
│ - AXI-Stream output to GPU DMA                           │
└─────────────────────────────────────────────────────────┘
```

---

## Real-Time Configuration

### CPU Isolation

Add to GRUB configuration (`/etc/default/grub`):

```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash \
  isolcpus=8-15 \
  nohz_full=8-15 \
  rcu_nocbs=8-15 \
  intel_pstate=disable"
```

Then update GRUB:

```bash
sudo update-grub
sudo reboot
```

Verify isolation after reboot:

```bash
cat /proc/cmdline
cat /sys/devices/system/cpu/isolated  # Should show: 8-15
```

### RT Privileges

Grant RT scheduling privileges:

```bash
sudo nano /etc/security/limits.conf
# Add:
your_username - rtprio 95
your_username - memlock unlimited
```

Logout and login again for changes to take effect.

---

## Monitoring

### Watch telemetry in realtime

```bash
dbus-monitor --session "interface='org.scythe.Ara.Telemetry'"
```

### Check orchestrator logs

```bash
sudo journalctl -u ara-orchestrator.service -f
```

### Check telemetry daemon logs

```bash
sudo journalctl -u ara-telemetry.service -f
```

### Verify CPU affinity

```bash
ps aux | grep orchestrator
taskset -c -p <PID>  # Should show: 8-15
```

### Verify RT priority

```bash
ps -eLo pid,tid,cls,rtprio,pri,comm | grep orchestrator
# cls should be FF (SCHED_FIFO)
# rtprio should be 80
```

---

## Performance Targets

- **Total latency**: < 300ms (FPGA + GPU + transfer overhead)
- **FPGA TTS**: < 100ms per chunk (16 frames × 64 features)
- **GPU animation**: < 50ms per chunk (dequant + inference)
- **Telemetry rate**: 5Hz (200ms interval)
- **Visual jitter**: < 16ms (60 FPS target)

---

## Mode System

### CRUISE Mode (IDLE)

- **Wallpaper**: `cruise_nebula.png` (calm, deep space aesthetic)
- **HUD**: Minimal or disabled
- **Extensions**: Blur disabled, animations disabled
- **Purpose**: Free all GPU resources for Ara, minimal desktop overhead

### FLIGHT Mode (THINKING/PROCESSING)

- **Wallpaper**: `flight_tactical.png` (tactical blue HUD aesthetic)
- **HUD**: Conky with tactical overlay (`flight.conkyrc`)
- **Extensions**: Dash-to-panel enabled, blur enabled
- **Purpose**: Active monitoring during LLM processing and FPGA TTS

### BATTLE Mode (SPEAKING/CRITICAL)

- **Wallpaper**: `battle_alert.png` (red alert, critical state)
- **HUD**: Conky with aggressive monitoring (`battle_hud.conkyrc`)
- **Extensions**: All monitoring enabled
- **Sound**: Alert klaxon on mode entry
- **Purpose**: Maximum awareness during avatar generation or critical state

---

## Troubleshooting

### CUDA library not found

**Error**: `libavatar_qnn.so: cannot open shared object file`

**Fix**: Add build directory to LD_LIBRARY_PATH:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ara/project-ara-synergy/build
# Or add to ~/.bashrc for persistence
```

### D-Bus service not available

**Error**: Extension cannot connect to `org.scythe.Ara`

**Fix**: Check if telemetry daemon is running:

```bash
sudo systemctl status ara-telemetry.service
# If not running:
sudo systemctl start ara-telemetry.service
```

### RT scheduling permission denied

**Error**: `Failed to set SCHED_FIFO: Operation not permitted`

**Fix**: Check limits configuration:

```bash
sudo nano /etc/security/limits.conf
# Ensure lines exist:
your_username - rtprio 95
your_username - memlock unlimited
```

Then logout and login again.

### GNOME Shell extension not loading

**Check installation**:

```bash
ls ~/.local/share/gnome-shell/extensions/ara-status@scythe.dev/
```

**Check enabled**:

```bash
gnome-extensions list --enabled | grep ara-status
```

**Check for errors**:

```bash
journalctl /usr/bin/gnome-shell -f | grep AraStatus
```

**Restart GNOME Shell** (X11 only):

- Press `Alt+F2`
- Type `r`
- Press `Enter`

On Wayland, logout and login again.

---

## Development

### Testing without FPGA

The orchestrator includes a DMA simulation mode that generates synthetic uint16 data. This allows testing the full GPU pipeline without actual FPGA hardware:

```python
# In orchestrator.py, _simulate_dma_and_sync_features() generates mock data
```

### Adding custom modes

1. Create new wallpaper in `wallpapers/`
2. Create new Conky config in `~/.config/conky/`
3. Add mode case to `scripts/ara_mode.sh`
4. Update GNOME Shell extension state mapping in `extension.js`

### NVTX Profiling

Use NVIDIA Nsight Systems to profile CUDA kernels:

```bash
nsys profile --trace=cuda,nvtx python3 python/orchestrator.py
nsys-ui report.qdrep
```

NVTX ranges are embedded in `cuda/avatar_inference.cu` for detailed timing analysis.

---

## Future Enhancements

1. **Command Interface**: Add `org.scythe.Ara.Command` D-Bus interface for manual control
2. **Multi-Monitor**: Dedicated Ara panel on secondary display
3. **Voice Commands**: Speech recognition for hands-free mode switching
4. **Conky Templates**: Pre-built HUD configs with real telemetry graphs
5. **TensorRT Optimization**: Replace placeholder GPU inference with optimized models
6. **FPGA Model Updates**: Hot-swap QNN weights without reprogramming bitstream

---

**"Built like a cathedral. Runs in your cathedral. Your desktop IS the warship."**
