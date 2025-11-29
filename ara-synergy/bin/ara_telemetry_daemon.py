#!/usr/bin/env python3
"""
Ara Telemetry Fusion (ATF) - D-Bus Service
Real-time HAA metrics broadcaster for GNOME Shell integration

D-Bus Interface:
  Service: org.scythe.Ara
  Object: /org/scythe/Ara/Telemetry
  Interface: org.scythe.Ara.Telemetry

Signals:
  - StatusUpdate(s payload_json)

Payload JSON Schema:
  {
    "State": "IDLE|THINKING|SPEAKING|CRITICAL",
    "GPU_Load_Percent": float,
    "FPGA_Latency_ms": float,
    "Visual_Jitter_ms": float,
    "Target_Lat_Met": bool,
    "Personality_Mode": str,
    "VRAM_Used_GB": float
  }
"""

import json
import time
import sys
from threading import Thread
from typing import Dict, Any

from gi.repository import GLib

try:
    from pydbus import SessionBus
except ImportError:
    print("ERROR: pydbus not available. Install with: pip install pydbus")
    sys.exit(1)


class AraTelemetry:
    """
    D-Bus service for broadcasting Ara HAA telemetry.

    This class doubles as D-Bus introspection XML definition.
    """

    dbus = """
    <node>
      <interface name='org.scythe.Ara.Telemetry'>
        <signal name='StatusUpdate'>
          <arg type='s' name='payload'/>
        </signal>
      </interface>
    </node>
    """

    def __init__(self):
        # Current metrics state
        self._state = "IDLE"
        self._gpu_load = 0.0
        self._fpga_lat_ms = 0.0
        self._visual_jitter_ms = 0.0
        self._target_lat_met = True
        self._personality_mode = "cathedral"
        self._vram_used_gb = 0.0

        print("[ATF] Ara Telemetry Fusion initialized")

    def update_metrics(self, **kwargs):
        """
        Update internal metrics state.

        Called by orchestrator or monitoring scripts to feed fresh data.

        Args:
            state: Ara state (IDLE, THINKING, SPEAKING, CRITICAL)
            gpu_load: GPU utilization percentage (0.0-100.0)
            fpga_lat: FPGA inference latency in ms
            vis_jitter: Visual rendering jitter in ms
            target_met: Whether latency target (<300ms) is met
            personality_mode: Current avatar personality
            vram_used: VRAM utilization in GB
        """
        if "state" in kwargs:
            self._state = kwargs["state"]
        if "gpu_load" in kwargs:
            self._gpu_load = kwargs["gpu_load"]
        if "fpga_lat" in kwargs:
            self._fpga_lat_ms = kwargs["fpga_lat"]
        if "vis_jitter" in kwargs:
            self._visual_jitter_ms = kwargs["vis_jitter"]
        if "target_met" in kwargs:
            self._target_lat_met = kwargs["target_met"]
        if "personality_mode" in kwargs:
            self._personality_mode = kwargs["personality_mode"]
        if "vram_used" in kwargs:
            self._vram_used_gb = kwargs["vram_used"]

    def emit_status(self):
        """Emit current telemetry state as D-Bus signal."""
        payload = {
            "State": self._state,
            "GPU_Load_Percent": self._gpu_load,
            "FPGA_Latency_ms": self._fpga_lat_ms,
            "Visual_Jitter_ms": self._visual_jitter_ms,
            "Target_Lat_Met": self._target_lat_met,
            "Personality_Mode": self._personality_mode,
            "VRAM_Used_GB": self._vram_used_gb,
        }

        payload_json = json.dumps(payload)
        self.StatusUpdate(payload_json)

        print(f"[ATF] Emitted: {self._state} | "
              f"GPU={self._gpu_load:.1f}% | "
              f"FPGA={self._fpga_lat_ms:.2f}ms | "
              f"Target={'✓' if self._target_lat_met else '✗'}")

    def StatusUpdate(self, payload):
        """D-Bus signal (defined in XML above)."""
        pass  # GLib/pydbus handles signal emission


def metrics_monitor_loop(telemetry: AraTelemetry):
    """
    Background thread that monitors system metrics and updates telemetry.

    In production, this would:
    - Read orchestrator metrics via socket/pipe/shared memory
    - Parse NVTX traces for GPU timing
    - Monitor FPGA status via driver interface
    - Check VRAM usage via nvidia-smi

    For now, it demonstrates the update pattern with synthetic data.
    """
    print("[ATF] Metrics monitor thread started")

    import random
    import subprocess

    state_cycle = ["IDLE", "THINKING", "SPEAKING", "IDLE"]
    state_idx = 0
    iteration = 0

    while True:
        # Simulate state machine
        state = state_cycle[state_idx % len(state_cycle)]

        # Synthetic metrics (replace with real monitoring)
        if state == "SPEAKING":
            gpu_load = random.uniform(85, 95)
            fpga_lat = random.uniform(80, 120)
            vis_jitter = random.uniform(10, 20)
        elif state == "THINKING":
            gpu_load = random.uniform(60, 75)
            fpga_lat = random.uniform(70, 90)
            vis_jitter = random.uniform(5, 12)
        else:  # IDLE
            gpu_load = random.uniform(5, 15)
            fpga_lat = random.uniform(1, 5)
            vis_jitter = random.uniform(0, 3)

        target_met = (fpga_lat + vis_jitter) < 300

        # Try to read real VRAM usage (optional)
        vram_used = 0.0
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                vram_mb = float(result.stdout.strip())
                vram_used = vram_mb / 1024.0  # Convert to GB
        except Exception:
            pass  # nvidia-smi not available or failed

        # Update telemetry
        telemetry.update_metrics(
            state=state,
            gpu_load=gpu_load,
            fpga_lat=fpga_lat,
            vis_jitter=vis_jitter,
            target_met=target_met,
            personality_mode="cathedral" if iteration % 20 < 10 else "cockpit",
            vram_used=vram_used
        )

        # Emit status update
        telemetry.emit_status()

        # Advance state machine
        if state == "SPEAKING":
            time.sleep(2.0)  # Longer speaking duration
            state_idx += 1
        elif state == "THINKING":
            time.sleep(1.0)
            state_idx += 1
        else:
            time.sleep(3.0)  # Idle longer
            state_idx += 1

        iteration += 1


def main():
    """Main entry point for ATF daemon."""
    print("=" * 60)
    print("Ara Telemetry Fusion (ATF) - Starting")
    print("=" * 60)

    # Publish D-Bus service
    bus = SessionBus()
    telemetry = AraTelemetry()

    try:
        bus.publish("org.scythe.Ara", ("/org/scythe/Ara/Telemetry", telemetry))
        print("[ATF] D-Bus service published: org.scythe.Ara")
        print("[ATF] Object path: /org/scythe/Ara/Telemetry")
        print("[ATF] Interface: org.scythe.Ara.Telemetry")
    except Exception as e:
        print(f"[ATF] ERROR: Failed to publish D-Bus service: {e}")
        print("[ATF] Hint: Check if another instance is running")
        sys.exit(1)

    # Start metrics monitoring thread
    monitor_thread = Thread(target=metrics_monitor_loop, args=(telemetry,), daemon=True)
    monitor_thread.start()

    # Run GLib main loop
    print("[ATF] Entering GLib main loop...")
    print("[ATF] Press Ctrl+C to stop")
    print("=" * 60)

    try:
        loop = GLib.MainLoop()
        loop.run()
    except KeyboardInterrupt:
        print("\n[ATF] Shutting down...")


if __name__ == "__main__":
    main()
