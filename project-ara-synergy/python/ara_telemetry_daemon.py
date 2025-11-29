#!/usr/bin/env python3
"""
Ara Telemetry Daemon (ATF - Ara Telemetry Fusion)

D-Bus service that broadcasts HAA system metrics at 5Hz for consumption by:
- GNOME Shell extension (ara-status@scythe.dev)
- Touch console (ara_touch_console.py)
- External monitoring tools

Service: org.scythe.Ara
Object: /org/scythe/Ara/Telemetry
Interface: org.scythe.Ara.Telemetry

Signal: StatusUpdate(s payload_json)
"""

import json
import time
from threading import Thread

from gi.repository import GLib
from pydbus import SessionBus


class AraTelemetry(object):
    """
    <node>
      <interface name='org.scythe.Ara.Telemetry'>
        <signal name='StatusUpdate'>
          <arg type='s' name='payload'/>
        </signal>
      </interface>
    </node>
    """

    def __init__(self):
        self._state = "IDLE"
        self._gpu_load = 0.0
        self._fpga_lat_ms = 0.0
        self._visual_jitter_ms = 0.0
        self._target_lat_met = True

    def update_metrics(self, state, gpu_load, fpga_lat, vis_jitter, target_met):
        """Update internal metrics. Called by metrics thread."""
        self._state = state
        self._gpu_load = gpu_load
        self._fpga_lat_ms = fpga_lat
        self._visual_jitter_ms = vis_jitter
        self._target_lat_met = target_met

    def emit_status(self):
        """Broadcast current metrics via D-Bus signal."""
        payload = {
            "State": self._state,
            "GPU_Load_Percent": self._gpu_load,
            "FPGA_Latency_ms": self._fpga_lat_ms,
            "Visual_Jitter_ms": self._visual_jitter_ms,
            "Target_Lat_Met": self._target_lat_met,
        }
        self.StatusUpdate(json.dumps(payload))


def metrics_loop(telemetry: AraTelemetry):
    """
    Background thread that updates metrics at 5Hz (200ms interval).

    TODO: Wire this to real orchestrator metrics via IPC:
    - Read from shared memory segment
    - Parse NVTX trace output
    - Query nvidia-smi for GPU load
    - Read from orchestrator log pipe
    """
    counter = 0
    while True:
        # MOCK DATA: Simulate realistic HAA metrics
        # Replace with actual data sources

        # Cycle through states for demo
        states = ["IDLE", "THINKING", "THINKING", "SPEAKING", "IDLE"]
        state = states[counter % len(states)]

        # Simulate varying GPU load
        if state == "SPEAKING":
            gpu_load = 92.0
        elif state == "THINKING":
            gpu_load = 45.0
        else:
            gpu_load = 5.0

        # Simulate FPGA latency (target: <100ms)
        fpga_lat = 87.5 if state != "IDLE" else 0.0

        # Simulate visual jitter (target: <16ms for 60 FPS)
        vis_jitter = 12.3 if state == "SPEAKING" else 8.0

        # Target met if total latency < 300ms
        target_met = (fpga_lat + vis_jitter) < 300.0

        telemetry.update_metrics(
            state=state,
            gpu_load=gpu_load,
            fpga_lat=fpga_lat,
            vis_jitter=vis_jitter,
            target_met=target_met,
        )
        telemetry.emit_status()

        counter += 1
        time.sleep(0.2)  # 5 Hz (200ms interval)


def main():
    """Main entry point. Publishes D-Bus service and starts metrics loop."""
    print("[ATF] Starting Ara Telemetry Fusion daemon...")

    bus = SessionBus()
    telemetry = AraTelemetry()

    # Publish D-Bus service
    bus.publish("org.scythe.Ara", ("/org/scythe/Ara/Telemetry", telemetry))
    print("[ATF] D-Bus service published: org.scythe.Ara")
    print("[ATF] Object path: /org/scythe/Ara/Telemetry")
    print("[ATF] Broadcasting StatusUpdate signals at 5Hz...")

    # Start background metrics thread
    t = Thread(target=metrics_loop, args=(telemetry,), daemon=True)
    t.start()

    # Run GLib main loop (required for D-Bus)
    try:
        loop = GLib.MainLoop()
        loop.run()
    except KeyboardInterrupt:
        print("\n[ATF] Shutting down...")


if __name__ == "__main__":
    main()
