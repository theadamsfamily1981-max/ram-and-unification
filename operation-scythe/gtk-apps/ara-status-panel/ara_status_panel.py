#!/usr/bin/env python3
"""
Operation SCYTHE - Ara Status Panel
Real-time telemetry display for Cathedral Avatar HAA system

Displays FPGA/GPU metrics with conditional styling based on system health.
"""

import sys
import time
from collections import deque
import gi

gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')

from gi.repository import Gtk, Adw, Gio, GLib

# D-Bus integration for telemetry
try:
    from pydbus import SessionBus
    DBUS_AVAILABLE = True
except ImportError:
    DBUS_AVAILABLE = False
    print("Warning: pydbus not available, D-Bus telemetry disabled")


class AraStatusPanel(Adw.Application):
    """Main application class for Ara Status Panel."""

    def __init__(self):
        super().__init__(application_id='org.cathedral.AraStatusPanel',
                        flags=Gio.ApplicationFlags.FLAGS_NONE)

        # Message stack (LIFO) for status updates
        self.status_messages = deque(maxlen=10)
        self.status_label = None

        # Current notification ID for replacement
        self.notification_id = None

        # D-Bus setup for telemetry
        if DBUS_AVAILABLE:
            try:
                bus = SessionBus()

                # Subscribe to status updates
                bus.subscribe(
                    sender=None,
                    iface="org.cathedral.AraTelemetry.Status",
                    signal="StatusUpdate",
                    object="/org/cathedral/AraTelemetry/Status",
                    arg0=None,
                    signal_fired=self.on_status_update_signal
                )

                # Subscribe to performance metrics
                bus.subscribe(
                    sender=None,
                    iface="org.cathedral.AraTelemetry.Status",
                    signal="PerformanceMetrics",
                    object="/org/cathedral/AraTelemetry/Status",
                    arg0=None,
                    signal_fired=self.on_performance_metrics_signal
                )

                print("[ARA] D-Bus telemetry subscriptions active")

            except Exception as e:
                print(f"[ARA] D-Bus subscription failed: {e}")

    def do_activate(self):
        """Called when the application is activated."""
        window = self.create_main_window()
        window.present()

    def create_main_window(self):
        """Create the borderless, overlay-style status panel."""

        # Create main window
        window = Gtk.ApplicationWindow(application=self)
        window.set_title("Ara Status")
        window.set_default_size(800, 60)

        # Borderless for overlay aesthetic
        window.set_decorated(False)

        # Create horizontal box for status display
        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        main_box.set_margin_top(10)
        main_box.set_margin_bottom(10)
        main_box.set_margin_start(20)
        main_box.set_margin_end(20)

        # Ara icon/indicator
        indicator = Gtk.Label(label="◉")
        indicator.add_css_class("ara-indicator")
        indicator.add_css_class("title-2")
        main_box.append(indicator)

        # Status message label
        self.status_label = Gtk.Label(label="Ara Status: Initializing...")
        self.status_label.add_css_class("ara-status-normal")
        self.status_label.set_hexpand(True)
        self.status_label.set_xalign(0)  # Left align
        main_box.append(self.status_label)

        # Performance metrics display
        self.perf_label = Gtk.Label(label="")
        self.perf_label.add_css_class("ara-perf-metrics")
        self.perf_label.add_css_class("caption")
        main_box.append(self.perf_label)

        # Close button (minimal)
        close_btn = Gtk.Button(label="×")
        close_btn.add_css_class("circular")
        close_btn.add_css_class("ara-close-btn")
        close_btn.connect("clicked", lambda _: window.close())
        main_box.append(close_btn)

        window.set_child(main_box)

        # Apply custom CSS
        self.apply_custom_css()

        # Start demo telemetry if D-Bus unavailable
        if not DBUS_AVAILABLE:
            GLib.timeout_add_seconds(3, self.emit_demo_telemetry)

        return window

    def on_status_update_signal(self, sender, object_path, iface, signal, params):
        """D-Bus signal handler for status updates."""
        if len(params) >= 4:
            name, value, level, timestamp = params[:4]
            print(f"[ARA] Status: {name} = {value} ({level})")
            GLib.idle_add(self.update_status, name, value, level)

    def on_performance_metrics_signal(self, sender, object_path, iface, signal, params):
        """D-Bus signal handler for performance metrics."""
        if len(params) >= 3:
            fpga_latency, gpu_latency, total_latency = params[:3]
            print(f"[ARA] Perf: FPGA={fpga_latency:.1f}ms GPU={gpu_latency:.1f}ms Total={total_latency:.1f}ms")
            GLib.idle_add(self.update_performance, fpga_latency, gpu_latency, total_latency)

    def update_status(self, name, value, level):
        """Update status display with conditional styling."""

        # Build status message
        msg = f"{name}: {value}"

        # Push to message stack
        self.status_messages.append((msg, level, time.time()))

        # Display most recent message
        self.status_label.set_text(msg)

        # Remove old CSS classes
        for css_class in ["ara-status-normal", "ara-status-warning", "ara-status-critical"]:
            self.status_label.remove_css_class(css_class)

        # Apply new CSS class based on level
        if level == "Critical":
            self.status_label.add_css_class("ara-status-critical")
            self.send_notification(msg, urgency="critical")
        elif level == "Warning":
            self.status_label.add_css_class("ara-status-warning")
            self.send_notification(msg, urgency="normal")
        else:
            self.status_label.add_css_class("ara-status-normal")

    def update_performance(self, fpga_latency, gpu_latency, total_latency):
        """Update performance metrics display."""
        msg = f"FPGA: {fpga_latency:.1f}ms | GPU: {gpu_latency:.1f}ms | Total: {total_latency:.1f}ms"
        self.perf_label.set_text(msg)

        # Conditional styling based on latency targets
        if total_latency > 500:
            self.perf_label.remove_css_class("ara-perf-normal")
            self.perf_label.add_css_class("ara-perf-critical")
        elif total_latency > 300:
            self.perf_label.remove_css_class("ara-perf-normal")
            self.perf_label.add_css_class("ara-perf-warning")
        else:
            self.perf_label.remove_css_class("ara-perf-warning")
            self.perf_label.remove_css_class("ara-perf-critical")
            self.perf_label.add_css_class("ara-perf-normal")

    def send_notification(self, message, urgency="normal"):
        """Send GNOME notification with replacement ID."""
        try:
            # Use gdbus to send notification
            # Replace existing notification by using a fixed ID
            if self.notification_id is None:
                self.notification_id = 42  # Fixed ID for replacement

            urgency_level = 0 if urgency == "low" else 1 if urgency == "normal" else 2

            # Note: This is a simplified notification
            # Full implementation would use org.freedesktop.Notifications D-Bus interface
            import subprocess
            subprocess.run([
                "notify-send",
                "Cathedral Avatar Status",
                message,
                "--urgency", urgency,
                "--icon", "dialog-information"
            ], check=False)

        except Exception as e:
            print(f"[ARA] Notification failed: {e}")

    def emit_demo_telemetry(self):
        """Emit demo telemetry data (for testing without D-Bus)."""
        import random

        # Simulate FPGA latency
        fpga_latency = random.uniform(80, 120)
        level = "Normal"
        if fpga_latency > 110:
            level = "Warning"
        elif fpga_latency > 115:
            level = "Critical"

        self.update_status("FPGA TTS Latency", f"{fpga_latency:.1f} ms", level)

        # Simulate performance metrics
        gpu_latency = random.uniform(60, 90)
        total_latency = fpga_latency + gpu_latency

        self.update_performance(fpga_latency, gpu_latency, total_latency)

        return True  # Continue timer

    def apply_custom_css(self):
        """Apply custom CSS for Ara Status Panel aesthetic."""

        css_provider = Gtk.CssProvider()
        css_data = """
        /* Ara Status Panel Custom Styling */

        /* Main window background */
        window {
            background-color: rgba(0, 10, 20, 0.85);
            border: 1px solid rgba(0, 200, 255, 0.3);
            box-shadow: 0 0 20px rgba(0, 200, 255, 0.4);
        }

        /* Ara indicator (glowing dot) */
        .ara-indicator {
            color: rgba(0, 255, 255, 1.0);
            text-shadow: 0 0 10px rgba(0, 255, 255, 1.0);
            animation: pulse-indicator 2s infinite;
        }

        @keyframes pulse-indicator {
            0%, 100% { opacity: 1.0; }
            50% { opacity: 0.5; }
        }

        /* Status message styling */
        .ara-status-normal {
            color: rgba(0, 255, 255, 1.0);
            text-shadow: 0 0 8px rgba(0, 255, 255, 0.6);
            font-weight: bold;
        }

        .ara-status-warning {
            color: rgba(255, 200, 0, 1.0);
            text-shadow: 0 0 10px rgba(255, 200, 0, 0.8);
            font-weight: bold;
        }

        .ara-status-critical {
            color: rgba(255, 50, 50, 1.0);
            text-shadow: 0 0 15px rgba(255, 0, 0, 1.0);
            font-weight: bold;
            animation: pulse-critical 1.5s infinite;
        }

        @keyframes pulse-critical {
            0%, 100% { opacity: 1.0; transform: scale(1.0); }
            50% { opacity: 0.7; transform: scale(1.05); }
        }

        /* Performance metrics styling */
        .ara-perf-metrics {
            color: rgba(100, 200, 255, 0.9);
            font-family: monospace;
        }

        .ara-perf-normal {
            color: rgba(100, 255, 150, 1.0);
        }

        .ara-perf-warning {
            color: rgba(255, 200, 0, 1.0);
        }

        .ara-perf-critical {
            color: rgba(255, 100, 100, 1.0);
        }

        /* Close button */
        .ara-close-btn {
            font-size: 20px;
            color: rgba(150, 150, 150, 0.6);
            background-color: transparent;
            border: none;
        }

        .ara-close-btn:hover {
            color: rgba(255, 100, 100, 0.9);
        }
        """

        css_provider.load_from_data(css_data.encode())

        Gtk.StyleContext.add_provider_for_display(
            self.get_display(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )


def main():
    """Main entry point."""
    app = AraStatusPanel()
    return app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
