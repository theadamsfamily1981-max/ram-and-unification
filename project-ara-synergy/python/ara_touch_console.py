#!/usr/bin/env python3
"""
Ara Touch Console

Large-button GTK4/LibAdwaita macro pad for Cathedral Avatar control.
Designed for touch displays with sci-fi aesthetic.

Features:
- QUANTUM MODE: Trigger RT mode via D-Bus
- SHUTDOWN ARRAY: Graceful FPGA shutdown
- EVASIVE SIGNAL: Block all radios (rfkill)
"""

import subprocess
from gi.repository import Adw, Gtk, Gio

APP_ID = "org.scythe.Ara.TouchConsole"


class TouchConsole(Adw.Application):
    def __init__(self):
        super().__init__(
            application_id=APP_ID,
            flags=Gio.ApplicationFlags.FLAGS_NONE
        )

    def do_activate(self):
        win = Adw.ApplicationWindow(application=self)
        win.set_title("Ara Console")
        win.set_default_size(800, 480)

        # Create grid layout for macro buttons
        grid = Gtk.Grid(
            column_spacing=24,
            row_spacing=24,
            margin_top=32,
            margin_bottom=32,
            margin_start=32,
            margin_end=32
        )

        def make_btn(label, cb):
            """Create large touch-optimized button."""
            b = Gtk.Button(label=label)
            b.get_style_context().add_class("suggested-action")
            b.set_hexpand(True)
            b.set_vexpand(True)
            b.connect("clicked", cb)
            return b

        # Macro buttons
        btn_rt = make_btn("QUANTUM MODE", self.on_rt_mode)
        btn_shutdown = make_btn("SHUTDOWN ARRAY", self.on_shutdown)
        btn_evasive = make_btn("EVASIVE SIGNAL", self.on_evasive)

        grid.attach(btn_rt,       0, 0, 1, 1)
        grid.attach(btn_shutdown, 1, 0, 1, 1)
        grid.attach(btn_evasive,  0, 1, 2, 1)

        win.set_content(grid)
        win.present()

    def on_rt_mode(self, _btn):
        """Trigger RT mode via D-Bus command interface."""
        print("[CONSOLE] Triggering QUANTUM MODE (RT priority enforcement)")
        subprocess.Popen([
            "dbus-send", "--session", "--dest=org.scythe.Ara",
            "--type=method_call",
            "/org/scythe/Ara/Command",
            "org.scythe.Ara.Command.TriggerRTMode"
        ])

    def on_shutdown(self, _btn):
        """Gracefully shutdown FPGA via D-Bus command interface."""
        print("[CONSOLE] Issuing FPGA shutdown command")
        subprocess.Popen([
            "dbus-send", "--session", "--dest=org.scythe.Ara",
            "--type=method_call",
            "/org/scythe/Ara/Command",
            "org.scythe.Ara.Command.ShutdownFPGA"
        ])

    def on_evasive(self, _btn):
        """Block all wireless radios (evasive signal mode)."""
        print("[CONSOLE] EVASIVE SIGNAL - Blocking all radios")
        subprocess.Popen(["rfkill", "block", "all"])


if __name__ == "__main__":
    app = TouchConsole()
    app.run([])
