#!/usr/bin/env python3
"""
Operation SCYTHE - Tactical Macro Console
Touch-optimized GTK4/LibAdwaita mode switcher for Cathedral Avatar system

Coordinates GNOME state transitions with HAA avatar personality modes.
"""

import sys
import subprocess
import gi

gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')

from gi.repository import Gtk, Adw, Gio, GLib

# Optional: D-Bus integration for mode feedback
try:
    from pydbus import SessionBus
    DBUS_AVAILABLE = True
except ImportError:
    DBUS_AVAILABLE = False
    print("Warning: pydbus not available, D-Bus feedback disabled")


class ScytheMacroConsole(Adw.Application):
    """Main application class for SCYTHE Macro Console."""

    def __init__(self):
        super().__init__(application_id='org.scythe.MacroConsole',
                        flags=Gio.ApplicationFlags.FLAGS_NONE)

        self.current_mode = "standby"  # Default mode
        self.mode_buttons = {}

        # D-Bus setup for mode feedback
        if DBUS_AVAILABLE:
            try:
                bus = SessionBus()
                # Subscribe to mode change signals
                bus.subscribe(
                    sender=None,
                    iface="org.scythe.ModeControl",
                    signal="ModeActivated",
                    object="/org/scythe/ModeControl",
                    arg0=None,
                    signal_fired=self.on_mode_activated_signal
                )
            except Exception as e:
                print(f"D-Bus subscription failed: {e}")

    def do_activate(self):
        """Called when the application is activated."""
        window = self.create_main_window()
        window.present()

        # Set Always-on-Top after window is presented
        GLib.timeout_add(500, self.set_always_on_top, window)

    def create_main_window(self):
        """Create the main borderless, touch-optimized console window."""

        # Create main window
        window = Gtk.ApplicationWindow(application=self)
        window.set_title("SCYTHE Tactical Console")
        window.set_default_size(400, 300)

        # Borderless window for cockpit aesthetic
        window.set_decorated(False)

        # Create grid layout for macro buttons
        grid = Gtk.Grid()
        grid.set_row_spacing(15)
        grid.set_column_spacing(15)
        grid.set_margin_top(20)
        grid.set_margin_bottom(20)
        grid.set_margin_start(20)
        grid.set_margin_end(20)

        # Title label
        title_label = Gtk.Label(label="OPERATION SCYTHE")
        title_label.add_css_class("title-1")
        title_label.add_css_class("scythe-title")
        grid.attach(title_label, 0, 0, 3, 1)

        # Mode buttons (large, touch-optimized OSD style)
        self.create_mode_button(grid, "Standby", "standby", 0, 1)
        self.create_mode_button(grid, "Flight", "flight", 1, 1)
        self.create_mode_button(grid, "Alert", "alert", 2, 1)

        # Status label
        self.status_label = Gtk.Label(label=f"Current Mode: {self.current_mode.upper()}")
        self.status_label.add_css_class("caption")
        self.status_label.add_css_class("scythe-status")
        grid.attach(self.status_label, 0, 2, 3, 1)

        # Quit button (small, bottom corner)
        quit_btn = Gtk.Button(label="×")
        quit_btn.add_css_class("circular")
        quit_btn.add_css_class("scythe-quit-btn")
        quit_btn.connect("clicked", lambda _: window.close())
        grid.attach(quit_btn, 2, 3, 1, 1)

        # Set grid as window content
        window.set_child(grid)

        # Apply custom CSS
        self.apply_custom_css()

        # Update initial button state
        self.update_active_button()

        return window

    def create_mode_button(self, grid, label, mode, col, row):
        """Create a large, touch-optimized mode button."""

        button = Gtk.Button(label=label)

        # Apply OSD-style classes for large, glowing buttons
        button.add_css_class("osd")
        button.add_css_class("scythe-macro-button")
        button.add_css_class(f"mode-{mode}")

        # Set minimum size for touch targets
        button.set_size_request(100, 80)

        # Connect touch/click gesture
        gesture = Gtk.GestureClick.new()
        gesture.connect("pressed", lambda g, n, x, y: self.on_mode_button_pressed(mode))
        button.add_controller(gesture)

        # Also handle standard clicked signal for mouse input
        button.connect("clicked", lambda _: self.on_mode_button_pressed(mode))

        # Store button reference
        self.mode_buttons[mode] = button

        grid.attach(button, col, row, 1, 1)

    def on_mode_button_pressed(self, mode):
        """Handle mode button press."""
        print(f"[SCYTHE] Mode button pressed: {mode}")

        # Execute mode activation script
        script_path = f"/usr/local/bin/activate_{mode}.sh"

        try:
            subprocess.Popen([script_path],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
            print(f"[SCYTHE] Executed: {script_path}")

            # Update local state (will be confirmed by D-Bus signal)
            self.set_mode(mode)

        except FileNotFoundError:
            print(f"[SCYTHE] Error: Script not found: {script_path}")
            print(f"[SCYTHE] Hint: Copy mode scripts to /usr/local/bin/ and chmod +x")

            # Fallback: update UI anyway
            self.set_mode(mode)

        except Exception as e:
            print(f"[SCYTHE] Error executing script: {e}")

    def set_mode(self, mode):
        """Set the current mode and update UI."""
        if self.current_mode == mode:
            return

        self.current_mode = mode
        self.status_label.set_text(f"Current Mode: {mode.upper()}")
        self.update_active_button()

    def update_active_button(self):
        """Update button styling to reflect active mode."""
        for mode, button in self.mode_buttons.items():
            if mode == self.current_mode:
                button.add_css_class("active-mode")
            else:
                button.remove_css_class("active-mode")

    def on_mode_activated_signal(self, sender, object_path, iface, signal, params):
        """D-Bus signal handler for mode activation confirmation."""
        if len(params) >= 1:
            mode = params[0]
            print(f"[SCYTHE] D-Bus: Mode activated signal received: {mode}")
            GLib.idle_add(self.set_mode, mode)

    def set_always_on_top(self, window):
        """Set window to always-on-top using wmctrl."""
        # Get window XID
        try:
            surface = window.get_surface()
            if surface is None:
                # Retry if surface not ready
                return True

            # Use wmctrl to set always-on-top
            # Note: This requires wmctrl to be installed
            result = subprocess.run(
                ["wmctrl", "-l"],
                capture_output=True,
                text=True
            )

            # Find our window in wmctrl output
            for line in result.stdout.split('\n'):
                if "SCYTHE Tactical Console" in line:
                    xid = line.split()[0]
                    subprocess.run(["wmctrl", "-i", "-r", xid, "-b", "add,above"])
                    print(f"[SCYTHE] Set always-on-top for window {xid}")
                    return False  # Stop retrying

        except FileNotFoundError:
            print("[SCYTHE] wmctrl not found, install with: sudo apt install wmctrl")
        except Exception as e:
            print(f"[SCYTHE] Error setting always-on-top: {e}")

        return False  # Stop retrying

    def apply_custom_css(self):
        """Apply custom CSS for SCYTHE aesthetic."""

        css_provider = Gtk.CssProvider()
        css_data = """
        /* SCYTHE Macro Console Custom Styling */

        .scythe-title {
            color: rgba(0, 255, 255, 1.0);
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.8);
            font-weight: bold;
            letter-spacing: 3px;
        }

        .scythe-status {
            color: rgba(0, 200, 255, 0.9);
            text-shadow: 0 0 5px rgba(0, 200, 255, 0.5);
            margin-top: 10px;
        }

        /* Large touch-optimized mode buttons */
        .scythe-macro-button {
            font-size: 16px;
            font-weight: bold;
            padding: 20px;
            background-color: rgba(0, 30, 50, 0.6);
            border: 2px solid rgba(0, 200, 255, 0.5);
            box-shadow: 0 0 10px rgba(0, 200, 255, 0.6);
            border-radius: 8px;
            color: rgba(0, 255, 255, 1.0);
        }

        .scythe-macro-button:hover {
            background-color: rgba(0, 60, 100, 0.8);
            box-shadow: 0 0 25px rgba(0, 255, 255, 1.0);
            border-color: rgba(0, 255, 255, 0.8);
            transition: all 0.3s ease;
        }

        .scythe-macro-button:active {
            background-color: rgba(0, 100, 150, 0.9);
            box-shadow: 0 0 30px rgba(0, 255, 255, 1.0) inset;
        }

        /* Active mode highlighting */
        .scythe-macro-button.active-mode {
            background-color: rgba(0, 150, 255, 0.5);
            border-color: rgba(0, 255, 255, 1.0);
            box-shadow: 0 0 30px rgba(0, 255, 255, 1.0);
            border-width: 3px;
        }

        /* Mode-specific colors */
        .mode-standby.active-mode {
            border-color: rgba(100, 200, 255, 1.0);
            box-shadow: 0 0 30px rgba(100, 200, 255, 1.0);
        }

        .mode-flight.active-mode {
            border-color: rgba(255, 150, 0, 1.0);
            box-shadow: 0 0 30px rgba(255, 150, 0, 1.0);
        }

        .mode-alert.active-mode {
            border-color: rgba(255, 50, 50, 1.0);
            box-shadow: 0 0 30px rgba(255, 50, 50, 1.0);
            animation: pulse-alert 1.5s infinite;
        }

        @keyframes pulse-alert {
            0%, 100% { opacity: 1.0; }
            50% { opacity: 0.7; }
        }

        /* Quit button styling */
        .scythe-quit-btn {
            font-size: 24px;
            color: rgba(255, 100, 100, 0.8);
            background-color: rgba(50, 0, 0, 0.3);
            border: 1px solid rgba(255, 50, 50, 0.4);
        }

        .scythe-quit-btn:hover {
            background-color: rgba(100, 0, 0, 0.5);
            color: rgba(255, 50, 50, 1.0);
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
    app = ScytheMacroConsole()
    return app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
