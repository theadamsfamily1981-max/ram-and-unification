#!/usr/bin/env python3
# hud/tfan_hud.py
# Ara Brain HUD - GTK4/libadwaita Real-Time Visualization
#
# Polls ~/.tfan/metrics.json and displays T-FAN brain state in real time.
#
# Requirements:
#   - GTK 4
#   - libadwaita
#   - PyGObject (gi)
#
# Install on Fedora/GNOME:
#   sudo dnf install gtk4-devel libadwaita-devel python3-gobject
#
# Usage:
#   python hud/tfan_hud.py

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk, Pango


# ============================================================================
#  Constants
# ============================================================================

METRICS_PATH = Path(os.path.expanduser("~/.tfan/metrics.json"))
POLL_INTERVAL_MS = 500  # Poll every 500ms

APP_ID = "org.ara.brain.hud"
APP_TITLE = "Ara Brain HUD"


# ============================================================================
#  Metric Card Widget
# ============================================================================

class MetricCard(Gtk.Box):
    """A styled card showing a single metric with label and value."""

    def __init__(self, label: str, value: str = "—", color: str = "#88C0D0"):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.set_margin_top(8)
        self.set_margin_bottom(8)
        self.set_margin_start(8)
        self.set_margin_end(8)

        self.add_css_class("card")

        # Label
        self.label_widget = Gtk.Label(label=label)
        self.label_widget.add_css_class("dim-label")
        self.label_widget.set_halign(Gtk.Align.START)
        self.append(self.label_widget)

        # Value
        self.value_widget = Gtk.Label(label=value)
        self.value_widget.add_css_class("title-1")
        self.value_widget.set_halign(Gtk.Align.START)
        self.append(self.value_widget)

        self._color = color

    def set_value(self, value: str) -> None:
        self.value_widget.set_label(value)


# ============================================================================
#  Progress Bar Widget
# ============================================================================

class MetricProgressBar(Gtk.Box):
    """A labeled progress bar for bounded metrics."""

    def __init__(self, label: str, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.set_margin_top(4)
        self.set_margin_bottom(4)

        self.min_val = min_val
        self.max_val = max_val

        # Header row
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        header.set_homogeneous(False)

        self.label_widget = Gtk.Label(label=label)
        self.label_widget.add_css_class("dim-label")
        self.label_widget.set_halign(Gtk.Align.START)
        self.label_widget.set_hexpand(True)
        header.append(self.label_widget)

        self.value_label = Gtk.Label(label="—")
        self.value_label.set_halign(Gtk.Align.END)
        header.append(self.value_label)

        self.append(header)

        # Progress bar
        self.progress = Gtk.ProgressBar()
        self.progress.set_show_text(False)
        self.append(self.progress)

    def set_value(self, value: float) -> None:
        # Clamp and normalize
        clamped = max(self.min_val, min(self.max_val, value))
        fraction = (clamped - self.min_val) / (self.max_val - self.min_val + 1e-9)

        self.progress.set_fraction(fraction)
        self.value_label.set_label(f"{value:.4f}")


# ============================================================================
#  NCE Actions List
# ============================================================================

class NCEActionsList(Gtk.Box):
    """Shows recent NCE niche construction actions."""

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.set_margin_top(8)
        self.set_margin_bottom(8)

        # Header
        header = Gtk.Label(label="Recent NCE Actions")
        header.add_css_class("heading")
        header.set_halign(Gtk.Align.START)
        self.append(header)

        # List box for actions
        self.list_box = Gtk.ListBox()
        self.list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self.list_box.add_css_class("boxed-list")
        self.append(self.list_box)

        # Placeholder
        self.placeholder = Gtk.Label(label="No actions yet")
        self.placeholder.add_css_class("dim-label")
        self.list_box.set_placeholder(self.placeholder)

    def update_actions(self, actions: list) -> None:
        # Clear existing
        while True:
            row = self.list_box.get_first_child()
            if row is None:
                break
            self.list_box.remove(row)

        # Add new actions (most recent first)
        for action in reversed(actions[-5:]):
            row = Adw.ActionRow()
            row.set_title(action.get("action_type", "unknown"))

            benefit = action.get("benefit", 0)
            cost = action.get("cost_ext", 0)
            executed = action.get("executed", False)

            subtitle = f"Benefit: {benefit:.1f} | Cost: {cost:.1f}"
            if executed:
                subtitle += " ✓"
            row.set_subtitle(subtitle)

            self.list_box.append(row)


# ============================================================================
#  Main HUD Window
# ============================================================================

class BrainHUDWindow(Adw.ApplicationWindow):
    """Main HUD window showing T-FAN brain state."""

    def __init__(self, app: Adw.Application):
        super().__init__(application=app, title=APP_TITLE)
        self.set_default_size(450, 700)

        # Header bar
        header = Adw.HeaderBar()
        header.set_title_widget(Gtk.Label(label=APP_TITLE))

        # Main content
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        main_box.append(header)

        # Scrollable content
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll.set_vexpand(True)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        content.set_margin_top(16)
        content.set_margin_bottom(16)
        content.set_margin_start(16)
        content.set_margin_end(16)

        # Status row
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)

        self.step_card = MetricCard("Step", "—")
        status_box.append(self.step_card)

        self.loss_card = MetricCard("Loss", "—")
        status_box.append(self.loss_card)

        self.elapsed_card = MetricCard("Elapsed", "—")
        status_box.append(self.elapsed_card)

        content.append(status_box)

        # Separator
        content.append(Gtk.Separator())

        # UDK Section
        udk_label = Gtk.Label(label="UDK Proxies")
        udk_label.add_css_class("title-2")
        udk_label.set_halign(Gtk.Align.START)
        udk_label.set_margin_top(8)
        content.append(udk_label)

        # Sigma proxy
        self.sigma_bar = MetricProgressBar("σ (Entropy Production)", 0.0, 2.0)
        content.append(self.sigma_bar)

        # Epsilon proxy
        self.epsilon_bar = MetricProgressBar("ε (Dissipation Rate)", 0.0, 0.5)
        content.append(self.epsilon_bar)

        # L_topo
        self.ltopo_bar = MetricProgressBar("L_topo (Betti Instability)", 0.0, 1.0)
        content.append(self.ltopo_bar)

        # Kappa proxy
        self.kappa_bar = MetricProgressBar("κ (FIM Curvature)", 0.0, 1.0)
        content.append(self.kappa_bar)

        # Separator
        content.append(Gtk.Separator())

        # Extra metrics section
        extra_label = Gtk.Label(label="Training Metrics")
        extra_label.add_css_class("title-2")
        extra_label.set_halign(Gtk.Align.START)
        extra_label.set_margin_top(8)
        content.append(extra_label)

        # UTCF
        self.utcf_bar = MetricProgressBar("UTCF (Unified Cost)", 0.0, 5.0)
        content.append(self.utcf_bar)

        # Lambda topo
        self.lambda_topo_bar = MetricProgressBar("λ_topo (Dynamic Weight)", 0.0, 0.5)
        content.append(self.lambda_topo_bar)

        # Separator
        content.append(Gtk.Separator())

        # NCE Actions
        self.nce_list = NCEActionsList()
        content.append(self.nce_list)

        # Connection status
        self.status_label = Gtk.Label(label="Waiting for metrics...")
        self.status_label.add_css_class("dim-label")
        self.status_label.set_margin_top(16)
        content.append(self.status_label)

        scroll.set_child(content)
        main_box.append(scroll)

        self.set_content(main_box)

        # Start polling
        self._last_mtime = 0
        GLib.timeout_add(POLL_INTERVAL_MS, self._poll_metrics)

    def _poll_metrics(self) -> bool:
        """Poll metrics file and update UI."""
        try:
            if not METRICS_PATH.exists():
                self.status_label.set_label(f"Waiting for {METRICS_PATH}...")
                return True

            # Check if file changed
            mtime = METRICS_PATH.stat().st_mtime
            if mtime == self._last_mtime:
                return True
            self._last_mtime = mtime

            # Read and parse
            with open(METRICS_PATH, "r") as f:
                data = json.load(f)

            self._update_from_data(data)
            self.status_label.set_label(f"Live — {METRICS_PATH}")

        except Exception as e:
            self.status_label.set_label(f"Error: {e}")

        return True  # Continue polling

    def _update_from_data(self, data: Dict[str, Any]) -> None:
        """Update all widgets from metrics data."""
        # Step / Loss / Elapsed
        self.step_card.set_value(str(data.get("step", "—")))
        loss = data.get("loss", 0)
        self.loss_card.set_value(f"{loss:.4f}" if isinstance(loss, float) else "—")

        elapsed = data.get("elapsed_sec", 0)
        if elapsed > 3600:
            elapsed_str = f"{elapsed/3600:.1f}h"
        elif elapsed > 60:
            elapsed_str = f"{elapsed/60:.1f}m"
        else:
            elapsed_str = f"{elapsed:.0f}s"
        self.elapsed_card.set_value(elapsed_str)

        # UDK proxies
        udk = data.get("udk", {})
        self.sigma_bar.set_value(udk.get("sigma_proxy", 0))
        self.epsilon_bar.set_value(udk.get("epsilon_proxy", 0))
        self.ltopo_bar.set_value(udk.get("L_topo", 0))
        self.kappa_bar.set_value(udk.get("kappa_proxy", 0))

        # Extra metrics
        extra = data.get("extra", {})
        self.utcf_bar.set_value(extra.get("utcf", 0))
        self.lambda_topo_bar.set_value(extra.get("lambda_topo", 0))

        # NCE actions
        nce_actions = data.get("nce_actions", [])
        self.nce_list.update_actions(nce_actions)


# ============================================================================
#  Application
# ============================================================================

class BrainHUDApp(Adw.Application):
    """Main GTK4/Adwaita application."""

    def __init__(self):
        super().__init__(
            application_id=APP_ID,
            flags=Gio.ApplicationFlags.FLAGS_NONE,
        )
        self.connect("activate", self._on_activate)

    def _on_activate(self, app):
        win = BrainHUDWindow(app)
        win.present()


# ============================================================================
#  Entry Point
# ============================================================================

def main():
    print(f"Ara Brain HUD")
    print(f"Monitoring: {METRICS_PATH}")
    print(f"Poll interval: {POLL_INTERVAL_MS}ms")
    print()

    app = BrainHUDApp()
    return app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
