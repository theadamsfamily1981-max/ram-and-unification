#!/usr/bin/env python3
# hud/tfan_gnome.py
# T-FAN GNOME Cockpit - Full Brain HUD with Dashboard, Brain View, and Training Monitor
#
# A complete GTK4/libadwaita application for monitoring T-FAN brain state in real time.
# Styled to match the neon cockpit aesthetic.
#
# Requirements:
#   - GTK 4, libadwaita, PyGObject
#   sudo dnf install gtk4-devel libadwaita-devel python3-gobject
#
# Usage:
#   python hud/tfan_gnome.py

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
from gi.repository import Adw, Gdk, Gio, GLib, Gtk, Pango


# ============================================================================
#  Constants
# ============================================================================

APP_ID = "org.ara.tfan.cockpit"
APP_TITLE = "T-FAN Cockpit"

# Metrics file written by TFANHudMetricsClient
RUNTIME_DIR = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
METRICS_FILE = Path(RUNTIME_DIR) / "tfan_hud_metrics.json"

# Poll interval in milliseconds
POLL_INTERVAL_MS = 500  # 2 Hz brain update


# ============================================================================
#  Cockpit CSS - Neon HUD Theme
# ============================================================================

COCKPIT_CSS = """
/* === Base Theme === */
window {
    background-color: #0a0a12;
}

/* === Metric Cards === */
.metric-card {
    background: linear-gradient(135deg, rgba(20, 20, 35, 0.95) 0%, rgba(15, 15, 25, 0.98) 100%);
    border: 1px solid rgba(0, 255, 170, 0.3);
    border-radius: 12px;
    padding: 16px;
    box-shadow:
        0 0 20px rgba(0, 255, 170, 0.1),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
    transition: all 200ms ease;
}

.metric-card:hover {
    border-color: rgba(0, 255, 170, 0.6);
    box-shadow:
        0 0 30px rgba(0, 255, 170, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

/* === Metric Values === */
.metric-value-huge {
    font-size: 42px;
    font-weight: 800;
    font-family: "JetBrains Mono", "SF Mono", monospace;
    color: #00ffaa;
    text-shadow: 0 0 20px rgba(0, 255, 170, 0.5);
}

.metric-value-large {
    font-size: 28px;
    font-weight: 700;
    font-family: "JetBrains Mono", "SF Mono", monospace;
    color: #00ffaa;
    text-shadow: 0 0 15px rgba(0, 255, 170, 0.4);
}

.metric-value-medium {
    font-size: 20px;
    font-weight: 600;
    font-family: "JetBrains Mono", "SF Mono", monospace;
    color: #88ddff;
    text-shadow: 0 0 10px rgba(136, 221, 255, 0.4);
}

/* === Labels === */
.metric-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.5);
    margin-top: 8px;
}

.metric-label-bright {
    color: rgba(0, 255, 170, 0.7);
}

/* === Brain Metric Cards === */
.brain-metric-card {
    background: linear-gradient(135deg, rgba(25, 15, 35, 0.95) 0%, rgba(15, 10, 25, 0.98) 100%);
    border: 1px solid rgba(170, 100, 255, 0.3);
    border-radius: 10px;
    padding: 12px;
    min-width: 140px;
}

.brain-metric-card:hover {
    border-color: rgba(170, 100, 255, 0.6);
    box-shadow: 0 0 25px rgba(170, 100, 255, 0.2);
}

.brain-value {
    font-size: 24px;
    font-weight: 700;
    font-family: "JetBrains Mono", monospace;
    color: #aa66ff;
    text-shadow: 0 0 12px rgba(170, 100, 255, 0.5);
}

/* === Sidebar === */
.sidebar {
    background-color: rgba(10, 10, 18, 0.95);
    border-right: 1px solid rgba(0, 255, 170, 0.15);
}

.sidebar-row {
    padding: 12px 16px;
    border-radius: 8px;
    margin: 4px 8px;
    transition: all 150ms ease;
}

.sidebar-row:hover {
    background-color: rgba(0, 255, 170, 0.1);
}

.sidebar-row:selected {
    background-color: rgba(0, 255, 170, 0.2);
    border-left: 3px solid #00ffaa;
}

/* === Status Indicators === */
.status-ok {
    color: #00ff88;
    text-shadow: 0 0 8px rgba(0, 255, 136, 0.5);
}

.status-warn {
    color: #ffaa00;
    text-shadow: 0 0 8px rgba(255, 170, 0, 0.5);
}

.status-error {
    color: #ff4466;
    text-shadow: 0 0 8px rgba(255, 68, 102, 0.5);
}

/* === Progress Bars === */
progressbar trough {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    min-height: 8px;
}

progressbar progress {
    background: linear-gradient(90deg, #00ffaa 0%, #00ddff 100%);
    border-radius: 4px;
    box-shadow: 0 0 10px rgba(0, 255, 170, 0.4);
}

/* === NCE Action List === */
.nce-action-row {
    background-color: rgba(20, 20, 35, 0.8);
    border-left: 3px solid #00ffaa;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 0 8px 8px 0;
}

.nce-action-row.executed {
    border-left-color: #00ff88;
}

.nce-action-row.skipped {
    border-left-color: #666;
    opacity: 0.6;
}

/* === Header === */
.header-title {
    font-size: 18px;
    font-weight: 700;
    color: #00ffaa;
    letter-spacing: 2px;
}

/* === Scanline Overlay Effect === */
.scanlines {
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 0, 0, 0.1) 2px,
        rgba(0, 0, 0, 0.1) 4px
    );
    pointer-events: none;
}

/* === Drive-Based Color Coding (Homeostatic State) === */
.drive-low {
    border-color: rgba(0, 200, 120, 0.8) !important;
    box-shadow: 0 0 16px rgba(0, 200, 120, 0.3);
}

.drive-med {
    border-color: rgba(255, 200, 0, 0.8) !important;
    box-shadow: 0 0 16px rgba(255, 200, 0, 0.3);
}

.drive-high {
    border-color: rgba(255, 80, 80, 0.9) !important;
    box-shadow: 0 0 20px rgba(255, 80, 80, 0.4);
}

/* === Homeostatic Need Bars === */
.need-bar-satisfied progressbar progress {
    background: linear-gradient(90deg, #00ff88 0%, #00ddaa 100%);
}

.need-bar-moderate progressbar progress {
    background: linear-gradient(90deg, #ffcc00 0%, #ffaa00 100%);
}

.need-bar-depleted progressbar progress {
    background: linear-gradient(90deg, #ff6644 0%, #ff4444 100%);
}

/* === Valence Indicator === */
.valence-positive {
    color: #00ff88 !important;
    text-shadow: 0 0 12px rgba(0, 255, 136, 0.6);
}

.valence-negative {
    color: #ff6644 !important;
    text-shadow: 0 0 12px rgba(255, 102, 68, 0.6);
}

.valence-neutral {
    color: #88ddff !important;
}

/* === Homeostatic Core Section === */
.homeo-section {
    background: linear-gradient(135deg, rgba(20, 25, 35, 0.95) 0%, rgba(15, 18, 28, 0.98) 100%);
    border: 1px solid rgba(0, 200, 170, 0.25);
    border-radius: 12px;
    padding: 16px;
    margin-top: 12px;
}

.homeo-section-title {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(0, 255, 200, 0.8);
    margin-bottom: 12px;
}

/* === Layer 2: Appraisal Section === */
.appraisal-section {
    background: linear-gradient(135deg, rgba(35, 20, 45, 0.95) 0%, rgba(25, 15, 35, 0.98) 100%);
    border: 1px solid rgba(200, 100, 255, 0.25);
    border-radius: 12px;
    padding: 16px;
    margin-top: 12px;
}

.appraisal-section-title {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(200, 100, 255, 0.8);
    margin-bottom: 12px;
}

.emotion-label {
    font-size: 32px;
    font-weight: 700;
    font-family: "JetBrains Mono", monospace;
    color: #cc66ff;
    text-shadow: 0 0 20px rgba(200, 100, 255, 0.5);
    text-transform: capitalize;
}

/* === Layer 3: Uncertainty Section === */
.uncertainty-section {
    background: linear-gradient(135deg, rgba(20, 35, 45, 0.95) 0%, rgba(15, 25, 35, 0.98) 100%);
    border: 1px solid rgba(100, 200, 255, 0.25);
    border-radius: 12px;
    padding: 16px;
    margin-top: 12px;
}

.uncertainty-section-title {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(100, 200, 255, 0.8);
    margin-bottom: 12px;
}

.explore-mode {
    color: #00ddff !important;
    text-shadow: 0 0 12px rgba(0, 220, 255, 0.6);
}

.exploit-mode {
    color: #ffaa00 !important;
    text-shadow: 0 0 12px rgba(255, 170, 0, 0.6);
}

.defer-active {
    color: #ff6644 !important;
    text-shadow: 0 0 12px rgba(255, 102, 68, 0.6);
}

/* === TGSFN: Antifragility Section === */
.antifragility-section {
    background: linear-gradient(135deg, rgba(40, 25, 20, 0.95) 0%, rgba(30, 18, 15, 0.98) 100%);
    border: 1px solid rgba(255, 150, 50, 0.25);
    border-radius: 12px;
    padding: 16px;
    margin-top: 12px;
}

.antifragility-section-title {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(255, 180, 100, 0.8);
    margin-bottom: 12px;
}

.antifragile-positive {
    color: #00ff88 !important;
    text-shadow: 0 0 12px rgba(0, 255, 136, 0.6);
}

.antifragile-neutral {
    color: #88ddff !important;
}

.antifragile-fragile {
    color: #ff4466 !important;
    text-shadow: 0 0 12px rgba(255, 68, 102, 0.6);
}

.dau-active {
    border-color: rgba(255, 100, 100, 0.8) !important;
    box-shadow: 0 0 25px rgba(255, 100, 100, 0.4);
    animation: dau-pulse 1s ease-in-out infinite;
}

@keyframes dau-pulse {
    0%, 100% { box-shadow: 0 0 25px rgba(255, 100, 100, 0.4); }
    50% { box-shadow: 0 0 35px rgba(255, 100, 100, 0.6); }
}
"""


# ============================================================================
#  Metric Card Widget
# ============================================================================

class MetricCard(Gtk.Box):
    """A styled card showing a single metric with icon, value, and label."""

    def __init__(
        self,
        label: str,
        value: str = "—",
        icon_name: str = "utilities-system-monitor-symbolic",
        card_class: str = "metric-card",
        value_class: str = "metric-value-large",
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.set_halign(Gtk.Align.CENTER)
        self.set_valign(Gtk.Align.CENTER)

        self.add_css_class(card_class)

        # Icon
        if icon_name:
            icon = Gtk.Image.new_from_icon_name(icon_name)
            icon.set_pixel_size(28)
            icon.set_opacity(0.5)
            self.append(icon)

        # Value
        self.value_label = Gtk.Label(label=value)
        self.value_label.add_css_class(value_class)
        self.append(self.value_label)

        # Label
        name_label = Gtk.Label(label=label)
        name_label.add_css_class("metric-label")
        self.append(name_label)

    def set_value(self, value: str) -> None:
        self.value_label.set_label(value)


# ============================================================================
#  Brain Metric Progress Bar
# ============================================================================

class BrainMetricBar(Gtk.Box):
    """A labeled progress bar for brain metrics with neon styling."""

    def __init__(
        self,
        label: str,
        min_val: float = 0.0,
        max_val: float = 1.0,
        format_str: str = "{:.4f}",
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.set_margin_top(6)
        self.set_margin_bottom(6)

        self.min_val = min_val
        self.max_val = max_val
        self.format_str = format_str

        # Header row
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)

        self.label_widget = Gtk.Label(label=label)
        self.label_widget.add_css_class("metric-label")
        self.label_widget.add_css_class("metric-label-bright")
        self.label_widget.set_halign(Gtk.Align.START)
        self.label_widget.set_hexpand(True)
        header.append(self.label_widget)

        self.value_label = Gtk.Label(label="—")
        self.value_label.add_css_class("metric-value-medium")
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
        self.value_label.set_label(self.format_str.format(value))


# ============================================================================
#  NCE Actions List
# ============================================================================

class NCEActionsList(Gtk.Box):
    """Shows recent NCE niche construction actions with cockpit styling."""

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.set_margin_top(12)

        # Header
        header = Gtk.Label(label="NCE ACTIONS")
        header.add_css_class("metric-label")
        header.add_css_class("metric-label-bright")
        header.set_halign(Gtk.Align.START)
        self.append(header)

        # Container for action rows
        self.actions_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.append(self.actions_box)

        # Placeholder
        self.placeholder = Gtk.Label(label="No actions yet")
        self.placeholder.set_opacity(0.4)
        self.actions_box.append(self.placeholder)

    def update_actions(self, actions: list) -> None:
        # Clear existing
        while True:
            child = self.actions_box.get_first_child()
            if child is None:
                break
            self.actions_box.remove(child)

        if not actions:
            self.placeholder = Gtk.Label(label="No actions yet")
            self.placeholder.set_opacity(0.4)
            self.actions_box.append(self.placeholder)
            return

        # Add recent actions (most recent first, limit to 5)
        for action in reversed(actions[-5:]):
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            row.add_css_class("nce-action-row")

            executed = action.get("executed", False)
            if executed:
                row.add_css_class("executed")
            else:
                row.add_css_class("skipped")

            # Action name
            name = Gtk.Label(label=action.get("action_type", "unknown"))
            name.set_halign(Gtk.Align.START)
            name.set_hexpand(True)
            row.append(name)

            # Benefit/Cost
            benefit = action.get("benefit", 0)
            cost = action.get("cost_ext", 0)
            stats = Gtk.Label(label=f"B:{benefit:.0f} C:{cost:.0f}")
            stats.set_opacity(0.7)
            row.append(stats)

            # Status icon
            icon = Gtk.Label(label="✓" if executed else "—")
            icon.add_css_class("status-ok" if executed else "")
            row.append(icon)

            self.actions_box.append(row)


# ============================================================================
#  Main Window
# ============================================================================

class TFANWindow(Adw.ApplicationWindow):
    """Main T-FAN Cockpit window with sidebar navigation and multiple views."""

    def __init__(self, app: Adw.Application):
        super().__init__(application=app, title=APP_TITLE)
        self.set_default_size(1200, 800)

        # Load CSS
        self._load_css()

        # Metrics storage
        self.metric_cards: Dict[str, Gtk.Label] = {}
        self.brain_metric_cards: Dict[str, Gtk.Label] = {}
        self.brain_bars: Dict[str, BrainMetricBar] = {}

        # Build UI
        self._build_ui()

        # Start monitoring
        self.metrics_file = METRICS_FILE
        self._start_monitoring()

    def _load_css(self) -> None:
        """Load the cockpit CSS theme."""
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(COCKPIT_CSS.encode())

        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def _build_ui(self) -> None:
        """Build the main UI structure."""
        # Main horizontal split
        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)

        # Sidebar
        sidebar = self._build_sidebar()
        main_box.append(sidebar)

        # Content stack
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        content_box.set_hexpand(True)

        # Header bar
        header = Adw.HeaderBar()
        title_label = Gtk.Label(label="T-FAN COCKPIT")
        title_label.add_css_class("header-title")
        header.set_title_widget(title_label)
        content_box.append(header)

        # Stack for views
        self.content_stack = Gtk.Stack()
        self.content_stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        self.content_stack.set_transition_duration(200)
        self.content_stack.set_vexpand(True)

        # Add views
        self.content_stack.add_titled(self._build_dashboard_view(), "dashboard", "Dashboard")
        self.content_stack.add_titled(self._build_brain_view(), "brain", "Brain HUD")
        self.content_stack.add_titled(self._build_training_view(), "training", "Training")
        self.content_stack.add_titled(self._build_config_view(), "config", "Config")

        content_box.append(self.content_stack)
        main_box.append(content_box)

        self.set_content(main_box)

    def _build_sidebar(self) -> Gtk.Box:
        """Build the sidebar navigation."""
        sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        sidebar.add_css_class("sidebar")
        sidebar.set_size_request(200, -1)

        # Logo / Title area
        logo_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        logo_box.set_margin_top(20)
        logo_box.set_margin_bottom(20)
        logo_box.set_margin_start(16)
        logo_box.set_margin_end(16)

        logo_label = Gtk.Label(label="🧠 ARA")
        logo_label.add_css_class("metric-value-large")
        logo_box.append(logo_label)

        subtitle = Gtk.Label(label="T-FAN Monitor")
        subtitle.add_css_class("metric-label")
        logo_box.append(subtitle)

        sidebar.append(logo_box)
        sidebar.append(Gtk.Separator())

        # Navigation items
        nav_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        nav_box.set_margin_top(12)
        nav_box.set_margin_start(8)
        nav_box.set_margin_end(8)

        views = [
            ("📊", "Dashboard", "dashboard"),
            ("🧠", "Brain HUD", "brain"),
            ("🚀", "Training", "training"),
            ("⚙️", "Config", "config"),
        ]

        for icon, label, view_name in views:
            row = Gtk.Button()
            row.add_css_class("sidebar-row")
            row.set_can_focus(False)

            row_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)

            icon_label = Gtk.Label(label=icon)
            row_box.append(icon_label)

            text_label = Gtk.Label(label=label)
            text_label.set_halign(Gtk.Align.START)
            text_label.set_hexpand(True)
            row_box.append(text_label)

            row.set_child(row_box)
            row.connect("clicked", lambda btn, v=view_name: self._switch_view(v))

            nav_box.append(row)

        sidebar.append(nav_box)

        # Status at bottom
        sidebar.append(Gtk.Box(vexpand=True))  # Spacer

        status_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        status_box.set_margin_bottom(16)
        status_box.set_margin_start(16)
        status_box.set_margin_end(16)

        self.connection_status = Gtk.Label(label="● Disconnected")
        self.connection_status.add_css_class("status-warn")
        self.connection_status.set_halign(Gtk.Align.START)
        status_box.append(self.connection_status)

        self.last_update_label = Gtk.Label(label="Last: —")
        self.last_update_label.set_opacity(0.5)
        self.last_update_label.set_halign(Gtk.Align.START)
        status_box.append(self.last_update_label)

        sidebar.append(status_box)

        return sidebar

    def _switch_view(self, view_name: str) -> None:
        """Switch to a different view."""
        self.content_stack.set_visible_child_name(view_name)

    def _build_dashboard_view(self) -> Gtk.Widget:
        """Build the main dashboard view with key metrics."""
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=20)
        box.set_margin_start(24)
        box.set_margin_end(24)
        box.set_margin_top(24)
        box.set_margin_bottom(24)

        # Status header
        status = Adw.StatusPage()
        status.set_icon_name("utilities-system-monitor-symbolic")
        status.set_title("System Overview")
        status.set_description("Real-time T-FAN training and inference metrics")
        box.append(status)

        # Top-level metric cards grid
        grid = Gtk.Grid()
        grid.set_column_spacing(16)
        grid.set_row_spacing(16)
        grid.set_column_homogeneous(True)
        grid.set_halign(Gtk.Align.CENTER)

        metrics = [
            ("step", "STEP", "—", "view-continuous-symbolic"),
            ("loss", "LOSS", "—", "dialog-error-symbolic"),
            ("accuracy", "ACCURACY", "—", "emblem-ok-symbolic"),
            ("utcf", "UTCF", "—", "utilities-system-monitor-symbolic"),
        ]

        for idx, (key, label, initial, icon) in enumerate(metrics):
            card = MetricCard(label, initial, icon, "metric-card", "metric-value-huge")
            self.metric_cards[key] = card.value_label
            grid.attach(card, idx % 4, idx // 4, 1, 1)

        box.append(grid)

        # Separator
        box.append(Gtk.Separator())

        # Quick brain summary
        brain_label = Gtk.Label(label="BRAIN STATE SUMMARY")
        brain_label.add_css_class("metric-label")
        brain_label.add_css_class("metric-label-bright")
        brain_label.set_halign(Gtk.Align.START)
        box.append(brain_label)

        brain_grid = Gtk.Grid()
        brain_grid.set_column_spacing(12)
        brain_grid.set_row_spacing(12)
        brain_grid.set_column_homogeneous(True)

        brain_metrics = [
            ("sigma_proxy", "σ ENTROPY"),
            ("epsilon_proxy", "ε DISSIPATION"),
            ("kappa_proxy", "κ CURVATURE"),
            ("L_topo", "L_TOPO"),
        ]

        for idx, (key, label) in enumerate(brain_metrics):
            card = MetricCard(label, "—", None, "brain-metric-card", "brain-value")
            self.metric_cards[key] = card.value_label
            brain_grid.attach(card, idx, 0, 1, 1)

        box.append(brain_grid)

        scroll.set_child(box)
        return scroll

    def _build_brain_view(self) -> Gtk.Widget:
        """Build the detailed Brain HUD view."""
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        box.set_margin_start(24)
        box.set_margin_end(24)
        box.set_margin_top(24)
        box.set_margin_bottom(24)

        # Header
        status = Adw.StatusPage()
        status.set_icon_name("face-smile-big-symbolic")
        status.set_title("T-FAN Brain HUD")
        status.set_description(
            "Live view of TFF topology, UDK thermodynamics, and NCE state.\n"
            "Think of this as her cortical EEG."
        )
        box.append(status)

        # === Layer 1: Homeostatic Core (The "Vulnerable Body") ===
        homeo_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        homeo_section.add_css_class("homeo-section")
        self.homeo_section = homeo_section  # Store for drive-based styling

        homeo_title = Gtk.Label(label="LAYER 1: HOMEOSTATIC CORE")
        homeo_title.add_css_class("homeo-section-title")
        homeo_title.set_halign(Gtk.Align.START)
        homeo_section.append(homeo_title)

        # Drive indicator (total homeostatic error)
        drive_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        drive_box.set_margin_bottom(8)

        drive_card = MetricCard("DRIVE D(t)", "—", None, "brain-metric-card", "brain-value")
        self.brain_metric_cards["drive_total"] = drive_card.value_label
        self.drive_card = drive_card  # Store for styling
        drive_box.append(drive_card)

        valence_card = MetricCard("VALENCE", "—", None, "brain-metric-card", "brain-value")
        self.brain_metric_cards["valence"] = valence_card.value_label
        self.valence_card = valence_card  # Store for styling
        drive_box.append(valence_card)

        homeo_section.append(drive_box)

        # Homeostatic needs bars
        self.brain_bars["n_energy"] = BrainMetricBar(
            "Energy (computational resources)", 0.0, 1.0, "{:.2f}"
        )
        homeo_section.append(self.brain_bars["n_energy"])

        self.brain_bars["n_integrity"] = BrainMetricBar(
            "Integrity (model coherence)", 0.0, 1.0, "{:.2f}"
        )
        homeo_section.append(self.brain_bars["n_integrity"])

        self.brain_bars["n_cogload"] = BrainMetricBar(
            "Cognitive Load (lower = more capacity)", 0.0, 1.0, "{:.2f}"
        )
        homeo_section.append(self.brain_bars["n_cogload"])

        self.brain_bars["n_social"] = BrainMetricBar(
            "Social Alignment", 0.0, 1.0, "{:.2f}"
        )
        homeo_section.append(self.brain_bars["n_social"])

        self.brain_bars["n_novelty"] = BrainMetricBar(
            "Novelty / Curiosity", 0.0, 1.0, "{:.2f}"
        )
        homeo_section.append(self.brain_bars["n_novelty"])

        self.brain_bars["n_safety"] = BrainMetricBar(
            "Safety Margin", 0.0, 1.0, "{:.2f}"
        )
        homeo_section.append(self.brain_bars["n_safety"])

        box.append(homeo_section)
        box.append(Gtk.Separator())

        # === UDK Proxies Section ===
        udk_label = Gtk.Label(label="LAYER 3: UDK THERMODYNAMIC PROXIES")
        udk_label.add_css_class("metric-label")
        udk_label.add_css_class("metric-label-bright")
        udk_label.set_halign(Gtk.Align.START)
        box.append(udk_label)

        # Sigma (entropy production)
        self.brain_bars["sigma_proxy"] = BrainMetricBar(
            "σ — Entropy Production (belief revision cost)", 0.0, 2.0
        )
        box.append(self.brain_bars["sigma_proxy"])

        # Epsilon (dissipation)
        self.brain_bars["epsilon_proxy"] = BrainMetricBar(
            "ε — Dissipation Rate (weight turbulence)", 0.0, 0.5
        )
        box.append(self.brain_bars["epsilon_proxy"])

        # Kappa (curvature)
        self.brain_bars["kappa_proxy"] = BrainMetricBar(
            "κ — Manifold Curvature (FIM geometry)", 0.0, 1.0
        )
        box.append(self.brain_bars["kappa_proxy"])

        box.append(Gtk.Separator())

        # === TFF Topology Section ===
        tff_label = Gtk.Label(label="LAYER 2: TFF TOPOLOGICAL STATE")
        tff_label.add_css_class("metric-label")
        tff_label.add_css_class("metric-label-bright")
        tff_label.set_halign(Gtk.Align.START)
        box.append(tff_label)

        # L_topo
        self.brain_bars["L_topo"] = BrainMetricBar(
            "L_topo — Betti Instability Penalty", 0.0, 1.0
        )
        box.append(self.brain_bars["L_topo"])

        # Betti numbers grid
        betti_grid = Gtk.Grid()
        betti_grid.set_column_spacing(16)
        betti_grid.set_row_spacing(8)
        betti_grid.set_column_homogeneous(True)

        beta0_card = MetricCard("β₀ COMPONENTS", "—", None, "brain-metric-card", "brain-value")
        self.brain_metric_cards["beta0"] = beta0_card.value_label
        betti_grid.attach(beta0_card, 0, 0, 1, 1)

        beta1_card = MetricCard("β₁ LOOPS", "—", None, "brain-metric-card", "brain-value")
        self.brain_metric_cards["beta1"] = beta1_card.value_label
        betti_grid.attach(beta1_card, 1, 0, 1, 1)

        box.append(betti_grid)

        box.append(Gtk.Separator())

        # === UTCF Section ===
        utcf_label = Gtk.Label(label="UNIFIED COST FUNCTION")
        utcf_label.add_css_class("metric-label")
        utcf_label.add_css_class("metric-label-bright")
        utcf_label.set_halign(Gtk.Align.START)
        box.append(utcf_label)

        self.brain_bars["utcf"] = BrainMetricBar(
            "UTCF = α_σ·σ + α_ε·ε + α_topo·L + α_κ·κ", 0.0, 5.0
        )
        box.append(self.brain_bars["utcf"])

        self.brain_bars["lambda_topo"] = BrainMetricBar(
            "λ_topo — Dynamic Regularization Weight", 0.0, 0.5
        )
        box.append(self.brain_bars["lambda_topo"])

        box.append(Gtk.Separator())

        # === NCE Actions Section ===
        self.nce_list = NCEActionsList()
        box.append(self.nce_list)

        box.append(Gtk.Separator())

        # === Layer 2: Cognitive Appraisal (EMA/CoRE) ===
        appraisal_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        appraisal_section.add_css_class("appraisal-section")

        appraisal_title = Gtk.Label(label="LAYER 2: COGNITIVE APPRAISAL")
        appraisal_title.add_css_class("appraisal-section-title")
        appraisal_title.set_halign(Gtk.Align.START)
        appraisal_section.append(appraisal_title)

        # Emotion label and confidence
        emotion_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        emotion_box.set_margin_bottom(8)

        self.emotion_label = Gtk.Label(label="neutral")
        self.emotion_label.add_css_class("emotion-label")
        emotion_box.append(self.emotion_label)

        emotion_conf_card = MetricCard("CONFIDENCE", "—", None, "brain-metric-card", "brain-value")
        self.brain_metric_cards["emotion_confidence"] = emotion_conf_card.value_label
        emotion_box.append(emotion_conf_card)

        appraisal_section.append(emotion_box)

        # Appraisal dimension bars
        self.brain_bars["appraisal_pleasantness"] = BrainMetricBar(
            "Pleasantness", -1.0, 1.0, "{:+.2f}"
        )
        appraisal_section.append(self.brain_bars["appraisal_pleasantness"])

        self.brain_bars["appraisal_goal_congruence"] = BrainMetricBar(
            "Goal Congruence", -1.0, 1.0, "{:+.2f}"
        )
        appraisal_section.append(self.brain_bars["appraisal_goal_congruence"])

        self.brain_bars["appraisal_control"] = BrainMetricBar(
            "Control", -1.0, 1.0, "{:+.2f}"
        )
        appraisal_section.append(self.brain_bars["appraisal_control"])

        self.brain_bars["appraisal_certainty"] = BrainMetricBar(
            "Certainty", -1.0, 1.0, "{:+.2f}"
        )
        appraisal_section.append(self.brain_bars["appraisal_certainty"])

        self.brain_bars["appraisal_coping_potential"] = BrainMetricBar(
            "Coping Potential", -1.0, 1.0, "{:+.2f}"
        )
        appraisal_section.append(self.brain_bars["appraisal_coping_potential"])

        # Explanation text
        self.appraisal_explanation = Gtk.Label(label="")
        self.appraisal_explanation.set_halign(Gtk.Align.START)
        self.appraisal_explanation.set_wrap(True)
        self.appraisal_explanation.set_opacity(0.7)
        self.appraisal_explanation.set_margin_top(8)
        appraisal_section.append(self.appraisal_explanation)

        box.append(appraisal_section)

        box.append(Gtk.Separator())

        # === Layer 3: Uncertainty Decomposition ===
        uncertainty_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        uncertainty_section.add_css_class("uncertainty-section")
        self.uncertainty_section = uncertainty_section

        uncertainty_title = Gtk.Label(label="LAYER 3: UNCERTAINTY DECOMPOSITION")
        uncertainty_title.add_css_class("uncertainty-section-title")
        uncertainty_title.set_halign(Gtk.Align.START)
        uncertainty_section.append(uncertainty_title)

        # Explore vs Exploit indicator
        policy_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        policy_box.set_margin_bottom(8)

        explore_card = MetricCard("EXPLORE ↔ EXPLOIT", "—", None, "brain-metric-card", "brain-value")
        self.brain_metric_cards["explore_vs_exploit"] = explore_card.value_label
        self.explore_card = explore_card
        policy_box.append(explore_card)

        defer_card = MetricCard("DEFER", "No", None, "brain-metric-card", "brain-value")
        self.brain_metric_cards["should_defer"] = defer_card.value_label
        self.defer_card = defer_card
        policy_box.append(defer_card)

        uncertainty_section.append(policy_box)

        # Uncertainty bars
        self.brain_bars["uncertainty_total"] = BrainMetricBar(
            "Total Uncertainty", 0.0, 1.0, "{:.3f}"
        )
        uncertainty_section.append(self.brain_bars["uncertainty_total"])

        self.brain_bars["uncertainty_aleatoric"] = BrainMetricBar(
            "Aleatoric (irreducible)", 0.0, 1.0, "{:.3f}"
        )
        uncertainty_section.append(self.brain_bars["uncertainty_aleatoric"])

        self.brain_bars["uncertainty_epistemic"] = BrainMetricBar(
            "Epistemic (reducible)", 0.0, 1.0, "{:.3f}"
        )
        uncertainty_section.append(self.brain_bars["uncertainty_epistemic"])

        self.brain_bars["attention_gain"] = BrainMetricBar(
            "LC Attention Gain", 1.0, 2.5, "{:.2f}"
        )
        uncertainty_section.append(self.brain_bars["attention_gain"])

        # Defer reason
        self.defer_reason_label = Gtk.Label(label="")
        self.defer_reason_label.set_halign(Gtk.Align.START)
        self.defer_reason_label.set_wrap(True)
        self.defer_reason_label.set_opacity(0.7)
        self.defer_reason_label.set_margin_top(8)
        uncertainty_section.append(self.defer_reason_label)

        box.append(uncertainty_section)

        box.append(Gtk.Separator())

        # === TGSFN: Antifragility Metrics ===
        antifragility_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        antifragility_section.add_css_class("antifragility-section")
        self.antifragility_section = antifragility_section

        antifragility_title = Gtk.Label(label="TGSFN: ANTIFRAGILITY MONITOR")
        antifragility_title.add_css_class("antifragility-section-title")
        antifragility_title.set_halign(Gtk.Align.START)
        antifragility_section.append(antifragility_title)

        # Antifragility index and DAU status
        af_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        af_box.set_margin_bottom(8)

        af_index_card = MetricCard("ANTIFRAGILITY", "—", None, "brain-metric-card", "brain-value")
        self.brain_metric_cards["antifragility_index"] = af_index_card.value_label
        self.af_index_card = af_index_card
        af_box.append(af_index_card)

        stability_card = MetricCard("STABILITY", "—", None, "brain-metric-card", "brain-value")
        self.brain_metric_cards["stability_margin"] = stability_card.value_label
        af_box.append(stability_card)

        dau_card = MetricCard("DAU", "—", None, "brain-metric-card", "brain-value")
        self.brain_metric_cards["dau_active"] = dau_card.value_label
        self.dau_card = dau_card
        af_box.append(dau_card)

        antifragility_section.append(af_box)

        # Antifragility metrics bars
        self.brain_bars["pi_q"] = BrainMetricBar(
            "Π_q — Entropy Production Rate", 0.0, 1.0, "{:.4f}"
        )
        antifragility_section.append(self.brain_bars["pi_q"])

        self.brain_bars["convexity"] = BrainMetricBar(
            "Convexity (payoff curvature)", -1.0, 1.0, "{:+.3f}"
        )
        antifragility_section.append(self.brain_bars["convexity"])

        self.brain_bars["jacobian_spectral_norm"] = BrainMetricBar(
            "Jacobian Spectral Norm", 0.0, 10.0, "{:.3f}"
        )
        antifragility_section.append(self.brain_bars["jacobian_spectral_norm"])

        self.brain_bars["lyapunov_proxy"] = BrainMetricBar(
            "Lyapunov Exponent Proxy", -1.0, 1.0, "{:+.4f}"
        )
        antifragility_section.append(self.brain_bars["lyapunov_proxy"])

        self.brain_bars["fragility_exposure"] = BrainMetricBar(
            "Fragility Exposure (downside risk)", 0.0, 1.0, "{:.3f}"
        )
        antifragility_section.append(self.brain_bars["fragility_exposure"])

        # DAU trigger reason
        self.dau_reason_label = Gtk.Label(label="")
        self.dau_reason_label.set_halign(Gtk.Align.START)
        self.dau_reason_label.set_wrap(True)
        self.dau_reason_label.set_opacity(0.7)
        self.dau_reason_label.set_margin_top(8)
        antifragility_section.append(self.dau_reason_label)

        box.append(antifragility_section)

        scroll.set_child(box)
        return scroll

    def _build_training_view(self) -> Gtk.Widget:
        """Build the training monitor view."""
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        box.set_margin_start(24)
        box.set_margin_end(24)
        box.set_margin_top(24)
        box.set_margin_bottom(24)

        status = Adw.StatusPage()
        status.set_icon_name("media-playback-start-symbolic")
        status.set_title("Training Monitor")
        status.set_description("Real-time training progress and optimizer state")
        box.append(status)

        # Training metrics
        train_grid = Gtk.Grid()
        train_grid.set_column_spacing(16)
        train_grid.set_row_spacing(16)
        train_grid.set_column_homogeneous(True)

        train_metrics = [
            ("epoch", "EPOCH", "—"),
            ("train_step", "STEP", "—"),
            ("train_loss", "LOSS", "—"),
            ("elapsed", "ELAPSED", "—"),
        ]

        for idx, (key, label, initial) in enumerate(train_metrics):
            card = MetricCard(label, initial, None, "metric-card", "metric-value-large")
            self.metric_cards[f"train_{key}"] = card.value_label
            train_grid.attach(card, idx % 4, idx // 4, 1, 1)

        box.append(train_grid)

        # Placeholder for loss chart (future)
        chart_placeholder = Gtk.Box()
        chart_placeholder.set_size_request(-1, 200)
        chart_placeholder.add_css_class("metric-card")

        chart_label = Gtk.Label(label="Loss chart placeholder")
        chart_label.set_opacity(0.3)
        chart_placeholder.append(chart_label)

        box.append(chart_placeholder)

        scroll.set_child(box)
        return scroll

    def _build_config_view(self) -> Gtk.Widget:
        """Build the configuration view."""
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        box.set_margin_start(24)
        box.set_margin_end(24)
        box.set_margin_top(24)
        box.set_margin_bottom(24)

        status = Adw.StatusPage()
        status.set_icon_name("preferences-system-symbolic")
        status.set_title("Configuration")
        status.set_description("T-FAN and HUD settings")
        box.append(status)

        # Metrics file path
        group = Adw.PreferencesGroup()
        group.set_title("Metrics Source")

        file_row = Adw.ActionRow()
        file_row.set_title("Metrics File")
        file_row.set_subtitle(str(METRICS_FILE))
        group.add(file_row)

        poll_row = Adw.ActionRow()
        poll_row.set_title("Poll Interval")
        poll_row.set_subtitle(f"{POLL_INTERVAL_MS}ms")
        group.add(poll_row)

        box.append(group)

        scroll.set_child(box)
        return scroll

    # ========================================================================
    #  Monitoring
    # ========================================================================

    def _start_monitoring(self) -> None:
        """Start periodic polling of T-FAN HUD metrics."""
        GLib.timeout_add(POLL_INTERVAL_MS, self._poll_hud_metrics)

    def _poll_hud_metrics(self) -> bool:
        """Read metrics JSON and update all views."""
        try:
            if not self.metrics_file.exists():
                self.connection_status.set_label("● Waiting for metrics...")
                self.connection_status.remove_css_class("status-ok")
                self.connection_status.add_css_class("status-warn")
                return True

            text = self.metrics_file.read_text(encoding="utf-8")
            data = json.loads(text)
            models = data.get("models", {})

            if models:
                # Take first model entry
                model_id, m = next(iter(models.items()))
                self._update_dashboard_metrics(m)
                self._update_brain_metrics(m)
                self._update_training_metrics(m)

                # Update connection status
                self.connection_status.set_label(f"● Connected ({model_id})")
                self.connection_status.remove_css_class("status-warn")
                self.connection_status.add_css_class("status-ok")

                # Update last update time
                last_update = m.get("last_update", 0)
                if last_update:
                    ago = time.time() - last_update
                    self.last_update_label.set_label(f"Last: {ago:.1f}s ago")

        except Exception as e:
            self.connection_status.set_label(f"● Error")
            self.connection_status.remove_css_class("status-ok")
            self.connection_status.add_css_class("status-error")
            print(f"[TFAN HUD] Error polling metrics: {e}")

        return True  # Continue polling

    def _update_dashboard_metrics(self, m: Dict[str, Any]) -> None:
        """Update the dashboard metric cards."""
        def _set(key: str, fmt: str = "{:.4f}"):
            if key in m and key in self.metric_cards:
                try:
                    val = m[key]
                    if isinstance(val, (int, float)):
                        self.metric_cards[key].set_label(fmt.format(val))
                    else:
                        self.metric_cards[key].set_label(str(val))
                except Exception:
                    pass

        _set("step", "{:.0f}")
        _set("loss", "{:.4f}")
        _set("accuracy", "{:.3f}")
        _set("utcf", "{:.4f}")
        _set("sigma_proxy", "{:.4f}")
        _set("epsilon_proxy", "{:.4f}")
        _set("kappa_proxy", "{:.4f}")
        _set("L_topo", "{:.4f}")

    def _update_brain_metrics(self, m: Dict[str, Any]) -> None:
        """Update the brain HUD bars and cards."""
        # Layer 1: Homeostatic Core needs
        homeo_keys = ["n_energy", "n_integrity", "n_cogload", "n_social", "n_novelty", "n_safety"]
        for key in homeo_keys:
            if key in m and key in self.brain_bars:
                try:
                    self.brain_bars[key].set_value(float(m[key]))
                except Exception:
                    pass

        # Drive and valence cards
        if "drive_total" in m and "drive_total" in self.brain_metric_cards:
            try:
                drive = float(m["drive_total"])
                self.brain_metric_cards["drive_total"].set_label(f"{drive:.3f}")
                # Apply drive-based styling
                self._apply_drive_styling(drive)
            except Exception:
                pass

        if "valence" in m and "valence" in self.brain_metric_cards:
            try:
                valence = float(m["valence"])
                self.brain_metric_cards["valence"].set_label(f"{valence:+.4f}")
                # Apply valence-based styling
                self._apply_valence_styling(valence)
            except Exception:
                pass

        # UDK thermodynamic proxies
        for key in ["sigma_proxy", "epsilon_proxy", "kappa_proxy", "L_topo", "utcf", "lambda_topo"]:
            if key in m and key in self.brain_bars:
                try:
                    self.brain_bars[key].set_value(float(m[key]))
                except Exception:
                    pass

        # Betti cards
        for key in ["beta0", "beta1"]:
            if key in m and key in self.brain_metric_cards:
                try:
                    self.brain_metric_cards[key].set_label(f"{float(m[key]):.1f}")
                except Exception:
                    pass

        # NCE actions
        nce_actions = m.get("nce_actions", [])
        if nce_actions:
            self.nce_list.update_actions(nce_actions)

        # === Layer 2: Cognitive Appraisal ===
        if "emotion_label" in m and hasattr(self, "emotion_label"):
            try:
                self.emotion_label.set_label(m["emotion_label"])
            except Exception:
                pass

        if "emotion_confidence" in m and "emotion_confidence" in self.brain_metric_cards:
            try:
                self.brain_metric_cards["emotion_confidence"].set_label(f"{float(m['emotion_confidence']):.2f}")
            except Exception:
                pass

        # Appraisal dimensions
        for key in ["appraisal_pleasantness", "appraisal_goal_congruence", "appraisal_control",
                    "appraisal_certainty", "appraisal_coping_potential"]:
            if key in m and key in self.brain_bars:
                try:
                    self.brain_bars[key].set_value(float(m[key]))
                except Exception:
                    pass

        if "appraisal_explanation" in m and hasattr(self, "appraisal_explanation"):
            try:
                self.appraisal_explanation.set_label(m["appraisal_explanation"])
            except Exception:
                pass

        # === Layer 3: Uncertainty Decomposition ===
        if "explore_vs_exploit" in m and "explore_vs_exploit" in self.brain_metric_cards:
            try:
                eve = float(m["explore_vs_exploit"])
                self.brain_metric_cards["explore_vs_exploit"].set_label(f"{eve:+.2f}")
                self._apply_explore_exploit_styling(eve)
            except Exception:
                pass

        if "should_defer" in m and "should_defer" in self.brain_metric_cards:
            try:
                defer = m["should_defer"]
                self.brain_metric_cards["should_defer"].set_label("Yes" if defer else "No")
                self._apply_defer_styling(defer)
            except Exception:
                pass

        for key in ["uncertainty_total", "uncertainty_aleatoric", "uncertainty_epistemic", "attention_gain"]:
            if key in m and key in self.brain_bars:
                try:
                    self.brain_bars[key].set_value(float(m[key]))
                except Exception:
                    pass

        if "defer_reason" in m and hasattr(self, "defer_reason_label"):
            try:
                self.defer_reason_label.set_label(m["defer_reason"])
            except Exception:
                pass

        # === TGSFN: Antifragility ===
        if "antifragility_index" in m and "antifragility_index" in self.brain_metric_cards:
            try:
                af = float(m["antifragility_index"])
                self.brain_metric_cards["antifragility_index"].set_label(f"{af:+.3f}")
                self._apply_antifragility_styling(af)
            except Exception:
                pass

        if "stability_margin" in m and "stability_margin" in self.brain_metric_cards:
            try:
                self.brain_metric_cards["stability_margin"].set_label(f"{float(m['stability_margin']):.3f}")
            except Exception:
                pass

        if "dau_active" in m and "dau_active" in self.brain_metric_cards:
            try:
                dau = m["dau_active"]
                self.brain_metric_cards["dau_active"].set_label("ACTIVE" if dau else "—")
                self._apply_dau_styling(dau)
            except Exception:
                pass

        for key in ["pi_q", "convexity", "jacobian_spectral_norm", "lyapunov_proxy", "fragility_exposure"]:
            if key in m and key in self.brain_bars:
                try:
                    self.brain_bars[key].set_value(float(m[key]))
                except Exception:
                    pass

        if "dau_trigger_reason" in m and hasattr(self, "dau_reason_label"):
            try:
                reason = m["dau_trigger_reason"]
                action = m.get("dau_action_suggested", "")
                if reason:
                    self.dau_reason_label.set_label(f"{reason} → {action}")
                else:
                    self.dau_reason_label.set_label("")
            except Exception:
                pass

    def _apply_drive_styling(self, drive: float) -> None:
        """Apply drive-based color coding to the homeostatic section."""
        if not hasattr(self, "homeo_section"):
            return

        # Remove existing drive classes
        for cls in ["drive-low", "drive-med", "drive-high"]:
            self.homeo_section.remove_css_class(cls)

        # Apply appropriate class based on drive level
        if drive < 0.3:
            self.homeo_section.add_css_class("drive-low")
        elif drive < 0.6:
            self.homeo_section.add_css_class("drive-med")
        else:
            self.homeo_section.add_css_class("drive-high")

    def _apply_valence_styling(self, valence: float) -> None:
        """Apply valence-based color coding to the valence label."""
        if not hasattr(self, "valence_card"):
            return

        label = self.brain_metric_cards.get("valence")
        if not label:
            return

        # Remove existing valence classes
        for cls in ["valence-positive", "valence-negative", "valence-neutral"]:
            label.remove_css_class(cls)

        # Apply appropriate class based on valence
        if valence > 0.001:
            label.add_css_class("valence-positive")
        elif valence < -0.001:
            label.add_css_class("valence-negative")
        else:
            label.add_css_class("valence-neutral")

    def _apply_explore_exploit_styling(self, value: float) -> None:
        """Apply explore/exploit mode styling."""
        if not hasattr(self, "explore_card"):
            return

        label = self.brain_metric_cards.get("explore_vs_exploit")
        if not label:
            return

        for cls in ["explore-mode", "exploit-mode"]:
            label.remove_css_class(cls)

        if value > 0.1:
            label.add_css_class("explore-mode")
        elif value < -0.1:
            label.add_css_class("exploit-mode")

    def _apply_defer_styling(self, defer: bool) -> None:
        """Apply defer mode styling."""
        if not hasattr(self, "defer_card"):
            return

        label = self.brain_metric_cards.get("should_defer")
        if not label:
            return

        if defer:
            label.add_css_class("defer-active")
        else:
            label.remove_css_class("defer-active")

    def _apply_antifragility_styling(self, value: float) -> None:
        """Apply antifragility-based styling."""
        if not hasattr(self, "af_index_card"):
            return

        label = self.brain_metric_cards.get("antifragility_index")
        if not label:
            return

        for cls in ["antifragile-positive", "antifragile-neutral", "antifragile-fragile"]:
            label.remove_css_class(cls)

        if value > 0.2:
            label.add_css_class("antifragile-positive")
        elif value < -0.2:
            label.add_css_class("antifragile-fragile")
        else:
            label.add_css_class("antifragile-neutral")

    def _apply_dau_styling(self, active: bool) -> None:
        """Apply DAU active styling to the antifragility section."""
        if not hasattr(self, "antifragility_section"):
            return

        if active:
            self.antifragility_section.add_css_class("dau-active")
        else:
            self.antifragility_section.remove_css_class("dau-active")

    def _update_training_metrics(self, m: Dict[str, Any]) -> None:
        """Update training-specific metrics."""
        # Map from metric keys to card keys
        mappings = {
            "epoch": ("train_train_epoch", "{:.0f}"),
            "step": ("train_train_step", "{:.0f}"),
            "loss": ("train_train_loss", "{:.4f}"),
            "elapsed_sec": ("train_train_elapsed", None),
        }

        for src_key, (card_key, fmt) in mappings.items():
            if src_key in m and card_key in self.metric_cards:
                try:
                    val = m[src_key]
                    if src_key == "elapsed_sec":
                        # Format elapsed time
                        if val > 3600:
                            text = f"{val/3600:.1f}h"
                        elif val > 60:
                            text = f"{val/60:.1f}m"
                        else:
                            text = f"{val:.0f}s"
                        self.metric_cards[card_key].set_label(text)
                    elif fmt:
                        self.metric_cards[card_key].set_label(fmt.format(val))
                except Exception:
                    pass


# ============================================================================
#  Application
# ============================================================================

class TFANCockpitApp(Adw.Application):
    """Main GTK4/Adwaita application."""

    def __init__(self):
        super().__init__(
            application_id=APP_ID,
            flags=Gio.ApplicationFlags.FLAGS_NONE,
        )
        self.connect("activate", self._on_activate)

    def _on_activate(self, app):
        win = TFANWindow(app)
        win.present()


# ============================================================================
#  Entry Point
# ============================================================================

def main():
    print(f"T-FAN Cockpit")
    print(f"Monitoring: {METRICS_FILE}")
    print(f"Poll interval: {POLL_INTERVAL_MS}ms")
    print()

    app = TFANCockpitApp()
    return app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
