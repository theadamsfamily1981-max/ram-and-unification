# tfan/tfan_telemetry.py
# Canonical brain state telemetry for T-FAN → GNOME HUD pipeline
#
# This module provides:
#   - TfanBrainSnapshot: Full brain state dataclass (L1-L4)
#   - TelemetryExporter: Model-side atomic JSON writer
#   - TelemetryClient: HUD-side reader with staleness detection

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================================
#  Canonical Brain Snapshot (All Layers)
# ============================================================================

@dataclass
class TfanBrainSnapshot:
    """
    Complete T-FAN brain state snapshot for HUD visualization.

    Captures all four layers:
      - L1: Homeostatic Core (needs/drives)
      - L2: TFF Topology (Betti numbers, curvature)
      - L3: UDK Thermodynamics (σ, ε, UTCF)
      - L4: NCE/COS (niche construction actions)
    """

    # === Core bookkeeping ===
    step: int = 0
    epoch: int = 0
    wall_time: float = field(default_factory=time.time)
    phase: str = "train"  # "train", "eval", "inference", "idle"

    # === Utility / Loss ===
    loss_utility: float = 0.0
    loss_topo: float = 0.0
    loss_total: float = 0.0
    accuracy: float = 0.0

    # === Layer 1: Homeostatic Core (Vulnerable Body) ===
    # These are the "needs" that define the agent's intrinsic drives
    # Inspired by HRRL and the Affective Taxis hypothesis
    drive_total: float = 0.0      # D_t = Σ |need_i - setpoint_i|
    n_energy: float = 0.5         # Computational resource availability
    n_integrity: float = 0.5      # Model coherence / no corruption
    n_cogload: float = 0.5        # Cognitive load (inverse = available capacity)
    n_social: float = 0.5         # Social alignment / rapport
    n_novelty: float = 0.5        # Exploration drive
    n_safety: float = 0.5         # Safety margin / risk aversion

    # Homeostatic set-points (what the agent "wants" each need to be)
    setpoint_energy: float = 0.8
    setpoint_integrity: float = 0.9
    setpoint_cogload: float = 0.3  # Low cognitive load is preferred
    setpoint_social: float = 0.7
    setpoint_novelty: float = 0.4
    setpoint_safety: float = 0.8

    # === Layer 2: TFF Topology ===
    beta0: float = 0.0            # β₀: Connected components
    beta1: float = 0.0            # β₁: 1-dimensional holes (loops)
    beta2: float = 0.0            # β₂: 2-dimensional voids (optional)
    kappa_proxy: float = 0.0      # κ: Manifold curvature (FIM max eigenvalue)
    lambda_topo: float = 0.1      # Dynamic topological regularization weight

    # === Layer 3: UDK Thermodynamics ===
    sigma_proxy: float = 0.0      # σ: Entropy production (belief revision cost)
    epsilon_proxy: float = 0.0    # ε: Dissipation rate (weight turbulence)
    utcf: float = 0.0             # Unified Topo-Thermodynamic Cost Function

    # === Layer 4: NCE/COS (Niche Construction) ===
    offload_actions_total: int = 0
    offload_actions_executed: int = 0
    last_offload_action: str = "none"
    last_offload_benefit: float = 0.0
    last_offload_cost: float = 0.0

    # === Affective State (Derived) ===
    # Per Affective Taxis hypothesis:
    #   valence = -dσ/dt (pleasure when entropy decreasing)
    #   arousal = |UTCF| (magnitude of homeostatic error)
    valence: float = 0.0          # Computed: rate of σ change
    arousal: float = 0.0          # Computed: UTCF magnitude

    # === Training Metadata ===
    lr: float = 0.0
    grad_norm: float = 0.0
    batch_size: int = 0

    # === Optional: Homeostatic Coupling (Prosociality) ===
    # If coupled to external agent (user), track their inferred state
    coupled: bool = False
    coupled_agent_drive: float = 0.0
    coupling_weight: float = 0.0

    def compute_drive(self) -> float:
        """
        Compute total homeostatic drive from need-setpoint mismatches.
        D_t = Σ |need_i - setpoint_i|
        """
        drive = 0.0
        drive += abs(self.n_energy - self.setpoint_energy)
        drive += abs(self.n_integrity - self.setpoint_integrity)
        drive += abs(self.n_cogload - self.setpoint_cogload)
        drive += abs(self.n_social - self.setpoint_social)
        drive += abs(self.n_novelty - self.setpoint_novelty)
        drive += abs(self.n_safety - self.setpoint_safety)

        # Add coupled agent's drive if prosociality is enabled
        if self.coupled:
            drive += self.coupling_weight * self.coupled_agent_drive

        self.drive_total = drive
        return drive

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TfanBrainSnapshot":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
#  Telemetry Exporter (Model Side)
# ============================================================================

class TelemetryExporter:
    """
    Model/training-side helper: write the latest T-FAN brain snapshot
    to a JSON file atomically so the HUD can poll it.

    Usage:
        exporter = TelemetryExporter()

        for step in range(num_steps):
            snap = TfanBrainSnapshot(
                step=step,
                sigma_proxy=udk.state.sigma_proxy,
                ...
            )
            exporter.push(snap)
    """

    def __init__(self, path: Optional[str] = None):
        if path is None:
            runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
            path = os.path.join(runtime_dir, "tfan_brain_metrics.json")

        self.path = Path(path)
        self._prev_sigma: Optional[float] = None
        self._prev_time: Optional[float] = None

    def push(self, snap: TfanBrainSnapshot) -> None:
        """
        Atomically write the brain snapshot to disk.
        Also computes derived affective state (valence, arousal).
        """
        # Compute homeostatic drive
        snap.compute_drive()

        # Compute affective state
        snap.arousal = abs(snap.utcf)

        # Valence = -dσ/dt (positive when σ decreasing = "pleasure")
        if self._prev_sigma is not None and self._prev_time is not None:
            dt = snap.wall_time - self._prev_time
            if dt > 0:
                d_sigma = snap.sigma_proxy - self._prev_sigma
                snap.valence = -d_sigma / dt  # Pleasure when σ decreasing

        self._prev_sigma = snap.sigma_proxy
        self._prev_time = snap.wall_time

        # Atomic write
        data = snap.to_dict()
        tmp = self.path.with_suffix(".json.tmp")

        try:
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except Exception as e:
            print(f"[TelemetryExporter] Error writing: {e}")


# ============================================================================
#  Telemetry Client (HUD Side)
# ============================================================================

class TelemetryClient:
    """
    HUD-side helper: read the most recent brain snapshot from disk.

    Usage:
        client = TelemetryClient()

        # In GLib.timeout_add callback:
        data = client.get_latest()
        if data:
            update_labels(data)
    """

    def __init__(
        self,
        path: Optional[str] = None,
        ttl_seconds: float = 30.0,
    ):
        if path is None:
            runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
            path = os.path.join(runtime_dir, "tfan_brain_metrics.json")

        self.path = Path(path)
        self.ttl_seconds = ttl_seconds
        self._last_data: Optional[Dict[str, Any]] = None
        self._last_mtime: float = 0

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """
        Read the latest brain snapshot if available and not stale.

        Returns:
            Dict with brain state, or None if unavailable/stale.
        """
        try:
            if not self.path.exists():
                return None

            # Check if file changed
            mtime = self.path.stat().st_mtime
            if mtime == self._last_mtime and self._last_data:
                return self._last_data

            self._last_mtime = mtime
            text = self.path.read_text(encoding="utf-8")
            data = json.loads(text)

        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

        # Check staleness
        ts = data.get("wall_time", 0.0)
        now = time.time()
        if now - ts > self.ttl_seconds:
            # Data too old, training probably stopped
            data["_stale"] = True

        self._last_data = data
        return data

    def get_snapshot(self) -> Optional[TfanBrainSnapshot]:
        """Get latest as a TfanBrainSnapshot object."""
        data = self.get_latest()
        if data:
            return TfanBrainSnapshot.from_dict(data)
        return None

    def is_connected(self) -> bool:
        """Check if we're receiving fresh data."""
        data = self.get_latest()
        return data is not None and not data.get("_stale", False)


# ============================================================================
#  Convenience Functions
# ============================================================================

_default_exporter: Optional[TelemetryExporter] = None


def get_exporter() -> TelemetryExporter:
    """Get or create a default telemetry exporter."""
    global _default_exporter
    if _default_exporter is None:
        _default_exporter = TelemetryExporter()
    return _default_exporter


def push_brain_state(**kwargs) -> None:
    """
    Quick push using default exporter.

    Usage:
        push_brain_state(
            step=100,
            sigma_proxy=0.1,
            epsilon_proxy=0.01,
            ...
        )
    """
    snap = TfanBrainSnapshot(**kwargs)
    get_exporter().push(snap)
