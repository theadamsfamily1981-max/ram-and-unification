# tfan/telemetry.py
# Telemetry Emitter for T-FAN Training
#
# Writes metrics to ~/.tfan/metrics.json for the GNOME HUD to poll.

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional


class TelemetryEmitter:
    """
    Simple file-based telemetry emitter.

    Writes the most recent metrics into ~/.tfan/metrics.json
    so the GNOME HUD can poll and display them.

    Usage:
        telemetry = TelemetryEmitter()
        telemetry.log_step(step=100, loss=0.5, udk_state=udk.state)
    """

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            home = Path(os.path.expanduser("~"))
            self.metrics_dir = home / ".tfan"
            self.metrics_file = self.metrics_dir / "metrics.json"
        else:
            self.metrics_dir = path.parent
            self.metrics_file = path

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.last_payload: Dict[str, Any] = {}
        self.start_time = time.time()

    def _atomic_write(self, data: Dict[str, Any]) -> None:
        """Write to a temp file then rename for atomic behavior."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self.metrics_dir),
            delete=False,
        ) as tmp:
            json.dump(data, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name

        os.replace(tmp_name, self.metrics_file)

    def log_step(
        self,
        step: int,
        loss: float,
        udk_state: Dict[str, float],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log metrics for a training step.

        Args:
            step: global training step
            loss: current total loss (L_T-FAN)
            udk_state: a dict with sigma_proxy, epsilon_proxy, L_topo, kappa_proxy
            extra: any extra stuff you want (accuracy, PAD, lambda_topo, etc.)
        """
        payload: Dict[str, Any] = {
            "step": step,
            "timestamp": time.time(),
            "elapsed_sec": time.time() - self.start_time,
            "loss": float(loss),
            "udk": {
                "sigma_proxy": float(udk_state.get("sigma_proxy", 0.0)),
                "epsilon_proxy": float(udk_state.get("epsilon_proxy", 0.0)),
                "L_topo": float(udk_state.get("L_topo", 0.0)),
                "kappa_proxy": float(udk_state.get("kappa_proxy", 0.0)),
            },
        }

        if extra:
            payload["extra"] = extra

        self.last_payload = payload
        self._atomic_write(payload)

    def log_affective_state(
        self,
        step: int,
        pad: Dict[str, float],
        drives: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log affective architecture state (PAD + homeostatic drives).

        Args:
            step: global step
            pad: dict with valence, arousal, dominance
            drives: optional dict with homeostatic drive values
        """
        # Merge into existing payload
        payload = self.last_payload.copy()
        payload["step"] = step
        payload["timestamp"] = time.time()
        payload["affective"] = {
            "valence": float(pad.get("valence", 0.0)),
            "arousal": float(pad.get("arousal", 0.0)),
            "dominance": float(pad.get("dominance", 0.0)),
        }

        if drives:
            payload["drives"] = {k: float(v) for k, v in drives.items()}

        self.last_payload = payload
        self._atomic_write(payload)

    def log_nce_action(
        self,
        step: int,
        action_type: str,
        benefit: float,
        cost_ext: float,
        executed: bool,
    ) -> None:
        """
        Log an NCE niche construction action evaluation.

        Args:
            step: global step
            action_type: type of action evaluated
            benefit: computed benefit (delta_sigma * H)
            cost_ext: external cost
            executed: whether action was actually taken
        """
        payload = self.last_payload.copy()
        payload["step"] = step
        payload["timestamp"] = time.time()

        nce_entry = {
            "action_type": action_type,
            "benefit": float(benefit),
            "cost_ext": float(cost_ext),
            "executed": executed,
        }

        if "nce_actions" not in payload:
            payload["nce_actions"] = []
        payload["nce_actions"].append(nce_entry)

        # Keep only last 10 actions in the log
        payload["nce_actions"] = payload["nce_actions"][-10:]

        self.last_payload = payload
        self._atomic_write(payload)

    def clear(self) -> None:
        """Clear the metrics file."""
        if self.metrics_file.exists():
            self.metrics_file.unlink()
        self.last_payload = {}
