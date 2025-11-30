# tfan/tfan_hud_metrics.py
# Lightweight metrics writer for the T-FAN GNOME HUD.
# Writes a JSON blob that the GNOME cockpit polls.

from __future__ import annotations

import json
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional


class TFANHudMetricsClient:
    """
    Lightweight metrics writer for the T-FAN GNOME HUD.
    Writes a JSON blob that the GNOME cockpit polls.

    Usage:

        hud = TFANHudMetricsClient(model_id="tfan_main")

        for step, batch in enumerate(loader):
            ...
            hud.update(
                step=step,
                loss=loss.item(),
                accuracy=acc,
                sigma_proxy=float(udk.state.sigma_proxy),
                epsilon_proxy=float(udk.state.epsilon_proxy),
                kappa_proxy=float(udk.state.kappa_proxy),
                L_topo=float(udk.state.L_topo),
                beta0=float(beta_k[0].item()),
                beta1=float(beta_k[1].item()) if len(beta_k) > 1 else 0.0,
                utcf=float(udk.utcf_metrics_cost()),
            )
    """

    def __init__(
        self,
        model_id: str = "tfan_main",
        path: Optional[str] = None,
        flush_interval: float = 0.5,
    ):
        if path is None:
            runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
            path = os.path.join(runtime_dir, "tfan_hud_metrics.json")

        self.path = Path(path)
        self.model_id = model_id
        self.flush_interval = flush_interval

        self._lock = threading.Lock()
        self._buffer: Dict[str, Any] = {}
        self._stop = False
        self._start_time = time.time()

        self._thread = threading.Thread(target=self._flusher, daemon=True)
        self._thread.start()

    def update(self, **metrics: Any) -> None:
        """
        Update metrics for this model.

        Example:
            hud.update(
                step=100,
                loss=0.12,
                accuracy=0.87,
                sigma_proxy=0.03,
                epsilon_proxy=0.001,
                kappa_proxy=1.2,
                L_topo=0.004,
                beta0=15.0,
                beta1=3.0,
                utcf=0.91,
            )
        """
        with self._lock:
            m = self._buffer.setdefault(self.model_id, {})
            m.update(metrics)
            m["last_update"] = time.time()
            m["elapsed_sec"] = time.time() - self._start_time

    def _flusher(self) -> None:
        """Background thread that periodically writes metrics to disk."""
        while not self._stop:
            time.sleep(self.flush_interval)
            with self._lock:
                if not self._buffer:
                    continue

                tmp_path = self.path.with_suffix(".tmp")
                payload = {
                    "models": self._buffer.copy(),
                    "timestamp": time.time(),
                }

                try:
                    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    tmp_path.replace(self.path)
                except Exception as e:
                    print(f"[TFANHudMetricsClient] Error writing metrics: {e}")

    def close(self) -> None:
        """Stop the background flusher thread."""
        self._stop = True
        self._thread.join(timeout=1.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# Convenience singleton for simple usage
_default_client: Optional[TFANHudMetricsClient] = None


def get_hud_client(model_id: str = "tfan_main") -> TFANHudMetricsClient:
    """Get or create a default HUD metrics client."""
    global _default_client
    if _default_client is None:
        _default_client = TFANHudMetricsClient(model_id=model_id)
    return _default_client


def hud_update(**metrics: Any) -> None:
    """Quick update using default client."""
    get_hud_client().update(**metrics)
