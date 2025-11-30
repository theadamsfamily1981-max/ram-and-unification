# tfan/ - T-FAN Core Package
# Consolidated implementation for Kitten bootstrap

from .tfan_core import (
    # TFF Components
    TopFusionEncoder,
    TopologyHead,
    TopologicalRegularizer,
    # UDK Components
    DissipationMonitor,
    UDKState,
    UDKController,
    estimate_fim_eigenvalue,
    # NCE/COS Components
    NicheAction,
    CognitiveOffloadingSubsystem,
    # High-level model
    TFanCore,
    # Factory functions
    create_tfan_core,
    create_udk_controller,
    create_cos,
)

from .telemetry import TelemetryEmitter
from .tfan_hud_metrics import (
    TFANHudMetricsClient,
    get_hud_client,
    hud_update,
)

__version__ = "0.1.0"

__all__ = [
    # TFF
    "TopFusionEncoder",
    "TopologyHead",
    "TopologicalRegularizer",
    # UDK
    "DissipationMonitor",
    "UDKState",
    "UDKController",
    "estimate_fim_eigenvalue",
    # NCE/COS
    "NicheAction",
    "CognitiveOffloadingSubsystem",
    # Model
    "TFanCore",
    # Factories
    "create_tfan_core",
    "create_udk_controller",
    "create_cos",
    # Telemetry (legacy)
    "TelemetryEmitter",
    # HUD Metrics (new - for GNOME cockpit)
    "TFANHudMetricsClient",
    "get_hud_client",
    "hud_update",
]
