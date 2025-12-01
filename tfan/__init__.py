# tfan/ - T-FAN Core Package
# Consolidated implementation for Kitten bootstrap

from .tfan_core import (
    # TFF Components
    TopFusionEncoder,
    TopologyHead,
    TopologicalRegularizer,
    # Layer 1: Homeostatic Core
    HomeostaticNeeds,
    HomeostaticCore,
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
from .tfan_telemetry import (
    TfanBrainSnapshot,
    TelemetryExporter,
    TelemetryClient,
    get_exporter,
    push_brain_state,
)

# Layer 2: Cognitive Appraisal (EMA/CoRE)
from .appraisal_engine import (
    APPRAISAL_DIMENSIONS,
    EMOTION_PATTERNS,
    AppraisalResult,
    AppraisalEngineStub,
    AppraisalEngineClient,
    create_appraisal_engine,
)

# Layer 3: Uncertainty Decomposition
from .uncertainty_head import (
    UncertaintyEstimate,
    PolicyGateDecision,
    PolicyDecision,
    UncertaintyHead,
    create_uncertainty_head,
)

# TGSFN: Antifragility Monitor
from .antifragility_monitor import (
    AntifragilityMetrics,
    DAUTrigger,
    AntifragilityMonitor,
    create_antifragility_monitor,
)

__version__ = "0.1.0"

__all__ = [
    # TFF
    "TopFusionEncoder",
    "TopologyHead",
    "TopologicalRegularizer",
    # Layer 1: Homeostatic Core
    "HomeostaticNeeds",
    "HomeostaticCore",
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
    # HUD Metrics (GNOME cockpit - simple)
    "TFANHudMetricsClient",
    "get_hud_client",
    "hud_update",
    # Brain Telemetry (full 4-layer snapshot)
    "TfanBrainSnapshot",
    "TelemetryExporter",
    "TelemetryClient",
    "get_exporter",
    "push_brain_state",
    # Layer 2: Cognitive Appraisal
    "APPRAISAL_DIMENSIONS",
    "EMOTION_PATTERNS",
    "AppraisalResult",
    "AppraisalEngineStub",
    "AppraisalEngineClient",
    "create_appraisal_engine",
    # Layer 3: Uncertainty Decomposition
    "UncertaintyEstimate",
    "PolicyGateDecision",
    "PolicyDecision",
    "UncertaintyHead",
    "create_uncertainty_head",
    # TGSFN: Antifragility Monitor
    "AntifragilityMetrics",
    "DAUTrigger",
    "AntifragilityMonitor",
    "create_antifragility_monitor",
]
