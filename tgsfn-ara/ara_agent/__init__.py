# ara_agent/__init__.py
# Ara Agent - Autonomous Reasoning Agent with TGSFN Backend
#
# This package implements the agent-level components that interface
# with the TGSFN spiking network:
#   - Identity manifold (Poincaré ball embedding)
#   - L5 control law for action generation
#   - Homeostatic regulation
#   - Dynamic Axiom Updater (DAU) for antifragility

from .identity_manifold import IdentityManifold, HyperbolicEncoder
from .control_law import L5Controller, compute_control_action
from .homeostasis import HomeostaticRegulator, NeedsVector
from .dau import DAULite, create_dau

__all__ = [
    "IdentityManifold",
    "HyperbolicEncoder",
    "L5Controller",
    "compute_control_action",
    "HomeostaticRegulator",
    "NeedsVector",
    "DAULite",
    "create_dau",
]

__version__ = "0.1.0"
