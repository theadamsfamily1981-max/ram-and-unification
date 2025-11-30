# tfan_agent/__init__.py
# Phase 1: Homeostatic Affective Spiking Agent (Euclidean, non-geometric)
#
# This package implements the "vulnerable body" architecture:
#   - L1: HomeostatL1 - Homeostatic Core with needs, drives, PAD affect
#   - L2: AppraisalHeadL2 - Cognitive appraisal engine
#   - L3: GatingControllerL3 - Adaptive policy/gating controller
#   - Policy: SimpleSpikingPolicy - LIF-like policy network
#
# Phase 2 will add: Geometry/TGSFN, Π_q, DAU, real SNNs

from .homeostat import HomeostatL1, HomeostatConfig, HomeostatState
from .appraisal import AppraisalHeadL2, AppraisalConfig
from .gating import GatingControllerL3, GatingConfig
from .snn_policy import SimpleSpikingPolicy, PolicyConfig
from .buffers import Trajectory
from .simple_env import SimpleHomeostaticEnv, SimpleHomeostaticEnvConfig
from .training_loop import TfanAgent, train_phase1

__all__ = [
    # L1: Homeostatic Core
    "HomeostatL1",
    "HomeostatConfig",
    "HomeostatState",
    # L2: Appraisal Engine
    "AppraisalHeadL2",
    "AppraisalConfig",
    # L3: Gating Controller
    "GatingControllerL3",
    "GatingConfig",
    # Policy Network
    "SimpleSpikingPolicy",
    "PolicyConfig",
    # Buffers
    "Trajectory",
    # Environment
    "SimpleHomeostaticEnv",
    "SimpleHomeostaticEnvConfig",
    # Agent & Training
    "TfanAgent",
    "train_phase1",
]
