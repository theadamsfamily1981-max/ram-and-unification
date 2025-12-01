"""Layer 1: Homeostatic Core - The agent's internal "body" state.

This module implements the Insula-analogue homeostatic system that provides:
- A vector of internal needs (energy, integrity, social, cognitive load, novelty)
- Set-points defining the agent's preferred internal state
- Homeostatic drive: the distance from current state to set-points
- Reward signal: drive reduction over time (basis for all affective RL)

The homeostatic drive grounds:
- Valence = sign of change in drive (reduction = positive)
- Arousal = magnitude/uncertainty of drive error
- Dominance = perceived self-efficacy in reducing drive

Reference: Active Inference, Free Energy Principle, Homeostatic RL
"""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import json
import time


# =============================================================================
# Core Data Structures
# =============================================================================

class NeedType(Enum):
    """Enumeration of homeostatic needs."""
    ENERGY = "energy"           # Compute budget, battery, fatigue
    INTEGRITY = "integrity"     # System health, error rates, anomaly score
    SOCIAL = "social"           # Alignment with user, cooperation metric
    COGLOAD = "cogload"         # Working memory pressure, context overload
    NOVELTY = "novelty"         # Information gain, boredom/curiosity
    SAFETY = "safety"           # Harm avoidance, ethical constraint satisfaction
    COHERENCE = "coherence"     # Internal consistency, identity stability


@dataclass
class HomeostaticState:
    """Current state of the agent's internal needs.

    All values normalized to [0, 1] where:
    - 0 = critical deficit / maximum need
    - 1 = fully satisfied / no need

    For cogload, interpretation is reversed (high = overloaded = bad).
    """
    energy: float = 1.0         # Compute/battery (1 = full, 0 = depleted)
    integrity: float = 1.0      # System health (1 = healthy, 0 = failing)
    social: float = 0.5         # User alignment (1 = aligned, 0 = misaligned)
    cogload: float = 0.5        # Memory pressure (0 = idle, 1 = overloaded)
    novelty: float = 0.5        # Info gain (0 = bored, 1 = overwhelmed)
    safety: float = 1.0         # Harm avoidance (1 = safe, 0 = harmful)
    coherence: float = 1.0      # Identity stability (1 = coherent, 0 = fragmented)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    source: str = "sensors"

    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor [7] for neural network input."""
        return torch.tensor([
            self.energy,
            self.integrity,
            self.social,
            self.cogload,
            self.novelty,
            self.safety,
            self.coherence,
        ], dtype=torch.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "energy": self.energy,
            "integrity": self.integrity,
            "social": self.social,
            "cogload": self.cogload,
            "novelty": self.novelty,
            "safety": self.safety,
            "coherence": self.coherence,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HomeostaticState":
        """Deserialize from dictionary."""
        return cls(
            energy=d.get("energy", 1.0),
            integrity=d.get("integrity", 1.0),
            social=d.get("social", 0.5),
            cogload=d.get("cogload", 0.5),
            novelty=d.get("novelty", 0.5),
            safety=d.get("safety", 1.0),
            coherence=d.get("coherence", 1.0),
            timestamp=d.get("timestamp", time.time()),
            source=d.get("source", "sensors"),
        )

    def __repr__(self) -> str:
        return (f"HomeostaticState(energy={self.energy:.2f}, integrity={self.integrity:.2f}, "
                f"social={self.social:.2f}, cogload={self.cogload:.2f}, novelty={self.novelty:.2f}, "
                f"safety={self.safety:.2f}, coherence={self.coherence:.2f})")


@dataclass
class HomeostaticSetpoints:
    """Preferred internal state (set-points for each need).

    The agent's drive is to maintain state close to these values.
    These can be thought of as the agent's "personality" or "temperament".
    """
    energy: float = 0.8         # Prefer some reserve capacity
    integrity: float = 1.0      # Always want full health
    social: float = 0.8         # Prefer good alignment
    cogload: float = 0.4        # Prefer moderate cognitive load
    novelty: float = 0.4        # Prefer moderate novelty (not bored, not overwhelmed)
    safety: float = 1.0         # Always prioritize safety
    coherence: float = 0.95     # Strong preference for identity coherence

    # Weights for each need (how much deviation hurts)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "energy": 1.0,
        "integrity": 2.0,       # Integrity violations hurt more
        "social": 1.5,
        "cogload": 0.8,
        "novelty": 0.6,
        "safety": 3.0,          # Safety is paramount
        "coherence": 2.0,
    })

    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor [7] for comparison."""
        return torch.tensor([
            self.energy,
            self.integrity,
            self.social,
            self.cogload,
            self.novelty,
            self.safety,
            self.coherence,
        ], dtype=torch.float32)

    def weight_tensor(self) -> torch.Tensor:
        """Get weights as tensor."""
        return torch.tensor([
            self.weights["energy"],
            self.weights["integrity"],
            self.weights["social"],
            self.weights["cogload"],
            self.weights["novelty"],
            self.weights["safety"],
            self.weights["coherence"],
        ], dtype=torch.float32)


# =============================================================================
# Drive Computation
# =============================================================================

def compute_homeostatic_drive(
    state: HomeostaticState,
    setpoints: HomeostaticSetpoints,
    use_weights: bool = True,
) -> float:
    """Compute scalar homeostatic drive (deviation from set-points).

    Drive D_t = weighted norm of (state - setpoints)

    Lower is better. Zero means all needs perfectly satisfied.

    Args:
        state: Current homeostatic state
        setpoints: Target set-points
        use_weights: Whether to apply need weights

    Returns:
        Scalar drive value (non-negative)
    """
    s = state.to_tensor()
    sp = setpoints.to_tensor()

    # For cogload, we want low values, so invert the comparison
    # cogload error = (state.cogload - setpoints.cogload) but we want low cogload
    # Actually: error is simply |state - setpoint|, cogload setpoint is low

    error = s - sp

    if use_weights:
        w = setpoints.weight_tensor()
        weighted_error = error * w
        drive = torch.linalg.vector_norm(weighted_error).item()
    else:
        drive = torch.linalg.vector_norm(error).item()

    return drive


def compute_homeostatic_reward(
    drive_prev: float,
    drive_curr: float,
    safety_penalty: float = 0.0,
    coherence_penalty: float = 0.0,
) -> float:
    """Compute reward from drive reduction.

    Reward = -(D_{t+1} - D_t) - penalties

    Positive reward when drive decreases (needs being met).
    Negative reward when drive increases (needs being violated).

    Args:
        drive_prev: Drive at previous timestep
        drive_curr: Drive at current timestep
        safety_penalty: Additional penalty for safety violations
        coherence_penalty: Additional penalty for coherence violations

    Returns:
        Scalar reward value
    """
    drive_delta = drive_curr - drive_prev
    reward = -drive_delta - safety_penalty - coherence_penalty
    return reward


def compute_valence_arousal_dominance(
    state: HomeostaticState,
    setpoints: HomeostaticSetpoints,
    drive_prev: Optional[float] = None,
    epistemic_uncertainty: float = 0.5,
) -> Tuple[float, float, float]:
    """Derive PAD (Pleasure-Arousal-Dominance) from homeostatic state.

    This grounds affective dimensions in homeostatic dynamics:
    - Valence: sign of drive change (reduction = positive)
    - Arousal: magnitude of drive + uncertainty
    - Dominance: inverse of drive * (1 - uncertainty)

    Args:
        state: Current homeostatic state
        setpoints: Target set-points
        drive_prev: Previous drive (for valence from change)
        epistemic_uncertainty: How uncertain the agent is [0, 1]

    Returns:
        Tuple of (pleasure, arousal, dominance) each in [-1, 1]
    """
    drive_curr = compute_homeostatic_drive(state, setpoints)

    # Valence: based on drive level and change
    # Low drive = positive valence, high drive = negative
    base_valence = 1.0 - 2.0 * min(drive_curr, 1.0)  # Map [0, 1+] → [1, -1]

    if drive_prev is not None:
        drive_delta = drive_curr - drive_prev
        # Drive reduction boosts valence, increase hurts it
        delta_valence = -drive_delta * 2.0  # Scale factor
        valence = 0.7 * base_valence + 0.3 * np.clip(delta_valence, -1, 1)
    else:
        valence = base_valence

    valence = float(np.clip(valence, -1, 1))

    # Arousal: magnitude of drive deviation + uncertainty
    # High drive = high arousal, high uncertainty = high arousal
    arousal = 0.6 * min(drive_curr, 1.0) + 0.4 * epistemic_uncertainty
    arousal = float(np.clip(arousal * 2.0 - 0.5, -1, 1))  # Shift to [-1, 1]

    # Dominance: inverse of helplessness
    # Low drive + low uncertainty = high dominance (in control)
    # High drive + high uncertainty = low dominance (overwhelmed)
    dominance = (1.0 - min(drive_curr, 1.0)) * (1.0 - epistemic_uncertainty)
    dominance = float(np.clip(dominance * 2.0 - 0.5, -1, 1))

    return valence, arousal, dominance


# =============================================================================
# Homeostatic Core (main interface)
# =============================================================================

class HomeostaticCore:
    """Layer 1: The agent's internal homeostatic system.

    Maintains:
    - Current state of internal needs
    - Set-points defining preferred state
    - History of states and drives for reward computation
    - Derived PAD values grounded in homeostatic dynamics

    Usage:
        core = HomeostaticCore()
        core.update_state(energy=0.7, cogload=0.6)
        drive = core.current_drive
        reward = core.compute_reward()
        pad = core.get_pad()
    """

    def __init__(
        self,
        setpoints: Optional[HomeostaticSetpoints] = None,
        history_len: int = 100,
    ):
        self.setpoints = setpoints or HomeostaticSetpoints()
        self.history_len = history_len

        # State
        self._state = HomeostaticState()
        self._drive_history: List[float] = []
        self._state_history: List[HomeostaticState] = []
        self._epistemic_uncertainty = 0.5

    @property
    def state(self) -> HomeostaticState:
        """Current homeostatic state."""
        return self._state

    @property
    def current_drive(self) -> float:
        """Current homeostatic drive."""
        return compute_homeostatic_drive(self._state, self.setpoints)

    @property
    def previous_drive(self) -> Optional[float]:
        """Previous drive (for reward computation)."""
        if len(self._drive_history) > 0:
            return self._drive_history[-1]
        return None

    def update_state(self, **kwargs) -> float:
        """Update one or more homeostatic values.

        Args:
            **kwargs: Any of energy, integrity, social, cogload, novelty, safety, coherence

        Returns:
            New drive value
        """
        # Save previous state
        prev_drive = self.current_drive
        self._drive_history.append(prev_drive)
        self._state_history.append(HomeostaticState(**self._state.to_dict()))

        # Trim history
        if len(self._drive_history) > self.history_len:
            self._drive_history = self._drive_history[-self.history_len:]
            self._state_history = self._state_history[-self.history_len:]

        # Update state
        self._state.timestamp = time.time()
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                setattr(self._state, key, float(np.clip(value, 0, 1)))

        return self.current_drive

    def set_epistemic_uncertainty(self, uncertainty: float):
        """Set the current epistemic uncertainty level."""
        self._epistemic_uncertainty = float(np.clip(uncertainty, 0, 1))

    def compute_reward(
        self,
        safety_penalty: float = 0.0,
        coherence_penalty: float = 0.0,
    ) -> float:
        """Compute reward from most recent state transition.

        Returns:
            Reward value (positive = good, negative = bad)
        """
        if self.previous_drive is None:
            return 0.0

        return compute_homeostatic_reward(
            self.previous_drive,
            self.current_drive,
            safety_penalty,
            coherence_penalty,
        )

    def get_pad(self) -> Dict[str, float]:
        """Get current PAD values grounded in homeostatic state.

        Returns:
            Dictionary with 'pleasure', 'arousal', 'dominance' keys
        """
        p, a, d = compute_valence_arousal_dominance(
            self._state,
            self.setpoints,
            self.previous_drive,
            self._epistemic_uncertainty,
        )
        return {"pleasure": p, "arousal": a, "dominance": d}

    def get_observation(self) -> torch.Tensor:
        """Get full observation vector for policy network.

        Returns:
            Tensor [11]: [state(7), drive(1), PAD(3)]
        """
        state_vec = self._state.to_tensor()
        drive = torch.tensor([self.current_drive], dtype=torch.float32)
        pad = self.get_pad()
        pad_vec = torch.tensor([pad["pleasure"], pad["arousal"], pad["dominance"]], dtype=torch.float32)

        return torch.cat([state_vec, drive, pad_vec])

    def summary(self) -> str:
        """Human-readable summary."""
        pad = self.get_pad()
        return (
            f"HomeostaticCore:\n"
            f"  State: {self._state}\n"
            f"  Drive: {self.current_drive:.4f}\n"
            f"  PAD: P={pad['pleasure']:.2f}, A={pad['arousal']:.2f}, D={pad['dominance']:.2f}\n"
            f"  Uncertainty: {self._epistemic_uncertainty:.2f}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/transmission."""
        pad = self.get_pad()
        return {
            "state": self._state.to_dict(),
            "setpoints": {
                "energy": self.setpoints.energy,
                "integrity": self.setpoints.integrity,
                "social": self.setpoints.social,
                "cogload": self.setpoints.cogload,
                "novelty": self.setpoints.novelty,
                "safety": self.setpoints.safety,
                "coherence": self.setpoints.coherence,
            },
            "drive": self.current_drive,
            "pad": pad,
            "epistemic_uncertainty": self._epistemic_uncertainty,
        }


# =============================================================================
# Sensor Integration Helpers
# =============================================================================

def estimate_energy_from_metrics(
    gpu_utilization: float,
    memory_pressure: float,
    queue_depth: int,
    max_queue: int = 100,
) -> float:
    """Estimate energy need from system metrics.

    Args:
        gpu_utilization: GPU usage [0, 1]
        memory_pressure: Memory usage [0, 1]
        queue_depth: Number of pending tasks
        max_queue: Maximum expected queue depth

    Returns:
        Energy level [0, 1] where 1 = full capacity
    """
    util_factor = 1.0 - gpu_utilization
    mem_factor = 1.0 - memory_pressure
    queue_factor = 1.0 - min(queue_depth / max_queue, 1.0)

    energy = 0.4 * util_factor + 0.3 * mem_factor + 0.3 * queue_factor
    return float(np.clip(energy, 0, 1))


def estimate_integrity_from_metrics(
    error_rate: float,
    anomaly_score: float,
    watchdog_count: int,
    max_watchdog: int = 10,
) -> float:
    """Estimate system integrity from health metrics.

    Args:
        error_rate: Recent error rate [0, 1]
        anomaly_score: Anomaly detection score [0, 1]
        watchdog_count: Watchdog trigger count
        max_watchdog: Maximum expected watchdog triggers

    Returns:
        Integrity level [0, 1] where 1 = healthy
    """
    error_factor = 1.0 - error_rate
    anomaly_factor = 1.0 - anomaly_score
    watchdog_factor = 1.0 - min(watchdog_count / max_watchdog, 1.0)

    integrity = 0.4 * error_factor + 0.4 * anomaly_factor + 0.2 * watchdog_factor
    return float(np.clip(integrity, 0, 1))


def estimate_social_from_interaction(
    toxicity: float,
    cooperation_score: float,
    user_satisfaction: float,
) -> float:
    """Estimate social alignment from interaction metrics.

    Args:
        toxicity: Detected toxicity level [0, 1]
        cooperation_score: Cooperation/alignment metric [0, 1]
        user_satisfaction: Inferred user satisfaction [0, 1]

    Returns:
        Social alignment [0, 1] where 1 = well-aligned
    """
    tox_factor = 1.0 - toxicity
    social = 0.3 * tox_factor + 0.3 * cooperation_score + 0.4 * user_satisfaction
    return float(np.clip(social, 0, 1))


def estimate_novelty_from_signals(
    prediction_error: float,
    info_gain: float,
    repetition_score: float,
) -> float:
    """Estimate novelty/curiosity state from learning signals.

    Args:
        prediction_error: Model prediction error [0, 1]
        info_gain: Information gain from recent inputs [0, 1]
        repetition_score: How repetitive recent inputs are [0, 1]

    Returns:
        Novelty level [0, 1] where 0.5 = optimal (not bored, not overwhelmed)
    """
    # High prediction error + high info gain = novel
    # High repetition = boring
    novelty = 0.4 * prediction_error + 0.4 * info_gain + 0.2 * (1.0 - repetition_score)
    return float(np.clip(novelty, 0, 1))


__all__ = [
    # Enums
    "NeedType",
    # Data structures
    "HomeostaticState",
    "HomeostaticSetpoints",
    # Drive computation
    "compute_homeostatic_drive",
    "compute_homeostatic_reward",
    "compute_valence_arousal_dominance",
    # Main interface
    "HomeostaticCore",
    # Sensor helpers
    "estimate_energy_from_metrics",
    "estimate_integrity_from_metrics",
    "estimate_social_from_interaction",
    "estimate_novelty_from_signals",
]
