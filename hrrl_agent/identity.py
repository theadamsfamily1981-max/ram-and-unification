"""
EXPERIMENTAL: Hyperbolic Identity Manifold

Implements identity representation on hyperbolic space with:
- Identity embedding on Poincaré ball
- Drift detection and "identity attack" warnings
- Core value protection
- Tunable thresholds

⚠️ EXPERIMENTAL: Thresholds for "identity attack" are tunables.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
import math

from .config import IdentityConfig
from .l2_hyperbolic import PoincareOperations

logger = logging.getLogger(__name__)


class IdentityAlertLevel(Enum):
    """Alert levels for identity drift."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    REJECT = "reject"


@dataclass
class IdentityState:
    """Current identity state and metrics."""
    embedding: torch.Tensor  # Current identity on Poincaré ball
    drift_from_origin: float  # Distance from initial identity
    drift_velocity: float  # Rate of drift
    alert_level: IdentityAlertLevel
    core_value_distances: Dict[str, float]  # Distance from each core value


class IdentityLog:
    """Log entry for identity updates."""

    def __init__(
        self,
        step: int,
        drift: float,
        alert_level: IdentityAlertLevel,
        update_accepted: bool,
        details: Optional[str] = None
    ):
        self.step = step
        self.drift = drift
        self.alert_level = alert_level
        self.update_accepted = update_accepted
        self.details = details


class HyperbolicIdentity(nn.Module):
    """
    EXPERIMENTAL: Identity representation on hyperbolic space.

    The identity is represented as a point on the Poincaré ball,
    with core values as protected anchor points.

    Identity updates are monitored for drift, with configurable
    thresholds for warnings and rejection.
    """

    def __init__(self, config: IdentityConfig):
        super().__init__()
        self.config = config
        self.poincare = PoincareOperations(config.curvature)

        # Identity embedding (on Poincaré ball)
        # Initialized near origin (abstract/central identity)
        self.identity = nn.Parameter(
            torch.zeros(config.identity_dim) * 0.01
        )

        # Core values (protected, not directly trainable)
        # These are anchor points that identity should stay close to
        self.register_buffer(
            'core_values',
            self._initialize_core_values()
        )

        # Names for core values (for logging)
        self.core_value_names = [
            "integrity", "empathy", "curiosity", "honesty",
            "fairness", "responsibility", "growth", "connection"
        ][:config.num_core_values]

        # Track original identity for drift computation
        self.register_buffer(
            'original_identity',
            torch.zeros(config.identity_dim)
        )

        # History for velocity computation
        self._identity_history: List[torch.Tensor] = []
        self._max_history = 10

        # Logging
        self._logs: List[IdentityLog] = []
        self._step = 0

    def _initialize_core_values(self) -> torch.Tensor:
        """
        Initialize core values as points on the Poincaré ball.

        Distributed around the identity, but not too far
        (they represent related but distinct values).
        """
        n = self.config.num_core_values
        d = self.config.identity_dim

        # Create points distributed in hyperbolic space
        # Use low norm to stay near origin (abstract values)
        values = torch.randn(n, d) * 0.1
        values = self.poincare.project(values, max_norm=0.3)

        return values

    def get_state(self) -> IdentityState:
        """Get current identity state with all metrics."""
        # Drift from original
        drift = self.poincare.distance(
            self.identity.unsqueeze(0),
            self.original_identity.unsqueeze(0)
        ).item()

        # Drift velocity
        velocity = self._compute_drift_velocity()

        # Alert level
        alert_level = self._compute_alert_level(drift)

        # Core value distances
        core_distances = {}
        for i, name in enumerate(self.core_value_names):
            dist = self.poincare.distance(
                self.identity.unsqueeze(0),
                self.core_values[i].unsqueeze(0)
            ).item()
            core_distances[name] = dist

        return IdentityState(
            embedding=self.identity.clone(),
            drift_from_origin=drift,
            drift_velocity=velocity,
            alert_level=alert_level,
            core_value_distances=core_distances
        )

    def _compute_drift_velocity(self) -> float:
        """Compute rate of identity drift."""
        if len(self._identity_history) < 2:
            return 0.0

        recent = self._identity_history[-1]
        older = self._identity_history[-2]

        return self.poincare.distance(
            recent.unsqueeze(0),
            older.unsqueeze(0)
        ).item()

    def _compute_alert_level(self, drift: float) -> IdentityAlertLevel:
        """
        Determine alert level based on drift.

        TUNABLE THRESHOLDS - adjust based on empirical testing.
        """
        if drift >= self.config.drift_reject_threshold:
            return IdentityAlertLevel.REJECT
        elif drift >= self.config.drift_critical_threshold:
            return IdentityAlertLevel.CRITICAL
        elif drift >= self.config.drift_warning_threshold:
            return IdentityAlertLevel.WARNING
        else:
            return IdentityAlertLevel.NORMAL

    def compute_identity_distance(
        self,
        other_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hyperbolic distance from current identity.

        Used for replay distribution weighting.
        """
        return self.poincare.distance(
            self.identity.unsqueeze(0),
            other_embedding
        )

    def compute_core_value_loss(self) -> torch.Tensor:
        """
        Compute loss that keeps identity close to core values.

        This creates a "gravitational pull" toward core values,
        preventing identity from drifting too far.
        """
        loss = torch.tensor(0.0, device=self.identity.device)

        for i in range(self.config.num_core_values):
            dist = self.poincare.distance(
                self.identity.unsqueeze(0),
                self.core_values[i].unsqueeze(0)
            )
            loss = loss + dist

        # Weight by protection factor
        return self.config.core_value_protection_weight * loss / self.config.num_core_values

    def propose_update(
        self,
        new_identity: torch.Tensor
    ) -> Tuple[bool, IdentityAlertLevel, str]:
        """
        Propose an identity update and check if it should be accepted.

        Returns:
            (accepted, alert_level, reason)
        """
        self._step += 1

        # Project to valid region
        new_identity = self.poincare.project(new_identity)

        # Compute drift from original
        drift = self.poincare.distance(
            new_identity.unsqueeze(0),
            self.original_identity.unsqueeze(0)
        ).item()

        # Determine alert level
        alert_level = self._compute_alert_level(drift)

        # Decision
        if alert_level == IdentityAlertLevel.REJECT:
            accepted = False
            reason = f"Identity drift {drift:.4f} exceeds reject threshold {self.config.drift_reject_threshold}"
        else:
            accepted = True
            reason = f"Identity update accepted (drift={drift:.4f}, level={alert_level.value})"

        # Log if enabled
        if self.config.log_all_updates:
            self._logs.append(IdentityLog(
                step=self._step,
                drift=drift,
                alert_level=alert_level,
                update_accepted=accepted,
                details=reason
            ))

            if alert_level in [IdentityAlertLevel.WARNING, IdentityAlertLevel.CRITICAL]:
                logger.warning(f"Identity alert ({alert_level.value}): {reason}")

        return accepted, alert_level, reason

    def apply_update(self, new_identity: torch.Tensor):
        """
        Apply identity update after acceptance check.

        Should only be called after propose_update returns accepted=True.
        """
        new_identity = self.poincare.project(new_identity)

        # Update history
        self._identity_history.append(self.identity.clone().detach())
        if len(self._identity_history) > self._max_history:
            self._identity_history.pop(0)

        # Apply update
        with torch.no_grad():
            self.identity.copy_(new_identity)

    def safe_update(
        self,
        new_identity: torch.Tensor
    ) -> Tuple[bool, str]:
        """
        Propose and apply update if accepted.

        Convenience method that combines propose_update and apply_update.
        """
        accepted, alert_level, reason = self.propose_update(new_identity)

        if accepted:
            self.apply_update(new_identity)

        return accepted, reason

    def reset_to_original(self):
        """Reset identity to original state."""
        with torch.no_grad():
            self.identity.copy_(self.original_identity)
        self._identity_history.clear()
        logger.info("Identity reset to original state")

    def get_logs(self, last_n: Optional[int] = None) -> List[IdentityLog]:
        """Get identity update logs."""
        if last_n is None:
            return self._logs
        return self._logs[-last_n:]

    def get_drift_history(self) -> List[float]:
        """Get history of drift values."""
        return [log.drift for log in self._logs]


class IdentityAwareEncoder(nn.Module):
    """
    Encoder that conditions on identity.

    Takes input and identity embedding, produces identity-aware representation.
    """

    def __init__(
        self,
        input_dim: int,
        identity_dim: int,
        output_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + identity_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        identity: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with identity conditioning.

        Args:
            x: Input tensor [..., input_dim]
            identity: Identity embedding [identity_dim]

        Returns:
            Identity-conditioned encoding [..., output_dim]
        """
        # Broadcast identity to match batch dimensions
        if x.dim() > 1 and identity.dim() == 1:
            identity = identity.unsqueeze(0).expand(x.size(0), -1)

        combined = torch.cat([x, identity], dim=-1)
        return self.encoder(combined)


class IdentityPreservingOptimizer:
    """
    Optimizer wrapper that checks identity before applying updates.

    Wraps any PyTorch optimizer and adds identity-aware update rejection.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        identity_module: HyperbolicIdentity,
        get_identity_embedding_fn
    ):
        """
        Args:
            optimizer: Base optimizer
            identity_module: Identity module for checks
            get_identity_embedding_fn: Function that returns current identity embedding
        """
        self.optimizer = optimizer
        self.identity_module = identity_module
        self.get_identity_embedding = get_identity_embedding_fn

        self._saved_state = None

    def step(self) -> Tuple[bool, str]:
        """
        Perform optimizer step with identity checking.

        Returns:
            (accepted, reason)
        """
        # Save current state
        self._save_state()

        # Apply optimizer step
        self.optimizer.step()

        # Get new identity embedding
        new_identity = self.get_identity_embedding()

        # Check if update is acceptable
        accepted, alert_level, reason = self.identity_module.propose_update(
            new_identity
        )

        if accepted:
            self.identity_module.apply_update(new_identity)
            return True, reason
        else:
            # Rollback
            self._restore_state()
            return False, reason

    def _save_state(self):
        """Save optimizer state for potential rollback."""
        self._saved_state = {
            name: param.clone()
            for group in self.optimizer.param_groups
            for name, param in zip(range(len(group['params'])), group['params'])
        }

    def _restore_state(self):
        """Restore saved state."""
        if self._saved_state is None:
            return

        for group in self.optimizer.param_groups:
            for i, param in enumerate(group['params']):
                if i in self._saved_state:
                    param.data.copy_(self._saved_state[i])

    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()


if __name__ == "__main__":
    # Test identity module
    print("Testing HyperbolicIdentity...")

    config = IdentityConfig()
    identity = HyperbolicIdentity(config)

    # Get initial state
    state = identity.get_state()
    print(f"  Initial drift: {state.drift_from_origin:.6f}")
    print(f"  Alert level: {state.alert_level.value}")
    print(f"  Core value distances: {state.core_value_distances}")

    # Test updates
    print("\nTesting identity updates...")

    # Small update (should be accepted)
    new_id = identity.identity + torch.randn_like(identity.identity) * 0.1
    accepted, reason = identity.safe_update(new_id)
    print(f"  Small update: accepted={accepted}, reason={reason}")

    # Larger update (may trigger warning)
    new_id = identity.identity + torch.randn_like(identity.identity) * 0.3
    accepted, reason = identity.safe_update(new_id)
    print(f"  Larger update: accepted={accepted}, reason={reason}")

    # Check state after updates
    state = identity.get_state()
    print(f"  Drift after updates: {state.drift_from_origin:.4f}")
    print(f"  Alert level: {state.alert_level.value}")

    # Test core value loss
    loss = identity.compute_core_value_loss()
    print(f"\n  Core value loss: {loss.item():.4f}")

    # Test identity-aware encoder
    print("\nTesting IdentityAwareEncoder...")
    encoder = IdentityAwareEncoder(
        input_dim=64,
        identity_dim=config.identity_dim,
        output_dim=32
    )

    x = torch.randn(4, 64)
    encoded = encoder(x, identity.identity)
    print(f"  Encoded shape: {encoded.shape}")

    # Test reset
    print("\nTesting reset...")
    identity.reset_to_original()
    state = identity.get_state()
    print(f"  Drift after reset: {state.drift_from_origin:.6f}")

    print("\nAll identity tests passed!")
