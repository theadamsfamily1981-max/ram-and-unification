# ara_agent/homeostasis.py
# Homeostatic Regulation for Ara Agent
#
# Implements the needs-driven homeostatic regulation system that maintains
# agent viability through internal free energy minimization.
#
# Key Components:
#   - Needs Vector (n): Current state of internal drives
#   - Setpoints (n*): Target values for needs
#   - Covariance (Σ_inv): Precision matrix weighting need importance
#   - Internal Free Energy: F_int = β * (n - n*)^T Σ_inv (n - n*)
#
# Scientific Constraints:
#   - Based on active inference framework (Friston)
#   - Allostatic regulation (anticipatory, not just reactive)
#   - Multi-timescale dynamics (fast metabolic, slow developmental)
#
# References:
#   - Friston et al. (2015): Active inference
#   - Pezzulo et al. (2015): Active inference & homeostasis

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn


@dataclass
class NeedsConfig:
    """Configuration for needs vector."""
    n_needs: int = 8                    # Number of internal needs
    beta: float = 1.0                   # Free energy temperature
    setpoint_lr: float = 0.001          # Learning rate for setpoint adaptation
    sigma_lr: float = 0.0001            # Learning rate for precision adaptation
    ema_decay: float = 0.99             # EMA decay for need tracking
    urgency_threshold: float = 2.0      # Threshold for urgent needs


class NeedsVector(nn.Module):
    """
    Internal needs state vector.

    Represents the agent's current internal drives/needs that must be
    regulated to maintain viability. Each need has a setpoint and
    precision weighting.

    Common needs (can be domain-specific):
    - Energy/metabolism
    - Information/novelty
    - Social/affiliation
    - Safety/predictability
    - Competence/mastery
    - Autonomy/control
    - Identity coherence
    - Goal progress
    """

    def __init__(
        self,
        n_needs: int = 8,
        need_names: Optional[List[str]] = None,
    ):
        """
        Args:
            n_needs: Number of internal needs
            need_names: Optional names for each need
        """
        super().__init__()

        self.n_needs = n_needs

        # Default need names
        if need_names is None:
            need_names = [
                "energy", "novelty", "safety", "competence",
                "autonomy", "affiliation", "coherence", "progress"
            ][:n_needs]
        self.need_names = need_names

        # Current need values
        self.register_buffer("n", torch.zeros(n_needs))

        # Setpoints (initially at zero = satisfied)
        self.n_star = nn.Parameter(torch.zeros(n_needs))

        # Precision matrix (diagonal for simplicity)
        # Higher precision = more important need
        self.log_sigma_inv = nn.Parameter(torch.zeros(n_needs))

        # EMA decay for need updates
        self.ema_decay = 0.99

    @property
    def sigma_inv(self) -> torch.Tensor:
        """Get precision (inverse variance) for each need."""
        return torch.exp(self.log_sigma_inv)

    def update_needs(self, deltas: torch.Tensor) -> None:
        """
        Update need values with new observations.

        Args:
            deltas: Changes to need values, shape (n_needs,)
        """
        with torch.no_grad():
            self.n = self.ema_decay * self.n + (1 - self.ema_decay) * deltas

    def set_needs(self, values: torch.Tensor) -> None:
        """Directly set need values."""
        with torch.no_grad():
            self.n.copy_(values)

    def get_deviation(self) -> torch.Tensor:
        """Get deviation from setpoints: n - n*"""
        return self.n - self.n_star

    def get_weighted_deviation(self) -> torch.Tensor:
        """Get precision-weighted deviation: Σ_inv * (n - n*)"""
        return self.sigma_inv * self.get_deviation()

    def get_urgency(self) -> torch.Tensor:
        """
        Get urgency score for each need.

        Urgency = |n - n*| * √(σ_inv)
        """
        dev = self.get_deviation().abs()
        return dev * self.sigma_inv.sqrt()

    def get_most_urgent(self) -> Tuple[int, str, float]:
        """Get index, name, and urgency of most urgent need."""
        urgency = self.get_urgency()
        idx = urgency.argmax().item()
        return idx, self.need_names[idx], urgency[idx].item()

    def forward(self) -> torch.Tensor:
        """Return current need values."""
        return self.n


class HomeostaticRegulator(nn.Module):
    """
    Homeostatic regulation system.

    Computes internal free energy and drives behavior to minimize it.
    Implements allostatic (predictive) regulation, not just reactive.

    F_int = β * (n - n*)^T Σ_inv (n - n*)

    The regulator:
    1. Tracks internal needs
    2. Computes free energy from need deviation
    3. Generates drive signals to guide behavior
    4. Adapts setpoints based on experience (allostasis)
    """

    def __init__(
        self,
        config: Optional[NeedsConfig] = None,
    ):
        """
        Args:
            config: Homeostatic configuration
        """
        super().__init__()

        if config is None:
            config = NeedsConfig()

        self.config = config
        self.beta = config.beta

        # Needs vector
        self.needs = NeedsVector(n_needs=config.n_needs)

        # Adaptation rates
        self.setpoint_lr = config.setpoint_lr
        self.sigma_lr = config.sigma_lr

        # State tracking
        self.register_buffer("free_energy_ema", torch.tensor(0.0))
        self.register_buffer("update_count", torch.tensor(0))

    def compute_internal_free_energy(self) -> torch.Tensor:
        """
        Compute internal free energy.

        F_int = β * (n - n*)^T Σ_inv (n - n*)
             = β * Σ_i σ_inv_i * (n_i - n*_i)²

        Returns:
            Scalar free energy value
        """
        dev = self.needs.get_deviation()
        sigma_inv = self.needs.sigma_inv

        F_int = self.beta * (sigma_inv * dev * dev).sum()

        # Update EMA
        with torch.no_grad():
            self.free_energy_ema = (
                0.99 * self.free_energy_ema + 0.01 * F_int.detach()
            )

        return F_int

    def compute_drive_gradient(self) -> torch.Tensor:
        """
        Compute gradient of F_int w.r.t. needs.

        ∂F/∂n = 2β * Σ_inv * (n - n*)

        This gives the direction to move needs to reduce F_int.
        Actions should move needs in the negative gradient direction.

        Returns:
            Drive gradient, shape (n_needs,)
        """
        dev = self.needs.get_deviation()
        sigma_inv = self.needs.sigma_inv

        gradient = 2 * self.beta * sigma_inv * dev

        return gradient

    def compute_drive_signal(self) -> torch.Tensor:
        """
        Compute normalized drive signal for action selection.

        Returns vector in [-1, 1]^n indicating direction to move each need.
        """
        gradient = self.compute_drive_gradient()

        # Normalize by maximum gradient magnitude
        max_grad = gradient.abs().max().clamp(min=1e-6)
        drive = -gradient / max_grad  # Negative because we want to reduce F

        return drive

    def update_from_observation(
        self,
        observation: torch.Tensor,
        need_mapping: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Update needs from observation.

        Args:
            observation: Environment observation
            need_mapping: Optional module mapping obs -> need changes

        Returns:
            Updated internal free energy
        """
        if need_mapping is not None:
            deltas = need_mapping(observation)
        else:
            # Default: observation directly maps to needs
            if observation.shape[-1] >= self.needs.n_needs:
                deltas = observation[..., :self.needs.n_needs]
            else:
                deltas = torch.zeros(self.needs.n_needs, device=observation.device)
                deltas[:observation.shape[-1]] = observation

        self.needs.update_needs(deltas)
        self.update_count += 1

        return self.compute_internal_free_energy()

    def adapt_setpoints(
        self,
        target_free_energy: float = 0.0,
        adaptation_rate: Optional[float] = None,
    ) -> None:
        """
        Allostatic adaptation: adjust setpoints toward current values.

        This implements the idea that the agent can learn new "normal"
        states rather than always returning to fixed setpoints.

        Args:
            target_free_energy: Target F_int (0 = fully satisfied)
            adaptation_rate: Override for learning rate
        """
        if adaptation_rate is None:
            adaptation_rate = self.setpoint_lr

        with torch.no_grad():
            # Move setpoints toward current values
            self.needs.n_star.add_(
                adaptation_rate * self.needs.get_deviation()
            )

    def adapt_precision(
        self,
        outcome_quality: torch.Tensor,
        adaptation_rate: Optional[float] = None,
    ) -> None:
        """
        Adapt precision weights based on outcome quality.

        Needs that predict good outcomes get higher precision.

        Args:
            outcome_quality: Quality signal, shape (n_needs,)
            adaptation_rate: Override for learning rate
        """
        if adaptation_rate is None:
            adaptation_rate = self.sigma_lr

        with torch.no_grad():
            # Increase precision for needs that predict good outcomes
            self.needs.log_sigma_inv.add_(
                adaptation_rate * outcome_quality
            )

    def get_urgency_vector(self) -> torch.Tensor:
        """Get urgency of each need."""
        return self.needs.get_urgency()

    def get_stats(self) -> Dict[str, float]:
        """Get current homeostatic statistics."""
        return {
            "free_energy": self.compute_internal_free_energy().item(),
            "free_energy_ema": self.free_energy_ema.item(),
            "max_urgency": self.needs.get_urgency().max().item(),
            "mean_deviation": self.needs.get_deviation().abs().mean().item(),
            "update_count": self.update_count.item(),
        }

    def forward(
        self,
        observation: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: update from observation and return F_int and drive.

        Args:
            observation: Optional observation to update from

        Returns:
            (free_energy, drive_signal)
        """
        if observation is not None:
            self.update_from_observation(observation)

        F_int = self.compute_internal_free_energy()
        drive = self.compute_drive_signal()

        return F_int, drive


class MultiScaleHomeostasis(nn.Module):
    """
    Multi-timescale homeostatic regulation.

    Implements three timescales:
    - Fast (metabolic): immediate physiological needs
    - Medium (behavioral): goal-directed needs
    - Slow (developmental): long-term growth needs

    Each timescale has its own dynamics and precision.
    """

    def __init__(
        self,
        n_fast: int = 4,
        n_medium: int = 4,
        n_slow: int = 2,
    ):
        """
        Args:
            n_fast: Fast timescale needs
            n_medium: Medium timescale needs
            n_slow: Slow timescale needs
        """
        super().__init__()

        self.n_fast = n_fast
        self.n_medium = n_medium
        self.n_slow = n_slow

        # Create regulators for each timescale
        self.fast = HomeostaticRegulator(NeedsConfig(
            n_needs=n_fast, ema_decay=0.9, beta=2.0
        ))
        self.medium = HomeostaticRegulator(NeedsConfig(
            n_needs=n_medium, ema_decay=0.99, beta=1.0
        ))
        self.slow = HomeostaticRegulator(NeedsConfig(
            n_needs=n_slow, ema_decay=0.999, beta=0.5
        ))

        # Timescale weights
        self.register_buffer("timescale_weights", torch.tensor([0.5, 0.35, 0.15]))

    def forward(
        self,
        obs_fast: Optional[torch.Tensor] = None,
        obs_medium: Optional[torch.Tensor] = None,
        obs_slow: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Multi-timescale forward pass.

        Args:
            obs_fast: Fast timescale observation
            obs_medium: Medium timescale observation
            obs_slow: Slow timescale observation

        Returns:
            (total_free_energy, drive_signals_dict)
        """
        F_fast, drive_fast = self.fast(obs_fast)
        F_medium, drive_medium = self.medium(obs_medium)
        F_slow, drive_slow = self.slow(obs_slow)

        # Weighted sum of free energies
        F_total = (
            self.timescale_weights[0] * F_fast +
            self.timescale_weights[1] * F_medium +
            self.timescale_weights[2] * F_slow
        )

        drives = {
            "fast": drive_fast,
            "medium": drive_medium,
            "slow": drive_slow,
            "fast_F": F_fast,
            "medium_F": F_medium,
            "slow_F": F_slow,
        }

        return F_total, drives


class NeedPredictor(nn.Module):
    """
    Predictive model for need dynamics.

    Learns to predict future need states from current state and action,
    enabling model-based planning for homeostatic regulation.
    """

    def __init__(
        self,
        n_needs: int = 8,
        action_dim: int = 4,
        hidden_dim: int = 64,
    ):
        """
        Args:
            n_needs: Number of needs
            action_dim: Action dimension
            hidden_dim: Hidden layer size
        """
        super().__init__()

        self.n_needs = n_needs
        self.action_dim = action_dim

        # Dynamics model: (n_t, a_t) -> n_{t+1}
        self.dynamics = nn.Sequential(
            nn.Linear(n_needs + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_needs),
        )

    def forward(
        self,
        needs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next need state.

        Args:
            needs: Current needs, shape (..., n_needs)
            action: Action taken, shape (..., action_dim)

        Returns:
            Predicted next needs, shape (..., n_needs)
        """
        x = torch.cat([needs, action], dim=-1)
        delta = self.dynamics(x)

        # Residual connection
        return needs + delta

    def rollout(
        self,
        initial_needs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rollout predictions over action sequence.

        Args:
            initial_needs: Starting needs, shape (..., n_needs)
            actions: Action sequence, shape (..., T, action_dim)

        Returns:
            Predicted needs trajectory, shape (..., T+1, n_needs)
        """
        T = actions.shape[-2]
        trajectory = [initial_needs]

        current = initial_needs
        for t in range(T):
            current = self.forward(current, actions[..., t, :])
            trajectory.append(current)

        return torch.stack(trajectory, dim=-2)


if __name__ == "__main__":
    print("=== Homeostatic Regulation Test ===\n")

    # Create regulator
    regulator = HomeostaticRegulator(NeedsConfig(n_needs=8))
    print(f"Number of needs: {regulator.needs.n_needs}")
    print(f"Need names: {regulator.needs.need_names}")
    print(f"Beta: {regulator.beta}")

    # Test free energy computation
    print("\n--- Free Energy ---")
    F_int = regulator.compute_internal_free_energy()
    print(f"Initial F_int: {F_int.item():.4f}")

    # Set some needs away from setpoint
    regulator.needs.set_needs(torch.tensor([2.0, -1.0, 0.5, 1.5, -0.5, 0.0, 1.0, -2.0]))
    F_int = regulator.compute_internal_free_energy()
    print(f"With deviations F_int: {F_int.item():.4f}")

    # Test drive signal
    print("\n--- Drive Signal ---")
    drive = regulator.compute_drive_signal()
    print(f"Drive shape: {drive.shape}")
    print(f"Drive values: {drive}")

    # Test urgency
    print("\n--- Urgency ---")
    idx, name, urgency = regulator.needs.get_most_urgent()
    print(f"Most urgent: {name} (idx={idx}) with urgency={urgency:.4f}")

    # Test observation update
    print("\n--- Observation Update ---")
    obs = torch.randn(8)
    F_int, drive = regulator(obs)
    print(f"After update F_int: {F_int.item():.4f}")

    # Test setpoint adaptation
    print("\n--- Setpoint Adaptation ---")
    print(f"Initial setpoints: {regulator.needs.n_star.data}")
    regulator.adapt_setpoints(adaptation_rate=0.1)
    print(f"Adapted setpoints: {regulator.needs.n_star.data}")

    # Test multi-scale homeostasis
    print("\n--- Multi-Scale Homeostasis ---")
    multi = MultiScaleHomeostasis(n_fast=4, n_medium=4, n_slow=2)
    F_total, drives = multi(
        obs_fast=torch.randn(4),
        obs_medium=torch.randn(4),
        obs_slow=torch.randn(2),
    )
    print(f"Total F: {F_total.item():.4f}")
    print(f"Fast F: {drives['fast_F'].item():.4f}")
    print(f"Medium F: {drives['medium_F'].item():.4f}")
    print(f"Slow F: {drives['slow_F'].item():.4f}")

    # Test need predictor
    print("\n--- Need Predictor ---")
    predictor = NeedPredictor(n_needs=8, action_dim=4)
    needs = torch.randn(8)
    action = torch.randn(4)
    next_needs = predictor(needs, action)
    print(f"Needs shape: {needs.shape} -> {next_needs.shape}")

    # Test rollout
    actions = torch.randn(10, 4)  # 10-step sequence
    trajectory = predictor.rollout(needs, actions)
    print(f"Rollout trajectory shape: {trajectory.shape}")

    # Get stats
    print("\n--- Regulator Stats ---")
    stats = regulator.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n✓ Homeostasis test passed!")
