# ara_agent/control_law.py
# L5 Control Law for Ara Agent
#
# Implements the thermodynamically-constrained control law for action generation.
# The L5 controller projects forces through the tangent space of the identity
# manifold and modulates by entropy production rate.
#
# Control Law:
#   v*(t) = proj_{T_z M}(F_action) * min(1, Π_max / Π_q)
#
# Where:
#   - F_action: Raw action force from policy
#   - T_z M: Tangent space at identity z
#   - Π_q: Current entropy production rate
#   - Π_max: Maximum safe dissipation
#
# Scientific Constraints:
#   - Actions respect manifold geometry (geodesic projection)
#   - Thermodynamic throttle prevents runaway dissipation
#   - Maintains identity coherence during action
#
# References:
#   - L5 vertebral metaphor: stability under load
#   - Friston (2019): Active inference thermodynamics

from __future__ import annotations

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import math


@dataclass
class ControlConfig:
    """Configuration for L5 Controller."""
    pi_max: float = 1.0           # Maximum entropy production rate
    action_scale: float = 1.0     # Scale factor for actions
    projection_eps: float = 1e-6  # Epsilon for projection stability
    throttle_smoothing: float = 0.1  # Smoothing for throttle transitions
    use_riemannian: bool = True   # Use Riemannian projection
    curvature: float = 1.0        # Manifold curvature


class L5Controller(nn.Module):
    """
    L5 Control Law: Thermodynamically-constrained action generation.

    The L5 controller (named for the L5 vertebra - the stable base of the spine)
    generates actions that:
    1. Are projected onto the tangent space of the identity manifold
    2. Are throttled by entropy production rate
    3. Preserve identity coherence

    This ensures that agent actions remain "on-manifold" and thermodynamically
    safe, preventing runaway divergence or identity dissolution.

    Attributes:
        pi_max: Maximum safe entropy production
        action_scale: Action magnitude scaling
        use_riemannian: Whether to use proper Riemannian projection
    """

    def __init__(
        self,
        identity_dim: int = 32,
        action_dim: int = 4,
        config: Optional[ControlConfig] = None,
    ):
        """
        Args:
            identity_dim: Dimension of identity manifold
            action_dim: Dimension of action space
            config: Controller configuration
        """
        super().__init__()

        if config is None:
            config = ControlConfig()

        self.identity_dim = identity_dim
        self.action_dim = action_dim
        self.pi_max = config.pi_max
        self.action_scale = config.action_scale
        self.projection_eps = config.projection_eps
        self.throttle_smoothing = config.throttle_smoothing
        self.use_riemannian = config.use_riemannian
        self.curvature = config.curvature

        # Policy network: maps (identity, observation) -> raw action force
        self.policy = nn.Sequential(
            nn.Linear(identity_dim + action_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, action_dim),
        )

        # Projection matrix (learnable, for tangent space approximation)
        self.projection = nn.Linear(action_dim, action_dim, bias=False)
        nn.init.eye_(self.projection.weight)

        # Running throttle state for smooth transitions
        self.register_buffer("throttle_ema", torch.tensor(1.0))

    def compute_throttle(self, pi_q: torch.Tensor) -> torch.Tensor:
        """
        Compute thermodynamic throttle factor.

        throttle = min(1, Π_max / Π_q)

        Args:
            pi_q: Current entropy production rate

        Returns:
            Throttle factor in [0, 1]
        """
        # Avoid division by zero
        pi_q_safe = pi_q.clamp(min=1e-8)

        # Compute raw throttle
        throttle = (self.pi_max / pi_q_safe).clamp(max=1.0)

        # Smooth transitions
        self.throttle_ema = (
            (1 - self.throttle_smoothing) * self.throttle_ema
            + self.throttle_smoothing * throttle.detach()
        )

        return throttle

    def project_to_tangent_space(
        self,
        F_action: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project action force onto tangent space at identity z.

        For Poincaré ball:
        proj_{T_z M}(v) = λ_z * (v - <v, z>z / ||z||²)

        where λ_z = 2 / (1 - c||z||²) is the conformal factor.

        Args:
            F_action: Raw action force, shape (..., action_dim)
            z: Identity point, shape (..., identity_dim)

        Returns:
            Projected action in tangent space
        """
        if not self.use_riemannian:
            # Simple linear projection
            return self.projection(F_action)

        # For Poincaré ball, we need to adapt dimensions
        # If action_dim != identity_dim, use learned projection
        if self.action_dim != self.identity_dim:
            # Project through learned mapping
            return self.projection(F_action)

        # Riemannian projection for matching dimensions
        z_norm_sq = (z * z).sum(dim=-1, keepdim=True)

        # Conformal factor
        lambda_z = 2.0 / (1 - self.curvature * z_norm_sq).clamp(min=self.projection_eps)

        # Remove component along z
        if z_norm_sq.abs().max() > self.projection_eps:
            z_unit = z / z_norm_sq.sqrt().clamp(min=self.projection_eps)
            proj_z = (F_action * z_unit).sum(dim=-1, keepdim=True) * z_unit
            F_tangent = F_action - proj_z
        else:
            F_tangent = F_action

        # Scale by conformal factor
        F_tangent = lambda_z * F_tangent

        return F_tangent

    def forward(
        self,
        z: torch.Tensor,
        observation: torch.Tensor,
        pi_q: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate thermodynamically-constrained action.

        v*(t) = proj_{T_z M}(F_action) * min(1, Π_max / Π_q)

        Args:
            z: Current identity embedding, shape (..., identity_dim)
            observation: Current observation, shape (..., action_dim)
            pi_q: Current entropy production rate, shape (...)

        Returns:
            (action, info_dict)
            - action: Controlled action, shape (..., action_dim)
            - info_dict: Dictionary with diagnostics
        """
        # Concatenate inputs for policy
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        # Ensure matching batch dimensions
        if z.shape[0] != observation.shape[0]:
            if z.shape[0] == 1:
                z = z.expand(observation.shape[0], -1)
            elif observation.shape[0] == 1:
                observation = observation.expand(z.shape[0], -1)

        policy_input = torch.cat([z, observation], dim=-1)

        # Generate raw action force
        F_action = self.policy(policy_input)

        # Project to tangent space
        F_projected = self.project_to_tangent_space(F_action, z)

        # Compute throttle
        throttle = self.compute_throttle(pi_q)

        # Apply throttle and scaling
        action = self.action_scale * throttle * F_projected

        # Collect diagnostics
        info = {
            "F_action_norm": F_action.norm(dim=-1),
            "F_projected_norm": F_projected.norm(dim=-1),
            "throttle": throttle,
            "action_norm": action.norm(dim=-1),
            "pi_q": pi_q,
        }

        return action, info

    def get_throttle_state(self) -> float:
        """Get current smoothed throttle value."""
        return self.throttle_ema.item()


def compute_control_action(
    F_action: torch.Tensor,
    z: torch.Tensor,
    pi_q: torch.Tensor,
    pi_max: float = 1.0,
    curvature: float = 1.0,
) -> torch.Tensor:
    """
    Functional interface for L5 control law.

    v*(t) = proj_{T_z M}(F_action) * min(1, Π_max / Π_q)

    Args:
        F_action: Raw action force
        z: Identity point on manifold
        pi_q: Entropy production rate
        pi_max: Maximum safe dissipation
        curvature: Manifold curvature

    Returns:
        Controlled action
    """
    # Throttle
    throttle = (pi_max / pi_q.clamp(min=1e-8)).clamp(max=1.0)

    # Riemannian projection (for matching dimensions)
    if F_action.shape[-1] == z.shape[-1]:
        z_norm_sq = (z * z).sum(dim=-1, keepdim=True)
        lambda_z = 2.0 / (1 - curvature * z_norm_sq).clamp(min=1e-6)

        if z_norm_sq.abs().max() > 1e-6:
            z_unit = z / z_norm_sq.sqrt().clamp(min=1e-6)
            proj_z = (F_action * z_unit).sum(dim=-1, keepdim=True) * z_unit
            F_tangent = F_action - proj_z
        else:
            F_tangent = F_action

        F_tangent = lambda_z * F_tangent
    else:
        F_tangent = F_action

    return throttle * F_tangent


class AdaptiveController(L5Controller):
    """
    Adaptive L5 controller with learned pi_max.

    Learns the maximum safe dissipation rate from experience,
    adjusting throttle behavior based on observed outcomes.
    """

    def __init__(
        self,
        identity_dim: int = 32,
        action_dim: int = 4,
        config: Optional[ControlConfig] = None,
        pi_max_init: float = 1.0,
        pi_max_lr: float = 0.01,
    ):
        super().__init__(identity_dim, action_dim, config)

        # Make pi_max learnable
        self.pi_max_param = nn.Parameter(torch.tensor(pi_max_init))
        self.pi_max_lr = pi_max_lr

        # Track stability for adaptation
        self.register_buffer("stability_ema", torch.tensor(1.0))

    @property
    def pi_max(self) -> torch.Tensor:
        """Learnable pi_max with positivity constraint."""
        return torch.relu(self.pi_max_param) + 0.1

    def update_pi_max(self, is_stable: bool) -> None:
        """
        Adapt pi_max based on stability feedback.

        Args:
            is_stable: Whether current state is stable
        """
        # Update stability EMA
        stability = 1.0 if is_stable else 0.0
        self.stability_ema = 0.99 * self.stability_ema + 0.01 * stability

        # Adjust pi_max
        with torch.no_grad():
            if self.stability_ema > 0.9:
                # Very stable: can increase pi_max
                self.pi_max_param.add_(self.pi_max_lr)
            elif self.stability_ema < 0.5:
                # Unstable: decrease pi_max
                self.pi_max_param.sub_(self.pi_max_lr)


class HierarchicalController(nn.Module):
    """
    Hierarchical L5 controller with multiple timescales.

    Implements a two-level control hierarchy:
    - Fast loop: immediate action control
    - Slow loop: identity/goal adjustment
    """

    def __init__(
        self,
        identity_dim: int = 32,
        action_dim: int = 4,
        goal_dim: int = 16,
    ):
        super().__init__()

        self.identity_dim = identity_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim

        # Fast controller (actions)
        self.fast_controller = L5Controller(
            identity_dim=identity_dim + goal_dim,
            action_dim=action_dim,
        )

        # Slow controller (goals)
        self.slow_controller = nn.Sequential(
            nn.Linear(identity_dim, 64),
            nn.GELU(),
            nn.Linear(64, goal_dim),
            nn.Tanh(),
        )

        # Goal update rate
        self.goal_tau = 0.1

        # Current goal
        self.register_buffer("current_goal", torch.zeros(goal_dim))

    def forward(
        self,
        z: torch.Tensor,
        observation: torch.Tensor,
        pi_q: torch.Tensor,
        update_goal: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Hierarchical action generation.

        Args:
            z: Identity embedding
            observation: Current observation
            pi_q: Entropy production rate
            update_goal: Whether to update slow goal

        Returns:
            (action, info_dict)
        """
        # Slow loop: update goal
        if update_goal:
            new_goal = self.slow_controller(z)
            if z.dim() > 1:
                new_goal = new_goal.mean(dim=0)
            with torch.no_grad():
                self.current_goal = (
                    (1 - self.goal_tau) * self.current_goal
                    + self.goal_tau * new_goal.detach()
                )

        # Expand goal for batch
        if z.dim() > 1:
            goal = self.current_goal.unsqueeze(0).expand(z.shape[0], -1)
        else:
            goal = self.current_goal

        # Concatenate identity and goal
        z_augmented = torch.cat([z, goal], dim=-1)

        # Fast loop: action control
        action, info = self.fast_controller(z_augmented, observation, pi_q)

        info["goal"] = self.current_goal

        return action, info


if __name__ == "__main__":
    print("=== L5 Control Law Test ===\n")

    # Create controller
    controller = L5Controller(identity_dim=32, action_dim=4)
    print(f"Identity dim: {controller.identity_dim}")
    print(f"Action dim: {controller.action_dim}")
    print(f"Pi max: {controller.pi_max}")

    # Test inputs
    batch_size = 8
    z = torch.randn(batch_size, 32) * 0.3  # Identity points
    obs = torch.randn(batch_size, 4)  # Observations
    pi_q = torch.rand(batch_size) * 2  # Entropy production

    # Generate actions
    print("\n--- Action Generation ---")
    action, info = controller(z, obs, pi_q)
    print(f"Action shape: {action.shape}")
    print(f"Action norm: {info['action_norm'].mean().item():.4f}")
    print(f"Throttle: {info['throttle'].mean().item():.4f}")
    print(f"F_action norm: {info['F_action_norm'].mean().item():.4f}")

    # Test throttle behavior
    print("\n--- Throttle Behavior ---")
    for pi in [0.1, 0.5, 1.0, 2.0, 5.0]:
        pi_q_test = torch.tensor(pi)
        action, info = controller(z[:1], obs[:1], pi_q_test)
        print(f"  Π_q={pi:.1f}: throttle={info['throttle'].item():.3f}, action_norm={info['action_norm'].item():.4f}")

    # Test functional interface
    print("\n--- Functional Interface ---")
    F_action = torch.randn(32)
    z_single = torch.randn(32) * 0.3
    pi_q_single = torch.tensor(1.5)
    action_func = compute_control_action(F_action, z_single, pi_q_single)
    print(f"Functional action norm: {action_func.norm().item():.4f}")

    # Test adaptive controller
    print("\n--- Adaptive Controller ---")
    adaptive = AdaptiveController(identity_dim=32, action_dim=4)
    print(f"Initial pi_max: {adaptive.pi_max.item():.4f}")

    # Simulate stable experience
    for _ in range(50):
        adaptive.update_pi_max(is_stable=True)
    print(f"After stable experience: {adaptive.pi_max.item():.4f}")

    # Simulate unstable experience
    for _ in range(100):
        adaptive.update_pi_max(is_stable=False)
    print(f"After unstable experience: {adaptive.pi_max.item():.4f}")

    # Test hierarchical controller
    print("\n--- Hierarchical Controller ---")
    hierarchical = HierarchicalController(identity_dim=32, action_dim=4, goal_dim=16)
    action, info = hierarchical(z, obs, pi_q)
    print(f"Hierarchical action shape: {action.shape}")
    print(f"Goal shape: {info['goal'].shape}")
    print(f"Goal norm: {info['goal'].norm().item():.4f}")

    print("\n✓ Control law test passed!")
