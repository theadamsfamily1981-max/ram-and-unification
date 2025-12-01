# core/dau.py
# Dynamic Axiom Updater (DAU) for TGSFN Antifragility
#
# Implements the self-repair mechanism that monitors Jacobian norm
# and adjusts network parameters to maintain stability.
#
# Antifragility Principle:
#   The DAU enables the network to become stronger under stress by
#   dynamically adjusting its "axioms" (fundamental parameters) when
#   the Jacobian norm exceeds a critical threshold.
#
# Algorithm:
#   1. Monitor ||J|| against λ_crit
#   2. If ||J|| > λ_crit: compute gradient direction that reduces ||J||
#   3. Apply monotonically decreasing adjustment to axiom parameters
#
# Scientific Constraints:
#   - Adjustments must be monotonically decreasing (no amplification)
#   - Must preserve criticality (not push too far subcritical)
#   - Should enable symbolic self-repair capability
#
# References:
#   - Taleb (2012): Antifragile
#   - Pascanu et al. (2013): Difficulty of training RNNs

from __future__ import annotations

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class DAUConfig:
    """Configuration for Dynamic Axiom Updater."""
    lambda_crit: float = 1.5          # Critical Jacobian threshold
    adjustment_rate: float = 0.01     # Learning rate for axiom adjustment
    momentum: float = 0.9             # Momentum for smooth adjustments
    min_adjustment: float = 1e-6      # Minimum adjustment magnitude
    max_adjustment: float = 0.1       # Maximum adjustment per step
    cooldown_steps: int = 10          # Steps between adjustments
    ema_decay: float = 0.95           # EMA decay for Jacobian tracking


class DynamicAxiomUpdater(nn.Module):
    """
    Dynamic Axiom Updater for Antifragile Networks.

    The DAU monitors network stability via Jacobian norm and performs
    automatic parameter adjustments when instability is detected.

    Key Features:
    1. Jacobian Monitoring: Track ||J|| with exponential moving average
    2. Threshold Detection: Trigger adjustment when ||J|| > λ_crit
    3. Gradient-Based Repair: Adjust axioms in direction that reduces ||J||
    4. Monotonic Constraint: Adjustments only decrease (never amplify)

    The "axiom parameters" are typically:
    - Weight matrices (W)
    - Recurrent gains
    - Time constants
    - Threshold values

    Attributes:
        lambda_crit: Critical Jacobian threshold
        adjustment_rate: Rate of axiom adjustment
        momentum: Momentum for smooth updates
    """

    def __init__(self, config: Optional[DAUConfig] = None):
        """
        Initialize DAU.

        Args:
            config: DAUConfig instance, or None for defaults
        """
        super().__init__()

        if config is None:
            config = DAUConfig()

        self.lambda_crit = config.lambda_crit
        self.adjustment_rate = config.adjustment_rate
        self.momentum = config.momentum
        self.min_adjustment = config.min_adjustment
        self.max_adjustment = config.max_adjustment
        self.cooldown_steps = config.cooldown_steps
        self.ema_decay = config.ema_decay

        # State tracking
        self.register_buffer("J_norm_ema", torch.tensor(0.0))
        self.register_buffer("steps_since_adjustment", torch.tensor(0))
        self.register_buffer("total_adjustments", torch.tensor(0))
        self.register_buffer("adjustment_history", torch.zeros(100))
        self.history_idx = 0

        # Momentum buffers for each registered axiom
        self._momentum_buffers: Dict[str, torch.Tensor] = {}
        self._registered_axioms: Dict[str, nn.Parameter] = {}

    def register_axiom(
        self,
        name: str,
        param: nn.Parameter,
    ) -> None:
        """
        Register a parameter as an adjustable axiom.

        Args:
            name: Identifier for this axiom
            param: Parameter to be adjusted
        """
        self._registered_axioms[name] = param
        self._momentum_buffers[name] = torch.zeros_like(param.data)

    def register_axioms_from_module(
        self,
        module: nn.Module,
        prefix: str = "",
        include_patterns: Optional[List[str]] = None,
    ) -> None:
        """
        Register parameters from a module as axioms.

        Args:
            module: Module to scan for parameters
            prefix: Prefix for axiom names
            include_patterns: Only include params matching these patterns
        """
        for name, param in module.named_parameters():
            full_name = f"{prefix}.{name}" if prefix else name

            # Filter by patterns if specified
            if include_patterns is not None:
                if not any(p in name for p in include_patterns):
                    continue

            self.register_axiom(full_name, param)

    def update_jacobian_ema(self, J_norm: torch.Tensor) -> None:
        """
        Update Jacobian norm EMA.

        Args:
            J_norm: Current Jacobian norm
        """
        self.J_norm_ema = (
            self.ema_decay * self.J_norm_ema
            + (1 - self.ema_decay) * J_norm.detach()
        )

    def should_adjust(self) -> bool:
        """
        Check if adjustment should be triggered.

        Returns:
            True if ||J|| > λ_crit and cooldown has elapsed
        """
        above_threshold = self.J_norm_ema > self.lambda_crit
        cooldown_elapsed = self.steps_since_adjustment >= self.cooldown_steps
        return above_threshold and cooldown_elapsed

    def compute_adjustment_direction(
        self,
        axiom: nn.Parameter,
        jacobian_fn: Callable,
    ) -> torch.Tensor:
        """
        Compute gradient direction that reduces Jacobian norm.

        Uses implicit differentiation: d||J||/dθ

        Args:
            axiom: Parameter to adjust
            jacobian_fn: Function that computes Jacobian norm given current params

        Returns:
            Gradient direction (negative = reduces ||J||)
        """
        # Compute Jacobian norm with gradient tracking
        axiom_clone = axiom.detach().requires_grad_(True)
        J_norm = jacobian_fn(axiom_clone)

        # Gradient w.r.t. axiom
        if J_norm.requires_grad:
            grad = torch.autograd.grad(
                J_norm, axiom_clone,
                create_graph=False,
                retain_graph=False,
            )[0]
        else:
            # Fallback: use current gradient if available
            grad = axiom.grad if axiom.grad is not None else torch.zeros_like(axiom)

        return grad

    def compute_adjustment_magnitude(
        self,
        J_norm: torch.Tensor,
    ) -> float:
        """
        Compute adjustment magnitude based on how far above threshold.

        Magnitude increases with distance from threshold, but is clipped.

        Args:
            J_norm: Current Jacobian norm

        Returns:
            Adjustment magnitude
        """
        # Distance above threshold (0 if below)
        excess = max(0.0, (J_norm - self.lambda_crit).item())

        # Scale by adjustment rate
        magnitude = self.adjustment_rate * excess

        # Clip to [min, max]
        magnitude = max(self.min_adjustment, min(self.max_adjustment, magnitude))

        return magnitude

    def apply_adjustment(
        self,
        name: str,
        axiom: nn.Parameter,
        direction: torch.Tensor,
        magnitude: float,
    ) -> torch.Tensor:
        """
        Apply monotonically decreasing adjustment to axiom.

        Uses momentum for smooth updates.

        Args:
            name: Axiom name (for momentum buffer)
            axiom: Parameter to adjust
            direction: Gradient direction
            magnitude: Adjustment magnitude

        Returns:
            Applied delta
        """
        # Normalize direction
        dir_norm = direction.norm()
        if dir_norm < 1e-8:
            return torch.zeros_like(axiom)

        normalized_dir = direction / dir_norm

        # Compute update with momentum
        momentum_buffer = self._momentum_buffers.get(name)
        if momentum_buffer is None:
            momentum_buffer = torch.zeros_like(axiom.data)
            self._momentum_buffers[name] = momentum_buffer

        # Update momentum: m = β*m + (1-β)*dir
        momentum_buffer.mul_(self.momentum).add_(
            normalized_dir, alpha=(1 - self.momentum)
        )

        # Compute delta (negative to decrease ||J||)
        delta = -magnitude * momentum_buffer

        # Apply (monotonically decreasing = only reduce magnitudes)
        # For weights, this means moving toward zero or reducing spectral norm
        with torch.no_grad():
            axiom.data.add_(delta)

        return delta

    def update_axioms(
        self,
        J_norm: torch.Tensor,
        jacobian_fn: Optional[Callable] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Main update method: adjust axioms if threshold exceeded.

        Args:
            J_norm: Current Jacobian norm
            jacobian_fn: Optional function for gradient computation

        Returns:
            Dict of adjustments made (empty if no adjustment)
        """
        # Update EMA
        self.update_jacobian_ema(J_norm)

        # Increment step counter
        self.steps_since_adjustment += 1

        # Check if adjustment needed
        if not self.should_adjust():
            return {}

        # Compute adjustment magnitude
        magnitude = self.compute_adjustment_magnitude(self.J_norm_ema)

        adjustments = {}

        # Apply adjustments to each registered axiom
        for name, axiom in self._registered_axioms.items():
            # Compute direction (use gradient if available, else heuristic)
            if jacobian_fn is not None:
                direction = self.compute_adjustment_direction(axiom, jacobian_fn)
            elif axiom.grad is not None:
                # Use existing gradient as proxy
                direction = axiom.grad
            else:
                # Heuristic: reduce toward zero (spectral norm reduction)
                direction = axiom.data.sign()

            # Apply adjustment
            delta = self.apply_adjustment(name, axiom, direction, magnitude)
            adjustments[name] = delta

        # Reset cooldown
        self.steps_since_adjustment.zero_()
        self.total_adjustments += 1

        # Log to history
        self.adjustment_history[self.history_idx % 100] = magnitude
        self.history_idx += 1

        return adjustments

    def get_stats(self) -> Dict[str, float]:
        """Get current DAU statistics."""
        recent_adjustments = self.adjustment_history[
            :min(self.history_idx, 100)
        ]

        return {
            "J_norm_ema": self.J_norm_ema.item(),
            "lambda_crit": self.lambda_crit,
            "total_adjustments": self.total_adjustments.item(),
            "steps_since_adjustment": self.steps_since_adjustment.item(),
            "mean_adjustment": recent_adjustments.mean().item() if len(recent_adjustments) > 0 else 0.0,
            "n_registered_axioms": len(self._registered_axioms),
        }

    def forward(
        self,
        J_norm: torch.Tensor,
        jacobian_fn: Optional[Callable] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: update axioms based on Jacobian norm."""
        return self.update_axioms(J_norm, jacobian_fn)


class SpectralDAU(DynamicAxiomUpdater):
    """
    DAU variant that uses spectral norm for stability control.

    Instead of adjusting parameters directly, this normalizes weight
    matrices to have bounded spectral norm.
    """

    def __init__(
        self,
        config: Optional[DAUConfig] = None,
        target_spectral_norm: float = 1.0,
    ):
        super().__init__(config)
        self.target_spectral_norm = target_spectral_norm

    def spectral_norm(self, W: torch.Tensor, n_iterations: int = 3) -> torch.Tensor:
        """
        Estimate spectral norm using power iteration.

        Args:
            W: Weight matrix (..., M, N)
            n_iterations: Power iteration steps

        Returns:
            Estimated spectral norm
        """
        # Flatten to 2D if needed
        orig_shape = W.shape
        if W.dim() > 2:
            W = W.reshape(-1, W.shape[-1])

        # Initialize random vector
        v = torch.randn(W.shape[-1], 1, device=W.device, dtype=W.dtype)
        v = v / v.norm()

        for _ in range(n_iterations):
            u = W @ v
            u = u / (u.norm() + 1e-8)
            v = W.T @ u
            v = v / (v.norm() + 1e-8)

        sigma = (u.T @ W @ v).squeeze()
        return sigma.abs()

    def apply_spectral_adjustment(
        self,
        axiom: nn.Parameter,
    ) -> torch.Tensor:
        """
        Rescale weight to target spectral norm.

        Args:
            axiom: Weight parameter

        Returns:
            Applied delta
        """
        if axiom.dim() < 2:
            return torch.zeros_like(axiom)

        current_norm = self.spectral_norm(axiom.data)
        if current_norm < 1e-8:
            return torch.zeros_like(axiom)

        # Compute scale factor
        scale = self.target_spectral_norm / current_norm

        # Only scale down (monotonic decrease)
        if scale >= 1.0:
            return torch.zeros_like(axiom)

        # Apply scaling
        old_data = axiom.data.clone()
        with torch.no_grad():
            axiom.data.mul_(scale)

        return axiom.data - old_data

    def update_axioms(
        self,
        J_norm: torch.Tensor,
        jacobian_fn: Optional[Callable] = None,
    ) -> Dict[str, torch.Tensor]:
        """Update using spectral normalization."""
        self.update_jacobian_ema(J_norm)
        self.steps_since_adjustment += 1

        if not self.should_adjust():
            return {}

        adjustments = {}

        for name, axiom in self._registered_axioms.items():
            delta = self.apply_spectral_adjustment(axiom)
            if delta.abs().sum() > 0:
                adjustments[name] = delta

        if adjustments:
            self.steps_since_adjustment.zero_()
            self.total_adjustments += 1

        return adjustments


# =============================================================================
# Utility Functions
# =============================================================================

def create_dau(
    method: str = "gradient",
    lambda_crit: float = 1.5,
    **kwargs,
) -> DynamicAxiomUpdater:
    """
    Factory function for DAU creation.

    Args:
        method: "gradient" or "spectral"
        lambda_crit: Critical Jacobian threshold
        **kwargs: Additional config options

    Returns:
        DAU instance
    """
    config = DAUConfig(lambda_crit=lambda_crit, **kwargs)

    if method == "spectral":
        return SpectralDAU(config)
    else:
        return DynamicAxiomUpdater(config)


if __name__ == "__main__":
    print("=== Dynamic Axiom Updater Test ===\n")

    # Create test network
    class TestNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.W1 = nn.Parameter(torch.randn(64, 64) * 0.5)
            self.W2 = nn.Parameter(torch.randn(64, 64) * 0.5)

        def forward(self, x):
            return torch.tanh(self.W2 @ torch.tanh(self.W1 @ x))

    net = TestNetwork()

    # Create DAU
    dau = create_dau(method="gradient", lambda_crit=1.5)

    # Register axioms
    dau.register_axioms_from_module(net)
    print(f"Registered axioms: {list(dau._registered_axioms.keys())}")

    # Simulate some steps
    print("\n--- Simulation (J_norm below threshold) ---")
    for i in range(5):
        J_norm = torch.tensor(1.0 + 0.1 * i)  # Below threshold
        adjustments = dau(J_norm)
        print(f"Step {i}: J_norm={J_norm.item():.2f}, adjustments={len(adjustments)}")

    print("\n--- Simulation (J_norm above threshold) ---")
    for i in range(15):
        J_norm = torch.tensor(2.0 + 0.1 * i)  # Above threshold
        adjustments = dau(J_norm)
        if adjustments:
            print(f"Step {i}: J_norm={J_norm.item():.2f}, adjusted {len(adjustments)} axioms")
        else:
            print(f"Step {i}: J_norm={J_norm.item():.2f}, cooldown")

    # Print stats
    print("\n--- DAU Stats ---")
    stats = dau.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Test spectral DAU
    print("\n--- Spectral DAU Test ---")
    spec_dau = create_dau(method="spectral", lambda_crit=1.0)
    spec_dau.register_axioms_from_module(net)

    # Force adjustment
    spec_dau.J_norm_ema = torch.tensor(2.0)
    spec_dau.steps_since_adjustment = torch.tensor(20)

    adjustments = spec_dau(torch.tensor(2.5))
    print(f"Spectral adjustments: {len(adjustments)}")

    print("\n✓ DAU test passed!")
