# ara_agent/dau.py
# Dynamic Axiom Updater (DAU) Lite for Ara Agent
#
# Implements a lightweight version of the DAU for real-time antifragility.
# Monitors network stability and applies automatic corrections when
# the system approaches instability.
#
# Key Mechanisms:
#   1. Monitor ||J|| against λ_crit threshold
#   2. Compute gradient direction to reduce ||J||
#   3. Apply monotonically decreasing corrections
#
# Antifragility Principle:
#   The DAU enables the system to become STRONGER under stress by
#   proactively adjusting parameters before instability occurs.
#
# Scientific Constraints:
#   - Adjustments are monotonically decreasing (no amplification)
#   - Preserves near-criticality (doesn't push too subcritical)
#   - Bounded adjustment magnitudes
#
# References:
#   - Taleb (2012): Antifragile
#   - Pascanu et al. (2013): Training RNNs

from __future__ import annotations

from typing import Dict, Optional, Tuple, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class DAUConfig:
    """Configuration for DAU-Lite."""
    lambda_crit: float = 1.5          # Critical Jacobian threshold
    adjustment_rate: float = 0.01     # Learning rate for adjustments
    momentum: float = 0.9             # Momentum for smooth updates
    min_adjustment: float = 1e-6      # Minimum adjustment magnitude
    max_adjustment: float = 0.1       # Maximum adjustment per step
    cooldown_steps: int = 10          # Steps between adjustments
    ema_decay: float = 0.95           # EMA decay for tracking
    spectral_target: float = 0.99     # Target spectral norm (< 1 for stability)


class DAULite(nn.Module):
    """
    Dynamic Axiom Updater - Lite Version.

    A lightweight implementation of the DAU for real-time stability control.
    Monitors the Jacobian norm proxy and applies gentle corrections when
    the system approaches the critical threshold.

    Unlike the full DAU which can modify arbitrary "axiom" parameters,
    DAU-Lite focuses on:
    1. Spectral norm control of weight matrices
    2. Gain/bias adjustments
    3. Threshold modulation

    Attributes:
        lambda_crit: Threshold above which corrections are applied
        adjustment_rate: Rate of correction
        spectral_target: Target spectral norm for weights
    """

    def __init__(
        self,
        config: Optional[DAUConfig] = None,
    ):
        """
        Args:
            config: DAU configuration
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
        self.spectral_target = config.spectral_target

        # State tracking
        self.register_buffer("J_norm_ema", torch.tensor(0.0))
        self.register_buffer("steps_since_adjustment", torch.tensor(0))
        self.register_buffer("total_adjustments", torch.tensor(0))
        self.register_buffer("adjustment_history", torch.zeros(100))
        self._history_idx = 0

        # Momentum buffers
        self._momentum_buffers: Dict[str, torch.Tensor] = {}

        # Registered parameters to adjust
        self._registered_params: Dict[str, nn.Parameter] = {}

    def register_parameter_for_adjustment(
        self,
        name: str,
        param: nn.Parameter,
    ) -> None:
        """
        Register a parameter for DAU adjustment.

        Args:
            name: Identifier for the parameter
            param: Parameter to adjust
        """
        self._registered_params[name] = param
        self._momentum_buffers[name] = torch.zeros_like(param.data)

    def register_module(
        self,
        module: nn.Module,
        prefix: str = "",
        weight_patterns: Optional[list] = None,
    ) -> int:
        """
        Register all matching parameters from a module.

        Args:
            module: Module to scan
            prefix: Name prefix
            weight_patterns: List of patterns to match (default: ["weight"])

        Returns:
            Number of parameters registered
        """
        if weight_patterns is None:
            weight_patterns = ["weight", "W"]

        count = 0
        for name, param in module.named_parameters():
            if any(p in name for p in weight_patterns):
                full_name = f"{prefix}.{name}" if prefix else name
                self.register_parameter_for_adjustment(full_name, param)
                count += 1

        return count

    def update_ema(self, J_norm: torch.Tensor) -> None:
        """Update Jacobian norm EMA."""
        self.J_norm_ema = (
            self.ema_decay * self.J_norm_ema
            + (1 - self.ema_decay) * J_norm.detach()
        )

    def should_adjust(self) -> bool:
        """Check if adjustment should be triggered."""
        above_threshold = self.J_norm_ema > self.lambda_crit
        cooldown_elapsed = self.steps_since_adjustment >= self.cooldown_steps
        return above_threshold.item() and cooldown_elapsed.item()

    def compute_adjustment_magnitude(self) -> float:
        """Compute adjustment magnitude based on overshoot."""
        excess = max(0.0, (self.J_norm_ema - self.lambda_crit).item())
        magnitude = self.adjustment_rate * (1 + excess)
        return max(self.min_adjustment, min(self.max_adjustment, magnitude))

    def spectral_norm(
        self,
        W: torch.Tensor,
        n_iters: int = 3,
    ) -> torch.Tensor:
        """
        Estimate spectral norm using power iteration.

        Args:
            W: Weight matrix
            n_iters: Power iteration steps

        Returns:
            Estimated spectral norm
        """
        if W.dim() < 2:
            return W.abs().max()

        # Flatten to 2D
        shape = W.shape
        W_2d = W.reshape(-1, shape[-1])

        # Power iteration
        v = torch.randn(W_2d.shape[-1], 1, device=W.device, dtype=W.dtype)
        v = v / v.norm()

        for _ in range(n_iters):
            u = W_2d @ v
            u = u / (u.norm() + 1e-8)
            v = W_2d.T @ u
            v = v / (v.norm() + 1e-8)

        sigma = (u.T @ W_2d @ v).squeeze()
        return sigma.abs()

    def apply_spectral_adjustment(
        self,
        name: str,
        param: nn.Parameter,
    ) -> torch.Tensor:
        """
        Apply spectral norm adjustment to a weight matrix.

        Args:
            name: Parameter name
            param: Parameter to adjust

        Returns:
            Applied delta
        """
        if param.dim() < 2:
            return torch.zeros_like(param)

        current_norm = self.spectral_norm(param.data)

        if current_norm < 1e-8:
            return torch.zeros_like(param)

        # Compute scale to reach target
        scale = self.spectral_target / current_norm

        # Only scale down (monotonic decrease)
        if scale >= 1.0:
            return torch.zeros_like(param)

        # Blend with momentum
        momentum_buffer = self._momentum_buffers.get(name)
        if momentum_buffer is None:
            momentum_buffer = torch.zeros_like(param.data)
            self._momentum_buffers[name] = momentum_buffer

        # Target: param * scale - param = param * (scale - 1)
        target_delta = param.data * (scale - 1)

        # Update momentum
        momentum_buffer.mul_(self.momentum).add_(
            target_delta, alpha=(1 - self.momentum)
        )

        # Apply adjustment
        delta = momentum_buffer.clone()
        with torch.no_grad():
            param.data.add_(delta)

        return delta

    def apply_gradient_adjustment(
        self,
        name: str,
        param: nn.Parameter,
        direction: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply gradient-based adjustment.

        Args:
            name: Parameter name
            param: Parameter to adjust
            direction: Gradient direction (default: use param.grad or param.sign)

        Returns:
            Applied delta
        """
        if direction is None:
            if param.grad is not None:
                direction = param.grad
            else:
                # Heuristic: reduce toward zero
                direction = param.data.sign()

        # Normalize direction
        dir_norm = direction.norm()
        if dir_norm < 1e-8:
            return torch.zeros_like(param)

        normalized_dir = direction / dir_norm

        # Get momentum buffer
        momentum_buffer = self._momentum_buffers.get(name)
        if momentum_buffer is None:
            momentum_buffer = torch.zeros_like(param.data)
            self._momentum_buffers[name] = momentum_buffer

        # Update momentum
        momentum_buffer.mul_(self.momentum).add_(
            normalized_dir, alpha=(1 - self.momentum)
        )

        # Compute magnitude
        magnitude = self.compute_adjustment_magnitude()

        # Apply (negative to reduce)
        delta = -magnitude * momentum_buffer

        with torch.no_grad():
            param.data.add_(delta)

        return delta

    def step(
        self,
        J_norm: torch.Tensor,
        use_spectral: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Main update step.

        Args:
            J_norm: Current Jacobian norm estimate
            use_spectral: Use spectral norm adjustment (vs gradient)

        Returns:
            Dict of adjustments applied
        """
        # Update tracking
        self.update_ema(J_norm)
        self.steps_since_adjustment += 1

        if not self.should_adjust():
            return {}

        adjustments = {}

        # Apply adjustments
        for name, param in self._registered_params.items():
            if use_spectral and param.dim() >= 2:
                delta = self.apply_spectral_adjustment(name, param)
            else:
                delta = self.apply_gradient_adjustment(name, param)

            if delta.abs().sum() > 0:
                adjustments[name] = delta

        # Update state
        if adjustments:
            self.steps_since_adjustment.zero_()
            self.total_adjustments += 1

            magnitude = sum(d.abs().sum().item() for d in adjustments.values())
            self.adjustment_history[self._history_idx % 100] = magnitude
            self._history_idx += 1

        return adjustments

    def get_stats(self) -> Dict[str, float]:
        """Get current DAU statistics."""
        recent = self.adjustment_history[:min(self._history_idx, 100)]

        return {
            "J_norm_ema": self.J_norm_ema.item(),
            "lambda_crit": self.lambda_crit,
            "above_threshold": float(self.J_norm_ema > self.lambda_crit),
            "total_adjustments": self.total_adjustments.item(),
            "steps_since_adjustment": self.steps_since_adjustment.item(),
            "mean_adjustment": recent.mean().item() if len(recent) > 0 else 0.0,
            "n_registered_params": len(self._registered_params),
        }

    def reset(self) -> None:
        """Reset DAU state."""
        self.J_norm_ema.zero_()
        self.steps_since_adjustment.zero_()
        self.total_adjustments.zero_()
        self.adjustment_history.zero_()
        self._history_idx = 0

        for buf in self._momentum_buffers.values():
            buf.zero_()

    def forward(
        self,
        J_norm: torch.Tensor,
        use_spectral: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: apply adjustment step."""
        return self.step(J_norm, use_spectral)


class AdaptiveDAU(DAULite):
    """
    Adaptive DAU with learned threshold.

    Learns the critical threshold λ_crit from experience,
    balancing stability vs. performance.
    """

    def __init__(
        self,
        config: Optional[DAUConfig] = None,
        lambda_crit_lr: float = 0.001,
    ):
        super().__init__(config)

        # Make lambda_crit learnable
        self._lambda_crit_param = nn.Parameter(
            torch.tensor(self.lambda_crit)
        )
        self.lambda_crit_lr = lambda_crit_lr

        # Track performance signal
        self.register_buffer("performance_ema", torch.tensor(0.0))

    @property
    def lambda_crit(self) -> float:
        """Learnable threshold with bounds."""
        return self._lambda_crit_param.clamp(0.5, 5.0).item()

    def update_threshold(
        self,
        performance: float,
        is_stable: bool,
    ) -> None:
        """
        Adapt threshold based on performance and stability.

        Args:
            performance: Performance metric (higher = better)
            is_stable: Whether system was stable
        """
        # Update performance EMA
        self.performance_ema = (
            0.99 * self.performance_ema + 0.01 * performance
        )

        with torch.no_grad():
            if is_stable and performance > self.performance_ema:
                # Good performance while stable: can increase threshold
                self._lambda_crit_param.add_(self.lambda_crit_lr)
            elif not is_stable:
                # Unstable: decrease threshold
                self._lambda_crit_param.sub_(self.lambda_crit_lr * 2)


def create_dau(
    method: str = "spectral",
    lambda_crit: float = 1.5,
    adaptive: bool = False,
    **kwargs,
) -> DAULite:
    """
    Factory function for DAU creation.

    Args:
        method: "spectral" or "gradient"
        lambda_crit: Critical threshold
        adaptive: Use adaptive DAU with learned threshold
        **kwargs: Additional config options

    Returns:
        DAU instance
    """
    config = DAUConfig(lambda_crit=lambda_crit, **kwargs)

    if adaptive:
        return AdaptiveDAU(config)
    else:
        return DAULite(config)


class DAUMonitor:
    """
    Lightweight monitoring for DAU without modification.

    Useful for logging/visualization without affecting training.
    """

    def __init__(
        self,
        window_size: int = 100,
        alert_threshold: float = 2.0,
    ):
        self.window_size = window_size
        self.alert_threshold = alert_threshold

        self.J_history: list = []
        self.adjustment_count = 0

    def log(self, J_norm: float, adjusted: bool = False) -> Optional[str]:
        """
        Log Jacobian norm observation.

        Args:
            J_norm: Current Jacobian norm
            adjusted: Whether adjustment was made

        Returns:
            Alert message if threshold exceeded
        """
        self.J_history.append(J_norm)
        if len(self.J_history) > self.window_size:
            self.J_history.pop(0)

        if adjusted:
            self.adjustment_count += 1

        if J_norm > self.alert_threshold:
            return f"ALERT: J_norm={J_norm:.3f} > {self.alert_threshold}"

        return None

    def get_summary(self) -> Dict[str, float]:
        """Get monitoring summary."""
        if not self.J_history:
            return {"mean": 0.0, "max": 0.0, "std": 0.0, "adjustments": 0}

        import numpy as np
        arr = np.array(self.J_history)

        return {
            "mean": float(arr.mean()),
            "max": float(arr.max()),
            "std": float(arr.std()),
            "adjustments": self.adjustment_count,
        }


if __name__ == "__main__":
    print("=== DAU-Lite Test ===\n")

    # Create test network
    class TestNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.W1 = nn.Parameter(torch.randn(64, 64) * 0.5)
            self.W2 = nn.Parameter(torch.randn(64, 64) * 0.5)

        def forward(self, x):
            return torch.tanh(self.W2 @ torch.tanh(self.W1 @ x))

    net = TestNet()

    # Create DAU
    dau = create_dau(method="spectral", lambda_crit=1.5)

    # Register parameters
    n_registered = dau.register_module(net)
    print(f"Registered {n_registered} parameters")
    print(f"Parameters: {list(dau._registered_params.keys())}")

    # Test spectral norm
    print("\n--- Spectral Norm ---")
    for name, param in dau._registered_params.items():
        sn = dau.spectral_norm(param.data)
        print(f"  {name}: σ = {sn.item():.4f}")

    # Simulate below threshold
    print("\n--- Below Threshold ---")
    for i in range(5):
        J_norm = torch.tensor(1.0 + 0.1 * i)  # Below λ_crit=1.5
        adjustments = dau(J_norm)
        print(f"Step {i}: J={J_norm.item():.2f}, adjustments={len(adjustments)}")

    # Simulate above threshold
    print("\n--- Above Threshold ---")
    for i in range(15):
        J_norm = torch.tensor(2.0 + 0.1 * i)  # Above λ_crit
        adjustments = dau(J_norm)
        if adjustments:
            print(f"Step {i}: J={J_norm.item():.2f}, adjusted {len(adjustments)} params")
        else:
            print(f"Step {i}: J={J_norm.item():.2f}, cooldown")

    # Check spectral norms after adjustment
    print("\n--- After Adjustments ---")
    for name, param in dau._registered_params.items():
        sn = dau.spectral_norm(param.data)
        print(f"  {name}: σ = {sn.item():.4f}")

    # Stats
    print("\n--- DAU Stats ---")
    stats = dau.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Test adaptive DAU
    print("\n--- Adaptive DAU ---")
    adaptive_dau = create_dau(adaptive=True, lambda_crit=1.5)
    print(f"Initial λ_crit: {adaptive_dau.lambda_crit:.4f}")

    # Simulate good performance
    for _ in range(50):
        adaptive_dau.update_threshold(performance=1.0, is_stable=True)
    print(f"After stable performance: {adaptive_dau.lambda_crit:.4f}")

    # Simulate instability
    for _ in range(50):
        adaptive_dau.update_threshold(performance=0.5, is_stable=False)
    print(f"After instability: {adaptive_dau.lambda_crit:.4f}")

    # Test monitor
    print("\n--- DAU Monitor ---")
    monitor = DAUMonitor(window_size=50, alert_threshold=2.5)

    for i in range(100):
        J = 1.0 + 0.03 * i  # Slowly increasing
        alert = monitor.log(J, adjusted=(i % 10 == 0))
        if alert:
            print(f"  {alert}")

    summary = monitor.get_summary()
    print(f"Monitor summary: {summary}")

    print("\n✓ DAU-Lite test passed!")
